// Gradient-oracle test for RNN (feature 002, T021).
//
//   1. smoke   - forward() gives finite outputs and backward() returns the MSE it
//                reports and moves the weights (small lr: no Infinity).
//   2. oracle  - expectGradientMatches: the update backward() really performs follows
//                -dL/dw from central finite differences. Loss is the value backward()
//                returns, L = (1/(T*out)) sum (o - y)^2, measured with forward() only.
//                Default SGD optimizer (no withPlainSGD). Only the cosine is asserted (Q9).
//   3. learns  - expectLearns: the same loss on fixed data drops below half.
// backward() passes dOutputs = -dL/do (= -2*(o - y)/out), so the optimizers' `w + lr*g`
// step follows -dL/dw.
import { describe, it, expect } from 'vitest'
import { RNN } from '../../src/RNN'
import { bindParams, expectGradientMatches } from '../helpers/gradientOracle'
import { withSeed } from '../helpers/seed'
import { expectLearns } from '../helpers/learns'

const SEQ: number[][] = [
  [0.5, -0.8],
  [-0.4, 0.6],
  [0.9, 0.1],
  [0.2, -0.7],
]
const TARGETS: number[][] = [[0.4], [-0.3], [0.5], [-0.2]]
const LR = 1e-3

function build(outSize = 1): RNN {
  return new RNN(2, 3, outSize)
}

function params(net: RNN) {
  return bindParams(net.Wxh, net.Whh, net.Why, net.bh, net.by)
}

/** Same MSE backward() returns, measured with forward() only. */
function loss(net: RNN): number {
  net.reset()
  const { outputs } = net.forward(SEQ)
  let s = 0
  outputs.forEach((o, t) => o.forEach((v, k) => (s += (v - TARGETS[t][k]) ** 2)))
  return s / (SEQ.length * outputs[0].length)
}

describe('RNN oracle', () => {
  it('smoke: forward is finite, backward reports the MSE and moves the weights', () => {
    const net = build()
    net.reset()
    const { outputs, hiddens } = net.forward(SEQ)
    expect(outputs).toHaveLength(SEQ.length)
    expect(outputs.flat().every(Number.isFinite)).toBe(true)
    expect(hiddens.flat().every((v) => Math.abs(v) < 1)).toBe(true)

    const p = params(net)
    const before = p.getParams()
    const expected = loss(net)
    const reported = net.backward(SEQ, TARGETS, LR)
    expect(Number.isFinite(reported)).toBe(true)
    expect(reported).toBeCloseTo(expected, 12)
    const after = p.getParams()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: backward() follows -dL/dw from finite differences (cosine > 0.999)', () => {
    const net = build()
    const r = expectGradientMatches({
      ...params(net),
      loss: () => loss(net),
      step: (lr) => {
        net.backward(SEQ, TARGETS, lr)
      },
      lr: LR,
      resetRng: () => net.reset(),
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  // outputSize 2 and 3 (closes mutant M14: no /outputSize in dOutputs). The step has the
  // scale of the mean loss, so the ratio is checked too (lr x0.97/x1.03, 5 seeds).
  const MULTI_TARGETS: Record<number, number[][]> = {
    2: [[0.4, -0.5], [-0.3, 0.2], [0.5, 0.6], [-0.2, -0.4]],
    3: [[0.4, -0.5, 0.1], [-0.3, 0.2, -0.6], [0.5, 0.6, 0.3], [-0.2, -0.4, 0.7]],
  }
  for (const n of [2, 3]) {
    it(`oracle with outputSize ${n}: cosine > 0.999 and ratio in (0.9, 1.1)`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        for (const f of [0.97, 1.03]) {
          withSeed(seed, () => {
            const net = build(n)
            const targets = MULTI_TARGETS[n]
            const r = expectGradientMatches({
              ...params(net),
              loss: () => {
                net.reset()
                const { outputs } = net.forward(SEQ)
                let s = 0
                outputs.forEach((o, t) => o.forEach((v, k) => (s += (v - targets[t][k]) ** 2)))
                return s / (SEQ.length * n)
              },
              step: (lr) => {
                net.backward(SEQ, targets, lr)
              },
              lr: LR * f,
              resetRng: () => net.reset(),
            })
            expect(r.cosine).toBeGreaterThan(0.999)
            expect(r.ratio).toBeGreaterThan(0.9)
            expect(r.ratio).toBeLessThan(1.1)
          })
        }
      }
    })
  }

  it('learns: the loss drops below half after a few hundred steps', () => {
    const net = build()
    expectLearns({
      loss: () => loss(net),
      step: () => {
        net.backward(SEQ, TARGETS, 0.05)
      },
      steps: 300,
    })
  })
})
