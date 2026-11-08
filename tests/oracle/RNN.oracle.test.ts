// Gradient-oracle test for RNN (feature 002, T021).
//
//   1. smoke   - forward() gives finite outputs and backward() returns the MSE it
//                reports and moves the weights (small lr: no Infinity).
//   2. oracle  - expectGradientMatches: the update backward() really performs follows
//                -dL/dw from central finite differences. Loss is the value backward()
//                returns, L = (1/(T*out)) sum (o - y)^2, measured with forward() only.
//                Default SGD optimizer (no withPlainSGD). Only the cosine is asserted (Q9).
//   3. learns  - expectLearns: the same loss on fixed data drops below half.
//   4. control - the same update with its delta negated by hand follows -dL/dw
//                (cosine ~ +1): the oracle is sound and the only defect is the sign.
// AUDIT C1: backward() builds dOutputs = 2*(o - y)/out (= +dL/do) and hands it to
// optimizers that do `w + lr*g`, so every weight moves UP the loss (cosine -1).
// Remove `.fails` from oracle/learns when the defect is fixed (feature 003).
import { describe, it, expect } from 'vitest'
import { RNN } from '../../src/RNN'
import { bindParams, expectGradientMatches, gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const SEQ: number[][] = [
  [0.5, -0.8],
  [-0.4, 0.6],
  [0.9, 0.1],
  [0.2, -0.7],
]
const TARGETS: number[][] = [[0.4], [-0.3], [0.5], [-0.2]]
const LR = 1e-3

function build(): RNN {
  return new RNN(2, 3, 1)
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

/** Real update with its delta negated by hand: w1 = w0 - (w1 - w0). */
function negatedStep(net: RNN, lr: number): void {
  const p = params(net)
  const before = p.getParams()
  net.backward(SEQ, TARGETS, lr)
  const after = p.getParams()
  p.setParams(before.map((b, i) => b - (after[i] - b)))
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

  // AUDIT C1: src/RNN.ts dOutputs = 2*diff/out has the wrong sign for the
  // `w + lr*g` optimizers (measured cosine -1).
  it.fails('oracle: backward() follows -dL/dw from finite differences (cosine > 0.999)', () => {
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

  // AUDIT C1: the loss grows instead of dropping.
  it.fails('learns: the loss drops below half after a few hundred steps', () => {
    const net = build()
    expectLearns({
      loss: () => loss(net),
      step: () => {
        net.backward(SEQ, TARGETS, 0.05)
      },
      steps: 300,
    })
  })

  it('control: with the delta negated by hand the update follows -dL/dw (cosine ~ +1)', () => {
    const net = build()
    const r = gradientOracle({
      ...params(net),
      loss: () => loss(net),
      step: (lr) => negatedStep(net, lr),
      lr: LR,
      resetRng: () => net.reset(),
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })
})
