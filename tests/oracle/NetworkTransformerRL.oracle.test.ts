// Gradient-oracle test for NetworkTransformerRL (feature 002, T020).
//
// Same structure as TransformerBlock.oracle.test.ts:
//   1. smoke   - predict() gives finite Q-values and train() returns the MSE it
//                reports and moves the weights.
//   2. oracle  - expectGradientMatches: the update train() really performs follows
//                -dL/dw from central finite differences. Loss is the MSE that
//                train() returns, measured with predict() only. Adam is replaced by
//                plain `w + lr*g` (withPlainSGD, Q1). Only the cosine is asserted (Q9).
//   3. learns  - expectLearns: the same loss on fixed data drops below half.
// C1 fixed (feature 003, T007): train() passes -dL/dpred; inputProj updated once.
import { describe, it, expect } from 'vitest'
import { NetworkTransformerRL } from '../../src/NetworkTransformerRL'
import { expectGradientMatches, withPlainSGD } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const SEQ_LEN = 3
const INPUT_DIM = 2
// Magnitudes ~1-2 so ReLU pre-activations stay away from the kink.
const SEQ: number[][] = [
  [1.2, -0.8],
  [-1.5, 0.9],
  [0.7, 1.6],
]
const TARGET = [0.8, -0.5]
const LR = 1e-3

function build(): NetworkTransformerRL {
  return new NetworkTransformerRL(SEQ_LEN, INPUT_DIM, {
    d_model: 4,
    nHeads: 2,
    d_ff: 8,
    nBlocks: 1,
    nActions: 2,
  })
}

/** Same MSE train() returns, measured with predict() only. */
function loss(net: NetworkTransformerRL): number {
  const pred = net.predict(SEQ)
  return pred.reduce((s, p, c) => s + (p - TARGET[c]) ** 2, 0) / pred.length
}

function step(net: NetworkTransformerRL, lr: number): void {
  withPlainSGD(() => {
    net.train(SEQ, TARGET, lr)
  })
}

describe('NetworkTransformerRL oracle', () => {
  it('smoke: predicts finite Q-values, train reports the MSE and moves the weights', () => {
    const net = build()
    const q = net.predict(SEQ)
    expect(q).toHaveLength(2)
    expect(q.every(Number.isFinite)).toBe(true)

    const before = net.getWeights()
    const expected = loss(net)
    const reported = withPlainSGD(() => net.train(SEQ, TARGET, LR))
    expect(Number.isFinite(reported)).toBe(true)
    expect(reported).toBeCloseTo(expected, 12)
    const after = net.getWeights()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: train() follows -dL/dw from finite differences (cosine > 0.999)', () => {
    const net = build()
    const r = expectGradientMatches({
      getParams: () => net.getWeights(),
      setParams: (p) => net.setWeights(p),
      loss: () => loss(net),
      step: (lr) => step(net, lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: the loss drops below half after a few hundred steps', () => {
    const net = build()
    expectLearns({
      loss: () => loss(net),
      step: () => step(net, 0.01),
      steps: 300,
    })
  })
})
