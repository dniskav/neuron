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
//   4. control - the same update with its delta negated by hand follows -dL/dw
//                (cosine ~ +1): proves the oracle itself is sound and that the
//                only defect is the sign.
// AUDIT C1: train() builds dPred = 2*(pred - target)/n (= +dL/dpred) and hands it to
// optimizers that do `w + lr*g`, so every weight moves UP the loss (cosine -1).
// Remove `.fails` from oracle/learns when the defect is fixed (feature 003).
import { describe, it, expect } from 'vitest'
import { NetworkTransformerRL } from '../../src/NetworkTransformerRL'
import { expectGradientMatches, gradientOracle, withPlainSGD } from '../helpers/gradientOracle'
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

/** Real update with its delta negated by hand: w1 = w0 - (w1 - w0). */
function negatedStep(net: NetworkTransformerRL, lr: number): void {
  const before = net.getWeights()
  step(net, lr)
  const after = net.getWeights()
  net.setWeights(before.map((b, i) => b - (after[i] - b)))
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

  // AUDIT C1: src/NetworkTransformerRL.ts dPred = 2*(pred - target)/n has the wrong
  // sign for the `w + lr*g` optimizers (measured cosine -1).
  it.fails('oracle: train() follows -dL/dw from finite differences (cosine > 0.999)', () => {
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

  // AUDIT C1: the loss grows instead of dropping.
  it.fails('learns: the loss drops below half after a few hundred steps', () => {
    const net = build()
    expectLearns({
      loss: () => loss(net),
      step: () => step(net, 0.01),
      steps: 300,
    })
  })

  it('control: with the delta negated by hand the update follows -dL/dw (cosine ~ +1)', () => {
    const net = build()
    const r = gradientOracle({
      getParams: () => net.getWeights(),
      setParams: (p) => net.setWeights(p),
      loss: () => loss(net),
      step: (lr) => negatedStep(net, lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })
})
