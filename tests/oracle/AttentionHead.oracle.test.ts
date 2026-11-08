// Gradient-oracle test for AttentionHead (feature 002, T014).
//
// Template: tests/oracle/NetworkN.oracle.test.ts and GRU.oracle.test.ts.
//   1. smoke   - predicts a 3-token sequence with finite outputs and backward()
//                returns finite dX and moves the weights.
//   2. oracle  - expectGradientMatches: the update backward() really performs
//                (Wq, Wk, Wv through the plain `w + lr*g` step) follows -dL/dw
//                from central finite differences (cosine ~ 1).
//                Loss is L = 0.5 * sum (out)^2 over the output of predict().
//                backward receives dOut = -dL/dout = -out (negated convention).
//   3. learns  - expectLearns: the same loss, measured outside backward on fixed
//                data, drops below half after a few hundred steps.
// AttentionHead stores Wq/Wk/Wv as WeightMatrix (Adam), so every real update runs
// inside withPlainSGD (Q1). The cosine is the only criterion (Q9: the ratio/scale
// is not asserted). A fixed seed (tests/setup.ts) fixes the initial weights.
import { describe, it, expect } from 'vitest'
import { AttentionHead } from '../../src/AttentionHead'
import { expectGradientMatches, withPlainSGD } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const D_MODEL = 4
const D_K = 3
const D_V = 2
// Magnitudes ~1-3 so the scores are far from uniform: otherwise dQ/dWq is tiny
// and a sign error in the Wq block barely moves the cosine (see the mutation run).
const X: number[][] = [
  [1.5, -0.9, 2.4, 0.6],
  [-1.2, 1.8, 0.3, -2.1],
  [2.7, 0.6, -1.5, 0.9],
]
const LR = 0.05

/** L = 0.5 * sum_t sum_i out[t][i]^2, measured with predict() only. */
function loss(head: AttentionHead): number {
  const out = head.predict(X)
  let s = 0
  for (const row of out) for (const v of row) s += 0.5 * v * v
  return s
}

/** One real update: forward, then backward with dOut = -dL/dout = -out. */
function step(head: AttentionHead, lr: number): void {
  withPlainSGD(() => {
    const out = head.predict(X)
    head.backward(
      out.map((row) => row.map((v) => -v)),
      lr,
    )
  })
}

function build(): AttentionHead {
  return new AttentionHead(D_MODEL, D_K, D_V, false)
}

describe('AttentionHead oracle', () => {
  it('smoke: predicts finite outputs and backward moves the weights', () => {
    const head = build()
    const out = head.predict(X)
    expect(out).toHaveLength(X.length)
    for (const row of out) {
      expect(row).toHaveLength(D_V)
      expect(row.every(Number.isFinite)).toBe(true)
    }
    const before = head.getWeights()
    const dOut = out.map((row) => row.map((v) => -v))
    const dX = withPlainSGD(() => head.backward(dOut, LR))
    expect(dX).toHaveLength(X.length)
    for (const row of dX) {
      expect(row).toHaveLength(D_MODEL)
      expect(row.every(Number.isFinite)).toBe(true)
    }
    const after = head.getWeights()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: backward() follows -dL/dw from finite differences', () => {
    const head = build()
    const r = expectGradientMatches({
      getParams: () => head.getWeights(),
      setParams: (p) => head.setWeights(p),
      loss: () => loss(head),
      step: (lr) => step(head, lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: the output loss drops below half after a few hundred steps', () => {
    const head = build()
    expectLearns({
      loss: () => loss(head),
      step: () => step(head, 0.05),
      steps: 300,
    })
  })
})
