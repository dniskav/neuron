// Gradient-oracle test for MultiHeadAttention (feature 002, T015).
//
// Template: tests/oracle/AttentionHead.oracle.test.ts (same family).
//   1. smoke   - one-token case with hand-computed output (softmax over one key is 1,
//                so each head returns its V row and out = Wo · [V0 | V1]); backward
//                returns finite dX and moves the weights.
//   2. oracle  - expectGradientMatches over ALL weights (heads Wq/Wk/Wv and Wo, via
//                getWeights/setWeights): the update backward() really performs
//                (plain `w + lr*g` step) follows -dL/dw from central finite differences.
//                Loss is L = 0.5 * sum (out)^2 over predict(); backward receives
//                dOut = -dL/dout = -out (negated convention).
//   3. learns  - expectLearns: the same loss, measured outside backward on fixed
//                data, drops below half after a few hundred steps.
// The cosine is the only criterion (Q9: the ratio/scale is not asserted).
// Every weight in MultiHeadAttention is a WeightMatrix (Adam), so every real update
// runs inside withPlainSGD (Q1). A fixed seed (tests/setup.ts) fixes the initial weights.
// Inputs have magnitudes ~1-3 so the scores are far from uniform (see AttentionHead).
import { describe, it, expect } from 'vitest'
import { MultiHeadAttention } from '../../src/MultiHeadAttention'
import { expectGradientMatches, withPlainSGD } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const D_MODEL = 4
const N_HEADS = 2
const X: number[][] = [
  [1.5, -0.9, 2.4, 0.6],
  [-1.2, 1.8, 0.3, -2.1],
  [2.7, 0.6, -1.5, 0.9],
]
const LR = 1e-3

/** L = 0.5 * sum_t sum_i out[t][i]^2, measured with predict() only. */
function loss(mha: MultiHeadAttention): number {
  const out = mha.predict(X)
  let s = 0
  for (const row of out) for (const v of row) s += 0.5 * v * v
  return s
}

/** One real update: forward, then backward with dOut = -dL/dout = -out. */
function step(mha: MultiHeadAttention, lr: number): void {
  withPlainSGD(() => {
    const out = mha.predict(X)
    mha.backward(
      out.map((row) => row.map((v) => -v)),
      lr,
    )
  })
}

function build(): MultiHeadAttention {
  return new MultiHeadAttention(D_MODEL, N_HEADS, false)
}

describe('MultiHeadAttention oracle', () => {
  it('smoke: one-token output matches the hand computation and backward moves the weights', () => {
    const mha = build()
    // Flat layout (see getWeights): head0 Wq(2x4), Wk(2x4), Wv(2x4), head1 Wq, Wk, Wv, then Wo(4x4).
    const zeros8 = [0, 0, 0, 0, 0, 0, 0, 0]
    const wv0 = [1, 0, 0, 0, 0, 1, 0, 0] // V0 = [x0, x1]
    const wv1 = [0, 0, 1, 0, 0, 0, 0, 1] // V1 = [x2, x3]
    const wo = [
      1, 1, 0, 0, // out0 = V0[0] + V0[1]
      0, 0, 1, 0, // out1 = V1[0]
      0, 0, 0, 2, // out2 = 2 * V1[1]
      1, 0, 0, -1, // out3 = V0[0] - V1[1]
    ]
    mha.setWeights([...zeros8, ...zeros8, ...wv0, ...zeros8, ...zeros8, ...wv1, ...wo])

    // x = [1, 2, 0, -1]: V0 = [1, 2], V1 = [0, -1], concat = [1, 2, 0, -1].
    // out = [1+2, 0, 2*(-1), 1+1] = [3, 0, -2, 2].
    const out = mha.predict([[1, 2, 0, -1]])
    expect(out).toHaveLength(1)
    const expected = [3, 0, -2, 2]
    for (let i = 0; i < D_MODEL; i++) expect(out[0][i]).toBeCloseTo(expected[i], 10)

    // Fresh head with random weights: finite outputs, finite dX, weights move.
    const fresh = build()
    const freshOut = fresh.predict(X)
    expect(freshOut).toHaveLength(X.length)
    for (const row of freshOut) {
      expect(row).toHaveLength(D_MODEL)
      expect(row.every(Number.isFinite)).toBe(true)
    }
    const before = fresh.getWeights()
    const dOut = freshOut.map((row) => row.map((v) => -v))
    const dX = withPlainSGD(() => fresh.backward(dOut, LR))
    expect(dX).toHaveLength(X.length)
    for (const row of dX) {
      expect(row).toHaveLength(D_MODEL)
      expect(row.every(Number.isFinite)).toBe(true)
    }
    const after = fresh.getWeights()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: backward() follows -dL/dw from finite differences (heads and Wo)', () => {
    const mha = build()
    const r = expectGradientMatches({
      getParams: () => mha.getWeights(),
      setParams: (p) => mha.setWeights(p),
      loss: () => loss(mha),
      step: (lr) => step(mha, lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: the output loss drops below half after a few hundred steps', () => {
    const mha = build()
    expectLearns({
      loss: () => loss(mha),
      step: () => step(mha, 0.05),
      steps: 300,
    })
  })
})
