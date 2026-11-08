// Gradient-oracle test for the update rules in src/MatMul.ts (feature 002, T033).
//
// Sign convention (constitution): update() receives g = -dL/dw and moves the
// weights by `w + lr * g`. WeightMatrix and BiasVector go through Adam, so the
// real update runs inside withPlainSGD (Q1); EmbeddingMatrix is plain SGD already.
//
// clipValue convention (WeightMatrix.update only): per element, BEFORE the
// optimizer step, g' = clamp(g, -clipValue, +clipValue). It is not a norm clip and
// the default Infinity disables it. BiasVector.update has no clip argument.
//
// Loss used for the fit: L = 0.5 * sum_s |W x_s - t_s|^2, so -dL/dW = sum_s (t_s - W x_s) x_s^T.
import { describe, it, expect } from 'vitest'
import { WeightMatrix, BiasVector, EmbeddingMatrix } from '../../src/MatMul'
import { expectGradientMatches, withPlainSGD, bindParams } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const LR = 1e-3
const XS: number[][] = [
  [1.5, -0.9, 2.4],
  [-1.2, 1.8, 0.3],
  [0.7, 0.6, -1.5],
]
const TS: number[][] = [
  [1.0, -2.0],
  [0.5, 1.5],
  [-1.0, 0.25],
]

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)

function wmLoss(m: WeightMatrix): number {
  let s = 0
  XS.forEach((x, k) =>
    m.W.forEach((row, i) => {
      const r = dot(row, x) - TS[k][i]
      s += 0.5 * r * r
    }),
  )
  return s
}

/** -dL/dW, shape of W. */
function wmNegGrad(m: WeightMatrix): number[][] {
  const g = m.W.map((row) => row.map(() => 0))
  XS.forEach((x, k) =>
    m.W.forEach((row, i) => {
      const r = TS[k][i] - dot(row, x)
      x.forEach((xj, j) => (g[i][j] += r * xj))
    }),
  )
  return g
}

describe('MatMul update oracle', () => {
  it('smoke: updates are finite and move the parameters', () => {
    const w = new WeightMatrix(2, 3)
    const w0 = w.getWeights()
    w.update([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], 0.01)
    expect(w.getWeights().every(Number.isFinite)).toBe(true)
    expect(w.getWeights().some((v, i) => v !== w0[i])).toBe(true)

    const b = new BiasVector(3)
    b.update([0.1, -0.2, 0.3], 0.01)
    expect(b.values.every(Number.isFinite)).toBe(true)
    expect(b.values.some((v) => v !== 0)).toBe(true)

    const e = new EmbeddingMatrix(4, 3)
    const e0 = e.get(2)
    e.update(2, [1, 1, 1], 0.1)
    expect(e.get(2)).not.toEqual(e0)
  })

  it('WeightMatrix: plain-SGD update moves W exactly lr*g', () => {
    const m = new WeightMatrix(2, 3)
    const before = m.W.map((r) => [...r])
    const g = [[0.5, -1.5, 2], [0, 3, -0.25]]
    withPlainSGD(() => m.update(g, 0.1))
    m.W.forEach((row, i) =>
      row.forEach((v, j) => expect(v).toBeCloseTo(before[i][j] + 0.1 * g[i][j], 12)),
    )
  })

  it('WeightMatrix: clipValue clamps per element before the step (exact values)', () => {
    const m = new WeightMatrix(2, 3)
    const before = m.W.map((r) => [...r])
    const g = [[5, -5, 0.3], [-0.9, 1.0, 100]]
    withPlainSGD(() => m.update(g, 0.1, 1))
    const expected = [[1, -1, 0.3], [-0.9, 1, 1]]
    m.W.forEach((row, i) =>
      row.forEach((v, j) => expect(v).toBeCloseTo(before[i][j] + 0.1 * expected[i][j], 12)),
    )
    // Default (Infinity) does not clip.
    const m2 = new WeightMatrix(1, 1)
    const b2 = m2.W[0][0]
    withPlainSGD(() => m2.update([[100]], 0.1))
    expect(m2.W[0][0]).toBeCloseTo(b2 + 10, 12)
  })

  it('BiasVector: plain-SGD update moves values exactly lr*g', () => {
    const b = new BiasVector(3)
    b.values = [1, 2, 3]
    withPlainSGD(() => b.update([0.5, -1, 2], 0.1))
    expect(b.values[0]).toBeCloseTo(1.05, 12)
    expect(b.values[1]).toBeCloseTo(1.9, 12)
    expect(b.values[2]).toBeCloseTo(3.2, 12)
  })

  it('EmbeddingMatrix: SGD update is W += lr*grad on that row only', () => {
    const e = new EmbeddingMatrix(3, 2)
    const before = e.W.map((r) => [...r])
    e.update(1, [2, -4], 0.25)
    expect(e.W[1][0]).toBeCloseTo(before[1][0] + 0.5, 12)
    expect(e.W[1][1]).toBeCloseTo(before[1][1] - 1, 12)
    expect(e.W[0]).toEqual(before[0])
    expect(e.W[2]).toEqual(before[2])
  })

  it('oracle: WeightMatrix.update follows -dL/dW from finite differences', () => {
    const m = new WeightMatrix(2, 3)
    const r = expectGradientMatches({
      ...bindParams(m.W),
      loss: () => wmLoss(m),
      step: (lr) => withPlainSGD(() => m.update(wmNegGrad(m), lr)),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('oracle: BiasVector.update follows -dL/db from finite differences', () => {
    // L = 0.5 * sum_i (b_i - t_i)^2 with t = [1, -2, 0.5, 3]
    const t = [1, -2, 0.5, 3]
    const b = new BiasVector(4)
    const r = expectGradientMatches({
      ...bindParams(b.values),
      loss: () => b.values.reduce((s, v, i) => s + 0.5 * (v - t[i]) ** 2, 0),
      step: (lr) => withPlainSGD(() => b.update(b.values.map((v, i) => t[i] - v), lr)),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('oracle: EmbeddingMatrix.update follows -dL/dE from finite differences', () => {
    // L = 0.5 * sum over looked-up rows |E[idx] - t|^2 for idx 0 and 2 (row 1 untouched).
    const e = new EmbeddingMatrix(3, 3)
    const t = [0.5, -1, 2]
    const rows = [0, 2]
    const r = expectGradientMatches({
      ...bindParams(e.W),
      loss: () =>
        rows.reduce((s, i) => s + e.W[i].reduce((a, v, j) => a + 0.5 * (v - t[j]) ** 2, 0), 0),
      step: (lr) => rows.forEach((i) => e.update(i, e.W[i].map((v, j) => t[j] - v), lr)),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: WeightMatrix fits W*x to a target', () => {
    const m = new WeightMatrix(2, 3)
    expectLearns({
      loss: () => wmLoss(m),
      step: () => withPlainSGD(() => m.update(wmNegGrad(m), 0.05)),
      steps: 100,
    })
  })

  it('learns: Adam-backed WeightMatrix (real optimizer) fits W*x to a target', () => {
    const m = new WeightMatrix(2, 3)
    expectLearns({
      loss: () => wmLoss(m),
      step: () => m.update(wmNegGrad(m), 0.05),
      steps: 200,
    })
  })

  it('learns: BiasVector and EmbeddingMatrix fit a target', () => {
    const t = [1, -2, 0.5]
    const b = new BiasVector(3)
    expectLearns({
      loss: () => b.values.reduce((s, v, i) => s + 0.5 * (v - t[i]) ** 2, 0),
      step: () => b.update(b.values.map((v, i) => t[i] - v), 0.05),
      steps: 200,
    })
    const e = new EmbeddingMatrix(2, 3)
    expectLearns({
      loss: () => e.W[1].reduce((s, v, i) => s + 0.5 * (v - t[i]) ** 2, 0),
      step: () => e.update(1, e.W[1].map((v, i) => t[i] - v), 0.1),
      steps: 50,
    })
  })
})
