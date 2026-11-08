// Input-sensitivity oracle for the attention family (feature 002, T040, finding N1).
//
// Template: the dX sensitivity check of tests/oracle/AttentionHead.oracle.test.ts.
// AttentionHead.backward() computes dX from the pre-update weights (N1 fixed), so dX
// does not depend on lr.
//   1. control  - with lr = 0, dX matches -dL/dX from central finite differences
//                 (cosine > 0.999).
//   2. property - two identical copies, one backward with lr = 0 and one with
//                 lr > 0, must return the same dX (normal test).
// TransformerBlock used to depend on lr through LayerNorm.backwardOne (gamma updated at
// pos i was read at pos i+1); LayerNorm now keeps the forward-time gamma (T008).
// withPlainSGD makes the lr > 0 step deterministic (Adam would hide the size of lr).
// Convention: backward receives dOut = -dL/dout and returns dX = -dL/dX.
// A fixed seed (tests/setup.ts) fixes the initial weights.
import { describe, it, expect } from 'vitest'
import { MultiHeadAttention } from '../../src/MultiHeadAttention'
import { withPlainSGD } from '../helpers/gradientOracle'

const X: number[][] = [
  [1.5, -0.9, 2.4, 0.6],
  [-1.2, 1.8, 0.3, -2.1],
  [2.7, 0.6, -1.5, 0.9],
]
const LR = 0.05
const EPS = 1e-5

interface Model {
  predict(x: number[][]): number[][]
  backward(dOut: number[][], lr: number): number[][]
  getWeights(): number[]
  setWeights(w: number[]): void
}

/** Fixed target with magnitude ~1 and the shape of `out`. */
function target(out: number[][]): number[][] {
  return out.map((row, t) => row.map((_, i) => 0.9 * Math.sin(1 + 3 * t + i)))
}

/** L = 0.5 * sum (out - T)^2, measured with predict() only. */
function loss(m: Model, x: number[][]): number {
  const out = m.predict(x)
  const T = target(out)
  let s = 0
  out.forEach((row, t) => row.forEach((v, i) => (s += 0.5 * (v - T[t][i]) ** 2)))
  return s
}

/** backward() with dOut = -dL/dout = -(out - T); returns the dX it reports. */
function dXOf(m: Model, lr: number): number[][] {
  return withPlainSGD(() => {
    const out = m.predict(X)
    const T = target(out)
    return m.backward(
      out.map((row, t) => row.map((v, i) => -(v - T[t][i]))),
      lr,
    )
  })
}

/** -dL/dX by central differences on the input. */
function fdMinusDX(m: Model): number[][] {
  return X.map((row, t) =>
    row.map((_, i) => {
      const xp = X.map((r) => [...r])
      const xm = X.map((r) => [...r])
      xp[t][i] += EPS
      xm[t][i] -= EPS
      return -(loss(m, xp) - loss(m, xm)) / (2 * EPS)
    }),
  )
}

function cosine(a: number[][], b: number[][]): number {
  const u = a.flat()
  const v = b.flat()
  let dot = 0
  let nu = 0
  let nv = 0
  for (let i = 0; i < u.length; i++) {
    dot += u[i] * v[i]
    nu += u[i] * u[i]
    nv += v[i] * v[i]
  }
  return dot / (Math.sqrt(nu) * Math.sqrt(nv))
}

/** Largest |a - b| relative to the largest |b|. */
function maxRelDiff(a: number[][], b: number[][]): number {
  const u = a.flat()
  const v = b.flat()
  const scale = Math.max(...v.map(Math.abs))
  return Math.max(...u.map((x, i) => Math.abs(x - v[i]))) / scale
}

const MODELS: { name: string; build: () => Model }[] = [
  { name: 'MultiHeadAttention', build: () => new MultiHeadAttention(4, 2, false) },
]

for (const { name, build } of MODELS) {
  describe(`${name} dX (N1)`, () => {
    it('control: with lr = 0, dX matches -dL/dX from finite differences', () => {
      const m = build()
      const fd = fdMinusDX(m)
      const dX = dXOf(m, 0)
      expect(cosine(dX, fd)).toBeGreaterThan(0.999)
    })

    it('property: dX with lr > 0 equals dX with lr = 0', () => {
      const a = build()
      const b = build()
      b.setWeights(a.getWeights())
      const dX0 = dXOf(a, 0)
      const dXlr = dXOf(b, LR)
      expect(maxRelDiff(dXlr, dX0)).toBeLessThan(1e-9)
    })
  })
}
