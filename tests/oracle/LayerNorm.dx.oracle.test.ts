// Input-sensitivity oracle for LayerNorm (feature 003, T008, finding N1).
//
// LayerNorm.backwardOne() is called once per position and updates gamma/beta each time.
// The dX of position i+1 must still use the gamma that the forward pass saw, so dX
// does not depend on lr.
//   1. control  - with lr = 0, dX matches -dL/dX from central finite differences
//                 (cosine > 0.999).
//   2. property - two identical copies, one backward sequence with lr = 0 and one with
//                 lr > 0, must return the same dX at every position.
// Convention: backwardOne receives dOut = -dL/dy and returns dX = -dL/dX.
// LayerNorm uses no randomness: the initial weights are fixed.
import { describe, it, expect } from 'vitest'
import { LayerNorm } from '../../src/LayerNorm'

const X: number[][] = [
  [1.5, -0.9, 2.4, 0.6],
  [-1.2, 1.8, 0.3, -2.1],
  [2.7, 0.6, -1.5, 0.9],
]
const T: number[][] = [
  [0.8, -0.4, 1.1, -0.7],
  [-0.9, 0.5, -0.3, 1.0],
  [0.2, 1.2, -0.8, -0.5],
]
const GAMMA0 = [1.5, 0.8, 1.2, 0.6]
const BETA0 = [0.1, -0.2, 0.3, 0.0]
const LR = 0.05
const EPS = 1e-5

function build(): LayerNorm {
  const norm = new LayerNorm(4)
  norm.setWeights([...GAMMA0, ...BETA0])
  return norm
}

function forward(norm: LayerNorm, xs: number[][]): number[][] {
  norm.resetCache(xs.length)
  return xs.map((x, pos) => norm.predictOne(x, pos))
}

/** L = 0.5 * sum (y - T)^2, measured with the forward pass only. */
function loss(norm: LayerNorm, xs: number[][]): number {
  const ys = forward(norm, xs)
  let s = 0
  ys.forEach((row, t) => row.forEach((v, i) => (s += 0.5 * (v - T[t][i]) ** 2)))
  return s
}

/** Forward, then backwardOne per position with dOut = -(y - T); returns the dX list. */
function dXOf(norm: LayerNorm, lr: number): number[][] {
  const ys = forward(norm, X)
  return ys.map((y, p) =>
    norm.backwardOne(
      y.map((v, i) => -(v - T[p][i])),
      p,
      lr,
    ),
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

describe('LayerNorm dX (N1)', () => {
  it('control: with lr = 0, dX matches -dL/dX from finite differences', () => {
    const norm = build()
    const fd = X.map((row, t) =>
      row.map((_, i) => {
        const xp = X.map((r) => [...r])
        const xm = X.map((r) => [...r])
        xp[t][i] += EPS
        xm[t][i] -= EPS
        return -(loss(norm, xp) - loss(norm, xm)) / (2 * EPS)
      }),
    )
    expect(cosine(dXOf(norm, 0), fd)).toBeGreaterThan(0.999)
  })

  it('property: dX with lr > 0 equals dX with lr = 0', () => {
    const dX0 = dXOf(build(), 0).flat()
    const dXlr = dXOf(build(), LR).flat()
    const scale = Math.max(...dX0.map(Math.abs))
    const diff = Math.max(...dXlr.map((x, i) => Math.abs(x - dX0[i]))) / scale
    expect(diff).toBeLessThan(1e-9)
  })
})
