// Oráculo de dx de Flatten (feature 006, T014, US4/FR-006).
//
// Flatten no tiene parámetros ni optimizador, así que la convención aquí es la
// del gradiente de entrada: backward(dOut) con dOut = dL/dy devuelve dL/dx
// (sin negar). Se compara componente a componente con diferencias finitas
// centrales de L(x), con |a − b| < 1e-6.
//
// Dos pérdidas:
//   L1(x) = Σ_k c_k · flat_k(x)        (lineal: dL1/dx = c reordenado)
//   L2(x) = Σ_k c_k · flat_k(x)²       (no lineal: verifica la regla de la cadena)
// Para L2, dOut_k = 2 · c_k · flat_k.
//
// Los valores esperados se deducen a mano en cada `it`.
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Flatten } from '../../src/Flatten'
import { restoreRandom, seedRandom } from '../helpers/seed'

beforeEach(() => seedRandom(2024))
afterEach(() => restoreRandom())

const TOL = 1e-6
const EPS = 1e-5

// Entrada no cuadrada 2×3×2 con valores no enteros.
const X0: number[][][] = [
  [
    [0.5, -1.25],
    [2.0, 0.75],
    [-0.4, 1.1],
  ],
  [
    [0.9, -0.3],
    [-1.6, 0.2],
    [1.3, -0.7],
  ],
]

// Pesos fijos c_k, con k = h·W·C + w·C + c (orden de salida de Flatten).
const CW = [0.5, -1, 2, 0.25, -0.75, 1.5, 3, -2, 0.1, -0.6, 1, 0.8]

function clone(x: number[][][]): number[][][] {
  return x.map((row) => row.map((p) => [...p]))
}

function flatOf(x: number[][][]): number[] {
  return new Flatten().forward(x)
}

/** Lineal: L1 = Σ c_k · flat_k. */
function L1(x: number[][][]): number {
  return flatOf(x).reduce((s, v, k) => s + CW[k] * v, 0)
}

/** No lineal: L2 = Σ c_k · flat_k². */
function L2(x: number[][][]): number {
  return flatOf(x).reduce((s, v, k) => s + CW[k] * v * v, 0)
}

/** Derivada numérica dL/dx por diferencias finitas centrales, componente a componente. */
function finiteDiff(L: (x: number[][][]) => number, x: number[][][]): number[][][] {
  return x.map((row, h) =>
    row.map((p, w) =>
      p.map((_, c) => {
        const xp = clone(x)
        const xm = clone(x)
        xp[h][w][c] += EPS
        xm[h][w][c] -= EPS
        return (L(xp) - L(xm)) / (2 * EPS)
      }),
    ),
  )
}

function flat3(t: number[][][]): number[] {
  return t.flatMap((row) => row.flatMap((p) => p))
}

function maxAbsDiff(a: number[][][], b: number[][][]): number {
  const u = flat3(a)
  const v = flat3(b)
  expect(u).toHaveLength(v.length)
  return Math.max(...u.map((x, i) => Math.abs(x - v[i])))
}

describe('Flatten dx oracle', () => {
  it('L1 lineal: backward(c) coincide con diferencias finitas por componente', () => {
    const fd = finiteDiff(L1, X0)
    const bw = new Flatten()
    bw.forward(X0)
    const dx = bw.backward(CW)
    expect(flat3(fd)).toHaveLength(12)
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)
  })

  it('L1 lineal: el gradiente esperado a mano es c reordenado por (h, w, c)', () => {
    // dL1/dx[h][w][c] = c_k con k = 6h + 2w + c (permutación identidad sobre la forma).
    // h0: w0 → [c0, c1] = [0.5, -1]; w1 → [c2, c3] = [2, 0.25]; w2 → [c4, c5] = [-0.75, 1.5]
    // h1: w0 → [c6, c7] = [3, -2];  w1 → [c8, c9] = [0.1, -0.6]; w2 → [c10, c11] = [1, 0.8]
    const expected = [
      [
        [0.5, -1],
        [2, 0.25],
        [-0.75, 1.5],
      ],
      [
        [3, -2],
        [0.1, -0.6],
        [1, 0.8],
      ],
    ]
    const bw = new Flatten()
    bw.forward(X0)
    const dx = bw.backward(CW)
    expect(maxAbsDiff(dx, expected)).toBeLessThan(TOL)
    expect(maxAbsDiff(finiteDiff(L1, X0), expected)).toBeLessThan(TOL)
  })

  it('L2 no lineal: backward(2·c·flat) coincide con diferencias finitas por componente', () => {
    // flat_k de X0 (k = 0..11): 0.5, −1.25, 2.0, 0.75, −0.4, 1.1, 0.9, −0.3, −1.6, 0.2, 1.3, −0.7.
    // dOut_k = 2·c_k·flat_k:
    //   k0: 2·0.5·0.5 = 0.5        k6: 2·3·0.9 = 5.4
    //   k1: 2·(−1)·(−1.25) = 2.5   k7: 2·(−2)·(−0.3) = 1.2
    //   k2: 2·2·2.0 = 8            k8: 2·0.1·(−1.6) = −0.32
    //   k3: 2·0.25·0.75 = 0.375    k9: 2·(−0.6)·0.2 = −0.24
    //   k4: 2·(−0.75)·(−0.4) = 0.6 k10: 2·1·1.3 = 2.6
    //   k5: 2·1.5·1.1 = 3.3        k11: 2·0.8·(−0.7) = −1.12
    const dOut = [0.5, 2.5, 8, 0.375, 0.6, 3.3, 5.4, 1.2, -0.32, -0.24, 2.6, -1.12]
    const bw = new Flatten()
    bw.forward(X0)
    const dx = bw.backward(dOut)
    const fd = finiteDiff(L2, X0)
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)

    // Mismo resultado a mano: el dOut anterior, reordenado por (h, w, c).
    const expected = [
      [
        [0.5, 2.5],
        [8, 0.375],
        [0.6, 3.3],
      ],
      [
        [5.4, 1.2],
        [-0.32, -0.24],
        [2.6, -1.12],
      ],
    ]
    expect(maxAbsDiff(dx, expected)).toBeLessThan(TOL)
  })
})
