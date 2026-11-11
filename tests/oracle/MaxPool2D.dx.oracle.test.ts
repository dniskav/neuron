// Oráculo de dx de MaxPool2D (feature 006, T015, US4/FR-006).
//
// MaxPool2D no tiene parámetros: backward(dOut) con dOut = dL/dy devuelve dL/dx
// (sin negar). Se compara componente a componente con diferencias finitas
// centrales de L(x), con |a − b| < 1e-6.
//
// Dos pérdidas:
//   L1(x) = Σ_k c_k · out_k(x)     (dOut = c)
//   L2(x) = Σ_k c_k · out_k(x)²    (dOut = 2·c·out)
//
// Puntos de prueba: entradas sin empates, con separación ≥ 0.5 entre el máximo
// y el resto de cada ventana, de modo que una perturbación de EPS no cambie el
// argmax. Los primeros tests usan ventanas NO solapadas (stride = poolSize); el
// bloque final (T055) cubre ventanas solapadas (stride 1, poolSize 2 y 3) y un
// máximo compartido por varias ventanas (sus gradientes se suman).
//
// Los valores esperados se deducen a mano en cada `it`.
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { MaxPool2D } from '../../src/MaxPool2D'
import { restoreRandom, seedRandom } from '../helpers/seed'

beforeEach(() => seedRandom(2024))
afterEach(() => restoreRandom())

const TOL = 1e-6
const EPS = 1e-5

// Entrada 4×4×2 ([h][w][c]). Canal 0 y canal 1 por separado.
// Canal 0 (bloques 2×2, máximo y segundo valor):
//   (0,0) {3,0,2,9}: 9 en (1,1)     (0,1) {7,1,4,8}: 8 en (1,3)  [seg. 7]
//   (1,0) {6,5,−2,4.5}: 6 en (2,0)  [seg. 5]  (1,1) {0.5,−1,2.5,10}: 10 en (3,3)
// Canal 1:
//   (0,0) {−0.5,2.2,0.3,−1.4}: 2.2 en (0,1)  (0,1) {1.7,−3,6.1,0.9}: 6.1 en (1,2)
//   (1,0) {4.2,1.1,0.2,5.4}: 5.4 en (3,1)    (1,1) {−2.6,3.3,0.6,−0.8}: 3.3 en (2,3)
const X: number[][][] = [
  [
    [3, -0.5],
    [0, 2.2],
    [7, 1.7],
    [1, -3],
  ],
  [
    [2, 0.3],
    [9, -1.4],
    [4, 6.1],
    [8, 0.9],
  ],
  [
    [6, 4.2],
    [5, 1.1],
    [0.5, -2.6],
    [-1, 3.3],
  ],
  [
    [-2, 0.2],
    [4.5, 5.4],
    [2.5, 0.6],
    [10, -0.8],
  ],
]

// Salida esperada a mano, [oh][ow][c]:
//   O[0][0] = [9, 2.2]   O[0][1] = [8, 6.1]
//   O[1][0] = [6, 5.4]   O[1][1] = [10, 3.3]
// Pesos c, en orden (oh, ow, c) con k = 4·oh + 2·ow + c:
const CW = [1, -2, 0.5, 3, -1.5, 0.25, 2, -0.75]

function clone(x: number[][][]): number[][][] {
  return x.map((row) => row.map((p) => [...p]))
}

function poolOf(x: number[][][]): number[][][] {
  return new MaxPool2D(2).forward(x)
}

function flat3(t: number[][][]): number[] {
  return t.flatMap((row) => row.flatMap((p) => p))
}

function maxAbsDiff(a: number[][][], b: number[][][]): number {
  const u = flat3(a)
  const v = flat3(b)
  expect(u).toHaveLength(v.length)
  return Math.max(...u.map((val, i) => Math.abs(val - v[i])))
}

/** L1 = Σ c_k · out_k. */
function L1(x: number[][][]): number {
  return flat3(poolOf(x)).reduce((s, v, k) => s + CW[k] * v, 0)
}

/** L2 = Σ c_k · out_k². */
function L2(x: number[][][]): number {
  return flat3(poolOf(x)).reduce((s, v, k) => s + CW[k] * v * v, 0)
}

/** dL/dx por diferencias finitas centrales, componente a componente. */
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

// Gradiente a mano de L1: c_k se lleva a la posición del máximo de su ventana.
// Posiciones (h, w, c) de cada máximo, en el orden de CW:
//   k0 (0,0,0)→(1,1,0)=1     k1 (0,0,1)→(0,1,1)=−2
//   k2 (0,1,0)→(1,3,0)=0.5   k3 (0,1,1)→(1,2,1)=3
//   k4 (1,0,0)→(2,0,0)=−1.5  k5 (1,0,1)→(3,1,1)=0.25
//   k6 (1,1,0)→(3,3,0)=2     k7 (1,1,1)→(2,3,1)=−0.75
const GRAD_L1 = [
  [
    [0, 0],
    [0, -2],
    [0, 0],
    [0, 0],
  ],
  [
    [0, 0],
    [1, 0],
    [0, 3],
    [0.5, 0],
  ],
  [
    [-1.5, 0],
    [0, 0],
    [0, 0],
    [0, -0.75],
  ],
  [
    [0, 0],
    [0, 0.25],
    [0, 0],
    [2, 0],
  ],
]

describe('MaxPool2D dx oracle', () => {
  it('salida de referencia a mano (base del oráculo)', () => {
    const out = poolOf(X)
    expect(out).toEqual([
      [
        [9, 2.2],
        [8, 6.1],
      ],
      [
        [6, 5.4],
        [10, 3.3],
      ],
    ])
  })

  it('L1 lineal: backward(c) coincide con diferencias finitas por componente', () => {
    const fd = finiteDiff(L1, X)
    const p = new MaxPool2D(2)
    p.forward(X)
    const dx = p.backward([
      [
        [1, -2],
        [0.5, 3],
      ],
      [
        [-1.5, 0.25],
        [2, -0.75],
      ],
    ])
    expect(flat3(fd)).toHaveLength(32)
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)
  })

  it('L1 lineal: FD y backward coinciden con el gradiente a mano (c en la posición del máximo)', () => {
    const p = new MaxPool2D(2)
    p.forward(X)
    const dx = p.backward([
      [
        [1, -2],
        [0.5, 3],
      ],
      [
        [-1.5, 0.25],
        [2, -0.75],
      ],
    ])
    expect(maxAbsDiff(dx, GRAD_L1)).toBeLessThan(TOL)
    expect(maxAbsDiff(finiteDiff(L1, X), GRAD_L1)).toBeLessThan(TOL)
  })

  it('L2 no lineal: dOut = 2·c·out coincide con diferencias finitas y con el valor a mano', () => {
    // dOut_k = 2·c_k·out_k con out = [9, 2.2, 8, 6.1, 6, 5.4, 10, 3.3]:
    //   k0: 2·1·9 = 18        k4: 2·(−1.5)·6 = −18
    //   k1: 2·(−2)·2.2 = −8.8  k5: 2·0.25·5.4 = 2.7
    //   k2: 2·0.5·8 = 8       k6: 2·2·10 = 40
    //   k3: 2·3·6.1 = 36.6    k7: 2·(−0.75)·3.3 = −4.95
    // Como el máximo es el único punto con derivada no nula, dL2/dx = dOut en
    // la posición del máximo: el mismo patrón que GRAD_L1, con valores 2·c·out.
    const dOut = [
      [
        [18, -8.8],
        [8, 36.6],
      ],
      [
        [-18, 2.7],
        [40, -4.95],
      ],
    ]
    const p = new MaxPool2D(2)
    p.forward(X)
    const dx = p.backward(dOut)
    const expected = [
      [
        [0, 0],
        [0, -8.8],
        [0, 0],
        [0, 0],
      ],
      [
        [0, 0],
        [18, 0],
        [0, 36.6],
        [8, 0],
      ],
      [
        [-18, 0],
        [0, 0],
        [0, 0],
        [0, -4.95],
      ],
      [
        [0, 0],
        [0, 2.7],
        [0, 0],
        [40, 0],
      ],
    ]
    expect(maxAbsDiff(dx, expected)).toBeLessThan(TOL)
    expect(maxAbsDiff(dx, finiteDiff(L2, X))).toBeLessThan(TOL)
  })
})

// ── T055: ventanas solapadas ────────────────────────────────────────────────
// Entrada 4×5×1 sin empates (v = valores distintos, separación ≥ 0.5 con el
// máximo de cada ventana). Es una permutación "desordenada" para que haya
// posiciones que son máximo de una ventana pero no de otra:
//   fila0: 1.0  6.5  2.0  8.0  3.0
//   fila1: 5.0  0.5  9.5  1.5  7.0
//   fila2: 4.0  7.5  3.5  6.0  2.5
//   fila3: 8.5  2.2  5.5  0.2  4.5
const XO: number[][][] = [
  [[1.0], [6.5], [2.0], [8.0], [3.0]],
  [[5.0], [0.5], [9.5], [1.5], [7.0]],
  [[4.0], [7.5], [3.5], [6.0], [2.5]],
  [[8.5], [2.2], [5.5], [0.2], [4.5]],
]

function lossOverlap(pool: number, x: number[][][], w: number[]): number {
  return flat3(new MaxPool2D(pool, 1).forward(x)).reduce((s, v, k) => s + w[k] * v, 0)
}

function weightsFor(n: number): number[] {
  // pesos distintos y no nulos, deterministas: 0.5, -1.25, 2, -0.75, ...
  return Array.from({ length: n }, (_, k) => (k % 2 === 0 ? 1 : -1) * (0.5 + 0.75 * k))
}

describe('MaxPool2D dx oracle, ventanas solapadas (T055)', () => {
  for (const pool of [2, 3]) {
    it(`stride 1, poolSize ${pool}: backward(c) coincide con diferencias finitas`, () => {
      const p = new MaxPool2D(pool, 1)
      const out = p.forward(XO)
      const [oh, ow] = p.outputShape(4, 5, 1)
      const n = oh * ow
      expect(flat3(out)).toHaveLength(n)
      const w = weightsFor(n)
      const dOut = Array.from({ length: oh }, (_, i) =>
        Array.from({ length: ow }, (_, j) => [w[i * ow + j]]),
      )
      const dx = p.backward(dOut)
      const fd = finiteDiff((x) => lossOverlap(pool, x, w), XO)
      expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)
    })
  }

  it('poolSize 2, stride 1: valor a mano en una posición que es máximo de una ventana y no de otra', () => {
    // Ventana (0,0) ={1.0,6.5,5.0,0.5}: máx 6.5 en (0,1).
    // Ventana (0,1) = {6.5,2.0,0.5,9.5}: máx 9.5 en (1,2). 6.5 es máximo de (0,0) pero no de (0,1),
    // por lo que dX(0,1) solo recibe c(0,0), no c(0,1).
    const p = new MaxPool2D(2, 1)
    p.forward(XO)
    const dOut = [
      [[1], [10], [0], [0]],
      [[0], [0], [0], [0]],
      [[0], [0], [0], [0]],
    ]
    const dx = p.backward(dOut)
    expect(dx[0][1][0]).toBeCloseTo(1, 12)
    expect(dx[1][2][0]).toBeCloseTo(10, 12)
  })

  it('máximo compartido por varias ventanas: los gradientes se suman y coinciden con diferencias finitas', () => {
    // 9.5 en (1,2) es el máximo de las 4 ventanas 2×2 que lo contienen:
    // (0,1),(0,2),(1,1),(1,2). Con c = 1, 2, 3, 4 en ellas: dX(1,2) = 1+2+3+4 = 10.
    const p = new MaxPool2D(2, 1)
    p.forward(XO)
    const dOut = [
      [[0], [1], [2], [0]],
      [[0], [3], [4], [0]],
      [[0], [0], [0], [0]],
    ]
    const dx = p.backward(dOut)
    expect(dx[1][2][0]).toBeCloseTo(10, 12)
    const w = [0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0]
    const fd = finiteDiff((x) => lossOverlap(2, x, w), XO)
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)
  })
})
