// Gradient-oracle test for LearnedPositionalEncoding (feature 002, T034).
//
// Plantilla NeuronN.oracle.test.ts: humo, oraculo y expectLearns.
// Modelo: apply(X) = X + W[0..len-1]. Convencion: update(dW, lr) hace w + lr*dW,
// con dW = -dL/dW; un modelo correcto da coseno ~ +1.
// Perdida: L = sum_{pos<SEQ, d} 0.5 (T - (X + W))^2. Las filas de W fuera de SEQ no entran en L.
// Pesos iniciales fijos (W0): el test no depende de Math.random; la semilla se fija igualmente.
import { beforeEach, describe, it, expect } from 'vitest'
import { LearnedPositionalEncoding } from '../../src/PositionalEncoding'
import { bindParams, expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

const MAX = 4
const DIM = 3
const SEQ = 3

// Embeddings de entrada y objetivos (SEQ x DIM).
const X: number[][] = [
  [0.5, -0.3, 0.8],
  [0.1, 0.2, -0.4],
  [-0.6, 0.9, 0.3],
]
const T: number[][] = [
  [0.7, 0.1, -0.2],
  [-0.5, 0.4, 0.6],
  [0.2, -0.8, 0.1],
]

// Tabla de posiciones inicial fija (MAX x DIM).
const W0: number[][] = [
  [0.2, -0.1, 0.05],
  [-0.3, 0.25, 0.1],
  [0.15, 0.4, -0.2],
  [-0.05, 0.3, 0.35],
]
const ZERO: number[][] = Array.from({ length: MAX }, () => new Array<number>(DIM).fill(0))

function build(w: number[][] = W0): LearnedPositionalEncoding {
  const pe = new LearnedPositionalEncoding(MAX, DIM)
  pe.weights = w.map((row) => [...row])
  return pe
}

/** L = sum_{pos<SEQ, d} 0.5 (T - (X + W))^2 */
function loss(pe: LearnedPositionalEncoding): number {
  const ys = pe.apply(X)
  let s = 0
  for (let p = 0; p < SEQ; p++)
    for (let d = 0; d < DIM; d++) s += 0.5 * (T[p][d] - ys[p][d]) ** 2
  return s
}

/** Un paso real: dW = -dL/dW = T - Y en las posiciones usadas, 0 en el resto; luego update. */
function step(pe: LearnedPositionalEncoding, lr: number): void {
  const ys = pe.apply(X)
  const dW = ZERO.map((row) => [...row])
  for (let p = 0; p < SEQ; p++)
    for (let d = 0; d < DIM; d++) dW[p][d] = T[p][d] - ys[p][d]
  pe.update(dW, lr)
}

describe('LearnedPositionalEncoding oraculo', () => {
  beforeEach(() => seedRandom(0x5eed))

  it('humo: apply suma la tabla con valores calculados a mano y update aplica w + lr*dW', () => {
    // Forma y cota inicial: pesos aleatorios en [-sqrt(1/DIM), sqrt(1/DIM)].
    const fresh = new LearnedPositionalEncoding(MAX, DIM)
    expect(fresh.weights.length).toBe(MAX)
    expect(fresh.weights.every((row) => row.length === DIM)).toBe(true)
    const limit = Math.sqrt(1 / DIM)
    expect(fresh.weights.flat().every((v) => Math.abs(v) <= limit)).toBe(true)

    // apply([1,2,3]) con W0: [1+0.2, 2-0.1, 3+0.05] = [1.2, 1.9, 3.05]
    const ys = build().apply([[1, 2, 3]])
    expect(ys[0][0]).toBeCloseTo(1.2, 9)
    expect(ys[0][1]).toBeCloseTo(1.9, 9)
    expect(ys[0][2]).toBeCloseTo(3.05, 9)

    // Signo: con W = 0 y dW = [1, -1, 0.5] en la fila 0, lr = 0.1 => W[0] = [0.1, -0.1, 0.05].
    const pe = build(ZERO)
    pe.update([[1, -1, 0.5], [0, 0, 0], [0, 0, 0], [0, 0, 0]], 0.1)
    expect(pe.weights[0][0]).toBeCloseTo(0.1, 9)
    expect(pe.weights[0][1]).toBeCloseTo(-0.1, 9)
    expect(pe.weights[0][2]).toBeCloseTo(0.05, 9)
    expect(pe.weights[1]).toEqual([0, 0, 0])
  })

  it('oraculo: el paso sigue -dL/dw', () => {
    const pe = build()
    const params = bindParams(pe.weights)
    const r = expectGradientMatches({
      getParams: () => params.getParams(),
      setParams: (p) => params.setParams(p),
      loss: () => loss(pe),
      step: (lr) => step(pe, lr),
      lr: 0.1,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('expectLearns: la perdida baja a menos de la mitad al ajustar la tabla', () => {
    const pe = build()
    expectLearns({ loss: () => loss(pe), step: () => step(pe, 0.1), steps: 200 })
  })
})
