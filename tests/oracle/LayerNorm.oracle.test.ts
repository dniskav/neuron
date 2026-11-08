// Gradient-oracle test for LayerNorm (feature 002, T013).
//
// Template: tests/oracle/NetworkN.oracle.test.ts (same three blocks):
//   1. smoke   - predictOne con valores calculados a mano (x_norm y gamma * x_norm + beta).
//   2. oracle  - expectGradientMatches sobre gamma/beta (getWeights/setWeights) con 2 posiciones:
//                el paso REAL (forward + backwardOne por posicion, lr) sigue -dL/dw (coseno > 0.999).
//                L = sum_pos 0.5 * sum_i (t - y)^2 y dOut = t - y (= -dL/dy, convencion de LayerNorm).
//   3. learns  - expectLearns: la perdida sobre datos fijos baja a menos de la mitad.
// Extra: el valor que devuelve backwardOne es el gradiente respecto a la entrada en la convencion
// -dL/dx (la misma que usa TransformerBlock al sumarlo al resto de gradientes). Se comprueba con
// diferencias finitas sobre x (coseno > 0.999).
// LayerNorm no usa azar: los pesos iniciales son fijos, la semilla global no hace falta.
import { describe, it, expect } from 'vitest'
import { LayerNorm } from '../../src/LayerNorm'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const DIM = 3
const SEQ = 2

// Dos posiciones con varianza no nula.
const X: number[][] = [
  [1, 2, 3],
  [2, -1, 0],
]
const TARGETS: number[][] = [
  [0.5, -0.2, 0.8],
  [-0.6, 0.9, 0.1],
]
// Pesos iniciales no triviales para que el gradiente dependa de gamma y beta.
const GAMMA0 = [1.5, 0.8, 1.2]
const BETA0 = [0.1, -0.2, 0.3]

function build(gamma: number[] = [1, 1, 1], beta: number[] = [0, 0, 0]): LayerNorm {
  const norm = new LayerNorm(DIM)
  norm.setWeights([...gamma, ...beta])
  return norm
}

/** Forward de la secuencia: resetCache y predictOne por posicion. */
function forward(norm: LayerNorm, xs: number[][]): number[][] {
  norm.resetCache(xs.length)
  return xs.map((x, pos) => norm.predictOne(x, pos))
}

/** L = sum_pos 0.5 * sum_i (t - y)^2, medida solo con predictOne. */
function loss(norm: LayerNorm, xs: number[][] = X): number {
  const ys = forward(norm, xs)
  let s = 0
  for (let p = 0; p < SEQ; p++)
    for (let i = 0; i < DIM; i++) s += 0.5 * (TARGETS[p][i] - ys[p][i]) ** 2
  return s
}

/** Un paso real: forward, dOut = t - y para todas las posiciones, luego backwardOne por posicion. */
function step(norm: LayerNorm, lr: number): void {
  const ys = forward(norm, X)
  const dOut = ys.map((y, p) => y.map((v, i) => TARGETS[p][i] - v))
  dOut.forEach((d, p) => norm.backwardOne(d, p, lr))
}

function cosine(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  for (let k = 0; k < a.length; k++) {
    dot += a[k] * b[k]
    na += a[k] * a[k]
    nb += b[k] * b[k]
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

describe('LayerNorm oraculo', () => {
  it('humo: predictOne normaliza con valores calculados a mano y backwardOne mueve gamma/beta', () => {
    // x = [1, 2, 3]: media 2, varianza 2/3 => x_norm = [-1, 0, 1] / sqrt(2/3) = [-1.224745, 0, 1.224745]
    const norm = build()
    const ys = forward(norm, [[1, 2, 3]])
    expect(ys[0][0]).toBeCloseTo(-1.224745, 3)
    expect(ys[0][1]).toBeCloseTo(0, 3)
    expect(ys[0][2]).toBeCloseTo(1.224745, 3)

    // gamma = 2, beta = 1 => y = 2 * x_norm + 1
    const affine = build([2, 2, 2], [1, 1, 1])
    const ya = forward(affine, [[1, 2, 3]])
    expect(ya[0][0]).toBeCloseTo(-1.449490, 3)
    expect(ya[0][1]).toBeCloseTo(1, 3)
    expect(ya[0][2]).toBeCloseTo(3.449490, 3)

    const before = norm.getWeights()
    const dX = forward(norm, X).map((y, p) =>
      norm.backwardOne(
        y.map((v, i) => TARGETS[p][i] - v),
        p,
        0.01,
      ),
    )
    expect(dX.flat().every(Number.isFinite)).toBe(true)
    const after = norm.getWeights()
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oraculo: el paso sobre gamma/beta sigue -dL/dw', () => {
    const norm = build(GAMMA0, BETA0)
    const r = expectGradientMatches({
      getParams: () => norm.getWeights(),
      setParams: (p) => norm.setWeights(p),
      loss: () => loss(norm),
      step: (lr) => step(norm, lr),
      lr: 0.01,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('oraculo de entrada: backwardOne devuelve -dL/dx (coseno frente a diferencias finitas sobre x)', () => {
    const norm = build(GAMMA0, BETA0)
    // Analitico: forward, dOut = t - y, backwardOne con lr = 0 (no cambia gamma/beta).
    const ys = forward(norm, X)
    const dX = ys.map((y, p) =>
      norm.backwardOne(
        y.map((v, i) => TARGETS[p][i] - v),
        p,
        0,
      ),
    )

    // Numerico: -dL/dx por diferencias centrales en cada componente de x (orden p-mayor, i-menor).
    const eps = 1e-5
    const g: number[] = []
    for (let p = 0; p < SEQ; p++)
      for (let i = 0; i < DIM; i++) {
        const plus = X.map((r) => [...r])
        plus[p][i] += eps
        const minus = X.map((r) => [...r])
        minus[p][i] -= eps
        g.push(-(loss(norm, plus) - loss(norm, minus)) / (2 * eps))
      }

    expect(cosine(dX.flat(), g)).toBeGreaterThan(0.999)
  })

  it('expectLearns: la perdida baja a la mitad al ajustar gamma/beta', () => {
    const norm = build()
    expectLearns({ loss: () => loss(norm), step: () => step(norm, 0.05), steps: 200 })
  })
})
