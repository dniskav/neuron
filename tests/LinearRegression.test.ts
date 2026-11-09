// Tests de comportamiento de LinearRegression (src/LinearRegression.ts).
//
// Complementan, sin repetir, tests/oracle/LinearRegression.oracle.test.ts (gradiente de fitGD)
// y tests/api/Regression.*.test.ts (pesos a mano, validación de datos, entrenamiento incremental).
//
// Regla de valores (FR-004): cada cifra esperada lleva su derivación a mano en un comentario.
// Ninguna se copia de la salida del código.
//
// ─── fitNormal: y = 2x + 1 con 3 puntos exactos ───────────────────────────────
//   X = [0, 1, 2], y = [1, 3, 5]. Xa = [[0,1],[1,1],[2,1]].
//   XᵀX = [[5,3],[3,3]], Xᵀy = [13, 9], det = 15 − 9 = 6.
//   W = (1/6)·[[3,−3],[−3,5]]·[13, 9] = (1/6)·[12, 6] = [2, 1]  →  pendiente 2, sesgo 1.
//
// ─── fitNormal: mínimos cuadrados con ruido, X = [0, 1, 2], y = [0, 0, 3] ─────
//   XᵀX = [[5,3],[3,3]], Xᵀy = [0·0 + 1·0 + 2·3, 0 + 0 + 3] = [6, 3].
//   W = (1/6)·[[3,−3],[−3,5]]·[6, 3] = (1/6)·[9, −3] = [1.5, −0.5].
//   Comprobación de las ecuaciones normales: residuos r = y − (1.5x − 0.5) = [0.5, −1, 0.5];
//   Σr = 0 y Σx·r = 0·0.5 + 1·(−1) + 2·0.5 = 0. Luego W es el mínimo.
//
// ─── fitNormal singular: columna 2 = 2 · columna 1 ───────────────────────────────
//   XᵀX tiene rango 2 de 3 → sin inversa → lanza.
//
// ─── fitGD a mano: X = [0, 2], y = [1, 5], lr = 0.1, w = 0, b = 0 ────────────────
//   Época 1: ŷ = [0, 0], r = ŷ − y = [−1, −5], MSE = (1 + 25)/2 = 13.
//     dw = (2/2)·(0·(−1) + 2·(−5)) = −10;  db = (2/2)·(−1 − 5) = −6.
//     w = 0 − 0.1·(−10) = 1;  b = 0 − 0.1·(−6) = 0.6.
//   Época 2: ŷ = [0.6, 2.6], r = [−0.4, −2.4], MSE = (0.16 + 5.76)/2 = 2.96.
//     dw = 0·(−0.4) + 2·(−2.4) = −4.8;  db = −0.4 − 2.4 = −2.8.
//     w = 1 + 0.48 = 1.48;  b = 0.6 + 0.28 = 0.88.
//
// ─── Convergencia y aprendizaje (5 semillas × lr ×0.7 / ×1.3) ───────────────────
//   Datos exactos y = 2·x1 − x2 + 0.5 con X uniforme en [−1, 1), m = 20. Como y es exacta y X
//   tiene rango completo, la solución única es W* = [2, −1, 0.5] (se verifica en cada punto:
//   2·x1 − x2 + 0.5 = y). Base lr = 0.1, 300 épocas, semillas 1..5, lr ×0.7 / ×1.0 / ×1.3.
//   Cifras medidas (peor caso sobre 15 combinaciones):
//     - error máximo de pesos |w − W*|∞: 1.58e-4 (semilla 1, lr ×0.7)  → umbral 1e-3, margen 84 %.
//     - MSE final: 6.16e-9 (semilla 1, lr ×0.7); MSE inicial entre 1.34 y 3.43.
//       Razón final/inicial máxima: 4.6e-9 → umbral 0.5, exigido < 0.35 (margen ≥ 30 %).
//   Umbrales fijados a priori: pesos 1e-3, MSE 1e-6; el peor caso queda por debajo de 0.7 × umbral.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach (tests/helpers/seed.ts).

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { LinearRegression } from '../src/LinearRegression'
import { seedRandom, restoreRandom } from './helpers/seed'
import { expectLearns } from './helpers/learns'

beforeEach(() => seedRandom(0x5eed))
afterEach(restoreRandom)

const W_STAR = [2, -1, 0.5]
const LR_BASE = 0.1
const EPOCHS = 300
const SEEDS = [1, 2, 3, 4, 5]
const LR_SCALES = [0.7, 1, 1.3]
const THRESHOLD_WEIGHTS = 1e-3
const THRESHOLD_MSE = 1e-6

/** Datos exactos y = 2·x1 − x2 + 0.5 con X generado por Math.random (semilla fijada por el llamador). */
function exactData(m: number): { X: number[][]; y: number[] } {
  const X: number[][] = []
  const y: number[] = []
  for (let i = 0; i < m; i++) {
    const x1 = 2 * Math.random() - 1
    const x2 = 2 * Math.random() - 1
    X.push([x1, x2])
    y.push(2 * x1 - x2 + 0.5)
  }
  return { X, y }
}

/** MSE de la hipótesis actual del modelo sobre (X, y), calculado fuera de fitGD. */
function mseOf(model: LinearRegression, X: number[][], y: number[]): number {
  const nFeat = model.weights.length - 1
  let s = 0
  for (let i = 0; i < X.length; i++) {
    let yhat = model.weights[nFeat]
    for (let j = 0; j < nFeat; j++) yhat += model.weights[j] * X[i][j]
    s += (yhat - y[i]) ** 2
  }
  return s / X.length
}

// ─── predict y getCoefficients ───────────────────────────────────────────────

describe('predict con pesos a mano (varios puntos)', () => {
  // weights = [w1, w2, bias] = [1.5, −2, 0.25]
  it('1.5·2 + (−2)·1 + 0.25 = 1.25', () => {
    const m = new LinearRegression()
    m.weights = [1.5, -2, 0.25]
    expect(m.predict([2, 1])).toBeCloseTo(1.25, 12)
  })

  it('con x = [0, 0] devuelve el sesgo 0.25', () => {
    const m = new LinearRegression()
    m.weights = [1.5, -2, 0.25]
    expect(m.predict([0, 0])).toBe(0.25)
  })

  it('1.5·(−1) + (−2)·3 + 0.25 = −7.25 (valores negativos)', () => {
    const m = new LinearRegression()
    m.weights = [1.5, -2, 0.25]
    expect(m.predict([-1, 3])).toBeCloseTo(-7.25, 12)
  })

  it('lanza con la longitud esperada y la recibida si x tiene un rasgo de más o de menos', () => {
    const m = new LinearRegression()
    m.weights = [1.5, -2, 0.25]
    expect(() => m.predict([1])).toThrow(/LinearRegression\.predict: expected 2 features, got 1/)
    expect(() => m.predict([1, 2, 3])).toThrow(/LinearRegression\.predict: expected 2 features, got 3/)
  })

  it('sin ajustar lanza "not been fitted yet"', () => {
    expect(() => new LinearRegression().predict([1, 2])).toThrow(/has not been fitted yet/)
  })
})

describe('getCoefficients', () => {
  it('separa pesos y sesgo: {weights: [1.5, −2], bias: 0.25}', () => {
    const m = new LinearRegression()
    m.weights = [1.5, -2, 0.25]
    expect(m.getCoefficients()).toEqual({ weights: [1.5, -2], bias: 0.25 })
  })

  it('devuelve copias: modificar el resultado no cambia el modelo', () => {
    const m = new LinearRegression()
    m.weights = [1.5, -2, 0.25]
    const c = m.getCoefficients()
    c.weights[0] = 99
    c.bias = 99
    expect(m.weights).toEqual([1.5, -2, 0.25])
    expect(m.predict([2, 1])).toBeCloseTo(1.25, 12)
  })

  it('sin ajustar lanza "not been fitted yet"', () => {
    expect(() => new LinearRegression().getCoefficients()).toThrow(/has not been fitted yet/)
  })
})

// ─── fitNormal: solución cerrada ─────────────────────────────────────────────

describe('fitNormal: solución exacta a mano', () => {
  it('y = 2x + 1 con 3 puntos → pendiente 2 y sesgo 1 (ver cabecera)', () => {
    const m = new LinearRegression()
    m.fitNormal([[0], [1], [2]], [1, 3, 5])
    const { weights, bias } = m.getCoefficients()
    expect(weights).toHaveLength(1)
    expect(weights[0]).toBeCloseTo(2, 9)
    expect(bias).toBeCloseTo(1, 9)
  })

  it('mínimos cuadrados con ruido: X = [0,1,2], y = [0,0,3] → pesos 1.5 y sesgo −0.5', () => {
    const m = new LinearRegression()
    m.fitNormal([[0], [1], [2]], [0, 0, 3])
    const { weights, bias } = m.getCoefficients()
    expect(weights[0]).toBeCloseTo(1.5, 9)
    expect(bias).toBeCloseTo(-0.5, 9)
  })

  it('dos rasgos con datos exactos → W* = [2, −1, 0.5]', () => {
    // Puntos: (1,0)→2.5, (0,1)→−0.5, (1,1)→1.5, (2,1)→3.5; todos cumplen 2·x1 − x2 + 0.5.
    const m = new LinearRegression()
    m.fitNormal(
      [
        [1, 0],
        [0, 1],
        [1, 1],
        [2, 1],
      ],
      [2.5, -0.5, 1.5, 3.5],
    )
    expect(m.weights).toHaveLength(3)
    W_STAR.forEach((w, i) => expect(m.weights[i]).toBeCloseTo(w, 9))
  })

  it('X con columnas linealmente dependientes lanza "singular"', () => {
    // columna 2 = 2 · columna 1 → XᵀX singular (ver cabecera)
    const m = new LinearRegression()
    expect(() =>
      m.fitNormal(
        [
          [1, 2],
          [2, 4],
          [3, 6],
        ],
        [1, 2, 3],
      ),
    ).toThrow(/LinearRegression\.fitNormal: XᵀX is singular/)
  })

  it('X vacía lanza', () => {
    expect(() => new LinearRegression().fitNormal([], [])).toThrow(
      /LinearRegression\.fitNormal: X is empty/,
    )
  })

  it('X e y de longitud distinta lanzan con ambas longitudes', () => {
    expect(() => new LinearRegression().fitNormal([[1], [2]], [1])).toThrow(
      /LinearRegression\.fitNormal: X has 2 rows but y has 1 elements/,
    )
  })
})

// Comportamiento ACTUAL de fitNormal en entradas inválidas. T037 (validateDataset) lo cambiará:
// entonces estos casos deben pasar a lanzar y este bloque se reescribe.
describe('fitNormal con entradas inválidas: comportamiento actual (T037 lo cambia)', () => {
  it('X con filas irregulares no lanza y deja los pesos en NaN', () => {
    const m = new LinearRegression()
    m.fitNormal(
      [
        [1, 2],
        [3],
      ],
      [1, 2],
    )
    expect(m.weights).toHaveLength(3)
    expect(m.weights.every((w) => Number.isNaN(w))).toBe(true)
  })

  it('y con NaN no lanza y deja los pesos en NaN', () => {
    const m = new LinearRegression()
    m.fitNormal([[0], [1], [2]], [1, NaN, 5])
    expect(m.weights).toHaveLength(2)
    expect(m.weights.every((w) => Number.isNaN(w))).toBe(true)
  })
})

// ─── fitGD: valores a mano y argumentos ──────────────────────────────────────

describe('fitGD: pasos a mano (ver cabecera)', () => {
  it('una época desde cero: pérdida 13, pesos [1, 0.6]', () => {
    const m = new LinearRegression()
    const losses = m.fitGD([[0], [2]], [1, 5], 0.1, 1)
    expect(losses).toHaveLength(1)
    expect(losses[0]).toBeCloseTo(13, 12)
    expect(m.weights[0]).toBeCloseTo(1, 12)
    expect(m.weights[1]).toBeCloseTo(0.6, 12)
  })

  it('dos épocas desde cero: pérdidas [13, 2.96], pesos [1.48, 0.88]', () => {
    const m = new LinearRegression()
    const losses = m.fitGD([[0], [2]], [1, 5], 0.1, 2)
    expect(losses[0]).toBeCloseTo(13, 12)
    expect(losses[1]).toBeCloseTo(2.96, 12)
    expect(m.weights[0]).toBeCloseTo(1.48, 12)
    expect(m.weights[1]).toBeCloseTo(0.88, 12)
  })

  it('devuelve una pérdida por época, en orden', () => {
    const m = new LinearRegression()
    expect(m.fitGD([[0], [2]], [1, 5], 0.1, 4)).toHaveLength(4)
  })
})

describe('fitGD: argumentos inválidos', () => {
  it('lr = 0 lanza "lr must be positive"', () => {
    expect(() => new LinearRegression().fitGD([[1]], [1], 0, 1)).toThrow(
      /LinearRegression\.fitGD: lr must be positive/,
    )
  })

  it('lr = NaN lanza "expected finite number"', () => {
    expect(() => new LinearRegression().fitGD([[1]], [1], NaN, 1)).toThrow(
      /LinearRegression\.fitGD: expected finite number/,
    )
  })

  it('epochs = 0 y epochs = 1.5 lanzan "positive integer"', () => {
    expect(() => new LinearRegression().fitGD([[1]], [1], 0.1, 0)).toThrow(
      /LinearRegression\.fitGD: epochs must be a positive integer/,
    )
    expect(() => new LinearRegression().fitGD([[1]], [1], 0.1, 1.5)).toThrow(
      /LinearRegression\.fitGD: epochs must be a positive integer/,
    )
  })

  it('X vacía lanza', () => {
    expect(() => new LinearRegression().fitGD([], [], 0.1, 1)).toThrow(
      /LinearRegression\.fitGD: X is empty/,
    )
  })

  it('X e y de longitud distinta lanzan', () => {
    expect(() => new LinearRegression().fitGD([[1], [2]], [1], 0.1, 1)).toThrow(
      /LinearRegression\.fitGD: X has 2 rows but y has 1 elements/,
    )
  })

  it('X[0] que no es array lanza', () => {
    const X = [1] as unknown as number[][]
    expect(() => new LinearRegression().fitGD(X, [1], 0.1, 1)).toThrow(
      /LinearRegression\.fitGD: X\[0\]: expected array, got number/,
    )
  })

  it('filas sin rasgos lanzan "at least 1 feature"', () => {
    expect(() => new LinearRegression().fitGD([[]], [1], 0.1, 1)).toThrow(
      /LinearRegression\.fitGD: X rows need at least 1 feature/,
    )
  })
})

// ─── Convergencia y aprendizaje ──────────────────────────────────────────────

describe('fitGD converge a la solución exacta (5 semillas × lr ×0.7 / ×1.0 / ×1.3)', () => {
  it('tras 300 épocas, |w − W*|∞ < 1e-3 (peor caso medido 1.58e-4, ≤ 0.7 × umbral)', () => {
    for (const seed of SEEDS) {
      for (const scale of LR_SCALES) {
        seedRandom(seed)
        const { X, y } = exactData(20)
        const m = new LinearRegression()
        m.fitGD(X, y, LR_BASE * scale, EPOCHS)
        const err = Math.max(...W_STAR.map((w, j) => Math.abs(m.weights[j] - w)))
        expect(err, `semilla ${seed}, lr x${scale}`).toBeLessThanOrEqual(0.7 * THRESHOLD_WEIGHTS)
      }
    }
  })

  it('fitGD y fitNormal dan la misma solución (1e-6) sobre los mismos datos', () => {
    seedRandom(3)
    const { X, y } = exactData(20)
    const gd = new LinearRegression()
    gd.fitGD(X, y, LR_BASE, EPOCHS)
    const ne = new LinearRegression()
    ne.fitNormal(X, y)
    ne.weights.forEach((w, j) => expect(Math.abs(gd.weights[j] - w)).toBeLessThan(1e-6))
  })
})

describe('fitGD aprende: la pérdida baja con margen (5 semillas × lr ×0.7 / ×1.0 / ×1.3)', () => {
  it('expectLearns: razón final/inicial < 0.35 (umbral 0.5 con margen 30 %; peor caso 4.6e-9)', () => {
    for (const seed of SEEDS) {
      for (const scale of LR_SCALES) {
        seedRandom(seed)
        const { X, y } = exactData(20)
        const m = new LinearRegression()
        m.weights = [0, 0, 0] // punto de partida explícito (la pérdida inicial necesita pesos)
        expectLearns({
          loss: () => mseOf(m, X, y),
          step: () => void m.fitGD(X, y, LR_BASE * scale, 1),
          steps: EPOCHS,
          ratio: 0.35,
        })
      }
    }
  })

  it('MSE final < 1e-6 (peor caso medido 6.2e-9, ≤ 0.7 × umbral)', () => {
    for (const seed of SEEDS) {
      for (const scale of LR_SCALES) {
        seedRandom(seed)
        const { X, y } = exactData(20)
        const m = new LinearRegression()
        const losses = m.fitGD(X, y, LR_BASE * scale, EPOCHS)
        expect(losses[EPOCHS - 1], `semilla ${seed}, lr x${scale}`).toBeLessThanOrEqual(
          0.7 * THRESHOLD_MSE,
        )
      }
    }
  })
})
