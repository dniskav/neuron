// Tests de comportamiento de LogisticRegression y SoftmaxRegression (feature 006, T026).
//
// Complementan tests/oracle/LogisticRegression.oracle.test.ts (gradiente por diferencias
// finitas, humo y un expectLearns) y tests/api/Regression.* (pesos asignados a mano, forma
// de los pesos). Aquí van: predicción con pesos a mano, el umbral 0.5 exacto, la
// estabilidad de softmax, pasos de entrenamiento a mano, los errores de validación y el
// "aprende" de AND/OR (binario) y de 3 blobs lineales (multiclase).
//
// Cifras de "aprende" (cabecera):
//  - Criterio a priori: pérdida media < UMBRAL = 0.05 con acierto del 100 %.
//    La media de −ln p(clase correcta) < 0.05 equivale a una media geométrica de
//    p(clase correcta) ≥ e^{-0.05} ≈ 0.951 (BCE y CE, igual).
//  - Margen exigido: peor caso < 0.7 · 0.05 = 0.035 (30 % de margen).
//  - Configuración: 1000 épocas, semillas 1..5 y lr ∈ {0.35, 0.5, 0.65}
//    (0.5 · 0.7, 0.5 y 0.5 · 1.3).
//  - Peor caso medido al validar (script temporal t026_validate, fuera del repo):
//      AND  0.012170 → 76 % por debajo del umbral 0.05
//      OR   0.006568 → 87 % por debajo del umbral
//      blobs (CE) 0.000136 → 99.7 % por debajo del umbral
//    Pérdida inicial de AND: ≈ 0.75–0.90 según semilla y lr; las 15 corridas de cada caso
//    acaban con acierto 1.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach (tests/helpers/seed.ts).
// Dentro de cada caso se vuelve a fijar la semilla de la corrida.
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { LogisticRegression, SoftmaxRegression } from '../src/LogisticRegression'
import { seedRandom, restoreRandom } from './helpers/seed'

beforeEach(() => seedRandom())
afterEach(() => restoreRandom())

// ─── Referencias independientes de src/ ──────────────────────────────────────

const sigmoidRef = (z: number) => 1 / (1 + Math.exp(-z))

const SEEDS = [1, 2, 3, 4, 5]
const LRS = [0.35, 0.5, 0.65]
const EPOCHS = 1000
const UMBRAL = 0.05

function bceMean(m: LogisticRegression, X: number[][], Y: number[]): number {
  let s = 0
  for (let i = 0; i < X.length; i++) {
    const z = m.weights.reduce((acc, w, j) => acc + w * X[i][j], m.bias)
    const p = sigmoidRef(z)
    s += -(Y[i] * Math.log(p) + (1 - Y[i]) * Math.log(1 - p))
  }
  return s / X.length
}

function ceMean(m: SoftmaxRegression, X: number[][], Y: number[]): number {
  let s = 0
  for (let i = 0; i < X.length; i++) {
    const scores = m.weights.map((wk, k) => m.biases[k] + wk.reduce((a, w, j) => a + w * X[i][j], 0))
    const z = scores.reduce((a, v) => a + Math.exp(v), 0)
    s += -Math.log(Math.exp(scores[Y[i]]) / z)
  }
  return s / X.length
}

// ═════════════════════════════════════════════════════════════════════════════
// LogisticRegression
// ═════════════════════════════════════════════════════════════════════════════

describe('LogisticRegression: predicción con pesos a mano', () => {
  it('predictProba = σ(w·x + b): w = [0.5, −1], b = −1, x = [2, 1] → z = −1', () => {
    // z = 0.5·2 + (−1)·1 + (−1) = −1. p = σ(−1) = 1 / (1 + e).
    const m = new LogisticRegression()
    m.weights = [0.5, -1]
    m.bias = -1
    expect(m.predict([2, 1])).toBeCloseTo(1 / (1 + Math.E), 12)
  })

  it('la sigmoide es creciente y simétrica: σ(2) = 1/(1+e⁻²) y σ(−2) = 1 − σ(2)', () => {
    // w = [2], b = 0. x = [1] → z = 2; x = [−1] → z = −2.
    const m = new LogisticRegression()
    m.weights = [2]
    m.bias = 0
    const s2 = 1 / (1 + Math.exp(-2))
    expect(m.predict([1])).toBeCloseTo(s2, 12)
    expect(m.predict([-1])).toBeCloseTo(1 - s2, 12)
    expect(m.predict([1])).toBeGreaterThan(0.5)
  })

  it('en z = 0 la probabilidad es exactamente 0.5', () => {
    // w = [1, −1], b = 0, x = [1, 1] → z = 1 − 1 + 0 = 0 → σ(0) = 1/(1+1) = 0.5.
    const m = new LogisticRegression()
    m.weights = [1, -1]
    m.bias = 0
    expect(m.predict([1, 1])).toBe(0.5)
  })

  it('classify usa el umbral 0.5 inclusivo: z = 0 → 1; z ligeramente negativo → 0', () => {
    // z = 0 → p = 0.5 → clase 1 (la comparación es >= 0.5).
    // z = −1e−9 → p = 0.5 − 2.5e−10 < 0.5 → clase 0.
    const m = new LogisticRegression()
    m.weights = [1]
    m.bias = 0
    expect(m.classify([0])).toBe(1)
    m.bias = -1e-9
    expect(m.classify([0])).toBe(0)
  })
})

describe('LogisticRegression: un paso de train a mano', () => {
  it('w = [0, 0], b = 0, x = [3, −2], y = 0, lr = 0.4 → w = [−0.6, 0.4], b = −0.2', () => {
    // z = 0 → p = 0.5. Paso: w += lr·(y − p)·x, b += lr·(y − p).
    // (y − p) = −0.5. w0 = 0.4·(−0.5)·3 = −0.6; w1 = 0.4·(−0.5)·(−2) = 0.4; b = 0.4·(−0.5) = −0.2.
    const m = new LogisticRegression()
    m.weights = [0, 0]
    m.bias = 0
    m.train([[3, -2]], [0], 0.4, 1)
    expect(m.weights[0]).toBeCloseTo(-0.6, 12)
    expect(m.weights[1]).toBeCloseTo(0.4, 12)
    expect(m.bias).toBeCloseTo(-0.2, 12)
  })

  it('lossHistory[0] es la BCE con los pesos iniciales: w = [0.5], b = 0, x = [2], y = 1', () => {
    // z = 0.5·2 = 1, p = σ(1). BCE = −ln σ(1) = ln(1 + e⁻¹) ≈ 0.31326.
    // La pérdida de cada muestra se calcula antes de su actualización.
    const m = new LogisticRegression()
    m.weights = [0.5]
    m.bias = 0
    const hist = m.train([[2]], [1], 0.1, 3)
    expect(hist).toHaveLength(3)
    expect(hist[0]).toBeCloseTo(Math.log(1 + Math.exp(-1)), 12)
    expect(hist[2]).toBeLessThan(hist[0])
  })

  it('con pesos ya del tamaño correcto continúa desde ellos (no los reinicia)', () => {
    // x = [0, 0] no mueve los pesos (w += lr·δ·0). z = b = 0.25, p = σ(0.25).
    // b' = 0.25 + lr·(1 − σ(0.25)) = 0.25 + 0.1·(1 − σ(0.25)).
    const m = new LogisticRegression()
    m.weights = [0.5, -0.5]
    m.bias = 0.25
    m.train([[0, 0]], [1], 0.1, 1)
    expect(m.weights).toEqual([0.5, -0.5])
    expect(m.bias).toBeCloseTo(0.25 + 0.1 * (1 - sigmoidRef(0.25)), 12)
  })

  it('pesos iniciales aleatorios: acotados por √(2/n) y sesgo 0 antes del primer paso', () => {
    // Sin pesos previos, n = 4 → límite √(2/4) = √0.5. Con x = 0, z = 0 → p = 0.5,
    // y = 1 → b' = lr·(1 − 0.5) = 0.5·lr = 0.1 con lr = 0.2. Los pesos no cambian (x = 0).
    const limit = Math.sqrt(2 / 4)
    const m = new LogisticRegression()
    m.train([[0, 0, 0, 0]], [1], 0.2, 1)
    expect(m.weights).toHaveLength(4)
    for (const w of m.weights) {
      expect(Math.abs(w)).toBeLessThanOrEqual(limit)
    }
    expect(m.bias).toBeCloseTo(0.1, 12)
  })

  it('la inicialización aleatoria es determinista con la misma semilla', () => {
    const X = [[1, 2], [-1, 0.5]]
    const Y = [1, 0]
    seedRandom(7)
    const a = new LogisticRegression()
    a.train(X, Y, 0.1, 5)
    seedRandom(7)
    const b = new LogisticRegression()
    b.train(X, Y, 0.1, 5)
    expect(a.weights).toEqual(b.weights)
    expect(a.bias).toBe(b.bias)
  })
})

describe('LogisticRegression: errores', () => {
  it('predict sin entrenar lanza', () => {
    expect(() => new LogisticRegression().predict([1, 2])).toThrow(
      'LogisticRegression.predict: model has not been trained yet',
    )
  })

  it('predict con longitud de x distinta de los pesos lanza', () => {
    const m = new LogisticRegression()
    m.weights = [1, 2]
    expect(() => m.predict([1])).toThrow('LogisticRegression.predict: expected array of length 2, got 1')
  })

  it('train: X vacía, filas y etiquetas de distinto tamaño', () => {
    const m = new LogisticRegression()
    expect(() => m.train([], [], 0.1, 1)).toThrow('LogisticRegression.train: X is empty')
    expect(() => m.train([[1], [2]], [1], 0.1, 1)).toThrow(
      'LogisticRegression.train: X has 2 rows but y has 1 labels',
    )
  })

  it('train: lr no finito, lr ≤ 0 y epochs no entero positivo', () => {
    const m = new LogisticRegression()
    expect(() => m.train([[1]], [1], NaN, 1)).toThrow('LogisticRegression.train: expected finite number')
    expect(() => m.train([[1]], [1], 0, 1)).toThrow('LogisticRegression.train: lr must be positive')
    expect(() => m.train([[1]], [1], -0.1, 1)).toThrow('LogisticRegression.train: lr must be positive')
    for (const bad of [0, -2, 1.5]) {
      expect(() => m.train([[1]], [1], 0.1, bad)).toThrow(
        'LogisticRegression.train: epochs must be a positive integer',
      )
    }
  })

  it('train: X[0] no es un array, filas sin rasgos y filas de distinta longitud', () => {
    const m = new LogisticRegression()
    expect(() => m.train([1 as unknown as number[]], [1], 0.1, 1)).toThrow(
      'LogisticRegression.train: X[0]: expected array, got number',
    )
    expect(() => m.train([[]], [1], 0.1, 1)).toThrow('LogisticRegression.train: X rows need at least 1 feature')
    expect(() => m.train([[1, 2], [1]], [1, 0], 0.1, 1)).toThrow(
      'LogisticRegression.train: X[1]: expected array of length 2, got 1',
    )
  })

  it('train: etiquetas fuera de [0, 1] y no finitas', () => {
    const m = new LogisticRegression()
    expect(() => m.train([[1]], [1.5], 0.1, 1)).toThrow('LogisticRegression.train: y[0] = 1.5 is outside [0, 1]')
    expect(() => m.train([[1]], [-0.1], 0.1, 1)).toThrow('is outside [0, 1]')
    expect(() => m.train([[1]], [NaN], 0.1, 1)).toThrow('LogisticRegression.train: y: invalid value at index 0: NaN')
  })

  it('una llamada a train rechazada no cambia los pesos ni el sesgo', () => {
    const m = new LogisticRegression()
    m.weights = [0.3]
    m.bias = 0.1
    expect(() => m.train([[1]], [2], 0.1, 1)).toThrow()
    expect(() => m.train([[1]], [1], 0, 1)).toThrow()
    expect(m.weights).toEqual([0.3])
    expect(m.bias).toBe(0.1)
  })
})

// ─── Aprende: AND y OR (binario) ─────────────────────────────────────────────

describe('LogisticRegression: aprende AND y OR (pérdida < 0.05, margen 30 %)', () => {
  const AND_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
  const AND_Y = [0, 0, 0, 1]
  const OR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
  const OR_Y = [0, 1, 1, 1]

  function worstCase(X: number[][], Y: number[]) {
    let worstLoss = 0
    let minAcc = 1
    for (const seed of SEEDS) {
      for (const lr of LRS) {
        seedRandom(seed)
        const m = new LogisticRegression()
        const hist = m.train(X, Y, lr, EPOCHS)
        expect(hist[hist.length - 1]).toBeLessThan(hist[0])
        const loss = bceMean(m, X, Y)
        const acc = X.filter((x, i) => m.classify(x) === Y[i]).length / X.length
        worstLoss = Math.max(worstLoss, loss)
        minAcc = Math.min(minAcc, acc)
      }
    }
    return { worstLoss, minAcc }
  }

  it('AND: con 5 semillas y lr 0.35 / 0.5 / 0.65 la pérdida media queda bajo 0.7·0.05 y acierto 100 %', () => {
    const { worstLoss, minAcc } = worstCase(AND_X, AND_Y)
    expect(worstLoss).toBeLessThan(0.7 * UMBRAL)
    expect(minAcc).toBe(1)
  })

  it('OR: con 5 semillas y lr 0.35 / 0.5 / 0.65 la pérdida media queda bajo 0.7·0.05 y acierto 100 %', () => {
    const { worstLoss, minAcc } = worstCase(OR_X, OR_Y)
    expect(worstLoss).toBeLessThan(0.7 * UMBRAL)
    expect(minAcc).toBe(1)
  })
})

// ═════════════════════════════════════════════════════════════════════════════
// SoftmaxRegression
// ═════════════════════════════════════════════════════════════════════════════

describe('SoftmaxRegression: predicción con pesos a mano', () => {
  it('softmax de [1, 2, 3] = eᵏ / (e + e² + e³)', () => {
    // Pesos W_k = [k+1] (una característica), b = 0, x = [1] → scores = [1, 2, 3].
    // Σ = e + e² + e³ ≈ 30.1929; p = [e, e², e³] / Σ ≈ [0.0900, 0.2447, 0.6652].
    const m = new SoftmaxRegression()
    m.weights = [[1], [2], [3]]
    m.biases = [0, 0, 0]
    const e = Math.E
    const sum = e + e * e + e ** 3
    const probs = m.predictProba([1])
    expect(probs).toHaveLength(3)
    expect(probs[0]).toBeCloseTo(e / sum, 12)
    expect(probs[1]).toBeCloseTo((e * e) / sum, 12)
    expect(probs[2]).toBeCloseTo((e ** 3) / sum, 12)
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('predict devuelve el índice de la clase con mayor score (argmax)', () => {
    // Scores [1, 2, 3] del caso anterior → clase 2. Con x = [−1] → scores [−1, −2, −3] → clase 0.
    const m = new SoftmaxRegression()
    m.weights = [[1], [2], [3]]
    m.biases = [0, 0, 0]
    expect(m.predict([1])).toBe(2)
    expect(m.predict([-1])).toBe(0)
  })

  it('estable con logits grandes: scores [1000, 1001, 1002] dan softmax([−2, −1, 0]), sin NaN', () => {
    // Sin restar el máximo, exp(1000) = Infinity y el resultado sería NaN.
    // Con invariancia a la traslación: softmax([1000,1001,1002]) = softmax([−2,−1,0])
    //   = [e⁻², e⁻¹, 1] / (e⁻² + e⁻¹ + 1).
    const m = new SoftmaxRegression()
    m.weights = [[1000], [1001], [1002]]
    m.biases = [0, 0, 0]
    const d = Math.exp(-2) + Math.exp(-1) + 1
    const probs = m.predictProba([1])
    expect(probs.every(Number.isFinite)).toBe(true)
    expect(probs[0]).toBeCloseTo(Math.exp(-2) / d, 12)
    expect(probs[1]).toBeCloseTo(Math.exp(-1) / d, 12)
    expect(probs[2]).toBeCloseTo(1 / d, 12)
    expect(m.predict([1])).toBe(2)
  })
})

describe('SoftmaxRegression: un paso de train a mano', () => {
  it('2 clases, W = [[0],[0]], b = [0,0], x = [2], y = 1, lr = 0.5 → W = [[−0.5],[0.5]], b = [−0.25, 0.25]', () => {
    // scores = [0, 0] → P = [0.5, 0.5]. δ_k = P_k − 1{k=1}: δ = [0.5, −0.5].
    // W_k −= lr·δ_k·x: W_0 = 0 − 0.5·0.5·2 = −0.5; W_1 = 0 − 0.5·(−0.5)·2 = 0.5.
    // b_k −= lr·δ_k: b_0 = −0.25; b_1 = 0.25.
    // Pérdida de la muestra antes del paso: −ln 0.5 = ln 2 ≈ 0.693147.
    const m = new SoftmaxRegression()
    m.weights = [[0], [0]]
    m.biases = [0, 0]
    const hist = m.train([[2]], [1], 0.5, 1)
    expect(hist[0]).toBeCloseTo(Math.log(2), 12)
    expect(m.weights[0][0]).toBeCloseTo(-0.5, 12)
    expect(m.weights[1][0]).toBeCloseTo(0.5, 12)
    expect(m.biases[0]).toBeCloseTo(-0.25, 12)
    expect(m.biases[1]).toBeCloseTo(0.25, 12)
  })

  it('inicializa K = max(y) + 1 clases con pesos acotados por √(2/n)', () => {
    // y = [0, 2] → K = 3 clases. n = 1 → límite inicial √(2/1) = √2.
    // Con x = [0] la actualización lr·δ·x es 0, así que los pesos quedan en su valor inicial.
    const m = new SoftmaxRegression()
    m.train([[0], [0]], [0, 2], 0.1, 1)
    expect(m.weights).toHaveLength(3)
    expect(m.biases).toHaveLength(3)
    const limit = Math.sqrt(2)
    for (const row of m.weights) {
      expect(row).toHaveLength(1)
      expect(Math.abs(row[0])).toBeLessThanOrEqual(limit)
    }
  })
})

describe('SoftmaxRegression: errores', () => {
  it('predictProba sin entrenar lanza', () => {
    expect(() => new SoftmaxRegression().predictProba([1])).toThrow(
      'SoftmaxRegression.predictProba: model has not been trained yet',
    )
  })

  it('predictProba con número de sesgos distinto de K lanza', () => {
    const m = new SoftmaxRegression()
    m.weights = [[1], [2]]
    m.biases = [0]
    expect(() => m.predictProba([1])).toThrow('SoftmaxRegression.predictProba: expected 2 biases, got 1')
  })

  it('predictProba con filas de pesos de distinta longitud lanza', () => {
    const m = new SoftmaxRegression()
    m.weights = [[1], [2, 3]]
    m.biases = [0, 0]
    expect(() => m.predictProba([1])).toThrow(
      'SoftmaxRegression.predictProba: weights row 1 has length 2, expected 1',
    )
  })

  it('predictProba con x de longitud distinta o no finita lanza', () => {
    const m = new SoftmaxRegression()
    m.weights = [[1], [2]]
    m.biases = [0, 0]
    expect(() => m.predictProba([1, 2])).toThrow('SoftmaxRegression.predictProba: expected array of length 1, got 2')
    expect(() => m.predictProba([NaN])).toThrow('SoftmaxRegression.predictProba: invalid value at index 0: NaN')
  })

  it('train: X vacía, filas y etiquetas de distinto tamaño, lr y epochs inválidos', () => {
    const m = new SoftmaxRegression()
    expect(() => m.train([], [], 0.1, 1)).toThrow('SoftmaxRegression.train: X is empty')
    expect(() => m.train([[1], [2]], [0], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: X has 2 rows but y has 1 labels',
    )
    expect(() => m.train([[1], [2]], [0, 1], Infinity, 1)).toThrow('SoftmaxRegression.train: expected finite number')
    expect(() => m.train([[1], [2]], [0, 1], 0, 1)).toThrow('SoftmaxRegression.train: lr must be positive')
    expect(() => m.train([[1], [2]], [0, 1], 0.1, 0)).toThrow(
      'SoftmaxRegression.train: epochs must be a positive integer',
    )
    expect(() => m.train([[1], [2]], [0, 1], 0.1, 2.5)).toThrow(
      'SoftmaxRegression.train: epochs must be a positive integer',
    )
  })

  it('train: etiquetas no enteras o negativas, y una sola clase lanzan', () => {
    const m = new SoftmaxRegression()
    expect(() => m.train([[1], [2]], [0.5, 1], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: label y[0] = 0.5 must be an integer >= 0',
    )
    expect(() => m.train([[1], [2]], [-1, 0], 0.1, 1)).toThrow('must be an integer >= 0')
    expect(() => m.train([[1], [2]], [0, 0], 0.1, 1)).toThrow('SoftmaxRegression.train: need at least 2 classes in y')
  })

  it('train: una etiqueta >= K del modelo existente lanza; no altera el modelo', () => {
    const m = new SoftmaxRegression()
    m.weights = [[0], [0]]
    m.biases = [0, 0]
    expect(() => m.train([[1]], [2], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: label y[0] = 2 is >= 2 classes of the existing model',
    )
    expect(m.weights).toEqual([[0], [0]])
    expect(m.biases).toEqual([0, 0])
  })

  it('train rechaza X[0] no array y filas de distinta longitud antes de tocar el modelo', () => {
    const m = new SoftmaxRegression()
    expect(() => m.train([1 as unknown as number[]], [0], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: X[0]: expected array, got number',
    )
    expect(() => m.train([[1, 2], [1]], [0, 1], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: X[1]: expected array of length 2, got 1',
    )
    expect(m.weights).toEqual([])
    expect(m.biases).toEqual([])
  })
})

// ─── Aprende: 3 blobs lineales (multiclase) ──────────────────────────────────

describe('SoftmaxRegression: aprende 3 blobs lineales (pérdida CE < 0.05, margen 30 %)', () => {
  // Centros (−3, 0), (3, 0), (0, 3); 4 puntos por blob con desplazamientos (±0.5, ±0.5).
  // Separación entre blobs ≥ 5, así que son linealmente separables.
  const OFFS = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
  const CENTERS = [[-3, 0], [3, 0], [0, 3]]
  const BLOB_X: number[][] = []
  const BLOB_Y: number[] = []
  CENTERS.forEach((c, k) => {
    OFFS.forEach((o) => {
      BLOB_X.push([c[0] + o[0], c[1] + o[1]])
      BLOB_Y.push(k)
    })
  })

  it('con 5 semillas y lr 0.35 / 0.5 / 0.65 la CE media queda bajo 0.7·0.05 y acierto 100 %', () => {
    let worstLoss = 0
    let minAcc = 1
    for (const seed of SEEDS) {
      for (const lr of LRS) {
        seedRandom(seed)
        const m = new SoftmaxRegression()
        const hist = m.train(BLOB_X, BLOB_Y, lr, EPOCHS)
        expect(hist[hist.length - 1]).toBeLessThan(hist[0])
        worstLoss = Math.max(worstLoss, ceMean(m, BLOB_X, BLOB_Y))
        const acc = BLOB_X.filter((x, i) => m.predict(x) === BLOB_Y[i]).length / BLOB_X.length
        minAcc = Math.min(minAcc, acc)
      }
    }
    expect(worstLoss).toBeLessThan(0.7 * UMBRAL)
    expect(minAcc).toBe(1)
  })
})
