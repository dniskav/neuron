// Tests de GaussianNaiveBayes (src/NaiveBayes.ts).
//
// Regla de valores (FR-004): cada cifra numérica lleva su derivación a mano en
// un comentario. Ninguna se copia de la salida del código.
//
// Modelo: por clase c y rasgo j, μ = media, σ² = varianza poblacional (÷ n),
// con σ² mínimo 1e-9 (suavizado del código). Prior π_c = n_c / m.
// Log-verosimilitud de un rasgo: −½·ln(2πσ²) − (x−μ)²/(2σ²).
//
// Conjuntos de referencia (1-D salvo que se indique):
//
//   D1: X = [-1, 1, 3, 5], y = [0, 0, 1, 1]
//       clase 0: valores -1, 1 → μ = 0, σ² = (1+1)/2 = 1
//       clase 1: valores 3, 5  → μ = 4, σ² = (1+1)/2 = 1
//       priores: 2/4 = 1/2 y 1/2 (iguales).
//       Cociente de log-verosimilitudes (clase 0 frente a clase 1), con σ² = 1 en ambas:
//         ln N(x;0,1) − ln N(x;4,1) = −x²/2 + (x−4)²/2 = −4x + 8
//       Con priores iguales, la frontera está en x = 2, y en ella P(0|x) = P(1|x) = 1/2.
//       P(0|x) = 1 / (1 + e^{4x−8}).
//         x = 3: P(0|3) = 1/(1+e⁴) ≈ 0.0179862100
//
//   D2: X = [-1, -1, 1, 1, 3, 5], y = [0, 0, 0, 0, 1, 1]
//       clase 0: valores -1, -1, 1, 1 → μ = 0, σ² = 4/4 = 1
//       clase 1: valores 3, 5 → μ = 4, σ² = 1
//       priores: 4/6 = 2/3 (clase 0) y 2/6 = 1/3 (clase 1); ln(π0/π1) = ln 2.
//       Cociente de log-posteriores: ln 2 − 4x + 8 = 0 →
//         frontera x* = 2 + ln2 / 4 ≈ 2.1732867951
//       Las priores desplazan la frontera hacia la clase 1 (con priores iguales era 2).
//       P(0|x) = 1 / (1 + ½·e^{4x−8}) = 2 / (2 + e^{4x−8}).
//         x = 2.1: P(0|2.1) = 2/(2+e^{0.4}) ≈ 0.5727664397
//         x = x*:  P(0|x*) = 2/(2+e^{ln2}) = 2/4 = 1/2
//
//   D3 (2-D): X = [[-1,-2], [1,2], [3,6], [5,10]], y = [0, 0, 1, 1]
//       clase 0: rasgo 1 → μ = 0, σ² = 1; rasgo 2 → valores -2, 2: μ = 0, σ² = (4+4)/2 = 4
//       clase 1: rasgo 1 → μ = 4, σ² = 1; rasgo 2 → valores 6, 10: μ = 8, σ² = (4+4)/2 = 4
//       priores 1/2 y 1/2. Las varianzas coinciden entre clases, así que los términos
//       ln(2πσ²) se cancelan. Punto x = (2, 8):
//         rasgo 1: −(2−0)²/2 − [−(2−4)²/2] = −2 − (−2) = 0
//         rasgo 2: −(8−0)²/(2·4) − [−(8−8)²/(2·4)] = −8 − 0 = −8
//         log-odds(0 frente a 1) = −8  →  P(0|x) = 1/(1+e⁸) ≈ 0.00033535013
//       Si el modelo ignorase el segundo rasgo, P(0|x) sería 1/2.
//
//   D4 (rasgo constante por clase): X = [[1], [1], [5], [5]], y = [0, 0, 1, 1]
//       clase 0: σ² = 0 → suavizado a 1e-9; μ = 1
//       clase 1: σ² = 0 → suavizado a 1e-9; μ = 5
//       Con σ² = 1e-9 en ambas clases, log-odds(1 frente a 0) en x es
//         [(x−1)² − (x−5)²] / (2e-9) = (8x − 24) / (2e-9).
//       En x = 5: (40 − 24)/(2e-9) = +8e9 → P(1|5) = 1 y P(0|5) ≈ 0.
//       Sin suavizado (σ² = 0) habría 0/0 = NaN en la puntuación de la clase 1 en x = 5.
//
//   D5 (tres clases, etiquetas no contiguas): X = [-1, 1, 3, 5, 9, 11], y = [5, 5, 9, 9, 12, 12]
//       clase 5: μ = 0, σ² = 1; clase 9: μ = 4, σ² = 1; clase 12: μ = 10, σ² = 1.
//       priores 2/6 = 1/3 cada una (iguales). En x = 4, puntuación relativa:
//         clase 5: −(4−0)²/2 = −8; clase 9: 0; clase 12: −(4−10)²/2 = −18.
//         P(9|4) = 1 / (1 + e⁻⁸ + e⁻¹⁸) ≈ 0.9996646346
//       Debe devolver la etiqueta (5, 9 o 12), no el índice.
//
// Aprendizaje: fit es cerrado (sin iteraciones), así que la variación de lr
// ×0.7 / ×1.3 no aplica. Se valida con 5 semillas: los grupos están en
// [−1, 1) y en [9, 11) con centros 0 y 10. Cada punto de prueba dista ≤ 1 de su
// centro. La frontera de la clase 0 frente a la 1 queda en torno a 5 ± 0.5 para
// cualquier muestra de 40 puntos. El cociente de log-verosimilitudes en el borde
// de cada grupo es ≥ 100 en magnitud (ver `blobs`), así que la precisión es
// 1.0 y el error peor-caso (0) queda por debajo de 0.7·(1 − 0.9) = 0.07.
//
// Semilla: seedRandom en beforeEach, restoreRandom en afterEach. Los datos del
// aprendizaje salen de mulberry32 con semilla explícita.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { GaussianNaiveBayes } from '../src/NaiveBayes'
import { mulberry32, seedRandom, restoreRandom } from './helpers/seed'

beforeEach(() => seedRandom(0x5eed))
afterEach(() => restoreRandom())

const D1X = [[-1], [1], [3], [5]]
const D1Y = [0, 0, 1, 1]
const D2X = [[-1], [-1], [1], [1], [3], [5]]
const D2Y = [0, 0, 0, 0, 1, 1]
const D3X = [[-1, -2], [1, 2], [3, 6], [5, 10]]
const D3Y = [0, 0, 1, 1]
const D4X = [[1], [1], [5], [5]]
const D4Y = [0, 0, 1, 1]
const D5X = [[-1], [1], [3], [5], [9], [11]]
const D5Y = [5, 5, 9, 9, 12, 12]

function fitted(X: number[][], y: number[]): GaussianNaiveBayes {
  const model = new GaussianNaiveBayes()
  model.fit(X, y)
  return model
}

// ─── fit + predict: frontera con priores iguales ─────────────────────────────

describe('predict con priores iguales (D1)', () => {
  it('las medias 0 y 4 con varianza 1 dan frontera en 2', () => {
    const model = fitted(D1X, D1Y)
    expect(model.predict([0])).toBe(0) // lejos de 2, del lado de la media 0
    expect(model.predict([4])).toBe(1) // del lado de la media 4
    expect(model.predict([1.9])).toBe(0) // 4·1.9−8 = −0.4 < 0 → clase 0 (ver cabecera)
    expect(model.predict([2.1])).toBe(1) // 4·2.1−8 = 0.4 > 0 → clase 1
  })

  it('en la frontera x = 2 la probabilidad es 0.5 para cada clase', () => {
    const p = fitted(D1X, D1Y).predictProba([2])
    // 1/(1+e^{4·2−8}) = 1/(1+e⁰) = 1/2
    expect(p.get(0)).toBeCloseTo(0.5, 9)
    expect(p.get(1)).toBeCloseTo(0.5, 9)
  })

  it('en x = 3 la probabilidad de la clase 0 es 1/(1+e⁴)', () => {
    const p = fitted(D1X, D1Y).predictProba([3])
    // P(0|3) = 1/(1+e^{4·3−8}) = 1/(1+e⁴) ≈ 0.0179862100
    expect(p.get(0)).toBeCloseTo(1 / (1 + Math.exp(4)), 9)
    expect(p.get(1)).toBeCloseTo(Math.exp(4) / (1 + Math.exp(4)), 9)
  })
})

// ─── priores desiguales desplazan la frontera ────────────────────────────────

describe('priores desiguales (D2)', () => {
  it('la frontera se desplaza a 2 + ln2/4 ≈ 2.1733 hacia la clase 1', () => {
    const model = fitted(D2X, D2Y)
    // Con priores iguales, x = 2.1 sería clase 1 (ver D1). Aquí la prior de la
    // clase 0 (2/3) gana: ln2 + 8 − 4·2.1 = 0.6931 − 0.4 = 0.2931 > 0 → clase 0.
    expect(model.predict([2.1])).toBe(0)
    // x = 2.25: ln2 + 8 − 9 = −0.3069 < 0 → clase 1.
    expect(model.predict([2.25])).toBe(1)
  })

  it('en x = 2.1 la probabilidad de la clase 0 es 2/(2+e^{0.4}) ≈ 0.5727664397', () => {
    const p = fitted(D2X, D2Y).predictProba([2.1])
    expect(p.get(0)).toBeCloseTo(2 / (2 + Math.exp(0.4)), 9)
  })

  it('en la frontera x* = 2 + ln2/4 la probabilidad vale 1/2', () => {
    // Derivación: 2/(2+e^{4x*−8}) = 2/(2+e^{ln2}) = 2/4 = 1/2.
    const xStar = 2 + Math.log(2) / 4
    const p = fitted(D2X, D2Y).predictProba([xStar])
    expect(p.get(0)).toBeCloseTo(0.5, 9)
  })
})

// ─── varias características: la verosimilitud es un producto ────────────────

describe('varias características (D3)', () => {
  it('el punto (2, 8) es de la clase 1 por el segundo rasgo', () => {
    const model = fitted(D3X, D3Y)
    expect(model.predict([2, 8])).toBe(1)
    expect(model.predict([0, 0])).toBe(0)
  })

  it('P(0|2,8) = 1/(1+e⁸) ≈ 0.00033535013 (el rasgo 1 no aporta diferencia)', () => {
    const p = fitted(D3X, D3Y).predictProba([2, 8])
    expect(p.get(0)).toBeCloseTo(1 / (1 + Math.exp(8)), 9)
  })
})

// ─── tres clases con etiquetas no contiguas ─────────────────────────────────

describe('tres clases con etiquetas 5, 9 y 12 (D5)', () => {
  it('predict devuelve la etiqueta original', () => {
    const model = fitted(D5X, D5Y)
    expect(model.predict([0])).toBe(5)
    expect(model.predict([4])).toBe(9)
    expect(model.predict([10])).toBe(12)
  })

  it('P(9|4) = 1/(1+e⁻⁸+e⁻¹⁸) ≈ 0.9996646346 y las claves son las etiquetas', () => {
    const p = fitted(D5X, D5Y).predictProba([4])
    expect([...p.keys()]).toEqual([5, 9, 12])
    expect(p.get(9)).toBeCloseTo(1 / (1 + Math.exp(-8) + Math.exp(-18)), 9)
  })
})

// ─── predictProba: distribución válida ──────────────────────────────────────

describe('predictProba', () => {
  it('devuelve un Map cuyas probabilidades suman 1 en cada punto', () => {
    const model = fitted(D2X, D2Y)
    for (const x of [-3, 0, 2, 2.1, 4, 7]) {
      const p = model.predictProba([x])
      expect(p).toBeInstanceOf(Map)
      const total = [...p.values()].reduce((s, v) => s + v, 0)
      expect(total).toBeCloseTo(1, 9) // suma 1
      for (const v of p.values()) {
        expect(v).toBeGreaterThanOrEqual(0)
        expect(v).toBeLessThanOrEqual(1)
      }
    }
  })

  it('suma 1 también con tres clases', () => {
    const p = fitted(D5X, D5Y).predictProba([7])
    const total = [...p.values()].reduce((s, v) => s + v, 0)
    expect(total).toBeCloseTo(1, 9)
  })
})

// ─── varianza mínima (suavizado 1e-9) ───────────────────────────────────────

describe('rasgo constante dentro de cada clase (D4)', () => {
  it('no produce NaN y clasifica en los valores de la media', () => {
    const model = fitted(D4X, D4Y)
    expect(model.predict([1])).toBe(0)
    expect(model.predict([5])).toBe(1)
  })

  it('P(1|5) ≈ 1 y todas las probabilidades son finitas', () => {
    const p = fitted(D4X, D4Y).predictProba([5])
    expect(Number.isFinite(p.get(0)!)).toBe(true)
    expect(Number.isFinite(p.get(1)!)).toBe(true)
    // log-odds(1 frente a 0) = (5−1)²/(2e-9) ≈ 8e9 → P(1|5) = 1 con precisión de doble.
    expect(p.get(1)).toBeCloseTo(1, 9)
  })
})

// ─── refit reemplaza el modelo anterior ─────────────────────────────────────

describe('refit', () => {
  it('un segundo fit con otras clases sustituye a las anteriores', () => {
    const model = fitted(D1X, D1Y)
    model.fit(D5X, D5Y)
    expect([...model.predictProba([4]).keys()]).toEqual([5, 9, 12])
    expect(model.predict([4])).toBe(9)
  })
})

// ─── aprendizaje con semillas ───────────────────────────────────────────────

/** Dos grupos 1-D: centro 0 y centro 10, ruido uniforme en [−1, 1). */
function blobs(seed: number, perClass: number): { X: number[][]; y: number[] } {
  const rnd = mulberry32(seed)
  const X: number[][] = []
  const y: number[] = []
  for (let i = 0; i < perClass; i++) {
    X.push([rnd() * 2 - 1]) // centro 0
    y.push(0)
    X.push([10 + (rnd() * 2 - 1)]) // centro 10
    y.push(1)
  }
  return { X, y }
}

describe('aprende dos grupos separados (5 semillas)', () => {
  it.each([1, 2, 3, 4, 5])('semilla %i: precisión 1.0 en datos nuevos', (seed) => {
    const train = blobs(seed, 40)
    const test = blobs(seed + 100, 40)
    const model = fitted(train.X, train.y)
    const hits = test.X.filter((x, i) => model.predict(x) === test.y[i]).length
    const accuracy = hits / test.X.length
    // Peor caso: error 0 ≤ 0.7·(1 − 0.9) = 0.07 → precisión ≥ 0.93.
    expect(accuracy).toBeGreaterThanOrEqual(0.93)
    expect(accuracy).toBe(1)
  })
})

// ─── errores ────────────────────────────────────────────────────────────────

describe('errores', () => {
  it('fit lanza con X vacía', () => {
    expect(() => new GaussianNaiveBayes().fit([], [])).toThrow(/X is empty/)
  })

  it('fit lanza si X e y tienen longitudes distintas (2 filas, 1 etiqueta)', () => {
    expect(() => new GaussianNaiveBayes().fit([[1], [2]], [0])).toThrow(
      /2 rows but y has 1 labels/
    )
  })

  it('predict y predictProba lanzan antes de fit', () => {
    const model = new GaussianNaiveBayes()
    expect(() => model.predict([1])).toThrow(/not been fitted/)
    expect(() => model.predictProba([1])).toThrow(/not been fitted/)
  })

  it('predict lanza con un vector de dimensión distinta (1 rasgo, se pasan 2)', () => {
    expect(() => fitted(D1X, D1Y).predict([1, 2])).toThrow(/expected 1 features, got 2/)
  })

  it('predictProba lanza con un vector vacío', () => {
    expect(() => fitted(D1X, D1Y).predictProba([])).toThrow(/expected 1 features, got 0/)
  })
})
