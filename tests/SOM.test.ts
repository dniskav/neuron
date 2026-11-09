// Tests de SOM (src/SOM.ts; mapa autoorganizado de Kohonen). Feature 006, T024.
//
// Regla de valores (FR-004): cada cifra numérica lleva su derivación a mano en un
// comentario. Ninguna se copia de la salida del código.
//
// Complementa tests/oracle/SOM.oracle.test.ts (regla 1×3 y "aprende" de 30 pasos);
// aquí se prueban la rejilla 2×2, la distancia del error de cuantización, el
// decaimiento temporal de lr y σ, y el aprendizaje con 5 semillas y 3 tasas.
//
// Fórmulas (src/SOM.ts): BMU = argmin ‖x − w‖² (empate: la primera en orden
// fila→columna, comparación estricta); h = exp(−d²/2σ²) con d² = Δfila² + Δcol²;
// w ← w + lr·h·(x − w); lr(t) = lr₀·(lr_f/lr₀)^t; σ(t) = σ₀·(σ_f/σ₀)^t con
// t = paso/(épocas·n) ∈ [0,1). QE = (1/n)·Σ ‖xᵢ − w[BMU(xᵢ)]‖.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach. Los pesos
// iniciales y el orden de presentación usan Math.random; los datos de "aprende"
// usan mulberry32 con semilla explícita.
//
// Aprendizaje ("aprende"): 40 puntos en 4 grupos de 10 alrededor de (5,5), (9,1),
// (1,9) y (9,9), con ruido uniforme en [−0.5, 0.5) por coordenada. Se entrena una
// rejilla 3×3 durante 20 épocas. Umbral: QE_final / QE_inicial < 0.7 (bajar ≥ 30 %).
// Margen: el peor caso de 5 semillas × lr ×0.7 / ×1.0 / ×1.3 debe quedar ≤ 0.7·0.7 =
// 0.49. Cifra medida en la validación (no es un expect): peor ratio 0.169, con
// QE_inicial ≈ 8.4 y QE_final ≈ 1.41.
//
// Entradas inválidas (NaN, QE de una X vacía, tamaños no enteros, filas posteriores
// erróneas) lanzan error y se prueban en tests/api/SOM.validation.test.ts (T056).

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { SOM } from '../src/SOM'
import { mulberry32, seedRandom, restoreRandom } from './helpers/seed'

/** Compara un vector con una tolerancia de 10^-digits por coordenada. */
function expectVec(actual: number[], expected: number[], digits = 12): void {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], digits))
}

/** Cuatro grupos de 10 puntos alrededor de (5,5), (9,1), (1,9) y (9,9). */
function blobs(): number[][] {
  const centers = [[5, 5], [9, 1], [1, 9], [9, 9]]
  const rnd = mulberry32(99)
  const X: number[][] = []
  for (const c of centers) {
    for (let i = 0; i < 10; i++) X.push([c[0] + rnd() - 0.5, c[1] + rnd() - 0.5])
  }
  return X
}

describe('SOM', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  describe('constructor', () => {
    it('forma [rows][cols][inputSize]; pesos iniciales en [0,1) en orden filas→columnas→dimensión', () => {
      // El k-ésimo sorteo de mulberry32(7) va a weights[r][c][i] con
      // k = (r·cols + c)·inputSize + i (rows = 2, cols = 3, inputSize = 4).
      seedRandom(7)
      const som = new SOM(2, 3, 4)
      const draws = mulberry32(7)
      const flat = Array.from({ length: 2 * 3 * 4 }, () => draws())
      expect(som.weights.length).toBe(2)
      for (let r = 0; r < 2; r++) {
        expect(som.weights[r].length).toBe(3)
        for (let c = 0; c < 3; c++) {
          expect(som.weights[r][c].length).toBe(4)
          for (let i = 0; i < 4; i++) {
            expect(som.weights[r][c][i]).toBe(flat[(r * 3 + c) * 4 + i])
            expect(som.weights[r][c][i]).toBeGreaterThanOrEqual(0)
            expect(som.weights[r][c][i]).toBeLessThan(1)
          }
        }
      }
    })

    it('misma semilla → mismos pesos iniciales', () => {
      seedRandom(7)
      const a = new SOM(2, 2, 3)
      seedRandom(7)
      const b = new SOM(2, 2, 3)
      expect(a.weights).toEqual(b.weights)
    })

    it('rechaza rows, cols o inputSize menores que 1', () => {
      expect(() => new SOM(0, 2, 1)).toThrow(/rows, cols and inputSize must be positive integers/)
      expect(() => new SOM(2, 0, 1)).toThrow(/rows, cols and inputSize must be positive integers/)
      expect(() => new SOM(2, 2, 0)).toThrow(/rows, cols and inputSize must be positive integers/)
      expect(() => new SOM(-1, 2, 1)).toThrow(/rows, cols and inputSize must be positive integers/)
    })
  })

  describe('getBMU y predict', () => {
    // Rejilla 2×2, pesos fijados: (0,0)=[1,2], (0,1)=[5,5], (1,0)=[−3,0], (1,1)=[2,−2].
    function fixedGrid(): SOM {
      const som = new SOM(2, 2, 2)
      som.weights = [
        [[1, 2], [5, 5]],
        [[-3, 0], [2, -2]],
      ]
      return som
    }

    it('elige la neurona de menor distancia² en una rejilla 2×2 (cuentas a mano)', () => {
      const som = fixedGrid()
      // x = [1,1]:  d² a (0,0)=0²+1²=1; (0,1)=4²+4²=32; (1,0)=4²+1²=17; (1,1)=1²+3²=10 → (0,0)
      expect(som.getBMU([1, 1])).toEqual([0, 0])
      // x = [4,4]:  d² = 13; 2; 65; 40 → (0,1)
      expect(som.getBMU([4, 4])).toEqual([0, 1])
      // x = [−2,0]: d² = 13; 74; 1; 20 → (1,0)
      expect(som.getBMU([-2, 0])).toEqual([1, 0])
      // x = [2,−3]: d² = 26; 73; 34; 1 → (1,1)
      expect(som.getBMU([2, -3])).toEqual([1, 1])
    })

    it('el caso [1,1] no coincide con la neurona más lejana (32): no es argmax', () => {
      // Si la búsqueda fuera de máxima distancia, [1,1] elegiría (0,1) con d² = 32.
      expect(fixedGrid().getBMU([1, 1])).not.toEqual([0, 1])
    })

    it('predict devuelve lo mismo que getBMU', () => {
      const som = fixedGrid()
      for (const x of [[1, 1], [4, 4], [-2, 0], [2, -3]]) {
        expect(som.predict(x)).toEqual(som.getBMU(x))
      }
    })

    it('empate: gana la primera neurona en orden fila→columna (comparación estricta)', () => {
      // x = [1]: d² = 1 para w = 0 y para w = 2 → gana (0,0), la primera.
      const som = new SOM(1, 2, 1)
      som.weights = [[[0], [2]]]
      expect(som.getBMU([1])).toEqual([0, 0])
      // Misma situación en orden columna: (0,0) y (1,0) empatan → gana (0,0).
      const col = new SOM(2, 1, 1)
      col.weights = [[[0]], [[2]]]
      expect(col.getBMU([1])).toEqual([0, 0])
    })

    it('lanza si la longitud de la entrada no coincide con inputSize (getBMU y predict)', () => {
      const som = new SOM(2, 2, 2)
      expect(() => som.getBMU([1])).toThrow(/expected input of length 2, got 1/)
      expect(() => som.predict([1, 2, 3])).toThrow(/expected input of length 2, got 3/)
    })
  })

  describe('quantizationError', () => {
    it('media de distancias euclídeas (no cuadradas) al BMU: 1×2, w = {0, 10}', () => {
      // x = 1 → BMU (0,0), d = 1. x = 4 → BMU (0,0), d = 4 (|4−10| = 6 es mayor).
      // x = 9 → BMU (0,1), d = 1 (|9−0| = 9 es mayor).
      // QE = (1 + 4 + 1)/3 = 2. Con distancias al cuadrado sería (1 + 16 + 1)/3 = 6.
      const som = new SOM(1, 2, 1)
      som.weights = [[[0], [10]]]
      expect(som.quantizationError([[1], [4], [9]])).toBeCloseTo(2, 12)
    })

    it('2-D: una neurona en (3,4) con puntos (0,0) y (3,4): QE = (5 + 0)/2 = 2.5', () => {
      // d((0,0),(3,4)) = √(9+16) = 5; d((3,4),(3,4)) = 0.
      const som = new SOM(1, 1, 2)
      som.weights = [[[3, 4]]]
      expect(som.quantizationError([[0, 0], [3, 4]])).toBeCloseTo(2.5, 12)
    })

    it('vale 0 cuando cada punto coincide con su prototipo', () => {
      const som = new SOM(1, 1, 2)
      som.weights = [[[3, 4]]]
      expect(som.quantizationError([[3, 4], [3, 4]])).toBe(0)
    })
  })

  describe('train: un paso con lr y radio conocidos', () => {
    it('rejilla 2×2: BMU y vecinas con h = exp(−d²/2σ²), lr y σ constantes', () => {
      // lr = 0.5 (inicial = final), σ = 1 (inicial = final), t = 0.
      // Pesos: (0,0)=[0,0], (0,1)=[2,0], (1,0)=[0,2], (1,1)=[2,2]. x = [1.8, 0.2].
      // d²(x, w): (0,0) = 3.24+0.04 = 3.28; (0,1) = 0.04+0.04 = 0.08; (1,0) = 3.24+3.24 = 6.48;
      // (1,1) = 0.04+3.24 = 3.28. BMU = (0,1) (único mínimo).
      // Distancias de rejilla desde (0,1): (0,0) d²=1; (1,1) d²=1; (1,0) d²=2; (0,1) d²=0.
      // h(d²=0) = 1; h(d²=1) = e^(−1/2); h(d²=2) = e^(−1).
      const e = Math.exp(-0.5)
      const f = Math.exp(-1)
      const som = new SOM(2, 2, 2, { initialLr: 0.5, finalLr: 0.5, initialSigma: 1, finalSigma: 1 })
      som.weights = [
        [[0, 0], [2, 0]],
        [[0, 2], [2, 2]],
      ]
      som.train([[1.8, 0.2]], 1)
      // BMU (h = 1): [2, 0] + 0.5·1·([1.8,0.2] − [2,0]) = [1.9, 0.1]
      expectVec(som.weights[0][1], [2 + 0.5 * (1.8 - 2), 0 + 0.5 * (0.2 - 0)])
      // (0,0) (h = e^(−1/2)): [0,0] + 0.5·e·[1.8,0.2]
      expectVec(som.weights[0][0], [0 + 0.5 * e * (1.8 - 0), 0 + 0.5 * e * (0.2 - 0)])
      // (1,1) (h = e^(−1/2)): [2,2] + 0.5·e·([1.8,0.2] − [2,2])
      expectVec(som.weights[1][1], [2 + 0.5 * e * (1.8 - 2), 2 + 0.5 * e * (0.2 - 2)])
      // (1,0) (h = e^(−1)): [0,2] + 0.5·f·([1.8,0.2] − [0,2])
      expectVec(som.weights[1][0], [0 + 0.5 * f * (1.8 - 0), 2 + 0.5 * f * (0.2 - 2)])
    })

    it('1×1: la única neurona avanza lr·(x − w) (h = 1 siempre)', () => {
      // w = 2, x = 4, lr = 0.25 constante: 2 + 0.25·(4 − 2) = 2.5.
      const som = new SOM(1, 1, 1, { initialLr: 0.25, finalLr: 0.25, initialSigma: 1, finalSigma: 1 })
      som.weights = [[[2]]]
      som.train([[4]], 1)
      expect(som.weights[0][0][0]).toBeCloseTo(2.5, 12)
    })

    it('corte de vecindad: h < 1e-6 deja la neurona intacta; h ≥ 1e-6 se actualiza', () => {
      // 1×7 rejilla, σ = 1 constante, lr = 0.5 constante. x = [0]; pesos: BMU (0,0) = 0 y
      // el resto = 1. BMU = (0,0) sin empate. Para la neurona (0,c): d² = c².
      //   c = 1: h = e^(−1/2) ≈ 0.607 ⇒ w = 1 − 0.5·e^(−1/2).
      //   c = 5: d² = 25, h = e^(−12.5) ≈ 3.7e−6 ≥ 1e−6 ⇒ w = 1 − 0.5·e^(−12.5).
      //   c = 6: d² = 36, h = e^(−18) ≈ 1.5e−8 < 1e−6 ⇒ se salta; w = 1 exacto.
      // Sin el corte, c = 6 cambiaría en 0.5·1.5e−8 ≈ 7.6e−9 y toBe(1) lo detectaría.
      const som = new SOM(1, 7, 1, { initialLr: 0.5, finalLr: 0.5, initialSigma: 1, finalSigma: 1 })
      som.weights = [[[0], [1], [1], [1], [1], [1], [1]]]
      som.train([[0]], 1)
      expect(som.weights[0][0][0]).toBe(0)
      expect(som.weights[0][1][0]).toBeCloseTo(1 - 0.5 * Math.exp(-0.5), 12)
      expect(som.weights[0][5][0]).toBeCloseTo(1 - 0.5 * Math.exp(-12.5), 12)
      expect(som.weights[0][6][0]).toBe(1)
    })

    it('decaimiento temporal: lr y σ bajan entre la muestra 1 (t = 0) y la 2 (t = 1/2)', () => {
      // 1×2 rejilla, pesos {0, 1}; X = [[0],[0]] (las dos muestras idénticas hacen que el
      // orden de presentación no importe), 1 época ⇒ total = 2 pasos.
      // Paso 1 (t = 0): lr₀ = 0.5, σ₀ = 1, 2σ² = 2. BMU = (0,0) (d = 0, w = 0 no cambia).
      //   Vecina (0,1), d² = 1, h = e^(−1/2): w₁ = 1 + 0.5·e^(−1/2)·(0 − 1) = 1 − 0.5·e^(−1/2).
      // Paso 2 (t = 1/2): lr = 0.5·(0.01/0.5)^(1/2) = 0.5·√0.02;
      //   σ = 1·(0.5/1)^(1/2) = √0.5, 2σ² = 1, d² = 1 ⇒ h = e^(−1).
      //   Vecina: w₁ ← w₁·(1 − 0.5·√0.02·e^(−1)).
      // Resultado: w₁ = (1 − 0.5·e^(−1/2))·(1 − 0.5·√0.02·e^(−1)).
      // Si σ no decayera, h en el paso 2 sería e^(−1/2); si lr no decayera, lr = 0.5.
      const som = new SOM(1, 2, 1, { initialLr: 0.5, finalLr: 0.01, initialSigma: 1, finalSigma: 0.5 })
      som.weights = [[[0], [1]]]
      som.train([[0], [0]], 1)
      const expected = (1 - 0.5 * Math.exp(-0.5)) * (1 - 0.5 * Math.sqrt(0.02) * Math.exp(-1))
      expect(som.weights[0][1][0]).toBeCloseTo(expected, 12)
      expect(som.weights[0][0][0]).toBeCloseTo(0, 12)
    })
  })

  describe('train: errores de entrada', () => {
    it('lanza si X está vacía o es null', () => {
      const som = new SOM(2, 2, 1)
      expect(() => som.train([], 1)).toThrow(/dataset X must be non-empty/)
      expect(() => som.train(null as unknown as number[][], 1)).toThrow(/dataset X must be non-empty/)
    })

    it('lanza si la primera fila no tiene inputSize elementos', () => {
      const som = new SOM(2, 2, 2)
      expect(() => som.train([[1, 2, 3]], 1)).toThrow(/expected input size 2, got 3/)
    })
  })

  describe('aprende: el error de cuantización baja (ver cabecera)', () => {
    it('QE_final / QE_inicial < 0.7 en el peor caso de 5 semillas × lr ×0.7/×1.0/×1.3 (peor ≤ 0.49)', () => {
      const X = blobs()
      for (const factor of [0.7, 1, 1.3]) {
        for (const seed of [1, 2, 3, 4, 5]) {
          seedRandom(seed)
          const som = new SOM(3, 3, 2, { initialLr: 0.5 * factor })
          const q0 = som.quantizationError(X)
          som.train(X, 20)
          const q1 = som.quantizationError(X)
          const ratio = q1 / q0
          expect(ratio, `lr×${factor}, semilla ${seed}`).toBeLessThanOrEqual(0.49)
        }
      }
    })
  })
})
