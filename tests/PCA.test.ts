// Tests de PCA (src/PCA.ts; iteración de potencias + deflación de Hotelling).
//
// Regla de valores (FR-004): cada cifra numérica lleva su derivación a mano en un
// comentario. Ninguna se copia de la salida del código.
//
// ─── Conjunto D (covarianza diagonal) ─────────────────────────────────────────
//   D = {(2,1), (2,−1), (−2,1), (−2,−1)}, media (0,0).
//   Σ = (1/n)·Σ xᵢxᵢᵀ:  Σx² = 4·4 = 16 → 16/4 = 4;  Σy² = 4·1 = 4 → 4/4 = 1;
//   Σxy = 2·1 + 2·(−1) + (−2)·1 + (−2)·(−1) = 0.   Luego Σ = diag(4, 1).
//   Autovalores 4 (eje e1) y 1 (eje e2). Ratios 4/5 = 0.8 y 1/5 = 0.2.
//   DS = D trasladado por (10, −3): media (10, −3), misma covarianza. Ejercita el centrado.
//
// ─── Conjunto L (línea y = x) ─────────────────────────────────────────────────
//   L = {(1,1), (2,2), (3,3)}, media (2,2). Centrados: (−1,−1), (0,0), (1,1).
//   Σ = (1/3)·[[2, 2], [2, 2]] = [[2/3, 2/3], [2/3, 2/3]].
//   Autovalores 4/3 (vector (1,1)/√2) y 0. Traza 4/3.
//   Primer componente ±(1/√2, 1/√2), λ₁ = 4/3, ratio 1 con un único componente.
//
// ─── Error de reconstrucción con un componente ─────────────────────────────────
//   PCA(1) sobre DS: x − x̂ = (y centrada)·e2, cuyo cuadrado es y² = 1 en cada
//   punto. Error cuadrático medio = 1 = λ₂ (varianza descartada).
//
// ─── Transformada de un punto ────────────────────────────────────────────────
//   (12, −2) − (10, −3) = (2, 1): en la base de ejes, z = (±2, ±1).
//
// ─── Tolerancias ─────────────────────────────────────────────────────────────
//   El criterio de parada de la iteración es |dot| − 1 < 1e-10, que deja los
//   vectores a ~1e-5 del eje exacto (medido, 5 semillas). Por eso:
//   - autovalores y ratios (cociente de Rayleigh, error O(θ²)): precisión 1e-6;
//   - ejes y componentes: |·| dentro de 1e-4 (cota ~7× por encima del error medido).
//
// ─── Afirmaciones NO incluidas (HALLAZGOS, ver informe de T005) ────────────────
//   - Reconstrucción exacta con todos los componentes con tolerancia 1e-9: el código
//     da un error máximo de ~2e-5 sobre DS. No se fija aquí.
//   - explainedVarianceRatio con nComponents < p: el código divide por la suma de los
//     autovalores retenidos (siempre suma 1). La fórmula documentada, λₖ / Σⱼ λⱼ sobre
//     toda la varianza, daría 0.8 para DS con un componente. No se fija aquí.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { PCA } from '../src/PCA'
import { seedRandom, restoreRandom } from './helpers/seed'

const D = [[2, 1], [2, -1], [-2, 1], [-2, -1]]
const DS = D.map(([x, y]) => [x + 10, y - 3])
const L = [[1, 1], [2, 2], [3, 3]]

const TOL_EIG = 6 // cifras de precisión de toBeCloseTo para autovalores y ratios
const TOL_AXIS = 1e-4 // tolerancia absoluta para componentes frente a ejes

/** Comprueba |actual − expected| < tol para cada elemento de un vector. */
function expectNearVec(actual: number[], expected: number[], tol: number): void {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(tol))
}

/** Error cuadrático medio por muestra entre dos matrices de filas. */
function meanSquaredRowError(A: number[][], B: number[][]): number {
  let s = 0
  A.forEach((row, i) => row.forEach((v, j) => { s += (v - B[i][j]) ** 2 }))
  return s / A.length
}

describe('PCA', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  describe('constructor', () => {
    it('acepta nComponents entero positivo y empieza sin ajustar', () => {
      for (const k of [1, 2, 5]) {
        const pca = new PCA(k)
        expect(pca.components).toEqual([])
        expect(pca.explainedVariance).toEqual([])
        expect(pca.mean).toEqual([])
      }
    })

    it('rechaza nComponents no entero o no positivo', () => {
      for (const k of [0, -1, 1.5, NaN]) {
        expect(() => new PCA(k)).toThrow(/nComponents must be a positive integer/)
      }
    })
  })

  describe('fit: covarianza diag(4, 1)', () => {
    it('la media es (10, −3): suma 40/4 y −12/4', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expect(pca.mean).toEqual([10, -3])
    })

    it('autovalores 4 y 1 (varianza a lo largo de cada eje)', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expect(pca.explainedVariance[0]).toBeCloseTo(4, TOL_EIG)
      expect(pca.explainedVariance[1]).toBeCloseTo(1, TOL_EIG)
    })

    it('componentes = ejes e1 y e2, con signo libre (se compara |·|)', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const [c1, c2] = pca.components
      expectNearVec([Math.abs(c1[0]), Math.abs(c1[1])], [1, 0], TOL_AXIS)
      expectNearVec([Math.abs(c2[0]), Math.abs(c2[1])], [0, 1], TOL_AXIS)
    })

    it('ratios 4/5 = 0.8 y 1/5 = 0.2 con dos componentes', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const r = pca.explainedVarianceRatio()
      expect(r[0]).toBeCloseTo(0.8, TOL_EIG)
      expect(r[1]).toBeCloseTo(0.2, TOL_EIG)
    })
  })

  describe('fit: línea y = x', () => {
    it('primer componente ±(1/√2, 1/√2), λ₁ = 4/3, ratio 1 con un componente', () => {
      const pca = new PCA(1)
      pca.fit(L)
      const c = pca.components[0]
      const s = 1 / Math.SQRT2
      expectNearVec([Math.abs(c[0]), Math.abs(c[1])], [s, s], TOL_AXIS)
      expect(c[0] * c[1]).toBeGreaterThan(0) // misma dirección que (1,1)
      expect(pca.mean).toEqual([2, 2])
      expect(pca.explainedVariance[0]).toBeCloseTo(4 / 3, TOL_EIG)
      expect(pca.explainedVarianceRatio()[0]).toBeCloseTo(1, 9)
    })
  })

  describe('transform e inverseTransform', () => {
    it('transform de la media es el origen', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expectNearVec(pca.transform([[10, -3]])[0], [0, 0], 1e-12)
    })

    it('transform de (12, −2) da coordenadas (±2, ±1) en la base de ejes', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const z = pca.transform([[12, -2]])
      expect(z.length).toBe(1)
      expect(z[0].length).toBe(2)
      expect(Math.abs(Math.abs(z[0][0]) - 2)).toBeLessThan(TOL_AXIS)
      expect(Math.abs(Math.abs(z[0][1]) - 1)).toBeLessThan(TOL_AXIS)
    })

    it('transform devuelve n filas de nComponents columnas', () => {
      const pca = new PCA(1)
      pca.fit(DS)
      const z = pca.transform(DS)
      expect(z.length).toBe(4)
      z.forEach(row => expect(row.length).toBe(1))
    })

    it('inverseTransform([[0, 0]]) devuelve la media (sin proyección no hay desplazamiento)', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expectNearVec(pca.inverseTransform([[0, 0]])[0], [10, -3], 1e-12)
    })

    it('inverseTransform(transform(X)) tiene forma n × nFeatures', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const R = pca.inverseTransform(pca.transform(DS))
      expect(R.length).toBe(4)
      R.forEach(row => expect(row.length).toBe(2))
    })

    it('reconstrucción con 1 componente: error cuadrático medio = λ₂ = 1', () => {
      // Ver cabecera: x − x̂ = (y centrada)·e2, y² = 1 en cada punto.
      const pca = new PCA(1)
      pca.fit(DS)
      const R = pca.inverseTransform(pca.transform(DS))
      expect(meanSquaredRowError(DS, R)).toBeCloseTo(1, TOL_EIG)
    })
  })

  describe('fitTransform', () => {
    it('equivale a fit seguido de transform con la misma semilla', () => {
      seedRandom(3)
      const a = new PCA(2).fitTransform(DS)
      seedRandom(3)
      const b = new PCA(2)
      b.fit(DS)
      expect(a).toEqual(b.transform(DS))
    })
  })

  describe('explainedVarianceRatio', () => {
    it('sin ajustar devuelve []', () => {
      expect(new PCA(1).explainedVarianceRatio()).toEqual([])
    })

    it('datos sin varianza: ratio 0 (convención del código cuando el total es 0)', () => {
      const pca = new PCA(1)
      pca.fit([[1, 2], [1, 2], [1, 2]])
      expect(pca.explainedVariance).toEqual([0])
      expect(pca.explainedVarianceRatio()).toEqual([0])
    })
  })

  describe('errores', () => {
    it('fit lanza con menos de 2 muestras', () => {
      expect(() => new PCA(1).fit([[1, 2]])).toThrow(/at least 2 samples/)
    })

    it('fit lanza si nComponents supera el número de características', () => {
      expect(() => new PCA(3).fit(DS)).toThrow(/cannot exceed number of features \(2\)/)
    })

    it('transform lanza sin fit()', () => {
      expect(() => new PCA(1).transform([[0, 0]])).toThrow(/call fit\(\) before transform/)
    })

    it('inverseTransform lanza sin fit()', () => {
      expect(() => new PCA(1).inverseTransform([[0]])).toThrow(/call fit\(\) before inverseTransform/)
    })
  })

  describe('robustez: semillas y determinismo', () => {
    it('5 semillas dan los mismos autovalores y ejes', () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const pca = new PCA(2)
        pca.fit(DS)
        expect(pca.explainedVariance[0]).toBeCloseTo(4, TOL_EIG)
        expect(pca.explainedVariance[1]).toBeCloseTo(1, TOL_EIG)
        const [c1, c2] = pca.components
        expectNearVec([Math.abs(c1[0]), Math.abs(c1[1])], [1, 0], TOL_AXIS)
        expectNearVec([Math.abs(c2[0]), Math.abs(c2[1])], [0, 1], TOL_AXIS)
      }
    })

    it('misma semilla → mismos componentes y autovalores', () => {
      seedRandom(42)
      const a = new PCA(2)
      a.fit(DS)
      seedRandom(42)
      const b = new PCA(2)
      b.fit(DS)
      expect(a.components).toEqual(b.components)
      expect(a.explainedVariance).toEqual(b.explainedVariance)
    })
  })
})
