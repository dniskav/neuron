// Validación de entradas de KMeans, PCA y SOM. Feature 006, T038 (US5, principio V).
//
// Regla: un error lanza ANTES de tocar el estado (centroides, media, componentes,
// pesos). Los mensajes que ya existían se conservan.
//
// Golden (T038, regla "no cambia"): salida de la base, antes de tocar src/, con
// entradas válidas y semillas fijas (seedRandom 1, 2, 3 y KMeans seed 42). No son
// cifras derivadas a mano: sirven para detectar cambios en entradas válidas.
//
// Cifra a mano (PCA): puntos (0,0),(1,1),(2,2),(3,3) están sobre y = x. Media (1.5, 1.5).
// Varianza por eje (población) = 1.25, traza = 2.5. Con 1 componente, λ₁ = 2.5 → ratio 1.
//
// SOM.getBMU ya lanzaba con dimensión errónea antes de T038 (SOM.ts, comprobación de
// longitud). Su test fija ese comportamiento; T038 solo añade la validación de epochs.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { KMeans } from '../../src/KMeans'
import { PCA } from '../../src/PCA'
import { SOM } from '../../src/SOM'
import { mulberry32, seedRandom, restoreRandom } from '../helpers/seed'

/** Mismo generador que el golden de la base (mulberry32(123)): X2 = 30 puntos 2-D, X3 = 12 puntos 3-D. */
function goldenData(): { X2: number[][]; X3: number[][] } {
  const rnd = mulberry32(123)
  const X2: number[][] = []
  for (const c of [[0, 0], [6, 0], [3, 6]]) {
    for (let i = 0; i < 10; i++) X2.push([c[0] + rnd() - 0.5, c[1] + rnd() - 0.5])
  }
  const X3: number[][] = Array.from({ length: 12 }, () => [rnd() * 4, rnd() * 2 - 1, rnd()])
  return { X2, X3 }
}

/** Copia profunda; structuredClone conserva NaN, JSON no. */
function snap<T>(v: T): T {
  return structuredClone(v)
}

describe('Unsupervised validation (T038)', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  describe('golden: entradas válidas no cambian', () => {
    it('KMeans (seed 42) separa los tres grupos en las mismas etiquetas que la base', () => {
      const { X2 } = goldenData()
      const km = new KMeans(3, { seed: 42 })
      km.fit(X2)
      // Etiquetas de la base: grupo 1 → 2, grupo 2 → 0, grupo 3 → 1 (permutación de los clústeres).
      expect(km.predictBatch(X2)).toEqual([
        ...Array(10).fill(2), ...Array(10).fill(0), ...Array(10).fill(1),
      ])
    })

    it('KMeans inertia con Math.random sembrado (base: 4.318071422922517)', () => {
      seedRandom(1)
      const { X2 } = goldenData()
      const km = new KMeans(3)
      km.fit(X2)
      expect(km.inertia(X2)).toBeCloseTo(4.318071422922517, 12)
    })

    it('PCA explainedVarianceRatio (base: 0.79466787757, 0.16704295332)', () => {
      seedRandom(2)
      const { X3 } = goldenData()
      const pca = new PCA(2)
      pca.fitTransform(X3)
      const r = pca.explainedVarianceRatio()
      expect(r[0]).toBeCloseTo(0.7946678775731625, 12)
      expect(r[1]).toBeCloseTo(0.1670429533208342, 12)
    })

    it('SOM quantizationError y BMU tras train (base: 2.8859259049897594)', () => {
      seedRandom(3)
      const { X2 } = goldenData()
      const som = new SOM(2, 2, 2)
      som.train(X2, 3)
      expect(som.quantizationError(X2)).toBeCloseTo(2.8859259049897594, 12)
      const bmu = X2.map(x => som.getBMU(x))
      expect(bmu.slice(0, 10).every(b => b[0] === 0 && b[1] === 1)).toBe(true)
      expect(bmu.slice(10, 20).every(b => b[0] === 0 && b[1] === 0)).toBe(true)
      expect(bmu.slice(20).every(b => b[0] === 1 && b[1] === 0)).toBe(true)
    })

    it('PCA a mano: puntos sobre y = x → ratio del primer componente = 1', () => {
      const pca = new PCA(1)
      pca.fit([[0, 0], [1, 1], [2, 2], [3, 3]])
      expect(pca.mean).toEqual([1.5, 1.5])
      expect(pca.explainedVarianceRatio()[0]).toBeCloseTo(1, 12)
    })
  })

  describe('KMeans.fit: X rectangular y finita', () => {
    it('filas de distinta longitud lanzan y no cambian los centroides', () => {
      const km = new KMeans(2, { seed: 5 })
      km.fit([[0, 0], [1, 1], [5, 5], [6, 6]])
      const before = snap(km.centroids)
      expect(() => km.fit([[0, 0], [1], [2, 2]])).toThrow(/KMeans\.fit: X\[1\]: expected array of length 2, got 1/)
      expect(km.centroids).toEqual(before)
    })

    it('un NaN en X lanza con el índice de la fila', () => {
      expect(() => new KMeans(2).fit([[0, 0], [NaN, 1], [2, 2]])).toThrow(/KMeans\.fit: X\[1\]: invalid value at index 0/)
    })

    it('un modelo sin ajustar no queda ajustado tras un fit fallido', () => {
      const km = new KMeans(2)
      expect(() => km.fit([[0, 0], [1]])).toThrow(/KMeans\.fit/)
      expect(km.centroids).toEqual([])
    })

    it('X vacía conserva el mensaje existente', () => {
      expect(() => new KMeans(1).fit([])).toThrow(/KMeans\.fit: dataset X must be non-empty/)
    })
  })

  describe('KMeans.predict: dimensión de la entrada', () => {
    it('una entrada más corta que los centroides lanza (antes devolvía un clúster)', () => {
      const km = new KMeans(2, { seed: 5 })
      km.fit([[0, 0], [1, 1], [5, 5], [6, 6]])
      expect(() => km.predict([1])).toThrow(/KMeans\.predict: expected array of length 2, got 1/)
    })

    it('una entrada más larga, un NaN y predictBatch con una fila errónea lanzan', () => {
      const km = new KMeans(2, { seed: 5 })
      km.fit([[0, 0], [1, 1], [5, 5], [6, 6]])
      expect(() => km.predict([0, 0, 0])).toThrow(/KMeans\.predict: expected array of length 2, got 3/)
      expect(() => km.predict([NaN, 0])).toThrow(/KMeans\.predict: invalid value at index 0/)
      expect(() => km.predictBatch([[0, 0], [1]])).toThrow(/KMeans\.predict/)
    })
  })

  describe('PCA.fit: X rectangular y finita', () => {
    it('filas de distinta longitud lanzan y no cambian media ni componentes', () => {
      const pca = new PCA(1)
      pca.fit([[0, 0], [1, 1], [2, 2], [3, 3]])
      const before = snap({ mean: pca.mean, c: pca.components, ev: pca.explainedVariance })
      expect(() => pca.fit([[0, 0], [1, 1], [2]])).toThrow(/PCA\.fit: X\[2\]: expected array of length 2, got 1/)
      expect({ mean: pca.mean, c: pca.components, ev: pca.explainedVariance }).toEqual(before)
    })

    it('un NaN en X lanza con el índice de la fila', () => {
      expect(() => new PCA(1).fit([[NaN, 1], [2, 3]])).toThrow(/PCA\.fit: X\[0\]: invalid value at index 0/)
    })

    it('menos de 2 muestras conserva el mensaje existente', () => {
      expect(() => new PCA(1).fit([[1, 2]])).toThrow(/PCA\.fit: need at least 2 samples/)
    })
  })

  describe('PCA.transform: ancho de cada fila', () => {
    it('una fila de ancho distinto al ajustado lanza con su índice', () => {
      const pca = new PCA(1)
      pca.fit([[0, 0], [1, 1], [2, 2], [3, 3]])
      expect(() => pca.transform([[1, 2, 3]])).toThrow(/PCA\.transform: X\[0\]: expected array of length 2, got 3/)
      expect(() => pca.transform([[1, 2], [3]])).toThrow(/PCA\.transform: X\[1\]: expected array of length 2, got 1/)
    })

    it('un NaN en una fila lanza', () => {
      const pca = new PCA(1)
      pca.fit([[0, 0], [1, 1], [2, 2], [3, 3]])
      expect(() => pca.transform([[NaN, 0]])).toThrow(/PCA\.transform: X\[0\]: invalid value at index 0/)
    })
  })

  describe('SOM.train: epochs entero y >= 1', () => {
    it('epochs 0 lanza y no toca los pesos', () => {
      const som = new SOM(2, 2, 2)
      const before = snap(som.weights)
      expect(() => som.train([[0.5, 0.5]], 0)).toThrow(/SOM\.train: epochs must be an integer >= 1, got 0/)
      expect(som.weights).toEqual(before)
    })

    it('epochs fraccionario, negativo o NaN lanzan', () => {
      const som = new SOM(2, 2, 2)
      expect(() => som.train([[0.5, 0.5]], 1.5)).toThrow(/epochs must be an integer >= 1, got 1.5/)
      expect(() => som.train([[0.5, 0.5]], -1)).toThrow(/epochs must be an integer >= 1, got -1/)
      expect(() => som.train([[0.5, 0.5]], NaN)).toThrow(/epochs must be an integer >= 1/)
    })

    it('epochs 1 se acepta', () => {
      const som = new SOM(2, 2, 2)
      expect(() => som.train([[0.5, 0.5]], 1)).not.toThrow()
    })
  })

  describe('SOM.getBMU / predict: dimensión (ya existía; se fija)', () => {
    it('entrada de longitud distinta a inputSize lanza', () => {
      const som = new SOM(2, 2, 2)
      expect(() => som.getBMU([1])).toThrow(/SOM\.getBMU: expected input of length 2, got 1/)
      expect(() => som.predict([1, 2, 3])).toThrow(/SOM\.getBMU: expected input of length 2, got 3/)
    })
  })
})
