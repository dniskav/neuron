// Tests de KMeans (src/KMeans.ts; Lloyd + K-Means++).
//
// Regla de valores (FR-004): cada cifra numérica lleva su derivación a mano en
// un comentario. Ninguna se copia de la salida del código.
//
// Conjuntos de referencia:
//   A = {(0,0), (2,0), (0,2), (2,2)}          media μA = (1, 1)
//   B = {(10,10), (12,10), (10,12), (12,12)}  media μB = (11, 11)
//   Inercia con centroides en las medias: cada punto de A dista² 2 de (1,1)
//   ((0−1)²+(0−1)² = 2; (2−1)²+(0−1)² = 2; (0−1)²+(2−1)² = 2; (2−1)²+(2−1)² = 2),
//   luego Σ_A = 4·2 = 8. Lo mismo para B (cada punto dista² 2 de (11,11)).
//   J = 8 + 8 = 16.
//
// Conjunto de k = 1: C = {(0,0), (3,0), (0,6)}, media (3/3, 6/3) = (1, 2).
//   J = (0−1)²+(0−2)² + (3−1)²+(0−2)² + (0−1)²+(6−2)² = 5 + 8 + 17 = 30.
//
// Aprendizaje ("dos blobs"): 20 puntos uniformes en [−0.5, 0.5)² alrededor de (0,0)
// y 20 alrededor de (10,0). Desviación típica por eje σ = √(1/12) ≈ 0.2887.
// Distancia entre centros 10 = 10/0.2887 ≈ 34.6 σ ≥ 5 σ. Hueco entre blobs ≥ 9
// (x ∈ [−0.5, 0.5) frente a x ∈ [9.5, 10.5)).
// Umbral: precisión ≥ 0.9 (salvo permutación de etiquetas). La precisión se satura
// en 1.0, así que el margen del 30 % se expresa sobre el error permitido: el peor
// caso de 5 semillas debe tener error ≤ 0.7 · (1 − 0.9) = 0.07, es decir,
// precisión ≥ 0.93. KMeans no tiene tasa de aprendizaje, así que la variación
// lr ×0.7 / ×1.3 no aplica.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach. Los datos de
// los blobs usan mulberry32 con una semilla explícita y KMeans recibe `seed`.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { KMeans } from '../src/KMeans'
import { mulberry32, seedRandom, restoreRandom } from './helpers/seed'

const A = [[0, 0], [2, 0], [0, 2], [2, 2]]
const B = [[10, 10], [12, 10], [10, 12], [12, 12]]
const AB = [...A, ...B]

/** Compara un punto con una tolerancia de 5e-10 por coordenada. */
function expectPoint(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 9))
}

/** Ordena centroides por la coordenada x (el orden de los clusters no es fijo). */
function sortByX(points: number[][]): number[][] {
  return [...points].sort((p, q) => p[0] - q[0])
}

/** Índice del centroide más próximo a `target` (los centroides ya están verificados). */
function clusterNear(km: KMeans, target: number[]): number {
  return km.centroids.findIndex(c => Math.abs(c[0] - target[0]) < 1e-9 && Math.abs(c[1] - target[1]) < 1e-9)
}

/** Dos blobs de 20 puntos cada uno, generados con una semilla explícita. */
function blobs(seed: number): { X: number[][]; y: number[] } {
  const rnd = mulberry32(seed)
  const X: number[][] = []
  const y: number[] = []
  for (let i = 0; i < 20; i++) {
    X.push([rnd() - 0.5, rnd() - 0.5]); y.push(0)
    X.push([10 + (rnd() - 0.5), rnd() - 0.5]); y.push(1)
  }
  return { X, y }
}

/** Precisión con k = 2, salvo permutación de etiquetas: max(aciertos, n − aciertos) / n. */
function bestAccuracy(pred: number[], y: number[]): number {
  let same = 0
  for (let i = 0; i < y.length; i++) if (pred[i] === y[i]) same++
  return Math.max(same, y.length - same) / y.length
}

describe('KMeans', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  describe('constructor', () => {
    it('acepta k entero positivo y empieza sin centroides', () => {
      for (const k of [1, 2, 5]) {
        const km = new KMeans(k, { seed: 1 })
        expect(km.centroids).toEqual([])
      }
    })

    it('rechaza k no entero o no positivo', () => {
      for (const k of [0, -1, 1.5, NaN]) {
        expect(() => new KMeans(k)).toThrow(/k must be a positive integer/)
      }
    })
  })

  describe('fit: centroides a mano', () => {
    it('dos grupos separados: centroides = medias (1,1) y (11,11), 5 semillas', () => {
      // Cada semilla da una inicialización K-Means++ distinta. La partición de Lloyd
      // es la de los grupos A y B, así que cada centroide es la media del grupo:
      // μA = (0+2+0+2, 0+0+2+2)/4 = (1,1); μB = (44, 44)/4 = (11,11).
      for (const seed of [1, 2, 3, 4, 5]) {
        const km = new KMeans(2, { seed })
        km.fit(AB)
        const sorted = sortByX(km.centroids)
        expect(km.centroids.length).toBe(2)
        expectPoint(sorted[0], [1, 1])
        expectPoint(sorted[1], [11, 11])
      }
    })

    it('k = 1: el único centroide es la media de todos los puntos', () => {
      // Con k = 1 todos los puntos van al mismo cluster; tras la primera iteración
      // el centroide es la media de C: (0+3+0, 0+0+6)/3 = (1, 2).
      const km = new KMeans(1, { seed: 7 })
      km.fit([[0, 0], [3, 0], [0, 6]])
      expectPoint(km.centroids[0], [1, 2])
    })

    it('maxIter = 1 con k = 1 aplica la actualización de centroides una vez', () => {
      // Una iteración basta para que el centroide sea la media (1, 2). Si el
      // paso de actualización se omitiera, el centroide seguiría siendo un punto de X.
      const km = new KMeans(1, { seed: 7, maxIter: 1 })
      km.fit([[0, 0], [3, 0], [0, 6]])
      expectPoint(km.centroids[0], [1, 2])
    })

    it('maxIter = 0 no itera: el centroide queda en un punto de X, no en la media', () => {
      // Sin iteraciones, el único centroide es el punto elegido al azar por K-Means++.
      // La media (1, 2) no pertenece a X, así que no puede aparecer.
      const X = [[0, 0], [3, 0], [0, 6]]
      const km = new KMeans(1, { seed: 7, maxIter: 0 })
      km.fit(X)
      const found = X.some(p => p[0] === km.centroids[0][0] && p[1] === km.centroids[0][1])
      expect(found).toBe(true)
      expect(km.centroids[0]).not.toEqual([1, 2])
    })

    it('k = n: cada punto es su propio centroide', () => {
      // Con k = n, cada punto acaba en su cluster: centroide = punto, distancia 0.
      const X = [[0, 0], [1, 0], [0, 1]]
      const km = new KMeans(3, { seed: 3 })
      km.fit(X)
      expect(km.centroids.length).toBe(3)
      for (const p of X) {
        expect(km.centroids.some(c => c[0] === p[0] && c[1] === p[1])).toBe(true)
      }
      expect(km.inertia(X)).toBe(0)
    })

    it('los centroides tienen k filas y d columnas', () => {
      const km = new KMeans(2, { seed: 1 })
      km.fit(AB)
      expect(km.centroids.length).toBe(2)
      for (const c of km.centroids) expect(c.length).toBe(2)
    })
  })

  describe('predict / predictBatch', () => {
    it('asigna cada punto al centroide más próximo', () => {
      // (3,3): dist² a (1,1) = 4+4 = 8; a (11,11) = 64+64 = 128 → cluster de (1,1).
      // (9,9): dist² a (1,1) = 64+64 = 128; a (11,11) = 4+4 = 8 → cluster de (11,11).
      const km = new KMeans(2, { seed: 1 })
      km.fit(AB)
      const cA = clusterNear(km, [1, 1])
      const cB = clusterNear(km, [11, 11])
      expect(cA).not.toBe(-1)
      expect(cB).not.toBe(-1)
      expect(km.predict([3, 3])).toBe(cA)
      expect(km.predict([9, 9])).toBe(cB)
    })

    it('predictBatch equivale a predict punto a punto', () => {
      // (0,0) y (3,3) → cluster de (1,1); (9,9) y (12,12) → cluster de (11,11).
      const km = new KMeans(2, { seed: 1 })
      km.fit(AB)
      const cA = clusterNear(km, [1, 1])
      const cB = clusterNear(km, [11, 11])
      expect(km.predictBatch([[0, 0], [3, 3], [9, 9], [12, 12]])).toEqual([cA, cA, cB, cB])
    })
  })

  describe('inertia', () => {
    it('dos grupos: J = 16 (ver cabecera)', () => {
      const km = new KMeans(2, { seed: 1 })
      km.fit(AB)
      expect(km.inertia(AB)).toBeCloseTo(16, 9)
    })

    it('calcula la suma de distancias² al centroide más próximo de cada punto', () => {
      // Con centroides (1,1) y (11,11): (0,0) → 2; (3,3) → 8. Total 10.
      const km = new KMeans(2, { seed: 1 })
      km.fit(AB)
      expect(km.inertia([[0, 0], [3, 3]])).toBeCloseTo(10, 9)
    })

    it('k = 1: J = 30 para C = {(0,0),(3,0),(0,6)} (ver cabecera)', () => {
      const C = [[0, 0], [3, 0], [0, 6]]
      const km = new KMeans(1, { seed: 7 })
      km.fit(C)
      expect(km.inertia(C)).toBeCloseTo(30, 9)
    })
  })

  describe('errores', () => {
    it('fit lanza con X vacía o nula', () => {
      expect(() => new KMeans(1).fit([])).toThrow(/non-empty/)
      expect(() => new KMeans(1).fit(null as unknown as number[][])).toThrow(/non-empty/)
    })

    it('fit lanza si k supera el número de muestras', () => {
      // k = 3 > n = 2
      expect(() => new KMeans(3).fit([[0, 0], [1, 1]])).toThrow(/cannot exceed number of samples/)
    })

    it('predict e inertia lanzan sin fit()', () => {
      const km = new KMeans(2)
      expect(() => km.predict([0, 0])).toThrow(/fit\(\)/)
      expect(() => km.inertia([[0, 0]])).toThrow(/fit\(\)/)
    })
  })

  describe('determinismo', () => {
    it('misma semilla → mismos centroides y mismas asignaciones', () => {
      const a = new KMeans(2, { seed: 42 })
      const b = new KMeans(2, { seed: 42 })
      a.fit(AB)
      b.fit(AB)
      expect(a.centroids).toEqual(b.centroids)
      expect(a.predictBatch(AB)).toEqual(b.predictBatch(AB))
    })
  })

  describe('aprende: dos blobs separados', () => {
    it('precisión ≥ 0.93 (margen 30 % sobre 0.9) y centroides = medias reales, 5 semillas', () => {
      // Por cada semilla: el blob 0 queda centrado en (0,0) y el 1 en (10,0), con un
      // hueco de ≥ 9 entre ambos; por tanto la partición correcta es la de y. Los
      // centroides deben coincidir con la media de cada grupo verdadero, calculada aquí.
      const worst: number[] = []
      for (const seed of [1, 2, 3, 4, 5]) {
        const { X, y } = blobs(seed)
        const km = new KMeans(2, { seed })
        km.fit(X)

        const acc = bestAccuracy(km.predictBatch(X), y)
        worst.push(acc)

        const mean = (label: number): number[] => {
          const pts = X.filter((_, i) => y[i] === label)
          const sx = pts.reduce((s, p) => s + p[0], 0)
          const sy = pts.reduce((s, p) => s + p[1], 0)
          return [sx / pts.length, sy / pts.length]
        }
        const m0 = mean(0)
        const m1 = mean(1)
        const found0 = km.centroids.some(c => Math.abs(c[0] - m0[0]) < 1e-9 && Math.abs(c[1] - m0[1]) < 1e-9)
        const found1 = km.centroids.some(c => Math.abs(c[0] - m1[0]) < 1e-9 && Math.abs(c[1] - m1[1]) < 1e-9)
        expect(found0).toBe(true)
        expect(found1).toBe(true)
      }
      expect(Math.min(...worst)).toBeGreaterThanOrEqual(0.93)
    })
  })
})
