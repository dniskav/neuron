// Tests de DataAugmentation (src/DataAugmentation.ts). Tarea T009 de specs/006-coverage.
//
// Cifras a mano (no copiadas de la salida del código):
//   · Stub U1 = e^(-1/2): en Box-Muller, sqrt(-2·ln U1) = 1, así que z = cos(2π·U2).
//     U2 = 0.5 → z = -1; U2 = 0.25 → z = 0. Con U1 fijo, addNoise es exacto.
//   · jitter con stub r: x + (2r - 1)·δ.
//   · Tabla de normalize: [[0,10],[5,20],[10,30]] → min [0,10], max [10,30],
//     normalizado [[0,0],[0.5,0.5],[1,1]].
//   · standardize de [1,2,3,4,5]: media 3; varianza poblacional (4+1+0+1+4)/5 = 2;
//     desviación √2 ≈ 1.41421356 (el código divide por n, no por n−1).
//   · shuffle con stub [0, 0.34, 0.99] sobre 4 elementos (Fisher-Yates desde i=3):
//       i=3, j=floor(0·4)=0 → [3,1,2,0]; i=2, j=floor(0.34·3)=1 → [3,2,1,0];
//       i=1, j=floor(0.99·2)=1 → sin cambio. Permutación final: [3,2,1,0].
//   · split de 10 elementos: trainEnd = floor(10·0.7) = 7; valEnd = 7 + floor(10·0.15) = 8;
//     partición 7 / 1 / 2.
//
// Estadística con semilla: 5 semillas (1..5), 10 000 muestras.
//   · Media de N(0, σ²) con σ=0.5: error típico 0.5/√10000 = 0.005. Tolerancia 0.02 (≈4σ).
//   · Desviación de N(0, σ²): error típico ≈ σ/√(2n) = 0.0035. Tolerancia 3 % de σ (≈4σ).
//   · Jitter δ=0.05: media de U(-δ,δ) = 0, error típico δ/√3/√n ≈ 2.9e-4. Tolerancia 0.002 (≈7σ).
//
// Casos degenerados (longitudes distintas en shuffle, X vacía, filas irregulares, NaN)
// NO se fijan aquí: se reportan como hallazgos en el informe de T009.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { DataAugmentation } from '../src/DataAugmentation'
import { seedRandom, restoreRandom } from './helpers/seed'

const SEEDS = [1, 2, 3, 4, 5]
// e^(-1/2): en Box-Muller da sqrt(-2·ln U1) = 1 (ln U1 = -1/2).
const U1 = Math.exp(-0.5)

// Reemplaza Math.random por una secuencia fija; lanza si se pide más de la cuenta.
function stubRandom(values: number[]): void {
  let k = 0
  Math.random = () => {
    if (k >= values.length) throw new Error('stubRandom: secuencia agotada')
    return values[k++]
  }
}

function meanOf(xs: number[]): number {
  return xs.reduce((s, v) => s + v, 0) / xs.length
}

function stdOf(xs: number[]): number {
  const m = meanOf(xs)
  return Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / xs.length)
}

describe('DataAugmentation', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  describe('addNoise', () => {
    it('with stub U1, U2 = 0.5 gives z = -1, so 2 + (-1)·0.1 = 1.9', () => {
      stubRandom([U1, 0.5])
      const out = DataAugmentation.addNoise([2], 0.1)
      expect(out[0]).toBeCloseTo(1.9, 9)
    })

    it('with stub U2 = 0.25 gives z = cos(π/2) = 0, so the input is unchanged', () => {
      stubRandom([U1, 0.25])
      expect(DataAugmentation.addNoise([2], 0.1)[0]).toBeCloseTo(2, 9)
    })

    it('each feature consumes two draws: [1,1] with σ=0.5 gives [1-0.5, 1+0] = [0.5, 1]', () => {
      stubRandom([U1, 0.5, U1, 0.25])
      const out = DataAugmentation.addNoise([1, 1], 0.5)
      expect(out[0]).toBeCloseTo(0.5, 9)
      expect(out[1]).toBeCloseTo(1, 9)
    })

    it('default sigma is 0.01: z = -1 gives 0 - 0.01 = -0.01', () => {
      stubRandom([U1, 0.5])
      expect(DataAugmentation.addNoise([0])[0]).toBeCloseTo(-0.01, 9)
    })

    it('does not mutate the input', () => {
      const x = [1, 2, 3]
      DataAugmentation.addNoise(x, 0.5)
      expect(x).toEqual([1, 2, 3])
    })

    it('with seeds 1..5 and 10 000 samples, mean ≈ 0 and std ≈ σ (tolerances 0.02 and 3 %)', () => {
      for (const seed of SEEDS) {
        seedRandom(seed)
        const xs = Array.from({ length: 10000 }, () => DataAugmentation.addNoise([0], 0.5)[0])
        expect(Math.abs(meanOf(xs))).toBeLessThan(0.02)
        expect(Math.abs(stdOf(xs) - 0.5) / 0.5).toBeLessThan(0.03)
      }
    })
  })

  describe('jitter', () => {
    it('with stub r = 0, 0.5, 0.75 and δ = 0.2 gives offsets -0.2, 0, +0.1', () => {
      stubRandom([0, 0.5, 0.75])
      const out = DataAugmentation.jitter([1, 1, 1], 0.2)
      expect(out[0]).toBeCloseTo(0.8, 9)
      expect(out[1]).toBeCloseTo(1.0, 9)
      expect(out[2]).toBeCloseTo(1.1, 9)
    })

    it('default delta is 0.01: r = 0 gives -0.01', () => {
      stubRandom([0])
      expect(DataAugmentation.jitter([0])[0]).toBeCloseTo(-0.01, 9)
    })

    it('with seeds 1..5 every |offset| ≤ δ, mean ≈ 0 (tolerance 0.002), and the range is reached', () => {
      const delta = 0.05
      for (const seed of SEEDS) {
        seedRandom(seed)
        const xs = Array.from({ length: 10000 }, () => DataAugmentation.jitter([0], delta)[0])
        for (const v of xs) expect(Math.abs(v)).toBeLessThanOrEqual(delta + 1e-12)
        expect(Math.abs(meanOf(xs))).toBeLessThan(0.002)
        // Con 10 000 muestras uniformes el máximo está muy cerca de δ (> 0.9·δ).
        expect(Math.max(...xs.map(Math.abs))).toBeGreaterThan(0.9 * delta)
      }
    })
  })

  describe('flipHorizontal', () => {
    it('reverses the order: [1,2,3] → [3,2,1]', () => {
      expect(DataAugmentation.flipHorizontal([1, 2, 3])).toEqual([3, 2, 1])
    })

    it('flipping twice returns the original vector', () => {
      const x = [4, 5, 6, 7]
      expect(DataAugmentation.flipHorizontal(DataAugmentation.flipHorizontal(x))).toEqual(x)
    })

    it('does not mutate the input', () => {
      const x = [1, 2, 3]
      DataAugmentation.flipHorizontal(x)
      expect(x).toEqual([1, 2, 3])
    })
  })

  describe('normalize / normalizePoint (min-max)', () => {
    // Derivación: min = [0,10], max = [10,30]; fila 1: (5-0)/10 = 0.5, (20-10)/20 = 0.5.
    const X = [[0, 10], [5, 20], [10, 30]]

    it('maps [[0,10],[5,20],[10,30]] to [[0,0],[0.5,0.5],[1,1]]', () => {
      const { normalized } = DataAugmentation.normalize(X)
      expect(normalized).toEqual([[0, 0], [0.5, 0.5], [1, 1]])
    })

    it('returns the fitted per-feature min and max', () => {
      const { min, max } = DataAugmentation.normalize(X)
      expect(min).toEqual([0, 10])
      expect(max).toEqual([10, 30])
    })

    it('a constant feature (min = max) maps to 0', () => {
      // Columna 0 constante 5: range 0 → 0. Columna 1: (1-1)/2 = 0, (3-1)/2 = 1.
      const { normalized } = DataAugmentation.normalize([[5, 1], [5, 3]])
      expect(normalized).toEqual([[0, 0], [0, 1]])
    })

    it('normalizePoint applies the training min/max to a new sample: (7-0)/10 = 0.7', () => {
      expect(DataAugmentation.normalizePoint([7, 20], [0, 10], [10, 30])[0]).toBeCloseTo(0.7, 9)
      expect(DataAugmentation.normalizePoint([7, 20], [0, 10], [10, 30])[1]).toBeCloseTo(0.5, 9)
    })

    it('normalizePoint with a constant training feature gives 0 regardless of the value', () => {
      // min = max = 3 en la columna 1 → 0; columna 0: (7-0)/10 = 0.7.
      expect(DataAugmentation.normalizePoint([7, 3], [0, 3], [10, 3])).toEqual([0.7, 0])
    })

    it('normalizePoint does not clip values outside the training range: (20-0)/10 = 2', () => {
      expect(DataAugmentation.normalizePoint([20], [0], [10])).toEqual([2])
    })
  })

  describe('standardize / standardizePoint (z-score)', () => {
    // X = [1,2,3,4,5]: media 3, varianza poblacional 2, desviación √2 ≈ 1.41421356.
    // z = (x-3)/√2 → -√2, -1/√2, 0, 1/√2, √2.
    const SQRT2 = Math.SQRT2
    const X = [[1], [2], [3], [4], [5]]

    it('fits mean 3 and population std √2 on [1..5]', () => {
      const { mean, std } = DataAugmentation.standardize(X)
      expect(mean[0]).toBeCloseTo(3, 12)
      expect(std[0]).toBeCloseTo(SQRT2, 12)
    })

    it('standardized values are -√2, -1/√2, 0, 1/√2, √2', () => {
      const { standardized } = DataAugmentation.standardize(X)
      expect(standardized[0][0]).toBeCloseTo(-SQRT2, 12)
      expect(standardized[1][0]).toBeCloseTo(-1 / SQRT2, 12)
      expect(standardized[2][0]).toBeCloseTo(0, 12)
      expect(standardized[3][0]).toBeCloseTo(1 / SQRT2, 12)
      expect(standardized[4][0]).toBeCloseTo(SQRT2, 12)
    })

    it('output has mean 0 and std 1 (Σz²/n = (2+0.5+0+0.5+2)/5 = 1)', () => {
      const { standardized } = DataAugmentation.standardize(X)
      const zs = standardized.map((r) => r[0])
      expect(meanOf(zs)).toBeCloseTo(0, 12)
      expect(zs.reduce((s, z) => s + z * z, 0) / zs.length).toBeCloseTo(1, 12)
    })

    it('a constant feature has std 0 and standardizes to 0', () => {
      const { standardized, mean, std } = DataAugmentation.standardize([[1, 7], [2, 7], [3, 7], [4, 7], [5, 7]])
      expect(mean[1]).toBe(7)
      expect(std[1]).toBe(0)
      for (const row of standardized) expect(row[1]).toBe(0)
    })

    it('standardizePoint uses the training mean and std: (1-3)/√2 = -√2', () => {
      expect(DataAugmentation.standardizePoint([1], [3], [SQRT2])[0]).toBeCloseTo(-SQRT2, 12)
    })

    it('standardizePoint with std 0 gives 0 whatever the value is', () => {
      expect(DataAugmentation.standardizePoint([3, 9], [3, 7], [SQRT2, 0])).toEqual([0, 0])
    })
  })

  describe('augmentBatch', () => {
    it('with sigma 0 the copies are exact: factor 3 gives 3 copies of each row, labels repeated', () => {
      const out = DataAugmentation.augmentBatch([[1, 2], [3, 4]], [0, 1], 3, 0)
      expect(out.X).toEqual([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]])
      expect(out.y).toEqual([0, 0, 0, 1, 1, 1])
    })

    it('default factor 2 keeps the original first and adds one noisy copy: 1 - 0.01 = 0.99', () => {
      stubRandom([U1, 0.5])
      const out = DataAugmentation.augmentBatch([[1]], [5])
      expect(out.X[0]).toEqual([1])
      expect(out.X[1][0]).toBeCloseTo(0.99, 9)
      expect(out.y).toEqual([5, 5])
    })
  })

  describe('shuffle', () => {
    it('stub [0, 0.34, 0.99] on 4 items gives the permutation [3,2,1,0] for X and y', () => {
      // Derivación en la cabecera del archivo.
      stubRandom([0, 0.34, 0.99])
      const out = DataAugmentation.shuffle([[0], [1], [2], [3]], [10, 11, 12, 13])
      expect(out.X).toEqual([[3], [2], [1], [0]])
      expect(out.y).toEqual([13, 12, 11, 10])
    })

    it('with seeds 1..5 X and y receive the same permutation and the multiset is kept', () => {
      for (const seed of SEEDS) {
        seedRandom(seed)
        const X = Array.from({ length: 50 }, (_, i) => [i])
        const y = X.map((r) => r[0] * 10)
        const out = DataAugmentation.shuffle(X, y)
        // Misma permutación: y_k = 10 · x_k en cada posición.
        out.X.forEach((row, k) => expect(out.y[k]).toBe(row[0] * 10))
        // Mismo multiconjunto de características.
        expect(out.X.map((r) => r[0]).sort((a, b) => a - b)).toEqual(X.map((r) => r[0]))
        expect(out.y.slice().sort((a, b) => a - b)).toEqual(y.slice().sort((a, b) => a - b))
      }
    })

    it('does not mutate the inputs', () => {
      const X = [[1], [2], [3]]
      const y = [10, 20, 30]
      DataAugmentation.shuffle(X, y)
      expect(X).toEqual([[1], [2], [3]])
      expect(y).toEqual([10, 20, 30])
    })
  })

  describe('split', () => {
    it('10 items with 0.7 / 0.15 give 7 train, 1 val, 2 test (floor(7) = 7, 7 + floor(1.5) = 8)', () => {
      const X = Array.from({ length: 10 }, (_, i) => [i])
      const y = X.map((r) => r[0])
      const s = DataAugmentation.split(X, y)
      expect(s.trainX.length).toBe(7)
      expect(s.valX.length).toBe(1)
      expect(s.testX.length).toBe(2)
      expect(s.trainY.length).toBe(7)
      expect(s.valY.length).toBe(1)
      expect(s.testY.length).toBe(2)
    })

    it('the three partitions are disjoint and their union is the original set', () => {
      const X = Array.from({ length: 10 }, (_, i) => [i])
      const y = X.map((r) => r[0])
      const s = DataAugmentation.split(X, y)
      const all = [...s.trainX, ...s.valX, ...s.testX].map((r) => r[0]).sort((a, b) => a - b)
      expect(all).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    })

    it('throws when trainRatio + valRatio ≥ 1 (0.8 + 0.2 = 1)', () => {
      expect(() => DataAugmentation.split([[1]], [1], 0.8, 0.2)).toThrow(/must be < 1/)
    })
  })
})
