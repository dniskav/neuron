// Validación de entradas de GaussianNaiveBayes y DecisionTree. Feature 006, T039 (US5, principio V).
//
// Regla: un error lanza ANTES de tocar el estado del modelo (medias, varianzas, priors,
// árbol). Los mensajes que ya existían se conservan.
//
// Golden (T039, regla "no cambia"): salida de la base, antes de tocar src/, con datos
// enteros fijos. Ambos modelos son deterministas (sin azar). Las probabilidades de
// NaiveBayes se comparan con tolerancia 1e-9.
//
// Cifra a mano (DecisionTree regresión): x = 0..3, y = [1, 2, 4, 8], maxDepth 2.
// La primera división en 1.5 deja {1, 2} (media 1.5) y {4, 8}; la segunda, en 2.5,
// separa 4 y 8. Predicciones a mano: 0.5 → 1.5, 1.5 → 1.5, 2.5 → 4, 3.5 → 8.

import { describe, it, expect } from 'vitest'
import { GaussianNaiveBayes } from '../../src/NaiveBayes'
import { DecisionTree } from '../../src/DecisionTree'

const NB_X = [[0, 0], [0, 1], [1, 0], [1, 1], [4, 4], [4, 5], [5, 4], [5, 5], [8, 0], [8, 1], [9, 0], [9, 1]]
const NB_Y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
const GRID = [[0.5, 0.5], [4.5, 4.5], [8.5, 0.5], [2, 2], [6, 2]]

function fittedNB(): GaussianNaiveBayes {
  const nb = new GaussianNaiveBayes()
  nb.fit(NB_X, NB_Y)
  return nb
}

function fittedDT(): DecisionTree {
  const t = new DecisionTree({ maxDepth: 2 })
  t.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
  return t
}

describe('golden (valid inputs, unchanged by T039)', () => {
  it('GaussianNaiveBayes predict on fixed grid', () => {
    expect(GRID.map(p => fittedNB().predict(p))).toEqual([0, 1, 2, 0, 1])
  })

  it('GaussianNaiveBayes predictProba on fixed grid (tolerance 1e-9)', () => {
    const nb = fittedNB()
    const expected: [number, number][][] = [
      [[0, 1], [1, 1.603810890548638e-28], [2, 2.572209372642415e-56]],
      [[0, 1.603810890548638e-28], [1, 1], [2, 1.603810890548638e-28]],
      [[0, 2.572209372642415e-56], [1, 1.603810890548638e-28], [2, 1]],
      [[0, 0.9999998874648376], [1, 1.1253516205509495e-7], [2, 1.8048511847361583e-35]],
      [[0, 7.125820413704712e-22], [1, 0.5000000000000009], [2, 0.5000000000000009]],
    ]
    GRID.forEach((p, i) => {
      const got = [...nb.predictProba(p).entries()]
      expect(got.map(([c]) => c)).toEqual(expected[i].map(([c]) => c))
      got.forEach(([, v], k) => {
        expect(Math.abs(v - expected[i][k][1])).toBeLessThan(1e-9)
      })
    })
  })

  it('DecisionTree XOR with maxDepth 2 separates all four points', () => {
    const t = fittedDT()
    expect(t.predictBatch([[0, 0], [0, 1], [1, 0], [1, 1]])).toEqual([0, 1, 1, 0])
  })

  it('DecisionTree multiclass on 1-D data', () => {
    const t = new DecisionTree({ maxDepth: 3 })
    t.fit([[0], [1], [2], [3], [4], [5]], [0, 0, 1, 1, 2, 2])
    expect(t.predictBatch([[0.4], [2.6], [4.9], [-3], [9]])).toEqual([0, 1, 2, 0, 2])
  })

  it('DecisionTree regression leaves are means (hand-derived in header)', () => {
    const t = new DecisionTree({ task: 'regression', maxDepth: 2 })
    t.fit([[0], [1], [2], [3]], [1, 2, 4, 8])
    expect(t.predictBatch([[0.5], [1.5], [2.5], [3.5]])).toEqual([1.5, 1.5, 4, 8])
  })

  it('DecisionTree default options (maxDepth 10, minSamplesSplit 2) still work', () => {
    const t = new DecisionTree()
    t.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
    expect(t.predictBatch([[0, 0], [0, 1], [1, 0], [1, 1]])).toEqual([0, 1, 1, 0])
  })
})

describe('GaussianNaiveBayes.fit validation', () => {
  it('rejects ragged rows before touching the model', () => {
    expect(() => new GaussianNaiveBayes().fit([[0, 0], [1]], [0, 1]))
      .toThrow(/^GaussianNaiveBayes\.fit: X\[1\]/)
  })

  it('rejects NaN and Infinity in X', () => {
    expect(() => new GaussianNaiveBayes().fit([[NaN], [1]], [0, 1]))
      .toThrow(/^GaussianNaiveBayes\.fit: X\[0\]/)
    expect(() => new GaussianNaiveBayes().fit([[0], [Infinity]], [0, 1]))
      .toThrow(/^GaussianNaiveBayes\.fit: X\[1\]/)
  })

  it('rejects non-finite labels', () => {
    expect(() => new GaussianNaiveBayes().fit([[0], [1]], [NaN, 1]))
      .toThrow(/^GaussianNaiveBayes\.fit: y/)
    expect(() => new GaussianNaiveBayes().fit([[0], [1]], [0, Infinity]))
      .toThrow(/^GaussianNaiveBayes\.fit: y/)
  })

  it('rejects rows with zero features', () => {
    expect(() => new GaussianNaiveBayes().fit([[], []], [0, 1]))
      .toThrow(/GaussianNaiveBayes\.fit: X rows need at least 1 feature/)
  })

  it('keeps the existing messages for empty X and length mismatch', () => {
    expect(() => new GaussianNaiveBayes().fit([], [])).toThrow(/X is empty/)
    expect(() => new GaussianNaiveBayes().fit([[1], [2]], [0]))
      .toThrow('GaussianNaiveBayes.fit: X has 2 rows but y has 1 labels')
  })

  it('a rejected fit leaves the previously fitted model intact', () => {
    const nb = fittedNB()
    const before = GRID.map(p => nb.predict(p))
    expect(() => nb.fit([[0, 0], [1]], [0, 1])).toThrow(/GaussianNaiveBayes\.fit/)
    expect(() => nb.fit(NB_X, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, NaN])).toThrow(/GaussianNaiveBayes\.fit/)
    expect(GRID.map(p => nb.predict(p))).toEqual(before)
  })
})

describe('DecisionTree constructor validation', () => {
  it('rejects non-integer maxDepth', () => {
    expect(() => new DecisionTree({ maxDepth: 1.5 })).toThrow('maxDepth must be positive')
    expect(() => new DecisionTree({ maxDepth: NaN })).toThrow('maxDepth must be positive')
  })

  it('rejects maxDepth -Infinity and mentions that Infinity is valid', () => {
    expect(() => new DecisionTree({ maxDepth: -Infinity })).toThrow('maxDepth must be positive')
    expect(() => new DecisionTree({ maxDepth: -Infinity })).toThrow(/Infinity/)
  })

  it('accepts maxDepth 1 (the smallest allowed value)', () => {
    expect(() => new DecisionTree({ maxDepth: 1 })).not.toThrow()
  })

  it('accepts maxDepth Infinity (no depth limit)', () => {
    expect(() => new DecisionTree({ maxDepth: Infinity })).not.toThrow()
  })

  it('rejects non-integer minSamplesSplit', () => {
    expect(() => new DecisionTree({ minSamplesSplit: 1.5 })).toThrow('minSamplesSplit must be at least 2')
    expect(() => new DecisionTree({ minSamplesSplit: NaN })).toThrow('minSamplesSplit must be at least 2')
  })
})

describe('DecisionTree.fit validation', () => {
  it('rejects ragged rows', () => {
    expect(() => new DecisionTree().fit([[0, 0], [1]], [0, 1]))
      .toThrow(/^DecisionTree\.fit: X\[1\]/)
  })

  it('rejects NaN and Infinity in X', () => {
    expect(() => new DecisionTree().fit([[NaN], [1]], [0, 1]))
      .toThrow(/^DecisionTree\.fit: X\[0\]/)
    expect(() => new DecisionTree().fit([[0], [-Infinity]], [0, 1]))
      .toThrow(/^DecisionTree\.fit: X\[1\]/)
  })

  it('rejects non-finite labels', () => {
    expect(() => new DecisionTree().fit([[0], [1]], [NaN, 1]))
      .toThrow(/^DecisionTree\.fit: y/)
  })

  it('keeps the existing messages for empty X and length mismatch', () => {
    expect(() => new DecisionTree().fit([], [])).toThrow('DecisionTree.fit: X is empty')
    expect(() => new DecisionTree().fit([[0], [1]], [0]))
      .toThrow('DecisionTree.fit: X has 2 rows but y has 1 labels')
  })

  it('a rejected fit leaves the previously fitted tree intact', () => {
    const t = fittedDT()
    const before = t.predictBatch([[0, 0], [0, 1], [1, 0], [1, 1]])
    expect(() => t.fit([[0, 0], [1]], [0, 1])).toThrow(/DecisionTree\.fit/)
    expect(t.predictBatch([[0, 0], [0, 1], [1, 0], [1, 1]])).toEqual(before)
  })
})

describe('DecisionTree.predict validation', () => {
  it('rejects an input of the wrong dimension', () => {
    const t = fittedDT()
    expect(() => t.predict([])).toThrow(/^DecisionTree\.predict: /)
    expect(() => t.predict([1])).toThrow(/^DecisionTree\.predict: /)
    expect(() => t.predict([1, 0, 0])).toThrow(/^DecisionTree\.predict: /)
  })

  it('rejects a non-finite input', () => {
    expect(() => fittedDT().predict([NaN, 0])).toThrow(/^DecisionTree\.predict: /)
  })

  it('accepts an input of the fitted dimension', () => {
    expect(fittedDT().predict([1, 1])).toBe(0)
  })
})
