import { describe, it, expect } from 'vitest'
import {
  validateArray,
  validateArrayMinLength,
  validate2DArray,
  validateNumber,
  validateDataset,
} from '../src/Validation'

// T033. Every expected value below is derived by hand from the message formats
// in src/Validation.ts. Branches previously uncovered (lines 36, 45, 59, 77) are
// exercised in 'uncovered branches'.

describe('uncovered branches', () => {
  it('validateArrayMinLength: non-array (exact message)', () => {
    expect(() => validateArrayMinLength('x', 1, 'm')).toThrow('m: expected array, got string')
  })
  it('validateArrayMinLength: NaN at index 1', () => {
    expect(() => validateArrayMinLength([1, NaN], 1, 'm')).toThrow('m: invalid value at index 1: NaN')
  })
  it('validate2DArray: non-array (exact message)', () => {
    expect(() => validate2DArray(5, 1, 1, 'm')).toThrow('m: expected 2D array, got number')
  })
  it('validate2DArray: Infinity at [1][0]', () => {
    expect(() => validate2DArray([[1], [Infinity]], 2, 1, 'm')).toThrow(
      'm: invalid value at [1][0]: Infinity'
    )
  })
})

describe('validateDataset', () => {
  const ok = [[1, 2], [3, 4], [5, 6]]

  it('returns nFeatures for valid X (y omitted)', () => {
    expect(validateDataset(ok, undefined, 'M.fit')).toBe(2)
  })
  it('returns nFeatures with finite y', () => {
    expect(validateDataset(ok, [-1.5, 0, 7], 'M.fit')).toBe(2)
  })
  it('accepts matching opts.nFeatures', () => {
    expect(validateDataset(ok, undefined, 'M.fit', { nFeatures: 2 })).toBe(2)
  })
  it('throws on mismatching opts.nFeatures', () => {
    expect(() => validateDataset(ok, undefined, 'M.fit', { nFeatures: 3 })).toThrow(
      'M.fit: X[0]: expected 3 features, got 2'
    )
  })
  it('throws on non-array X', () => {
    expect(() => validateDataset('a', undefined, 'M.fit')).toThrow('M.fit: X: expected array, got string')
  })
  it('throws on empty X', () => {
    expect(() => validateDataset([], [], 'M.fit')).toThrow('M.fit: X is empty')
  })
  it('throws when X[0] is not an array', () => {
    expect(() => validateDataset([1, 2], undefined, 'M.fit')).toThrow(
      'M.fit: X[0]: expected array, got number'
    )
  })
  it('throws on zero-width rows', () => {
    expect(() => validateDataset([[], []], undefined, 'M.fit')).toThrow(
      'M.fit: X rows need at least 1 feature'
    )
  })
  it('throws with row index on irregular row', () => {
    expect(() => validateDataset([[1, 2], [3, 4], [5]], undefined, 'M.fit')).toThrow(
      'M.fit: X[2]: expected array of length 2, got 1'
    )
  })
  it('throws with row index when a row is not an array', () => {
    expect(() => validateDataset([[1, 2], 7], undefined, 'M.fit')).toThrow(
      'M.fit: X[1]: expected array, got number'
    )
  })
  it('throws on NaN in X with row and column', () => {
    expect(() => validateDataset([[1, 2], [3, NaN]], undefined, 'M.fit')).toThrow(
      'M.fit: X[1]: invalid value at index 1: NaN'
    )
  })
  it('throws on Infinity in X', () => {
    expect(() => validateDataset([[Infinity, 2]], undefined, 'M.fit')).toThrow(
      'M.fit: X[0]: invalid value at index 0: Infinity'
    )
  })
  it('throws when y is not an array', () => {
    expect(() => validateDataset(ok, 3, 'M.fit')).toThrow('M.fit: y: expected array, got number')
  })
  it('throws when y length differs (shorter and longer)', () => {
    expect(() => validateDataset(ok, [1, 2], 'M.fit')).toThrow('M.fit: X has 3 rows but y has 2 labels')
    expect(() => validateDataset(ok, [1, 2, 3, 4], 'M.fit')).toThrow(
      'M.fit: X has 3 rows but y has 4 labels'
    )
  })
  it('throws on NaN / Infinity in y with index', () => {
    expect(() => validateDataset(ok, [1, NaN, 3], 'M.fit')).toThrow(
      'M.fit: y: invalid value at index 1: NaN'
    )
    expect(() => validateDataset(ok, [1, 2, -Infinity], 'M.fit')).toThrow(
      'M.fit: y: invalid value at index 2: -Infinity'
    )
  })

  describe('labels: integer', () => {
    it('accepts 0, 1, 5', () => {
      expect(validateDataset(ok, [0, 1, 5], 'S.train', { labels: 'integer' })).toBe(2)
    })
    it('rejects 1.5', () => {
      expect(() => validateDataset(ok, [0, 1.5, 2], 'S.train', { labels: 'integer' })).toThrow(
        'S.train: label y[1] = 1.5 must be an integer >= 0'
      )
    })
    it('rejects -1', () => {
      expect(() => validateDataset(ok, [0, 1, -1], 'S.train', { labels: 'integer' })).toThrow(
        'S.train: label y[2] = -1 must be an integer >= 0'
      )
    })
  })

  describe('labels: unit', () => {
    it('accepts 0, 0.3, 1 (boundaries inclusive)', () => {
      expect(validateDataset(ok, [0, 0.3, 1], 'L.train', { labels: 'unit' })).toBe(2)
    })
    it('rejects 1.0000001', () => {
      expect(() => validateDataset(ok, [0, 1.0000001, 1], 'L.train', { labels: 'unit' })).toThrow(
        'L.train: y[1] = 1.0000001 is outside [0, 1]'
      )
    })
    it('rejects -0.1', () => {
      expect(() => validateDataset(ok, [-0.1, 0, 1], 'L.train', { labels: 'unit' })).toThrow(
        'L.train: y[0] = -0.1 is outside [0, 1]'
      )
    })
  })

  describe('labels: finite (default)', () => {
    it('accepts 1.5 and -3 and 100', () => {
      expect(validateDataset(ok, [1.5, -3, 100], 'R.fit', { labels: 'finite' })).toBe(2)
      expect(validateDataset(ok, [1.5, -3, 100], 'R.fit')).toBe(2)
    })
  })
})

describe('validateArray', () => {
  it('passes for valid array', () => {
    expect(() => validateArray([1, 2, 3], 3, 'test')).not.toThrow()
  })

  it('throws for non-array', () => {
    expect(() => validateArray('not an array', 3, 'test')).toThrow('expected array')
  })

  it('throws for wrong length', () => {
    expect(() => validateArray([1, 2], 3, 'test')).toThrow('expected array of length 3')
  })

  it('throws for NaN value', () => {
    expect(() => validateArray([1, NaN, 3], 3, 'test')).toThrow('invalid value')
  })

  it('throws for Infinity value', () => {
    expect(() => validateArray([1, Infinity, 3], 3, 'test')).toThrow('invalid value')
  })

  it('throws for undefined value', () => {
    expect(() => validateArray([1, undefined, 3], 3, 'test')).toThrow('invalid value')
  })
})

describe('validateArrayMinLength', () => {
  it('passes for valid array', () => {
    expect(() => validateArrayMinLength([1, 2, 3], 2, 'test')).not.toThrow()
  })

  it('throws for too short array', () => {
    expect(() => validateArrayMinLength([1], 2, 'test')).toThrow('at least length 2')
  })
})

describe('validate2DArray', () => {
  it('passes for valid 2D array', () => {
    expect(() => validate2DArray([[1, 2], [3, 4]], 2, 2, 'test')).not.toThrow()
  })

  it('throws for wrong row count', () => {
    expect(() => validate2DArray([[1, 2]], 2, 2, 'test')).toThrow('expected 2 rows')
  })

  it('throws for wrong column count', () => {
    expect(() => validate2DArray([[1, 2, 3], [4, 5, 6]], 2, 2, 'test')).toThrow('expected 2 cols')
  })

  it('throws for non-array row', () => {
    expect(() => validate2DArray([1, [3, 4]], 2, 2, 'test')).toThrow('not an array')
  })
})

describe('validateNumber', () => {
  it('passes for valid number', () => {
    expect(() => validateNumber(1.5, 'test')).not.toThrow()
  })

  it('throws for NaN', () => {
    expect(() => validateNumber(NaN, 'test')).toThrow('expected finite number')
  })

  it('throws for Infinity', () => {
    expect(() => validateNumber(Infinity, 'test')).toThrow('expected finite number')
  })

  it('throws for string', () => {
    expect(() => validateNumber('hello', 'test')).toThrow('expected finite number')
  })
})
