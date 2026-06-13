import { describe, it, expect } from 'vitest'
import {
  validateArray,
  validateArrayMinLength,
  validate2DArray,
  validateNumber,
} from '../src/Validation'

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
