// ─── INPUT VALIDATION HELPERS ─────────────────────────────────────────────────
//
// Lightweight validators used by predict() and train() methods across all classes.
// Throws descriptive errors for invalid inputs to catch bugs early.
//
// ─────────────────────────────────────────────────────────────────────────────

export function validateArray(
  arr: unknown,
  expectedLength: number,
  methodName: string,
): asserts arr is number[] {
  if (!Array.isArray(arr)) {
    throw new Error(`${methodName}: expected array, got ${typeof arr}`)
  }
  if (arr.length !== expectedLength) {
    throw new Error(
      `${methodName}: expected array of length ${expectedLength}, got ${arr.length}`
    )
  }
  for (let i = 0; i < arr.length; i++) {
    if (typeof arr[i] !== 'number' || !isFinite(arr[i] as number)) {
      throw new Error(
        `${methodName}: invalid value at index ${i}: ${arr[i]}`
      )
    }
  }
}

export function validateArrayMinLength(
  arr: unknown,
  minLength: number,
  methodName: string,
): asserts arr is number[] {
  if (!Array.isArray(arr)) {
    throw new Error(`${methodName}: expected array, got ${typeof arr}`)
  }
  if (arr.length < minLength) {
    throw new Error(
      `${methodName}: expected array of at least length ${minLength}, got ${arr.length}`
    )
  }
  for (let i = 0; i < arr.length; i++) {
    if (typeof arr[i] !== 'number' || !isFinite(arr[i] as number)) {
      throw new Error(
        `${methodName}: invalid value at index ${i}: ${arr[i]}`
      )
    }
  }
}

export function validate2DArray(
  arr: unknown,
  expectedRows: number,
  expectedCols: number,
  methodName: string,
): asserts arr is number[][] {
  if (!Array.isArray(arr)) {
    throw new Error(`${methodName}: expected 2D array, got ${typeof arr}`)
  }
  if (arr.length !== expectedRows) {
    throw new Error(
      `${methodName}: expected ${expectedRows} rows, got ${arr.length}`
    )
  }
  for (let i = 0; i < arr.length; i++) {
    if (!Array.isArray(arr[i])) {
      throw new Error(`${methodName}: row ${i} is not an array`)
    }
    if ((arr[i] as number[]).length !== expectedCols) {
      throw new Error(
        `${methodName}: row ${i} expected ${expectedCols} cols, got ${(arr[i] as number[]).length}`
      )
    }
    for (let j = 0; j < (arr[i] as number[]).length; j++) {
      if (typeof (arr[i] as number[])[j] !== 'number' || !isFinite((arr[i] as number[])[j])) {
        throw new Error(
          `${methodName}: invalid value at [${i}][${j}]: ${(arr[i] as number[])[j]}`
        )
      }
    }
  }
}

export function validateNumber(
  value: unknown,
  methodName: string,
): asserts value is number {
  if (typeof value !== 'number' || !isFinite(value)) {
    throw new Error(`${methodName}: expected finite number, got ${value}`)
  }
}

// ─── DATASET VALIDATION ───────────────────────────────────────────────────────
//
// validateDataset(X, y, name, opts) checks a dataset BEFORE any model state is
// touched and returns the number of features (width of X[0]).
//   - X: non-empty array of arrays, all with the same width >= 1, all finite.
//   - y (undefined to skip): same length as X, all finite.
//   - opts.labels: 'integer' (integers >= 0), 'unit' (in [0, 1]) or 'finite' (default).
//   - opts.nFeatures: if given, the width X must have.
// Error messages start with `name` and carry the row index (X[i] / y[i]).

export interface DatasetOptions {
  labels?: 'integer' | 'unit' | 'finite'
  nFeatures?: number
}

export function validateDataset(
  X: unknown,
  y: unknown,
  name: string,
  opts: DatasetOptions = {},
): number {
  if (!Array.isArray(X)) {
    throw new Error(`${name}: X: expected array, got ${typeof X}`)
  }
  if (X.length === 0) throw new Error(`${name}: X is empty`)
  if (!Array.isArray(X[0])) {
    throw new Error(`${name}: X[0]: expected array, got ${typeof X[0]}`)
  }
  const nFeatures = (X[0] as unknown[]).length
  if (nFeatures === 0) throw new Error(`${name}: X rows need at least 1 feature`)
  if (opts.nFeatures !== undefined && nFeatures !== opts.nFeatures) {
    throw new Error(`${name}: X[0]: expected ${opts.nFeatures} features, got ${nFeatures}`)
  }
  if (y !== undefined) {
    if (!Array.isArray(y)) {
      throw new Error(`${name}: y: expected array, got ${typeof y}`)
    }
    if (X.length !== y.length) {
      throw new Error(`${name}: X has ${X.length} rows but y has ${y.length} labels`)
    }
  }
  for (let i = 0; i < X.length; i++) {
    validateArray(X[i], nFeatures, `${name}: X[${i}]`)
  }
  if (y !== undefined) {
    const ys = y as unknown[]
    validateArray(ys, X.length, `${name}: y`)
    const mode = opts.labels ?? 'finite'
    for (let i = 0; i < ys.length; i++) {
      const v = ys[i] as number
      if (mode === 'integer' && (!Number.isInteger(v) || v < 0)) {
        throw new Error(`${name}: label y[${i}] = ${v} must be an integer >= 0`)
      }
      if (mode === 'unit' && (v < 0 || v > 1)) {
        throw new Error(`${name}: y[${i}] = ${v} is outside [0, 1]`)
      }
    }
  }
  return nFeatures
}
