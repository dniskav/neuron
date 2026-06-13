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
