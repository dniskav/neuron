import { describe, it, expect, afterEach } from 'vitest'
import { defaultBackend, type Backend, type Vec } from '../src/core/tensor'
import { getBackend, setBackend } from '../src/core/backend'
import { mulberry32 } from './helpers/seed'

// ---- helpers -------------------------------------------------------------

// Deterministic values with zeros, negatives and magnitudes from 1e-8 to 1e8.
const SCALES = [1, 1e-8, 1e8, 0.5, 3]
function randomVec(n: number, rnd: () => number): Vec {
  const v = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if (i % 7 === 0) {
      v[i] = 0
      continue
    }
    const sign = rnd() < 0.5 ? -1 : 1
    v[i] = sign * rnd() * SCALES[i % SCALES.length]
  }
  return v
}

function randomMatrix(rows: number, cols: number, rnd: () => number): number[][] {
  const flat = randomVec(rows * cols, rnd)
  const m: number[][] = []
  for (let r = 0; r < rows; r++) m.push(Array.from(flat.subarray(r * cols, (r + 1) * cols)))
  return m
}

function flatten(m: number[][]): Vec {
  const cols = m.length === 0 ? 0 : m[0].length
  const out = new Float64Array(m.length * cols)
  m.forEach((row, r) => row.forEach((x, c) => (out[r * cols + c] = x)))
  return out
}

// |a - b| / max(1, |a|, |b|): the relative error the contract asks for.
function relErr(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1, Math.abs(a), Math.abs(b))
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tol = 1e-12): void {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) {
    const e = relErr(actual[i], expected[i])
    if (e > tol) {
      throw new Error(`index ${i}: got ${actual[i]}, expected ${expected[i]} (rel ${e})`)
    }
  }
}

// ---- naive references on number[][] (same summation order as the spec) ----

function naiveMatVec(W: number[][], x: number[], b: number[] | null): number[] {
  return W.map((row, k) => {
    let s = b === null ? 0 : b[k]
    for (let j = 0; j < row.length; j++) s += row[j] * x[j]
    return s
  })
}

function naiveMatTVec(W: number[][], d: number[]): number[] {
  const cols = W[0].length
  const out = new Array<number>(cols).fill(0)
  for (let k = 0; k < W.length; k++) for (let j = 0; j < cols; j++) out[j] += W[k][j] * d[k]
  return out
}

function naiveAddOuter(G: number[][], d: number[], x: number[], s: number): number[][] {
  return G.map((row, k) => row.map((g, j) => g + s * d[k] * x[j]))
}

function naiveMatMulBatch(W: number[][], X: number[][], b: number[] | null): number[][] {
  return X.map((x) => naiveMatVec(W, x, b))
}

function naiveMatMulBatchT(W: number[][], D: number[][]): number[][] {
  const cols = W[0].length
  return D.map((d) => {
    const out = new Array<number>(cols).fill(0)
    for (let j = 0; j < cols; j++) for (let k = 0; k < W.length; k++) out[j] += W[k][j] * d[k]
    return out
  })
}

function naiveAddOuterBatch(G: number[][], D: number[][], X: number[][], s: number): number[][] {
  return G.map((row, k) =>
    row.map((g, j) => {
      let acc = 0
      for (let i = 0; i < D.length; i++) acc += D[i][k] * X[i][j]
      return g + s * acc
    }),
  )
}

const SIZES: [number, number][] = [
  [1, 1],
  [3, 5],
  [17, 9],
]
const BATCHES = [1, 4, 33]

// ---- equivalence with the naive implementation ----------------------------

describe('defaultBackend matches a naive reference', () => {
  for (const [rows, cols] of SIZES) {
    for (const withBias of [false, true]) {
      it(`matVec ${rows}x${cols} bias=${withBias}`, () => {
        const rnd = mulberry32(rows * 100 + cols + (withBias ? 1 : 0))
        const W = randomMatrix(rows, cols, rnd)
        const x = randomVec(cols, rnd)
        const bArr = Array.from(randomVec(rows, rnd))
        const b = withBias ? Float64Array.from(bArr) : null
        const out = new Float64Array(rows).fill(999)
        defaultBackend.matVec(flatten(W), rows, cols, x, b, out)
        expectClose(out, naiveMatVec(W, Array.from(x), withBias ? bArr : null))
      })
    }

    it(`matTVec ${rows}x${cols}`, () => {
      const rnd = mulberry32(rows * 31 + cols)
      const W = randomMatrix(rows, cols, rnd)
      const d = randomVec(rows, rnd)
      const out = new Float64Array(cols).fill(999)
      defaultBackend.matTVec(flatten(W), rows, cols, d, out)
      expectClose(out, naiveMatTVec(W, Array.from(d)))
    })

    for (const s of [1, -0.5, 1e-8]) {
      it(`addOuter ${rows}x${cols} s=${s}`, () => {
        const rnd = mulberry32(rows * 17 + cols * 3)
        const d = randomVec(rows, rnd)
        const x = randomVec(cols, rnd)
        const G0 = randomMatrix(rows, cols, rnd)
        const G = flatten(G0)
        defaultBackend.addOuter(G, rows, cols, d, x, s)
        expectClose(G, flatten(naiveAddOuter(G0, Array.from(d), Array.from(x), s)))
      })
    }
  }

  for (const [rows, cols] of SIZES) {
    for (const n of BATCHES) {
      for (const withBias of [false, true]) {
        it(`matMulBatch ${rows}x${cols} n=${n} bias=${withBias}`, () => {
          const rnd = mulberry32(rows * 7 + cols * 11 + n + (withBias ? 5 : 0))
          const W = randomMatrix(rows, cols, rnd)
          const X = randomMatrix(n, cols, rnd)
          const bArr = Array.from(randomVec(rows, rnd))
          const b = withBias ? Float64Array.from(bArr) : null
          const OUT = new Float64Array(n * rows).fill(999)
          defaultBackend.matMulBatch(flatten(W), rows, cols, flatten(X), n, b, OUT)
          expectClose(OUT, flatten(naiveMatMulBatch(W, X, withBias ? bArr : null)))
        })
      }

      it(`matMulBatchT ${rows}x${cols} n=${n}`, () => {
        const rnd = mulberry32(rows * 13 + cols * 5 + n)
        const W = randomMatrix(rows, cols, rnd)
        const D = randomMatrix(n, rows, rnd)
        const OUT = new Float64Array(n * cols).fill(999)
        defaultBackend.matMulBatchT(flatten(W), rows, cols, flatten(D), n, OUT)
        expectClose(OUT, flatten(naiveMatMulBatchT(W, D)))
      })

      it(`addOuterBatch ${rows}x${cols} n=${n}`, () => {
        const rnd = mulberry32(rows * 19 + cols * 2 + n)
        const D = randomMatrix(n, rows, rnd)
        const X = randomMatrix(n, cols, rnd)
        const G0 = randomMatrix(rows, cols, rnd)
        const G = flatten(G0)
        defaultBackend.addOuterBatch(G, rows, cols, flatten(D), flatten(X), n, 0.25)
        expectClose(G, flatten(naiveAddOuterBatch(G0, D, X, 0.25)))
      })
    }
  }
})

// ---- buffer discipline ----------------------------------------------------

const SENTINEL = 12345.5

describe('backend only writes inside out / G', () => {
  it('matVec and matMulBatch leave the tail of a larger buffer alone', () => {
    const rows = 3
    const cols = 5
    const rnd = mulberry32(42)
    const W = flatten(randomMatrix(rows, cols, rnd))
    const x = randomVec(cols, rnd)
    const buf = new Float64Array(rows + 8).fill(SENTINEL)
    defaultBackend.matVec(W, rows, cols, x, null, buf.subarray(0, rows))
    for (let i = rows; i < buf.length; i++) expect(Object.is(buf[i], SENTINEL)).toBe(true)

    const n = 4
    const X = randomVec(n * cols, rnd)
    const big = new Float64Array(n * rows + 8).fill(SENTINEL)
    defaultBackend.matMulBatch(W, rows, cols, X, n, null, big.subarray(0, n * rows))
    for (let i = n * rows; i < big.length; i++) expect(Object.is(big[i], SENTINEL)).toBe(true)
  })

  it('matTVec and matMulBatchT leave the tail of a larger buffer alone', () => {
    const rows = 17
    const cols = 9
    const rnd = mulberry32(43)
    const W = flatten(randomMatrix(rows, cols, rnd))
    const d = randomVec(rows, rnd)
    const buf = new Float64Array(cols + 8).fill(SENTINEL)
    defaultBackend.matTVec(W, rows, cols, d, buf.subarray(0, cols))
    for (let i = cols; i < buf.length; i++) expect(Object.is(buf[i], SENTINEL)).toBe(true)

    const n = 4
    const D = randomVec(n * rows, rnd)
    const big = new Float64Array(n * cols + 8).fill(SENTINEL)
    defaultBackend.matMulBatchT(W, rows, cols, D, n, big.subarray(0, n * cols))
    for (let i = n * cols; i < big.length; i++) expect(Object.is(big[i], SENTINEL)).toBe(true)
  })

  it('addOuter and addOuterBatch touch only the first rows*cols of G', () => {
    const rows = 3
    const cols = 5
    const rnd = mulberry32(44)
    const d = randomVec(rows, rnd)
    const x = randomVec(cols, rnd)
    const buf = new Float64Array(rows * cols + 8).fill(SENTINEL)
    defaultBackend.addOuter(buf.subarray(0, rows * cols), rows, cols, d, x, 1)
    for (let i = rows * cols; i < buf.length; i++) expect(Object.is(buf[i], SENTINEL)).toBe(true)

    const n = 4
    const D = randomVec(n * rows, rnd)
    const X = randomVec(n * cols, rnd)
    const big = new Float64Array(rows * cols + 8).fill(SENTINEL)
    defaultBackend.addOuterBatch(big.subarray(0, rows * cols), rows, cols, D, X, n, 1)
    for (let i = rows * cols; i < big.length; i++) expect(Object.is(big[i], SENTINEL)).toBe(true)
  })

  it('overwrite ops ignore the previous contents of out', () => {
    const rows = 3
    const cols = 5
    const rnd = mulberry32(45)
    const W = flatten(randomMatrix(rows, cols, rnd))
    const x = randomVec(cols, rnd)
    const a = new Float64Array(rows).fill(0)
    const b = new Float64Array(rows).fill(1e6)
    defaultBackend.matVec(W, rows, cols, x, null, a)
    defaultBackend.matVec(W, rows, cols, x, null, b)
    expectClose(b, a)

    const d = randomVec(rows, rnd)
    const t1 = new Float64Array(cols).fill(0)
    const t2 = new Float64Array(cols).fill(1e6)
    defaultBackend.matTVec(W, rows, cols, d, t1)
    defaultBackend.matTVec(W, rows, cols, d, t2)
    expectClose(t2, t1)
  })
})

describe('addOuter accumulates', () => {
  it('a second call adds the same amount again', () => {
    const rows = 17
    const cols = 9
    const rnd = mulberry32(50)
    const d = randomVec(rows, rnd)
    const x = randomVec(cols, rnd)
    const G = new Float64Array(rows * cols)
    defaultBackend.addOuter(G, rows, cols, d, x, 0.5)
    const once = Float64Array.from(G)
    defaultBackend.addOuter(G, rows, cols, d, x, 0.5)
    expectClose(G, once.map((v) => 2 * v))
  })

  it('adds onto an existing base instead of replacing it', () => {
    const rows = 3
    const cols = 5
    const rnd = mulberry32(51)
    const d = randomVec(rows, rnd)
    const x = randomVec(cols, rnd)
    const base = randomVec(rows * cols, rnd)
    const G = Float64Array.from(base)
    defaultBackend.addOuter(G, rows, cols, d, x, 1)
    const delta = new Float64Array(rows * cols)
    defaultBackend.addOuter(delta, rows, cols, d, x, 1)
    expectClose(G, base.map((v, i) => v + delta[i]))
  })

  it('addOuterBatch also accumulates', () => {
    const rows = 3
    const cols = 5
    const n = 4
    const rnd = mulberry32(52)
    const D = randomVec(n * rows, rnd)
    const X = randomVec(n * cols, rnd)
    const G = new Float64Array(rows * cols)
    defaultBackend.addOuterBatch(G, rows, cols, D, X, n, 1)
    const once = Float64Array.from(G)
    defaultBackend.addOuterBatch(G, rows, cols, D, X, n, 1)
    expectClose(G, once.map((v) => 2 * v))
  })
})

// ---- setBackend / getBackend ---------------------------------------------

describe('setBackend / getBackend', () => {
  afterEach(() => {
    setBackend(null)
  })

  it('returns the default backend when nothing was set', () => {
    setBackend(null)
    expect(getBackend()).toBe(defaultBackend)
  })

  it('replaces only the operation given, keeping the rest', () => {
    let calls = 0
    const spy = (...args: Parameters<Backend['matVec']>): Vec => {
      calls++
      return defaultBackend.matVec(...args)
    }
    setBackend({ matVec: spy })
    const active = getBackend()
    expect(active.matVec).toBe(spy)
    expect(active.matTVec).toBe(defaultBackend.matTVec)
    expect(active.addOuter).toBe(defaultBackend.addOuter)
    expect(active.matMulBatch).toBe(defaultBackend.matMulBatch)
    expect(active.matMulBatchT).toBe(defaultBackend.matMulBatchT)
    expect(active.addOuterBatch).toBe(defaultBackend.addOuterBatch)

    const rows = 3
    const cols = 5
    const rnd = mulberry32(60)
    const W = flatten(randomMatrix(rows, cols, rnd))
    const x = randomVec(cols, rnd)
    const out = new Float64Array(rows)
    active.matVec(W, rows, cols, x, null, out)
    expect(calls).toBe(1)
    expectClose(out, naiveMatVec(randomMatrixFrom(W, rows, cols), Array.from(x), null))
  })

  it('mixes with the default backend, not with the previous one', () => {
    const spyA = (...args: Parameters<Backend['matVec']>): Vec => defaultBackend.matVec(...args)
    const spyB = (...args: Parameters<Backend['matTVec']>): Vec => defaultBackend.matTVec(...args)
    setBackend({ matVec: spyA })
    setBackend({ matTVec: spyB })
    expect(getBackend().matTVec).toBe(spyB)
    expect(getBackend().matVec).toBe(defaultBackend.matVec)
  })

  it('setBackend(null) restores the default backend', () => {
    setBackend({ matVec: (...args) => defaultBackend.matVec(...args) })
    setBackend(null)
    expect(getBackend()).toBe(defaultBackend)
  })

  it('an undefined entry keeps the default operation', () => {
    setBackend({ matVec: undefined })
    expect(getBackend().matVec).toBe(defaultBackend.matVec)
  })

  it('rejects a non-function operation with a clear message', () => {
    expect(() => setBackend({ matVec: 3 as unknown as Backend['matVec'] })).toThrow(
      'setBackend: matVec must be a function',
    )
    expect(getBackend()).toBe(defaultBackend)
  })
})

// Rebuilds the number[][] matrix from a flat row-major buffer.
function randomMatrixFrom(W: Vec, rows: number, cols: number): number[][] {
  const m: number[][] = []
  for (let r = 0; r < rows; r++) m.push(Array.from(W.subarray(r * cols, (r + 1) * cols)))
  return m
}
