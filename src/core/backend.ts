// Backend registry. The active backend is the default one, with any
// operations replaced by setBackend.
import { Backend, Vec, defaultBackend } from './tensor'
import { seededRandom } from '../random'

const OPS: (keyof Backend)[] = [
  'matVec',
  'matTVec',
  'addOuter',
  'matMulBatch',
  'matMulBatchT',
  'addOuterBatch',
]

let current: Backend = defaultBackend

/** Mixes `b` with defaultBackend: the operations in `b` replace the default ones. */
function mix(b: Partial<Backend>): Backend {
  const merged: Record<string, unknown> = { ...defaultBackend }
  for (const op of OPS) {
    const fn = b[op]
    if (fn === undefined) continue
    if (typeof fn !== 'function') {
      throw new Error(`setBackend: ${op} must be a function`)
    }
    merged[op] = fn
  }
  return merged as unknown as Backend
}

/**
 * Replaces the operations given in `b` and keeps the default ones for the
 * rest. The mix is always with defaultBackend, not with the previous backend,
 * so a second call does not keep the first call's replacements.
 * `null` restores defaultBackend.
 */
export function setBackend(b: Partial<Backend> | null): void {
  current = b === null ? defaultBackend : mix(b)
}

/** Returns the active backend. */
export function getBackend(): Backend {
  return current
}

// ---- checkBackend ---------------------------------------------------------
//
// Compares every operation of a backend (missing ones fall back to the
// default) with a naive reference, on several shapes and batch sizes.
//
// Error metric: |got - expected| / max(1, S), where S is the sum of the
// absolute values of the terms that make up the element (plus the initial
// value for accumulating operations). This is the usual scale for rounding
// in a sum: it stays small when terms cancel, and an absolute difference
// below 1 always passes with the default tolerance.
//
// Output buffers are allocated with PAD extra elements at the end, filled
// with SENTINEL. A backend that writes past its logical size changes that
// padding and fails. Inputs are snapshotted and must not change.

export interface BackendReport {
  ok: boolean
  failures: { op: keyof Backend; case: string; maxRelError: number }[]
}

const PAD = 3
const SENTINEL = 12345
const SHAPES: [number, number][] = [
  [1, 1],
  [3, 5],
  [17, 9],
  [64, 128],
]
const BATCHES = [1, 4, 33]

interface Reference {
  v: Float64Array
  s: Float64Array
}

interface Outcome {
  ret: unknown
  buf: Vec
  logical: number
  expected: Float64Array
  scale: Float64Array
  watched: Vec[]
  snapshots: Vec[]
}

interface CheckCase {
  op: keyof Backend
  label: string
  run: (impl: Backend) => Outcome
}

// Entries with zeros, both signs and magnitudes from 1e-8 to 1e8.
function sample(rnd: () => number): number {
  if (rnd() < 0.1) return 0
  const sign = rnd() < 0.5 ? -1 : 1
  return sign * Math.pow(10, rnd() * 16 - 8)
}

function values(n: number, rnd: () => number): Vec {
  const v = new Float64Array(n)
  for (let i = 0; i < n; i++) v[i] = sample(rnd)
  return v
}

function refMatVec(W: Vec, rows: number, cols: number, x: Vec, b: Vec | null): Reference {
  const v = new Float64Array(rows)
  const s = new Float64Array(rows)
  for (let k = 0; k < rows; k++) {
    let acc = b === null ? 0 : b[k]
    let sc = b === null ? 0 : Math.abs(b[k])
    for (let j = 0; j < cols; j++) {
      const p = W[k * cols + j] * x[j]
      acc += p
      sc += Math.abs(p)
    }
    v[k] = acc
    s[k] = sc
  }
  return { v, s }
}

function refMatTVec(W: Vec, rows: number, cols: number, d: Vec): Reference {
  const v = new Float64Array(cols)
  const s = new Float64Array(cols)
  for (let k = 0; k < rows; k++) {
    for (let j = 0; j < cols; j++) {
      const p = W[k * cols + j] * d[k]
      v[j] += p
      s[j] += Math.abs(p)
    }
  }
  return { v, s }
}

function refAddOuter(G0: Vec, rows: number, cols: number, d: Vec, x: Vec, sc: number): Reference {
  const v = new Float64Array(rows * cols)
  const s = new Float64Array(rows * cols)
  for (let k = 0; k < rows; k++) {
    for (let j = 0; j < cols; j++) {
      const idx = k * cols + j
      const p = sc * d[k] * x[j]
      v[idx] = G0[idx] + p
      s[idx] = Math.abs(G0[idx]) + Math.abs(p)
    }
  }
  return { v, s }
}

function refMatMulBatch(W: Vec, rows: number, cols: number, X: Vec, n: number, b: Vec | null): Reference {
  const v = new Float64Array(n * rows)
  const s = new Float64Array(n * rows)
  for (let i = 0; i < n; i++) {
    const r = refMatVec(W, rows, cols, X.subarray(i * cols, (i + 1) * cols), b)
    v.set(r.v, i * rows)
    s.set(r.s, i * rows)
  }
  return { v, s }
}

function refMatMulBatchT(W: Vec, rows: number, cols: number, D: Vec, n: number): Reference {
  const v = new Float64Array(n * cols)
  const s = new Float64Array(n * cols)
  for (let i = 0; i < n; i++) {
    const r = refMatTVec(W, rows, cols, D.subarray(i * rows, (i + 1) * rows))
    v.set(r.v, i * cols)
    s.set(r.s, i * cols)
  }
  return { v, s }
}

function refAddOuterBatch(
  G0: Vec,
  rows: number,
  cols: number,
  D: Vec,
  X: Vec,
  n: number,
  sc: number,
): Reference {
  const v = new Float64Array(rows * cols)
  const s = new Float64Array(rows * cols)
  for (let k = 0; k < rows; k++) {
    for (let j = 0; j < cols; j++) {
      const idx = k * cols + j
      let acc = 0
      let abs = 0
      for (let i = 0; i < n; i++) {
        const p = D[i * rows + k] * X[i * cols + j]
        acc += p
        abs += Math.abs(p)
      }
      v[idx] = G0[idx] + sc * acc
      s[idx] = Math.abs(G0[idx]) + Math.abs(sc) * abs
    }
  }
  return { v, s }
}

// Builds one case. `init` is the starting content of the output (for
// accumulating operations); otherwise the logical part starts as SENTINEL,
// so an operation that relies on a zeroed output is caught.
function caseOf(
  op: keyof Backend,
  label: string,
  watched: (Vec | null)[],
  outLen: number,
  init: Vec | null,
  ref: Reference,
  call: (impl: Backend, buf: Vec) => unknown,
): CheckCase {
  const live = watched.filter((a): a is Vec => a !== null)
  return {
    op,
    label,
    run(impl) {
      const snapshots = live.map((a) => a.slice())
      const buf = new Float64Array(outLen + PAD).fill(SENTINEL)
      if (init !== null) buf.set(init, 0)
      const ret = call(impl, buf)
      return {
        ret,
        buf,
        logical: outLen,
        expected: ref.v,
        scale: ref.s,
        watched: live,
        snapshots,
      }
    },
  }
}

function buildCases(rnd: () => number): CheckCase[] {
  const cases: CheckCase[] = []

  for (const [rows, cols] of SHAPES) {
    const shape = `${rows}x${cols}`
    const W = values(rows * cols, rnd)
    const x = values(cols, rnd)
    const d = values(rows, rnd)
    const bias = values(rows, rnd)
    const G0 = values(rows * cols, rnd)
    const scalar = sample(rnd)

    for (const b of [bias, null]) {
      const tag = b === null ? 'no bias' : 'bias'
      cases.push(
        caseOf('matVec', `${shape} ${tag}`, [W, x, b], rows, null,
          refMatVec(W, rows, cols, x, b),
          (impl, buf) => impl.matVec(W, rows, cols, x, b, buf)),
      )
    }

    cases.push(
      caseOf('matTVec', shape, [W, d], cols, null,
        refMatTVec(W, rows, cols, d),
        (impl, buf) => impl.matTVec(W, rows, cols, d, buf)),
    )

    cases.push(
      caseOf('addOuter', shape, [d, x], rows * cols, G0,
        refAddOuter(G0, rows, cols, d, x, scalar),
        (impl, buf) => impl.addOuter(buf, rows, cols, d, x, scalar)),
    )

    for (const n of BATCHES) {
      const X = values(n * cols, rnd)
      const D = values(n * rows, rnd)
      const Gb = values(rows * cols, rnd)
      const tag = `${shape} n=${n}`

      for (const b of [bias, null]) {
        const btag = b === null ? ' no bias' : ' bias'
        cases.push(
          caseOf('matMulBatch', tag + btag, [W, X, b], n * rows, null,
            refMatMulBatch(W, rows, cols, X, n, b),
            (impl, buf) => impl.matMulBatch(W, rows, cols, X, n, b, buf)),
        )
      }

      cases.push(
        caseOf('matMulBatchT', tag, [W, D], n * cols, null,
          refMatMulBatchT(W, rows, cols, D, n),
          (impl, buf) => impl.matMulBatchT(W, rows, cols, D, n, buf)),
      )

      cases.push(
        caseOf('addOuterBatch', tag, [D, X], rows * cols, Gb,
          refAddOuterBatch(Gb, rows, cols, D, X, n, scalar),
          (impl, buf) => impl.addOuterBatch(buf, rows, cols, D, X, n, scalar)),
      )
    }
  }

  return cases
}

/** Returns the worst error of an outcome, or Infinity for a structural failure. */
function score(o: Outcome, tolerance: number): number {
  if (o.ret !== o.buf) return Infinity

  let worst = 0
  for (let i = 0; i < o.logical; i++) {
    const e = Math.abs(o.buf[i] - o.expected[i]) / Math.max(1, o.scale[i])
    if (Number.isNaN(e)) return Infinity
    if (e > worst) worst = e
  }

  // Writes past the logical size are visible as changes in the padding.
  for (let i = o.logical; i < o.buf.length; i++) {
    if (!Object.is(o.buf[i], SENTINEL)) return Infinity
  }

  for (let w = 0; w < o.watched.length; w++) {
    const now = o.watched[w]
    const before = o.snapshots[w]
    for (let i = 0; i < now.length; i++) {
      if (!Object.is(now[i], before[i])) return Infinity
    }
  }

  return worst > tolerance ? worst : 0
}

/**
 * Checks a backend against a naive reference. Operations missing from `b`
 * are taken from defaultBackend. Does not change the active backend.
 *
 * Default tolerance 1e-12 (relative to the scale of each element, see above).
 * `seed` fixes the generated inputs.
 */
export function checkBackend(
  b: Partial<Backend>,
  opts: { tolerance?: number; seed?: number } = {},
): BackendReport {
  const tolerance = opts.tolerance ?? 1e-12
  const impl = mix(b)
  const cases = buildCases(seededRandom(opts.seed ?? 1))

  const failures: BackendReport['failures'] = []
  for (const c of cases) {
    let result: number
    try {
      result = score(c.run(impl), tolerance)
    } catch {
      result = Infinity
    }
    if (result !== 0) {
      failures.push({ op: c.op, case: c.label, maxRelError: result })
    }
  }

  return { ok: failures.length === 0, failures }
}
