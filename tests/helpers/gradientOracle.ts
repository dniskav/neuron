// Finite-difference gradient oracle (feature 002, T002).
//
// Compares the update a model REALLY performs (`step(lr)`) with the central
// finite-difference gradient of its loss. Convention (constitution): the
// optimizers do `w + lr * g`, so a correct model receives g = -dL/dw and
// moves along -grad L. A correct model therefore gives cosine ~ +1.
//
//   cosine = (delta . -g) / (|delta| |g|)     ratio = |delta| / (lr |g|)
//
// API follows plan.md. Adjustments: `resetRng` is a callback (no dependency
// on seed.ts); `withPlainSGD` is synchronous; failures throw `OracleError`
// (with a `code`), including expectGradientMatches mismatches.
import { Adam } from '../../src/optimizers'

export interface OracleOptions {
  getParams(): number[]
  setParams(p: number[]): void
  /** Deterministic loss; must reset any state/noise it depends on. */
  loss(): number
  /** ONE real update of the model. */
  step(lr: number): void
  lr?: number
  eps?: number
  maxParams?: number
  /** Called before every loss() and step() call. */
  resetRng?: () => void
}

export interface OracleResult {
  cosine: number
  ratio: number
  /** Number of sampled parameters. */
  n: number
}

export type OracleErrorCode = 'NAN' | 'ZERO_STEP' | 'FEW_GRADIENTS' | 'MISMATCH' | 'NO_PARAMS'

export class OracleError extends Error {
  constructor(
    readonly code: OracleErrorCode,
    message: string,
  ) {
    super(`[${code}] ${message}`)
    this.name = 'OracleError'
  }
}

const GRAD_EPS = 1e-12

function sampleIndices(total: number, max: number): number[] {
  if (total <= max) return Array.from({ length: total }, (_, i) => i)
  // Deterministic fixed-stride sampling, evenly spread over the vector.
  return Array.from({ length: max }, (_, k) => Math.floor((k * total) / max))
}

export function gradientOracle(o: OracleOptions): OracleResult {
  const lr = o.lr ?? 1e-3
  const eps = o.eps ?? 1e-5
  const maxParams = o.maxParams ?? 400
  const reset = o.resetRng ?? (() => {})

  const p0 = [...o.getParams()]
  if (p0.length === 0) throw new OracleError('NO_PARAMS', 'getParams() returned no parameters')
  if (p0.some((v) => !Number.isFinite(v)))
    throw new OracleError('NAN', 'initial parameters are not finite')

  const lossAt = (p: number[]): number => {
    o.setParams(p)
    reset()
    const l = o.loss()
    if (!Number.isFinite(l)) throw new OracleError('NAN', 'loss() is not finite')
    return l
  }

  const idx = sampleIndices(p0.length, maxParams)
  const g: number[] = []
  let delta: number[]
  try {
    for (const i of idx) {
      const plus = [...p0]
      plus[i] += eps
      const minus = [...p0]
      minus[i] -= eps
      g.push((lossAt(plus) - lossAt(minus)) / (2 * eps))
    }

    o.setParams([...p0])
    reset()
    o.step(lr)
    const p1 = o.getParams()
    delta = idx.map((i) => p1[i] - p0[i])
  } finally {
    o.setParams([...p0])
  }

  if (delta.some((v) => !Number.isFinite(v)))
    throw new OracleError('NAN', 'step() produced non-finite parameters')

  const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0))
  const nonZero = g.filter((v) => Math.abs(v) > GRAD_EPS).length
  if (nonZero < 3)
    throw new OracleError(
      'FEW_GRADIENTS',
      `only ${nonZero} non-zero numeric gradients (need >= 3); the check would be uninformative`,
    )
  const nd = norm(delta)
  if (nd === 0) throw new OracleError('ZERO_STEP', 'step() did not change any sampled parameter')
  const ng = norm(g)

  let dot = 0
  for (let k = 0; k < g.length; k++) dot += delta[k] * -g[k]
  return { cosine: dot / (nd * ng), ratio: nd / (lr * ng), n: idx.length }
}

export function expectGradientMatches(o: OracleOptions, min = 0.999): OracleResult {
  const r = gradientOracle(o)
  if (!(r.cosine > min))
    throw new OracleError(
      'MISMATCH',
      `cosine ${r.cosine} <= ${min} (ratio ${r.ratio}, n ${r.n}); the update does not follow -dL/dw`,
    )
  return r
}

type Tensor = number[] | number[][] | number[][][]

function flatten(t: Tensor, out: number[]): void {
  for (const x of t as unknown[]) {
    if (Array.isArray(x)) flatten(x as Tensor, out)
    else out.push(x as number)
  }
}

function writeInPlace(t: Tensor, p: number[], pos: number): number {
  const a = t as unknown[]
  for (let i = 0; i < a.length; i++) {
    if (Array.isArray(a[i])) pos = writeInPlace(a[i] as Tensor, p, pos)
    else a[i] = p[pos++]
  }
  return pos
}

/** Reads/writes (in place) the given nested arrays as one flat vector. */
export function bindParams(...tensors: Tensor[]): {
  getParams(): number[]
  setParams(p: number[]): void
} {
  return {
    getParams() {
      const out: number[] = []
      for (const t of tensors) flatten(t, out)
      return out
    },
    setParams(p: number[]) {
      let pos = 0
      for (const t of tensors) pos = writeInPlace(t, p, pos)
      if (pos !== p.length)
        throw new OracleError('NO_PARAMS', `setParams got ${p.length} values, expected ${pos}`)
    },
  }
}

/** Runs fn with Adam.prototype.step replaced by plain `w + lr * g` (Q1); always restores. */
export function withPlainSGD<T>(fn: () => T): T {
  const original = Adam.prototype.step
  Adam.prototype.step = function (_weight: number, gradient: number, lr: number): number {
    return _weight + lr * gradient
  }
  try {
    return fn()
  } finally {
    Adam.prototype.step = original
  }
}
