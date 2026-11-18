// Backend registry. The active backend is the default one, with any
// operations replaced by setBackend.
import { Backend, defaultBackend } from './tensor'

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
export function mix(b: Partial<Backend>): Backend {
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
