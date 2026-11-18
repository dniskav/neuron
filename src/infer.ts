// ─── INFERENCE ENTRY POINT (@dniskav/neuron/infer) ────────────────────────────
//
// Loads a format-2 JSON of a dense model (NetworkN, NetworkFast, Network) and
// predicts. It has its own reader and imports only core/format2 (pure
// validator), core/tensor, core/backend, activations and Validation. No
// optimizers, random, model classes, ModelSaver or Trainer: the bundle stays small.
import { Backend, Vec } from './core/tensor'
import { getBackend, mix } from './core/backend'
import { activationFromSpec, parseSavedModel, validateNetworkNConfig } from './core/format2'
import type { ActivationSpec } from './core/format2'
import { validateArray } from './Validation'

export interface InferenceModel {
  kind: string
  structure: number[]
  predict(x: number[]): number[]
  predictBatch(X: number[][]): number[][]
}

const WHO = 'loadForInference'

interface Layer {
  nIn: number
  nOut: number
  W: Vec
  b: Vec
  fn: (x: number) => number
  residual: boolean
}

function isCount(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 1
}

// Turns the saved config into structure, activations and residual flags.
function readConfig(kind: string, config: unknown): { structure: number[]; acts: ActivationSpec[]; res: boolean[] } {
  if (kind === 'Network') {
    const c = config as Record<string, unknown>
    if (!isCount(c.nInputs) || !isCount(c.nHidden) || !isCount(c.nOutputs)) {
      throw new Error(`${WHO}: Network config needs integer nInputs, nHidden and nOutputs >= 1`)
    }
    return {
      structure: [c.nInputs, c.nHidden, c.nOutputs],
      acts: ['sigmoid', 'sigmoid'],
      res: [false, false],
    }
  }
  const c = validateNetworkNConfig(config, `${WHO}: ${kind} config`)
  return { structure: c.structure, acts: c.activations, res: c.residual }
}

/**
 * Builds an inference-only model from a format-2 JSON text.
 * `opts.backend` replaces some backend operations for this model; without it
 * the global backend (`setBackend`) is read on every call.
 */
export function loadForInference(json: string, opts?: { backend?: Partial<Backend> }): InferenceModel {
  const saved = parseSavedModel(json, WHO)
  const { kind } = saved
  if (kind !== 'NetworkN' && kind !== 'NetworkFast' && kind !== 'Network') {
    throw new Error(
      `${WHO}: kind '${kind}' is not supported in infer (only NetworkN, NetworkFast and Network); use ModelSaver.load`,
    )
  }
  const { structure, acts, res } = readConfig(kind, saved.config)
  // Operations of opts.backend win; the rest come from the global backend (as in NetworkFast).
  let own: Partial<Backend> | null = null
  if (opts?.backend !== undefined) {
    mix(opts.backend) // validates that each given operation is a function
    own = {}
    for (const [op, fn] of Object.entries(opts.backend)) if (fn !== undefined) (own as Record<string, unknown>)[op] = fn
  }
  const backend = (): Backend => (own === null ? getBackend() : { ...getBackend(), ...own })

  let expected = 0
  for (let l = 0; l < structure.length - 1; l++) expected += structure[l + 1] * (structure[l] + 1)
  if (saved.weights.length !== expected) {
    throw new Error(`${WHO}: ${kind} with structure [${structure}] needs ${expected} weights, got ${saved.weights.length}`)
  }

  // Weight order: per layer, per neuron, its weights and then its bias.
  const layers: Layer[] = []
  let p = 0
  for (let l = 0; l < structure.length - 1; l++) {
    const nIn = structure[l]
    const nOut = structure[l + 1]
    const W = new Float64Array(nOut * nIn)
    const b = new Float64Array(nOut)
    for (let k = 0; k < nOut; k++) {
      for (let j = 0; j < nIn; j++) W[k * nIn + j] = saved.weights[p++]
      b[k] = saved.weights[p++]
    }
    layers.push({ nIn, nOut, W, b, fn: activationFromSpec(acts[l]).fn, residual: res[l] && nIn === nOut })
  }
  const nIn0 = structure[0]
  const nOut = structure[structure.length - 1]

  // Runs n rows (row-major in `cur`) through every layer.
  function run(cur: Vec, n: number): Vec {
    const be = backend()
    for (const L of layers) {
      const out = new Float64Array(n * L.nOut)
      be.matMulBatch(L.W, L.nOut, L.nIn, cur, n, L.b, out)
      for (let i = 0; i < out.length; i++) {
        const v = L.fn(out[i])
        out[i] = L.residual ? v + cur[i] : v
      }
      cur = out
    }
    return cur
  }

  return {
    kind,
    structure: [...structure],
    predict(x: number[]): number[] {
      validateArray(x, nIn0, 'infer.predict')
      const be = backend()
      let cur: Vec = Float64Array.from(x)
      for (const L of layers) {
        const out = new Float64Array(L.nOut)
        be.matVec(L.W, L.nOut, L.nIn, cur, L.b, out)
        for (let k = 0; k < out.length; k++) {
          const v = L.fn(out[k])
          out[k] = L.residual ? v + cur[k] : v
        }
        cur = out
      }
      return Array.from(cur)
    },
    predictBatch(X: number[][]): number[][] {
      if (!Array.isArray(X)) throw new Error(`infer.predictBatch: expected array, got ${typeof X}`)
      const n = X.length
      const flat = new Float64Array(n * nIn0)
      for (let i = 0; i < n; i++) {
        validateArray(X[i], nIn0, `infer.predictBatch: row ${i}`)
        flat.set(X[i], i * nIn0)
      }
      const out = run(flat, n)
      const rows: number[][] = []
      for (let i = 0; i < n; i++) rows.push(Array.from(out.subarray(i * nOut, (i + 1) * nOut)))
      return rows
    },
  }
}
