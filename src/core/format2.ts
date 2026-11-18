// ─── FORMAT 2 (pure validator) ────────────────────────────────────────────────
//
// Types and checks of the format-2 JSON of `ModelSaver.save` / `load`
// (contracts.md section 6). It imports NO model class and NO optimizer, so the
// inference entry point can reuse it. Every check runs before anything is built.
import { elu, leakyRelu, linear, makeElu, makeLeakyRelu, relu, sigmoid, tanh } from '../activations'
import type { Activation } from '../activations'

export type ActivationSpec =
  | 'sigmoid' | 'relu' | 'tanh' | 'linear' | 'leakyRelu' | 'elu'
  | { name: 'leakyRelu' | 'elu'; alpha: number }

export type OptimizerSpec =
  | { name: 'sgd' }
  | { name: 'momentum'; beta: number }
  | { name: 'adam'; beta1: number; beta2: number; epsilon: number }
  | { name: 'clip'; clipValue: number; inner: OptimizerSpec }
  | { name: 'custom' }

export interface NetworkNConfig {
  structure: number[]
  activations: ActivationSpec[]
  optimizer: OptimizerSpec
  residual: boolean[]
  dropoutRate: number
}

export interface SavedModel {
  format: 2
  kind: string
  config: unknown
  length: number
  weights: (number | null)[]
  timestamp: number
}

const PLAIN = ['sigmoid', 'relu', 'tanh', 'linear', 'leakyRelu', 'elu']

function show(v: unknown): string {
  return JSON.stringify(v) ?? String(v)
}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

/**
 * Parses and validates a format-2 JSON text. Returns the saved model with
 * `weights` known to be finite numbers and `length === weights.length`.
 * `who` is the prefix of the messages, e.g. 'ModelSaver.load'.
 */
export function parseSavedModel(
  json: string,
  who: string,
): { kind: string; config: unknown; weights: number[] } {
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch (e) {
    throw new Error(`${who}: the text is not valid JSON (${(e as Error).message})`)
  }
  if (!isObject(parsed)) throw new Error(`${who}: invalid model data`)
  if (parsed.format !== 2) {
    throw new Error(`${who}: unsupported format ${show(parsed.format)} (expected 2)`)
  }
  if (typeof parsed.kind !== 'string' || parsed.kind === '') {
    throw new Error(`${who}: kind must be a non-empty string, got ${show(parsed.kind)}`)
  }
  if (!isObject(parsed.config)) throw new Error(`${who}: config must be an object`)
  if (!Array.isArray(parsed.weights)) throw new Error(`${who}: weights must be an array`)
  const weights = parsed.weights as unknown[]
  for (let i = 0; i < weights.length; i++) {
    if (!isFiniteNumber(weights[i])) {
      throw new Error(
        `${who}: weights[${i}] is not a finite number (value: ${show(weights[i])}); NaN and Infinity are saved as null`,
      )
    }
  }
  if (parsed.length !== weights.length) {
    throw new Error(`${who}: length (${show(parsed.length)}) does not match weights (${weights.length})`)
  }
  return { kind: parsed.kind, config: parsed.config, weights: weights as number[] }
}

export function validateActivationSpec(spec: unknown, where: string): ActivationSpec {
  if (typeof spec === 'string') {
    if (PLAIN.includes(spec)) return spec as ActivationSpec
    throw new Error(`${where}: unknown activation ${show(spec)}`)
  }
  if (isObject(spec) && (spec.name === 'leakyRelu' || spec.name === 'elu') && isFiniteNumber(spec.alpha)) {
    return { name: spec.name, alpha: spec.alpha }
  }
  throw new Error(`${where}: invalid activation ${show(spec)}`)
}

export function validateOptimizerSpec(spec: unknown, where: string): OptimizerSpec {
  if (!isObject(spec)) throw new Error(`${where}: invalid optimizer ${show(spec)}`)
  switch (spec.name) {
    case 'sgd':
      return { name: 'sgd' }
    case 'custom':
      return { name: 'custom' }
    case 'momentum':
      if (!isFiniteNumber(spec.beta)) throw new Error(`${where}: momentum needs a finite beta`)
      return { name: 'momentum', beta: spec.beta }
    case 'adam':
      if (!isFiniteNumber(spec.beta1) || !isFiniteNumber(spec.beta2) || !isFiniteNumber(spec.epsilon)) {
        throw new Error(`${where}: adam needs finite beta1, beta2 and epsilon`)
      }
      return { name: 'adam', beta1: spec.beta1, beta2: spec.beta2, epsilon: spec.epsilon }
    case 'clip':
      if (!isFiniteNumber(spec.clipValue)) throw new Error(`${where}: clip needs a finite clipValue`)
      return { name: 'clip', clipValue: spec.clipValue, inner: validateOptimizerSpec(spec.inner, where) }
    default:
      throw new Error(`${where}: unknown optimizer ${show(spec.name)}`)
  }
}

/** True if the spec (or any wrapped one) is a custom optimizer. */
export function hasCustomOptimizer(spec: OptimizerSpec): boolean {
  if (spec.name === 'custom') return true
  return spec.name === 'clip' ? hasCustomOptimizer(spec.inner) : false
}

export function validateNetworkNConfig(config: unknown, where: string): NetworkNConfig {
  if (!isObject(config)) throw new Error(`${where}: config must be an object`)
  const s = config.structure
  if (!Array.isArray(s) || s.length < 2 || !s.every(n => Number.isInteger(n) && n >= 1)) {
    throw new Error(`${where}: structure must be an array of at least 2 integers >= 1, got ${show(s)}`)
  }
  const nLayers = s.length - 1
  if (!Array.isArray(config.activations) || config.activations.length !== nLayers) {
    throw new Error(`${where}: activations must be an array of ${nLayers} items`)
  }
  const activations = config.activations.map((a, i) => validateActivationSpec(a, `${where}: activations[${i}]`))
  const optimizer = validateOptimizerSpec(config.optimizer, `${where}: optimizer`)
  if (!Array.isArray(config.residual) || config.residual.length !== nLayers ||
      !config.residual.every(r => typeof r === 'boolean')) {
    throw new Error(`${where}: residual must be an array of ${nLayers} booleans`)
  }
  const d = config.dropoutRate
  if (!isFiniteNumber(d) || d < 0 || d >= 1) {
    throw new Error(`${where}: dropoutRate must be a number in [0, 1), got ${show(d)}`)
  }
  return {
    structure: [...(s as number[])],
    activations,
    optimizer,
    residual: [...(config.residual as boolean[])],
    dropoutRate: d,
  }
}

/** Builds the built-in activation a spec names. */
export function activationFromSpec(spec: ActivationSpec): Activation {
  if (typeof spec !== 'string') {
    return spec.name === 'leakyRelu' ? makeLeakyRelu(spec.alpha) : makeElu(spec.alpha)
  }
  switch (spec) {
    case 'sigmoid': return sigmoid
    case 'relu': return relu
    case 'tanh': return tanh
    case 'linear': return linear
    case 'leakyRelu': return leakyRelu
    default: return elu
  }
}

/** Spec of an activation, or an error that says why it cannot be saved. */
export function activationToSpec(act: Activation, where: string): ActivationSpec {
  const name = act.name
  if (name === undefined) {
    throw new Error(
      `${where}: the activation has no name, so it cannot be saved. Use a built-in ` +
      `(sigmoid, relu, tanh, linear, leakyRelu, elu, makeLeakyRelu, makeElu) or give your object a name and register a class that can rebuild it`,
    )
  }
  if (name === 'leakyRelu' || name === 'elu') {
    if (!isFiniteNumber(act.alpha)) throw new Error(`${where}: activation '${name}' has no finite alpha`)
    return { name, alpha: act.alpha }
  }
  if (PLAIN.includes(name)) return name as ActivationSpec
  throw new Error(`${where}: unknown activation name ${show(name)}`)
}
