// ─── Model Saver ─────────────────────────────────────────────────────────────
//
// Serializes and deserializes model weights.
// Since this is a browser/Node library without filesystem access by default,
// ModelSaver works with JSON strings. For file I/O, use the companion
// saveToFile/loadFromFile methods that accept a callback.
//
// Usage:
//   // Save
//   const json = ModelSaver.toJSON(model)
//   localStorage.setItem('model', json)
//
//   // Load
//   const json = localStorage.getItem('model')
//   ModelSaver.fromJSON(model, json)
//
//   // Or with file I/O (Node.js):
//   ModelSaver.saveToFile(model, 'model.json', fs.writeFileSync)
//   ModelSaver.loadFromFile(model, 'model.json', fs.readFileSync)
//
// Formato (format 1): { format, length, structure?, weights, timestamp }
//   - length:    número de pesos (weights.length).
//   - structure: solo si el modelo expone `structure` como array de números
//                (p. ej. NetworkN: [entradas, ...ocultas, salidas]).
//   No se guarda el nombre de la clase: los empaquetadores lo minimizan.
//   Los JSON antiguos ({ weights, timestamp }) siguen cargándose si la
//   longitud coincide con la del modelo.
//
// Formato 2 (save / load): { format: 2, kind, config, length, weights, timestamp }.
//   save(model) guarda el nombre de la clase (`kind`), su configuración
//   (`toConfig()`) y los pesos; load(json) crea una instancia nueva con la clase
//   registrada con ModelSaver.register(kind, Clase). El estado del optimizador no
//   se guarda: la instancia cargada empieza con el optimizador de cero.
//   El paquete no tiene efectos al importarse (sideEffects: false), así que las
//   clases integradas se registran en src/index.ts.
//
// ─────────────────────────────────────────────────────────────────────────────
import { hasCustomOptimizer, parseSavedModel, type SavedModel, type OptimizerSpec } from './core/format2'

export interface Serializable {
  getWeights(): number[]
  setWeights(weights: number[]): void
  /** Opcional: estructura del modelo, p. ej. [entradas, ...ocultas, salidas]. */
  structure?: readonly number[]
  /** Opcional: nombre de clase del formato 2 (solo se mira al cargar un JSON de formato 2). */
  kind?: string
}

/** Pesos planos: `getWeightsFlat` / `setWeightsFlat` (p. ej. NetworkLSTM). */
export interface FlatWeightsModel {
  getWeightsFlat(): number[]
  setWeightsFlat(weights: number[]): void
}

/** Pesos como array: `getWeights` / `setWeights`. */
export interface ArrayWeightsModel {
  getWeights(): number[]
  setWeights(weights: number[]): void
}

/** Un modelo tiene uno de los dos pares. Si tiene el plano, format 2 lo usa. */
export type WeightsModel = FlatWeightsModel | ArrayWeightsModel

/** Lo que `ModelSaver.save` necesita de un modelo. */
export type Saveable = { kind?: string; toConfig(): unknown } & WeightsModel

/** Lo que `ModelSaver.register` necesita de una clase. */
export interface Registrable {
  fromConfig(config: unknown): WeightsModel
}

export interface SaveOptions {
  /** Guarda un optimizador propio como { name: 'custom' }; al cargar se usa SGD y se avisa. */
  allowCustomOptimizer?: boolean
}

const registry = new Map<string, Registrable>()

// Optimizador descrito por una config (si tiene), sin lanzar.
function optimizerOf(config: unknown): OptimizerSpec | undefined {
  if (typeof config !== 'object' || config === null) return undefined
  const o = (config as Record<string, unknown>).optimizer
  return typeof o === 'object' && o !== null && typeof (o as { name?: unknown }).name === 'string'
    ? (o as OptimizerSpec)
    : undefined
}

function hasCustom(spec: OptimizerSpec | undefined): boolean {
  try {
    return spec !== undefined && hasCustomOptimizer(spec)
  } catch {
    return false
  }
}

const FORMAT_VERSION = 1

// Format 2 uses the flat pair when the model has both of its methods; otherwise the array pair.
function usesFlat(m: WeightsModel): m is FlatWeightsModel {
  const f = m as Partial<FlatWeightsModel>
  return typeof f.getWeightsFlat === 'function' && typeof f.setWeightsFlat === 'function'
}

function weightIO(m: WeightsModel): { read(): number[]; write(w: number[]): void } {
  if (usesFlat(m)) return { read: () => m.getWeightsFlat(), write: w => m.setWeightsFlat(w) }
  const a = m as ArrayWeightsModel
  return { read: () => a.getWeights(), write: w => a.setWeights(w) }
}

function isNumericArray(v: unknown): v is readonly number[] {
  return Array.isArray(v) && v.every(x => typeof x === 'number')
}

// Representación corta de un valor para los mensajes de error.
function show(v: unknown): string {
  return JSON.stringify(v) ?? String(v)
}

export class ModelSaver {
  // ── Serialize to JSON string ──────────────────────────────────────────────
  static toJSON(model: Serializable): string {
    const weights = model.getWeights()
    const payload: Record<string, unknown> = {
      format: FORMAT_VERSION,
      length: weights.length,
    }
    if (isNumericArray(model.structure)) payload.structure = [...model.structure]
    payload.weights = weights
    payload.timestamp = Date.now()
    return JSON.stringify(payload)
  }

  // ── Format 2: save ────────────────────────────────────────────────────────
  static save(model: Saveable, opts: SaveOptions = {}): string {
    if (typeof model.kind !== 'string' || model.kind === '' || typeof model.toConfig !== 'function') {
      throw new Error('ModelSaver.save: the model needs a kind and a toConfig() method (format 2)')
    }
    const config = model.toConfig()
    if (hasCustom(optimizerOf(config)) && !opts.allowCustomOptimizer) {
      throw new Error(
        'ModelSaver.save: the model uses a custom optimizer, which cannot be saved. ' +
        'Use SGD, Momentum, Adam or ClippedOptimizerFactory, or pass { allowCustomOptimizer: true } ' +
        '(the loaded model will use SGD)',
      )
    }
    const weights = weightIO(model).read()
    const saved: SavedModel = {
      format: 2,
      kind: model.kind,
      config,
      length: weights.length,
      weights,
      timestamp: Date.now(),
    }
    return JSON.stringify(saved)
  }

  // ── Format 2: load ────────────────────────────────────────────────────────
  // Everything is validated before the instance is returned; a failure creates nothing
  // that escapes. The optimizer state starts from zero (Q4).
  static load(json: string): unknown {
    const { kind, config, weights } = parseSavedModel(json, 'ModelSaver.load')
    const entry = registry.get(kind)
    if (entry === undefined) {
      const known = [...registry.keys()].join(', ') || 'none'
      throw new Error(
        `ModelSaver.load: unknown kind ${show(kind)} (registered: ${known}). ` +
        'Register the class with ModelSaver.register(kind, Class)',
      )
    }
    let model: WeightsModel
    try {
      model = entry.fromConfig(config)
    } catch (e) {
      const msg = (e as Error).message
      throw new Error(msg.startsWith('ModelSaver.load:') ? msg : `ModelSaver.load: invalid config for ${kind}: ${msg}`)
    }
    const io = weightIO(model)
    const expected = io.read().length
    if (weights.length !== expected) {
      throw new Error(`ModelSaver.load: the ${kind} in the file expects ${expected} weights and the file has ${weights.length}`)
    }
    if (hasCustom(optimizerOf(config))) {
      console.warn(`ModelSaver.load: ${kind} was saved with a custom optimizer; the loaded model uses SGD`)
    }
    io.write(weights)
    return model
  }

  // ── Format 2: class registry ──────────────────────────────────────────────
  static register(kind: string, entry: Registrable): void {
    if (typeof kind !== 'string' || kind === '') {
      throw new Error(`ModelSaver.register: kind must be a non-empty string, got ${show(kind)}`)
    }
    if (typeof entry?.fromConfig !== 'function') {
      throw new Error('ModelSaver.register: the class needs a static fromConfig(config)')
    }
    registry.set(kind, entry)
  }

  // ── Deserialize from JSON string ──────────────────────────────────────────
  // Validates everything BEFORE writing: on any mismatch it throws and the model is left untouched.
  static fromJSON(model: Serializable, json: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(json)
    } catch (e) {
      throw new Error(`ModelSaver.fromJSON: the text is not valid JSON (${(e as Error).message})`)
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new Error('ModelSaver.fromJSON: invalid model data')
    }
    const data = parsed as Record<string, unknown>

    // A format-2 text without `kind` is not a saved model: it falls to the unsupported-format error.
    if (data.format === 2 && data.kind !== undefined) {
      ModelSaver._fromFormat2(model, json)
      return
    }

    if (data.format !== undefined && data.format !== FORMAT_VERSION) {
      throw new Error(`ModelSaver.fromJSON: unsupported format ${show(data.format)} (expected ${FORMAT_VERSION})`)
    }

    if (!Array.isArray(data.weights)) {
      throw new Error('ModelSaver.fromJSON: invalid model data')
    }
    const weights: unknown[] = data.weights
    for (let i = 0; i < weights.length; i++) {
      const w = weights[i]
      if (typeof w !== 'number' || !Number.isFinite(w)) {
        throw new Error(
          `ModelSaver.fromJSON: weights[${i}] is not a finite number (value: ${show(w)}); NaN and Infinity are saved as null`,
        )
      }
    }

    if (data.length !== undefined && data.length !== weights.length) {
      throw new Error(
        `ModelSaver.fromJSON: length (${show(data.length)}) does not match weights (${weights.length})`,
      )
    }

    const expected = model.getWeights().length
    if (weights.length !== expected) {
      throw new Error(
        `ModelSaver.fromJSON: the model expects ${expected} weights and the file has ${weights.length}`,
      )
    }

    if (data.structure !== undefined) {
      if (!isNumericArray(data.structure)) {
        throw new Error('ModelSaver.fromJSON: structure must be an array of numbers')
      }
      if (isNumericArray(model.structure) && !sameNumbers(data.structure, model.structure)) {
        throw new Error(
          `ModelSaver.fromJSON: file structure [${data.structure.join(',')}] does not match the model structure [${model.structure.join(',')}]`,
        )
      }
    }

    model.setWeights(weights as number[])
  }

  // fromJSON on a format-2 text: same class (kind) and same structure, then the weights.
  private static _fromFormat2(model: Serializable, json: string): void {
    const { kind, config, weights } = parseSavedModel(json, 'ModelSaver.fromJSON')
    if (typeof model.kind !== 'string' || model.kind !== kind) {
      throw new Error(
        `ModelSaver.fromJSON: the file is a ${show(kind)} and the model is ${show(model.kind ?? 'a class without kind')}`,
      )
    }
    const structure = (config as Record<string, unknown>).structure
    if (isNumericArray(structure) && isNumericArray(model.structure) && !sameNumbers(structure, model.structure)) {
      throw new Error(
        `ModelSaver.fromJSON: file structure [${structure.join(',')}] does not match the model structure [${model.structure.join(',')}]`,
      )
    }
    const expected = model.getWeights().length
    if (weights.length !== expected) {
      throw new Error(`ModelSaver.fromJSON: the model expects ${expected} weights and the file has ${weights.length}`)
    }
    model.setWeights(weights)
  }

  // ── Save to file (requires write function) ────────────────────────────────
  static saveToFile(
    model: Serializable,
    path: string,
    writeFn: (path: string, data: string) => void,
  ): void {
    const json = ModelSaver.toJSON(model)
    writeFn(path, json)
  }

  // ── Load from file (requires read function) ───────────────────────────────
  static loadFromFile(
    model: Serializable,
    path: string,
    readFn: (path: string) => string,
  ): void {
    const json = readFn(path)
    ModelSaver.fromJSON(model, json)
  }
}

function sameNumbers(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i])
}
