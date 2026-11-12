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
// ─────────────────────────────────────────────────────────────────────────────

export interface Serializable {
  getWeights(): number[]
  setWeights(weights: number[]): void
  /** Opcional: estructura del modelo, p. ej. [entradas, ...ocultas, salidas]. */
  structure?: readonly number[]
}

const FORMAT_VERSION = 1

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
