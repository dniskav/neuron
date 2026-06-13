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
// ─────────────────────────────────────────────────────────────────────────────

export interface Serializable {
  getWeights(): number[]
  setWeights(weights: number[]): void
}

export class ModelSaver {
  // ── Serialize to JSON string ──────────────────────────────────────────────
  static toJSON(model: Serializable): string {
    return JSON.stringify({
      weights: model.getWeights(),
      timestamp: Date.now(),
    })
  }

  // ── Deserialize from JSON string ──────────────────────────────────────────
  static fromJSON(model: Serializable, json: string): void {
    const data = JSON.parse(json)
    if (!data.weights || !Array.isArray(data.weights)) {
      throw new Error('ModelSaver.fromJSON: invalid model data')
    }
    model.setWeights(data.weights)
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
