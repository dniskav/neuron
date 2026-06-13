// ─── DataLoader ──────────────────────────────────────────────────────────────
//
// Utility for batching and shuffling data.
// Despite the library being online (one sample at a time), DataLoader provides
// convenient iteration over datasets with optional shuffling.
//
// Usage:
//   const loader = new DataLoader({ inputs, targets }, 32)
//   while (loader.hasNext()) {
//     const batch = loader.next()
//     // process batch
//   }
//
// With validation split:
//   const loader = new DataLoader({ inputs, targets }, 32, 0.2)
//   // 80% for training, 20% for validation (shuffled before splitting)
//   const valData = loader.getValidationData()
//
// For sequences:
//   const seqLoader = DataLoader.sequences(data, seqLen)
//
// ─────────────────────────────────────────────────────────────────────────────

export interface DataPair {
  inputs: number[][]
  targets: number[][]
}

export class DataLoader {
  readonly data: DataPair
  readonly batchSize: number

  private _indices: number[]
  private _trainIndices: number[]
  private _valIndices: number[]
  private _pos: number
  private _validationSplit: number

  constructor(data: DataPair, batchSize = 1, validationSplit = 0) {
    if (data.inputs.length !== data.targets.length) {
      throw new Error('DataLoader: inputs and targets must have the same length')
    }
    if (validationSplit < 0 || validationSplit >= 1) {
      throw new Error(`DataLoader: validationSplit must be in [0, 1), got ${validationSplit}`)
    }

    this.data = data
    this.batchSize = batchSize
    this._validationSplit = validationSplit

    // Build full indices and shuffle once for consistent split
    const fullIndices = Array.from({ length: data.inputs.length }, (_, i) => i)
    // Fisher-Yates shuffle
    for (let i = fullIndices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [fullIndices[i], fullIndices[j]] = [fullIndices[j], fullIndices[i]]
    }

    if (validationSplit > 0) {
      const valSize = Math.round(data.inputs.length * validationSplit)
      const trainSize = data.inputs.length - valSize
      this._trainIndices = fullIndices.slice(0, trainSize)
      this._valIndices = fullIndices.slice(trainSize)
    } else {
      this._trainIndices = [...fullIndices]
      this._valIndices = []
    }

    this._indices = [...this._trainIndices]
    this._pos = 0
  }

  // ── Shuffle the training data ──────────────────────────────────────────────
  shuffle(): void {
    for (let i = this._trainIndices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this._trainIndices[i], this._trainIndices[j]] = [this._trainIndices[j], this._trainIndices[i]]
    }
    this._indices = [...this._trainIndices]
    this._pos = 0
  }

  // ── Check if more batches are available ───────────────────────────────────
  hasNext(): boolean {
    return this._pos < this._indices.length
  }

  // ── Get next batch ────────────────────────────────────────────────────────
  next(): DataPair {
    const end = Math.min(this._pos + this.batchSize, this._indices.length)
    const batchIndices = this._indices.slice(this._pos, end)
    this._pos = end

    return {
      inputs: batchIndices.map(i => this.data.inputs[i]),
      targets: batchIndices.map(i => this.data.targets[i]),
    }
  }

  // ── Reset iteration ───────────────────────────────────────────────────────
  reset(): void {
    this._pos = 0
  }

  // ── Get total number of training samples ───────────────────────────────────
  get length(): number {
    return this._trainIndices.length
  }

  // ── Get validation data as a DataPair ──────────────────────────────────────
  // Returns the validation samples (inputs + targets) in their shuffled order.
  // Returns empty arrays if no validation split was configured.
  getValidationData(): DataPair {
    return {
      inputs: this._valIndices.map(i => this.data.inputs[i]),
      targets: this._valIndices.map(i => this.data.targets[i]),
    }
  }

  // ── Get number of validation samples ───────────────────────────────────────
  get validationLength(): number {
    return this._valIndices.length
  }

  // ── Create sequence windows from a time series ────────────────────────────
  static sequences(data: number[][], seqLen: number, validationSplit = 0): DataLoader {
    if (data.length < seqLen + 1) {
      throw new Error('DataLoader.sequences: data length must be >= seqLen + 1')
    }

    const inputs: number[][] = []
    const targets: number[][] = []

    for (let i = 0; i <= data.length - seqLen - 1; i++) {
      inputs.push(data.slice(i, i + seqLen).flat())
      targets.push(data[i + seqLen])
    }

    return new DataLoader({ inputs, targets }, 1, validationSplit)
  }
}
