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
  private _pos: number

  constructor(data: DataPair, batchSize = 1) {
    if (data.inputs.length !== data.targets.length) {
      throw new Error('DataLoader: inputs and targets must have the same length')
    }
    this.data = data
    this.batchSize = batchSize
    this._indices = Array.from({ length: data.inputs.length }, (_, i) => i)
    this._pos = 0
  }

  // ── Shuffle the data ──────────────────────────────────────────────────────
  shuffle(): void {
    for (let i = this._indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this._indices[i], this._indices[j]] = [this._indices[j], this._indices[i]]
    }
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

  // ── Get total number of samples ───────────────────────────────────────────
  get length(): number {
    return this.data.inputs.length
  }

  // ── Create sequence windows from a time series ────────────────────────────
  static sequences(data: number[][], seqLen: number): DataLoader {
    if (data.length < seqLen + 1) {
      throw new Error('DataLoader.sequences: data length must be >= seqLen + 1')
    }

    const inputs: number[][] = []
    const targets: number[][] = []

    for (let i = 0; i <= data.length - seqLen - 1; i++) {
      inputs.push(data.slice(i, i + seqLen).flat())
      targets.push(data[i + seqLen])
    }

    return new DataLoader({ inputs, targets })
  }
}
