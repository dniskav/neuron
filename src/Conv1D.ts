// ─── Conv1D Layer ────────────────────────────────────────────────────────────
//
// 1D convolution over a sequence. Each filter slides across the input and
// computes dot products at each position.
//
// Input:  number[] of length inputLength
// Output: number[][] of shape [filters][outputLength]
//
// Parameters:
//   kernelSize: size of the sliding window
//   filters: number of output channels
//   stride: step size (default 1)
//   padding: 'valid' (no padding) or 'same' (pad to keep output length)
//
// ─────────────────────────────────────────────────────────────────────────────

export class Conv1D {
  readonly inputLength: number
  readonly kernelSize: number
  readonly filters: number
  readonly stride: number
  readonly padding: 'valid' | 'same'

  kernels: number[][][]  // [filters][kernelSize][1] (simplified: 1 input channel)
  biases: number[]       // [filters]

  private _input: number[] | null = null
  private _paddedInput: number[] | null = null

  constructor(
    inputLength: number,
    kernelSize: number,
    filters: number,
    stride = 1,
    padding: 'valid' | 'same' = 'valid',
  ) {
    if (inputLength <= 0 || kernelSize <= 0 || filters <= 0) {
      throw new Error('Conv1D: inputLength, kernelSize, and filters must be positive')
    }
    if (kernelSize > inputLength && padding === 'valid') {
      throw new Error('Conv1D: kernelSize cannot exceed inputLength with valid padding')
    }

    this.inputLength = inputLength
    this.kernelSize = kernelSize
    this.filters = filters
    this.stride = stride
    this.padding = padding

    // Xavier initialization
    const limit = Math.sqrt(2 / kernelSize)
    this.kernels = Array.from({ length: filters }, () =>
      Array.from({ length: kernelSize }, () =>
        [(Math.random() * 2 - 1) * limit]
      )
    )
    this.biases = new Array(filters).fill(0)
  }

  // ── Forward ───────────────────────────────────────────────────────────────
  forward(input: number[]): number[][] {
    if (input.length !== this.inputLength) {
      throw new Error(`Conv1D.forward: expected input of length ${this.inputLength}, got ${input.length}`)
    }

    this._input = [...input]

    // Apply padding if needed
    let padded: number[]
    if (this.padding === 'same') {
      const padSize = Math.floor((this.kernelSize - 1) / 2)
      padded = new Array(padSize).fill(0).concat(input).concat(new Array(padSize).fill(0))
    } else {
      padded = input
    }
    this._paddedInput = padded

    const outputLength = Math.floor((padded.length - this.kernelSize) / this.stride) + 1

    // Compute convolution for each filter
    const output: number[][] = Array.from({ length: this.filters }, () =>
      new Array(outputLength).fill(0)
    )

    for (let f = 0; f < this.filters; f++) {
      for (let pos = 0; pos < outputLength; pos++) {
        const start = pos * this.stride
        let sum = this.biases[f]
        for (let k = 0; k < this.kernelSize; k++) {
          sum += this.kernels[f][k][0] * padded[start + k]
        }
        output[f][pos] = sum
      }
    }

    return output
  }

  // ── Backward ──────────────────────────────────────────────────────────────
  backward(dOut: number[][]): number[] {
    if (!this._paddedInput || !this._input) {
      throw new Error('Conv1D.backward: call forward() first')
    }

    const padded = this._paddedInput
    const outputLength = dOut[0].length

    // Gradient w.r.t. kernels and biases
    const dKernels: number[][][] = Array.from({ length: this.filters }, () =>
      Array.from({ length: this.kernelSize }, () => [0])
    )
    const dBiases: number[] = new Array(this.filters).fill(0)

    // Gradient w.r.t. padded input
    const dPadded = new Array(padded.length).fill(0)

    for (let f = 0; f < this.filters; f++) {
      for (let pos = 0; pos < outputLength; pos++) {
        const start = pos * this.stride
        dBiases[f] += dOut[f][pos]
        for (let k = 0; k < this.kernelSize; k++) {
          dKernels[f][k][0] += dOut[f][pos] * padded[start + k]
          dPadded[start + k] += dOut[f][pos] * this.kernels[f][k][0]
        }
      }
    }

    // Update kernels and biases (SGD)
    // Note: caller should provide lr; for simplicity we use a small fixed lr here
    // In practice, this would be integrated with the optimizer system
    for (let f = 0; f < this.filters; f++) {
      for (let k = 0; k < this.kernelSize; k++) {
        // Store gradient for external update
        this.kernels[f][k][0] += dKernels[f][k][0] * 0.001
      }
      this.biases[f] += dBiases[f] * 0.001
    }

    // Remove padding from gradient
    if (this.padding === 'same') {
      const padSize = Math.floor((this.kernelSize - 1) / 2)
      return dPadded.slice(padSize, padSize + this.inputLength)
    }
    return dPadded.slice(0, this.inputLength)
  }

  // ── Output length ─────────────────────────────────────────────────────────
  getOutputLength(): number {
    if (this.padding === 'same') {
      return Math.ceil(this.inputLength / this.stride)
    }
    return Math.floor((this.inputLength - this.kernelSize) / this.stride) + 1
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: kernels (flattened), biases.
  getWeights(): number[] {
    const w: number[] = []
    for (const kernel of this.kernels)
      for (const k of kernel)
        w.push(k[0])
    w.push(...this.biases)
    return w
  }

  setWeights(weights: number[]): void {
    let idx = 0
    for (let f = 0; f < this.filters; f++)
      for (let k = 0; k < this.kernelSize; k++)
        this.kernels[f][k][0] = weights[idx++]
    for (let f = 0; f < this.filters; f++)
      this.biases[f] = weights[idx++]
  }
}
