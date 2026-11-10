// ─── Conv1D Layer ────────────────────────────────────────────────────────────
//
// 1D convolution over a sequence. Each filter slides across the input and
// computes dot products at each position.
//
// Input:  number[][] of shape [inputLength][inputChannels] (2D)
//         For backward compatibility, number[] is accepted when inputChannels=1.
// Output: number[][] of shape [filters][outputLength]
//
// Parameters:
//   kernelSize: size of the sliding window
//   filters: number of output channels
//   stride: step size (default 1)
//   padding: 'valid' (no padding) or 'same' (output length = ceil(L / stride)).
//     'same' pads total = k - 1 zeros: left = floor(total / 2), right = total - left
//     (asymmetric for even k, the extra zero goes to the right; symmetric for odd k).
//     The output length ceil(L / stride) matches TensorFlow, but with stride > 1
//     the windows are shifted differently (TF uses total = max((ceil(L/s)-1)*s + k - L, 0)).
//   inputChannels: number of input channels (default 1)
//
// ─────────────────────────────────────────────────────────────────────────────

import { OptimizerFactory, Optimizer, SGD } from "./optimizers";
import { validateArray } from "./Validation";

export class Conv1D {
  readonly inputLength: number
  readonly kernelSize: number
  readonly filters: number
  readonly stride: number
  readonly padding: 'valid' | 'same'
  readonly inputChannels: number

  kernels: number[][][]  // [filters][kernelSize][inputChannels]
  biases: number[]       // [filters]

  // Per-scalar optimizers
  private _kOpts: Optimizer[][][]  // [filters][kernelSize][inputChannels]
  private _bOpts: Optimizer[]       // [filters]

  private _input: number[][] | null = null
  private _paddedInput: number[][] | null = null

  constructor(
    inputLength: number,
    kernelSize: number,
    filters: number,
    stride = 1,
    padding: 'valid' | 'same' = 'valid',
    optimizerFactory: OptimizerFactory = () => new SGD(),
    inputChannels = 1,
  ) {
    if (inputLength <= 0 || kernelSize <= 0 || filters <= 0) {
      throw new Error('Conv1D: inputLength, kernelSize, and filters must be positive')
    }
    if (kernelSize > inputLength && padding === 'valid') {
      throw new Error('Conv1D: kernelSize cannot exceed inputLength with valid padding')
    }
    if (inputChannels < 1) {
      throw new Error('Conv1D: inputChannels must be >= 1')
    }
    if (!Number.isInteger(stride) || stride < 1) {
      throw new Error(`Conv1D: stride must be a finite integer >= 1, got ${stride}`)
    }

    this.inputLength = inputLength
    this.kernelSize = kernelSize
    this.filters = filters
    this.stride = stride
    this.padding = padding
    this.inputChannels = inputChannels

    // Xavier initialization
    const limit = Math.sqrt(2 / (kernelSize * inputChannels))
    this.kernels = Array.from({ length: filters }, () =>
      Array.from({ length: kernelSize }, () =>
        Array.from({ length: inputChannels }, () => (Math.random() * 2 - 1) * limit)
      )
    )
    this.biases = new Array(filters).fill(0)

    // Initialize per-scalar optimizers
    this._kOpts = Array.from({ length: filters }, () =>
      Array.from({ length: kernelSize }, () =>
        Array.from({ length: inputChannels }, () => optimizerFactory())
      )
    )
    this._bOpts = Array.from({ length: filters }, () => optimizerFactory())
  }

  // ── Forward ───────────────────────────────────────────────────────────────
  // Accepts either number[] (when inputChannels=1) or number[][] (multi-channel).
  forward(input: number[] | number[][]): number[][] {
    // Normalize input to 2D format
    const input2D: number[][] = this._normalizeInput(input)

    this._input = input2D.map(row => [...row])

    // Apply padding if needed
    let padded: number[][]
    if (this.padding === 'same') {
      const { left, right } = this._samePadding()
      const padRow = new Array(this.inputChannels).fill(0)
      padded = new Array(left).fill(null).map(() => [...padRow])
        .concat(input2D)
        .concat(new Array(right).fill(null).map(() => [...padRow]))
    } else {
      padded = input2D
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
          for (let c = 0; c < this.inputChannels; c++) {
            sum += this.kernels[f][k][c] * padded[start + k][c]
          }
        }
        output[f][pos] = sum
      }
    }

    return output
  }

  // ── Backward ──────────────────────────────────────────────────────────────
  backward(dOut: number[][], lr: number = 0.001): number[][] {
    if (!this._paddedInput || !this._input) {
      throw new Error('Conv1D.backward: call forward() first')
    }

    const padded = this._paddedInput
    const outputLength = Math.floor((padded.length - this.kernelSize) / this.stride) + 1

    // Validate dOut shape [filters][outputLength] before touching any state
    if (!Array.isArray(dOut) || dOut.length !== this.filters) {
      throw new Error(
        `Conv1D.backward: expected dOut with ${this.filters} filters, got ${Array.isArray(dOut) ? dOut.length : typeof dOut}`
      )
    }
    for (let f = 0; f < this.filters; f++) {
      validateArray(dOut[f], outputLength, 'Conv1D.backward')
    }

    // Gradient w.r.t. kernels and biases
    const dKernels: number[][][] = Array.from({ length: this.filters }, () =>
      Array.from({ length: this.kernelSize }, () =>
        new Array(this.inputChannels).fill(0)
      )
    )
    const dBiases: number[] = new Array(this.filters).fill(0)

    // Gradient w.r.t. padded input
    const dPadded: number[][] = padded.map(row => new Array(this.inputChannels).fill(0))

    for (let f = 0; f < this.filters; f++) {
      for (let pos = 0; pos < outputLength; pos++) {
        const start = pos * this.stride
        dBiases[f] += dOut[f][pos]
        for (let k = 0; k < this.kernelSize; k++) {
          for (let c = 0; c < this.inputChannels; c++) {
            dKernels[f][k][c] += dOut[f][pos] * padded[start + k][c]
            dPadded[start + k][c] += dOut[f][pos] * this.kernels[f][k][c]
          }
        }
      }
    }

    // Update kernels and biases via per-scalar optimizers
    for (let f = 0; f < this.filters; f++) {
      for (let k = 0; k < this.kernelSize; k++) {
        for (let c = 0; c < this.inputChannels; c++) {
          this.kernels[f][k][c] = this._kOpts[f][k][c].step(this.kernels[f][k][c], dKernels[f][k][c], lr)
        }
      }
      this.biases[f] = this._bOpts[f].step(this.biases[f], dBiases[f], lr)
    }

    // Remove padding from gradient
    if (this.padding === 'same') {
      const { left } = this._samePadding()
      return dPadded.slice(left, left + this.inputLength)
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
        for (const c of k)
          w.push(c)
    // Loop, not a push spread: a spread of >~125 000 arguments overflows the stack.
    for (const b of this.biases) w.push(b)
    return w
  }

  setWeights(weights: number[]): void {
    validateArray(
      weights,
      this.filters * this.kernelSize * this.inputChannels + this.filters,
      'Conv1D.setWeights',
    )
    let idx = 0
    for (let f = 0; f < this.filters; f++)
      for (let k = 0; k < this.kernelSize; k++)
        for (let c = 0; c < this.inputChannels; c++)
          this.kernels[f][k][c] = weights[idx++]
    for (let f = 0; f < this.filters; f++)
      this.biases[f] = weights[idx++]
  }

  // 'same' padding: total = k - 1, left = floor(total / 2), right = total - left
  // (asymmetric: the extra zero goes to the right). With odd k, left === right.
  // Not identical to TensorFlow for stride > 1 (TF: total = max((ceil(L/s)-1)*s + k - L, 0)).
  private _samePadding(): { left: number; right: number } {
    const total = this.kernelSize - 1
    const left = Math.floor(total / 2)
    return { left, right: total - left }
  }

  // ── Normalize input to 2D format ─────────────────────────────────────────
  private _normalizeInput(input: number[] | number[][]): number[][] {
    if (input.length === 0) {
      throw new Error('Conv1D.forward: input cannot be empty')
    }

    // Check if input is 1D (number[])
    if (typeof input[0] === 'number') {
      if (this.inputChannels !== 1) {
        throw new Error(`Conv1D.forward: expected 2D input with ${this.inputChannels} channels, got 1D`)
      }
      const input1D = input as number[]
      if (input1D.length !== this.inputLength) {
        throw new Error(`Conv1D.forward: expected input of length ${this.inputLength}, got ${input1D.length}`)
      }
      return input1D.map(v => [v])
    }

    // Input is 2D (number[][])
    const input2D = input as number[][]
    if (input2D.length !== this.inputLength) {
      throw new Error(`Conv1D.forward: expected input of length ${this.inputLength}, got ${input2D.length}`)
    }
    for (let i = 0; i < input2D.length; i++) {
      if (input2D[i].length !== this.inputChannels) {
        throw new Error(`Conv1D.forward: expected ${this.inputChannels} channels at position ${i}, got ${input2D[i].length}`)
      }
    }
    return input2D
  }
}
