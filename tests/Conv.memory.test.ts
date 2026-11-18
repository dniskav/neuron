import { describe, it, expect } from 'vitest'
import { Conv1D as Conv1DNew } from '../src/Conv1D'
import { Conv2D as Conv2DNew } from '../src/Conv2D'
import { CausalConv1D as CausalConv1DNew, TCN as TCNNew } from '../src/TCN'
import { relu as reluActivation } from '../src/activations'
import { random, seededRandom } from '../src/random'
import { validateArray } from '../src/Validation'
import {
  Adam,
  ClippedOptimizerFactory,
  Momentum,
  SGD,
  type Optimizer,
  type OptimizerFactory,
} from '../src/optimizers'
import { withSeed } from './helpers/seed'

// ─── T021: memory (vector optimizers) and exactness against the 0.4 layers ───
//
// The four 0.4 layers (Conv1D, Conv2D, CausalConv1D, TCN) are pasted below as the
// reference, with their original names and their original source. The 0.5 layers
// are imported under *New names. Weights are copied from the new layer into the
// reference, so the comparison does not depend on the initialization. Every
// comparison is Object.is, element by element, over 100 steps with a varying lr.

// ── 0.4 reference (pasted from `git show feat/0.5-core:src/<file>`, unchanged) ──

// ── 0.4 src/Conv1D.ts ──
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


class Conv1D {
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
        Array.from({ length: inputChannels }, () => (random() * 2 - 1) * limit)
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

// ── 0.4 src/Conv2D.ts ──
// ─── Conv2D Layer ─────────────────────────────────────────────────────────────
//
// 2D convolution over images or spatial feature maps.
// Each filter slides across the input height and width, computing dot products
// over a kH × kW × channels patch at every position.
//
// Input:   number[][][] of shape [H][W][C]  (height, width, channels)
// Output:  number[][][] of shape [outH][outW][filters]
// Kernels: number[][][][] of shape [filters][kH][kW][channels]
//
// Output size formula:
//   outH = floor((H - kH + padTop + padBottom) / stride) + 1
//   outW = floor((W - kW + padLeft + padRight) / stride) + 1
//
// Padding 'valid': no padding  → all four pads are 0
// Padding 'same':  zero-pad so that outH = ceil(H / stride), outW = ceil(W / stride)
//                  (independent per axis). Total padding per axis is k − 1, split
//                  asymmetrically (not exactly TensorFlow: the output size ceil(L / stride)
//                  matches, but with stride > 1 the windows are shifted differently,
//                  since TF uses total = max((ceil(L/s) − 1)·s + k − L, 0)):
//                    top  = left   = floor((k − 1) / 2)
//                    bottom = right = (k − 1) − top      (the extra row/column, if any,
//                                                         goes to bottom/right)
//                  With odd k this equals the symmetric floor(k / 2) on both sides;
//                  with even k the output now keeps the input size (4×4, k=4 → 4×4).
//
// Xavier initialization:  limit = sqrt(2 / (kH · kW · channels))
//
// ─────────────────────────────────────────────────────────────────────────────


class Conv2D {
  readonly inputHeight: number;
  readonly inputWidth: number;
  readonly channels: number;
  readonly kH: number;
  readonly kW: number;
  readonly filters: number;
  readonly stride: number;
  readonly padding: 'valid' | 'same';

  kernels: number[][][][];  // [filters][kH][kW][channels]
  biases: number[];         // [filters]

  // Per-scalar optimizers
  private _kOpts: Optimizer[][][][];  // [filters][kH][kW][channels]
  private _bOpts: Optimizer[];        // [filters]

  private _input: number[][][] | null = null;
  private _padded: number[][][] | null = null;

  constructor(
    inputHeight: number,
    inputWidth: number,
    channels: number,
    kernelSize: number | [number, number],
    filters: number,
    options?: {
      stride?: number;
      padding?: 'valid' | 'same';
      optimizerFactory?: OptimizerFactory;
    }
  ) {
    const [kH, kW] = Array.isArray(kernelSize)
      ? kernelSize
      : [kernelSize, kernelSize];

    if (inputHeight <= 0 || inputWidth <= 0 || channels <= 0 || filters <= 0) {
      throw new Error('Conv2D: dimensions and filters must be positive');
    }
    if (kH <= 0 || kW <= 0) {
      throw new Error('Conv2D: kernelSize must be positive');
    }
    const stride = options?.stride ?? 1;
    if (!Number.isInteger(stride) || stride < 1) {
      throw new Error(`Conv2D: stride must be a finite integer >= 1, got ${stride}`);
    }
    if ((options?.padding ?? 'valid') === 'valid' && (kH > inputHeight || kW > inputWidth)) {
      throw new Error(
        `Conv2D: kernel ${kH}x${kW} cannot exceed input ${inputHeight}x${inputWidth} with valid padding`
      );
    }

    this.inputHeight = inputHeight;
    this.inputWidth  = inputWidth;
    this.channels    = channels;
    this.kH          = kH;
    this.kW          = kW;
    this.filters     = filters;
    this.stride      = options?.stride  ?? 1;
    this.padding     = options?.padding ?? 'valid';

    const optimizerFactory: OptimizerFactory = options?.optimizerFactory ?? (() => new SGD());

    // Xavier initialization: fan-in = kH * kW * channels
    const limit = Math.sqrt(2 / (kH * kW * channels));
    this.kernels = Array.from({ length: filters }, () =>
      Array.from({ length: kH }, () =>
        Array.from({ length: kW }, () =>
          Array.from({ length: channels }, () => (random() * 2 - 1) * limit)
        )
      )
    );
    this.biases = new Array(filters).fill(0);

    // Initialize per-scalar optimizers
    this._kOpts = Array.from({ length: filters }, () =>
      Array.from({ length: kH }, () =>
        Array.from({ length: kW }, () =>
          Array.from({ length: channels }, () => optimizerFactory())
        )
      )
    );
    this._bOpts = Array.from({ length: filters }, () => optimizerFactory());
  }

  // ── Padding amounts (per axis, asymmetric for even kernels) ──────────────
  private _pads(): { top: number; bottom: number; left: number; right: number } {
    if (this.padding === 'valid') return { top: 0, bottom: 0, left: 0, right: 0 };
    const top  = Math.floor((this.kH - 1) / 2);
    const left = Math.floor((this.kW - 1) / 2);
    return { top, bottom: this.kH - 1 - top, left, right: this.kW - 1 - left };
  }

  // ── Padding helper ────────────────────────────────────────────────────────
  private _pad(input: number[][][]): number[][][] {
    if (this.padding === 'valid') return input;

    const { top, bottom, left, right } = this._pads();
    const H    = input.length;
    const W    = input[0].length;
    const C    = this.channels;

    const paddedH = H + top + bottom;
    const paddedW = W + left + right;

    const out: number[][][] = Array.from({ length: paddedH }, () =>
      Array.from({ length: paddedW }, () => new Array(C).fill(0))
    );

    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        for (let c = 0; c < C; c++) {
          out[h + top][w + left][c] = input[h][w][c];
        }
      }
    }

    return out;
  }

  // ── Output shape ──────────────────────────────────────────────────────────
  outputShape(): [number, number, number] {
    const { top, bottom, left, right } = this._pads();
    const outH = Math.floor((this.inputHeight - this.kH + top + bottom) / this.stride) + 1;
    const outW = Math.floor((this.inputWidth  - this.kW + left + right) / this.stride) + 1;
    return [outH, outW, this.filters];
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // output[h][w][f] = bias[f] + Σ_{kh,kw,c} kernel[f][kh][kw][c] · input[h·s+kh][w·s+kw][c]
  forward(input: number[][][]): number[][][] {
    if (input.length !== this.inputHeight) {
      throw new Error(`Conv2D.forward: expected height ${this.inputHeight}, got ${input.length}`);
    }
    if (input[0].length !== this.inputWidth) {
      throw new Error(`Conv2D.forward: expected width ${this.inputWidth}, got ${input[0].length}`);
    }

    this._input  = input;
    this._padded = this._pad(input);
    const padded = this._padded;

    const [outH, outW] = this.outputShape();

    const output: number[][][] = Array.from({ length: outH }, () =>
      Array.from({ length: outW }, () => new Array(this.filters).fill(0))
    );

    for (let f = 0; f < this.filters; f++) {
      for (let h = 0; h < outH; h++) {
        for (let w = 0; w < outW; w++) {
          let sum = this.biases[f];
          for (let kh = 0; kh < this.kH; kh++) {
            for (let kw = 0; kw < this.kW; kw++) {
              for (let c = 0; c < this.channels; c++) {
                sum += this.kernels[f][kh][kw][c] * padded[h * this.stride + kh][w * this.stride + kw][c];
              }
            }
          }
          output[h][w][f] = sum;
        }
      }
    }

    return output;
  }

  // ── Backward pass ─────────────────────────────────────────────────────────
  // dOutput: number[][][] of shape [outH][outW][filters]
  // Returns dInput: number[][][] of shape [H][W][channels]
  backward(dOutput: number[][][], lr: number): number[][][] {
    if (!this._padded || !this._input) {
      throw new Error('Conv2D.backward: call forward() first');
    }

    const [outH, outW] = this.outputShape();
    if (!Array.isArray(dOutput) || dOutput.length !== outH) {
      throw new Error(
        `Conv2D.backward: expected dOutput height ${outH}, got ${Array.isArray(dOutput) ? dOutput.length : typeof dOutput}`
      );
    }
    for (let h = 0; h < outH; h++) {
      const row = dOutput[h];
      if (!Array.isArray(row) || row.length !== outW) {
        throw new Error(
          `Conv2D.backward: expected dOutput[${h}] width ${outW}, got ${Array.isArray(row) ? row.length : typeof row}`
        );
      }
      for (let w = 0; w < outW; w++) {
        validateArray(row[w], this.filters, `Conv2D.backward dOutput[${h}][${w}]`);
      }
    }

    const padded = this._padded;

    // Gradient accumulators
    const dKernels: number[][][][] = Array.from({ length: this.filters }, () =>
      Array.from({ length: this.kH }, () =>
        Array.from({ length: this.kW }, () => new Array(this.channels).fill(0))
      )
    );
    const dBiases: number[] = new Array(this.filters).fill(0);

    // Gradient w.r.t. padded input
    const dPadded: number[][][] = Array.from({ length: padded.length }, () =>
      Array.from({ length: padded[0].length }, () => new Array(this.channels).fill(0))
    );

    for (let f = 0; f < this.filters; f++) {
      for (let h = 0; h < outH; h++) {
        for (let w = 0; w < outW; w++) {
          const dv = dOutput[h][w][f];
          dBiases[f] += dv;
          for (let kh = 0; kh < this.kH; kh++) {
            for (let kw = 0; kw < this.kW; kw++) {
              for (let c = 0; c < this.channels; c++) {
                const ph = h * this.stride + kh;
                const pw = w * this.stride + kw;
                dKernels[f][kh][kw][c] += dv * padded[ph][pw][c];
                dPadded[ph][pw][c]     += dv * this.kernels[f][kh][kw][c];
              }
            }
          }
        }
      }
    }

    // Apply optimizer updates
    for (let f = 0; f < this.filters; f++) {
      for (let kh = 0; kh < this.kH; kh++) {
        for (let kw = 0; kw < this.kW; kw++) {
          for (let c = 0; c < this.channels; c++) {
            this.kernels[f][kh][kw][c] = this._kOpts[f][kh][kw][c].step(
              this.kernels[f][kh][kw][c], dKernels[f][kh][kw][c], lr
            );
          }
        }
      }
      this.biases[f] = this._bOpts[f].step(this.biases[f], dBiases[f], lr);
    }

    // Strip padding to recover dInput
    const { top, left } = this._pads();
    return dPadded
      .slice(top, top + this.inputHeight)
      .map(row => row.slice(left, left + this.inputWidth));
  }

  // ── Weight serialization ──────────────────────────────────────────────────
  getWeights(): number[] {
    const w: number[] = [];
    for (const kf of this.kernels)
      for (const kh of kf)
        for (const kw of kh)
          for (const v of kw)
            w.push(v);
    // Loop, not a push spread: a spread of >~125 000 arguments overflows the stack.
    for (const b of this.biases) w.push(b);
    return w;
  }

  setWeights(weights: number[]): void {
    validateArray(
      weights,
      this.filters * this.kH * this.kW * this.channels + this.filters,
      'Conv2D.setWeights'
    );
    let idx = 0;
    for (let f = 0; f < this.filters; f++)
      for (let kh = 0; kh < this.kH; kh++)
        for (let kw = 0; kw < this.kW; kw++)
          for (let c = 0; c < this.channels; c++)
            this.kernels[f][kh][kw][c] = weights[idx++];
    for (let f = 0; f < this.filters; f++)
      this.biases[f] = weights[idx++];
  }
}

// ── 0.4 src/TCN.ts ──
// ─── TCN — Temporal Convolutional Network ────────────────────────────────────
//
// A stack of causal dilated 1D convolutions used as an alternative to RNNs
// for sequence modelling. Key properties:
//
//   Causal:     output at position t depends only on positions ≤ t (no future leak)
//   Dilated:    kernel positions are spaced d apart, so a kernel of size k covers
//               k + (k-1)·(d-1) positions with only k multiplications
//   Parallelizable: unlike RNNs, all positions are processed in parallel during
//                   training (no sequential dependency)
//   No vanishing gradient in depth: residual connections carry gradients through
//
// Dilations are typically doubled per level: [1, 2, 4, 8, ..., 2^(levels-1)]
//
// Receptive field of a single causal dilated conv:
//   rf = (kernelSize - 1) · dilation + 1
//
// Receptive field of the full TCN stack (levels layers with dilations 2^0..2^(L-1)):
//   RF = (kernelSize - 1) · (2^levels - 1) + 1
//
// Input/Output shape: number[][] of shape [time][channels]
//
// ─── LEARNING RATE ───────────────────────────────────────────────────────────
//
// train() takes plain SGD steps (default optimizer) on an MSE loss; the gradient
// is exact (it matches finite differences), so a rising loss means lr is too
// large for the problem, not that backprop is wrong. The stable range depends on
// the task, sequence length and initialization. One measured case, not a law:
// x[t] = sin(1.3 t + 0.4), target y[t] = x[t-1], T = 16, TCN(1, 4, 2, 2, 1),
// 300 train() calls per run, random() (Math.random por defecto) seeded with 1-5:
//   lr 0.1 - 0.39: final loss <= 6e-3 in all 5 seeds (start 0.48 - 0.85), with
//                  no spike above 1.5x the initial loss.
//   lr 0.5:        loss spikes to about 1.4 on the way but ends <= 1e-2.
//   lr 0.65 - 0.8: transient spikes (peak up to 6.8); most seeds end stuck at
//                  0.2 - 1.1.
//   lr 1.0:        loss ends higher than it started in all 5 seeds (2.5 - 26).
//   lr >= 1.5:     diverges (loss overflows toward 1e180 or NaN).
// Start around 0.1 - 0.3 and lower it if the loss spikes.
//
// ─────────────────────────────────────────────────────────────────────────────


// ── Input checks (T040). Run before any cache or weight is touched.
function checkLr(where: string, lr: number): void {
  if (!Number.isFinite(lr) || lr < 0) {
    throw new Error(`${where}: lr must be a finite number >= 0`);
  }
}

function checkRows(where: string, name: string, rows: unknown, width: number): void {
  if (!Array.isArray(rows)) {
    throw new Error(`${where}: ${name} must be an array of rows`);
  }
  rows.forEach((row, t) => {
    if (!Array.isArray(row) || row.length !== width) {
      throw new Error(`${where}: ${name}[${t}] must be an array of length ${width}`);
    }
    // T060: NaN/Infinity inside a row is rejected too (it used to flow through silently).
    validateArray(row, width, `${where}: ${name}[${t}]`);
  });
}

// Sizes must be finite positive integers: `NaN <= 0` is false, so a plain `<= 0` accepted NaN.
function isSize(n: unknown): boolean {
  return typeof n === "number" && Number.isInteger(n) && n > 0;
}

// ─── CausalConv1D ─────────────────────────────────────────────────────────────
//
// One causal dilated 1D convolution block.
//
// Causality is achieved by left-padding the input with (kernelSize - 1) · dilation
// zeros before applying a standard 1D convolution. This guarantees that the output
// at position t sees only input positions ≤ t.
//
// Forward:
//   For each output channel f and time position t:
//   out[t][f] = bias[f] + Σ_{k=0}^{K-1} Σ_{c} kernel[f][k][c] · paddedInput[t + k·d][c]
//
// where paddedInput is the causal (left-only) padded input, so position t in the
// padded sequence corresponds to position t - (K-1)·d in the original input.
//
class CausalConv1D {
  readonly inputChannels:  number;
  readonly outputChannels: number;
  readonly kernelSize:     number;
  readonly dilation:       number;

  // kernels[f][k][c]: f = output channel, k = kernel position, c = input channel
  kernels: number[][][];
  biases:  number[];

  // Per-scalar optimizers
  private _kOpts: Optimizer[][][];
  private _bOpts: Optimizer[];

  // Cache for backward pass
  private _paddedInput: number[][] | null = null;
  private _inputLen:    number = 0;

  constructor(
    inputChannels:  number,
    outputChannels: number,
    kernelSize:     number,
    dilation:       number,
    optimizerFactory: OptimizerFactory = () => new SGD()
  ) {
    if (!isSize(inputChannels) || !isSize(outputChannels) || !isSize(kernelSize) || !isSize(dilation)) {
      throw new Error('CausalConv1D: all dimensions must be positive integers');
    }

    this.inputChannels  = inputChannels;
    this.outputChannels = outputChannels;
    this.kernelSize     = kernelSize;
    this.dilation       = dilation;

    // Xavier initialization: fan-in = kernelSize * inputChannels
    const limit = Math.sqrt(2 / (kernelSize * inputChannels));
    this.kernels = Array.from({ length: outputChannels }, () =>
      Array.from({ length: kernelSize }, () =>
        Array.from({ length: inputChannels }, () => (random() * 2 - 1) * limit)
      )
    );
    this.biases = new Array(outputChannels).fill(0);

    this._kOpts = Array.from({ length: outputChannels }, () =>
      Array.from({ length: kernelSize }, () =>
        Array.from({ length: inputChannels }, () => optimizerFactory())
      )
    );
    this._bOpts = Array.from({ length: outputChannels }, () => optimizerFactory());
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // input: [T][inputChannels]
  // Returns: [T][outputChannels]  (same length — causal padding preserves T)
  forward(input: number[][]): number[][] {
    // T060: width and finiteness are checked before the cache (_paddedInput) is replaced.
    checkRows('CausalConv1D.forward', 'input', input, this.inputChannels);
    const T   = input.length;
    const pad = (this.kernelSize - 1) * this.dilation;

    // Left-only (causal) zero padding: [pad zeros][input]
    const zeroCh: number[] = new Array(this.inputChannels).fill(0);
    const padded: number[][] = [
      ...Array.from({ length: pad }, () => [...zeroCh]),
      ...input.map(row => [...row]),
    ];
    this._paddedInput = padded;
    this._inputLen    = T;

    const output: number[][] = Array.from({ length: T }, () =>
      new Array(this.outputChannels).fill(0)
    );

    for (let t = 0; t < T; t++) {
      for (let f = 0; f < this.outputChannels; f++) {
        let sum = this.biases[f];
        for (let k = 0; k < this.kernelSize; k++) {
          const srcPos = t + k * this.dilation;  // position in padded input
          for (let c = 0; c < this.inputChannels; c++) {
            sum += this.kernels[f][k][c] * padded[srcPos][c];
          }
        }
        output[t][f] = sum;
      }
    }

    return output;
  }

  // ── Backward pass ─────────────────────────────────────────────────────────
  // dOutput: [T][outputChannels]
  // Returns dInput: [T][inputChannels]
  backward(dOutput: number[][], lr: number): number[][] {
    if (!this._paddedInput) {
      throw new Error('CausalConv1D.backward: call forward() first');
    }
    checkLr('CausalConv1D.backward', lr);
    if (!Array.isArray(dOutput) || dOutput.length !== this._inputLen) {
      throw new Error(
        `CausalConv1D.backward: dOutput must have ${this._inputLen} rows (one per time step)`
      );
    }
    checkRows('CausalConv1D.backward', 'dOutput', dOutput, this.outputChannels);

    const T      = this._inputLen;
    const pad    = (this.kernelSize - 1) * this.dilation;
    const padded = this._paddedInput;

    const dKernels: number[][][] = Array.from({ length: this.outputChannels }, () =>
      Array.from({ length: this.kernelSize }, () =>
        new Array(this.inputChannels).fill(0)
      )
    );
    const dBiases: number[] = new Array(this.outputChannels).fill(0);

    // Gradient w.r.t. padded input (will strip pad later)
    const dPadded: number[][] = Array.from({ length: padded.length }, () =>
      new Array(this.inputChannels).fill(0)
    );

    for (let t = 0; t < T; t++) {
      for (let f = 0; f < this.outputChannels; f++) {
        const dv = dOutput[t][f];
        dBiases[f] += dv;
        for (let k = 0; k < this.kernelSize; k++) {
          const srcPos = t + k * this.dilation;
          for (let c = 0; c < this.inputChannels; c++) {
            dKernels[f][k][c] += dv * padded[srcPos][c];
            dPadded[srcPos][c] += dv * this.kernels[f][k][c];
          }
        }
      }
    }

    // Apply optimizer updates
    for (let f = 0; f < this.outputChannels; f++) {
      for (let k = 0; k < this.kernelSize; k++) {
        for (let c = 0; c < this.inputChannels; c++) {
          this.kernels[f][k][c] = this._kOpts[f][k][c].step(
            this.kernels[f][k][c], dKernels[f][k][c], lr
          );
        }
      }
      this.biases[f] = this._bOpts[f].step(this.biases[f], dBiases[f], lr);
    }

    // Strip the causal padding to recover dInput of shape [T][inputChannels]
    return dPadded.slice(pad);
  }
}

// ─── TCN ──────────────────────────────────────────────────────────────────────
//
// A stack of `levels` CausalConv1D layers with exponentially increasing dilation:
//   layer 0: dilation = 1
//   layer 1: dilation = 2
//   layer 2: dilation = 4
//   ...
//   layer L-1: dilation = 2^(L-1)
//
// Total receptive field: RF = (kernelSize - 1) · (2^levels - 1) + 1
//
// After the dilated stack, the last hidden-state vector at each time step is
// projected to `outputSize` via a linear layer (NetworkN, single layer).
//
// ReLU activations are applied between conv layers (not at the final projection).
//
// Residual connections: when inputChannels == channels, the input of each level
// is added to the level output AFTER the ReLU, so the sum is not passed through
// the ReLU (ResNet-style). This further stabilizes gradients in deep stacks.
//
class TCN {
  readonly layers: CausalConv1D[];
  private _outputW: number[][];  // [outputSize][channels]  — linear projection
  private _outputB: number[];    // [outputSize]
  private _outOpts: Optimizer[][];
  private _bOutOpts: Optimizer[];

  readonly inputChannels: number;
  readonly channels:      number;
  readonly kernelSize:    number;
  readonly levels:        number;
  readonly outputSize:    number;

  // Cache for backward pass
  private _layerInputs: number[][][] = [];   // inputs to each conv layer
  private _layerOutputs: number[][][] = [];  // outputs from each conv layer (pre-relu)
  private _lastHidden: number[][] = [];      // post-relu output of last conv layer
  private _finalOutputs: number[][] = [];    // linear projection outputs

  constructor(
    inputChannels: number,
    channels:      number,
    kernelSize:    number,
    levels:        number,
    outputSize:    number,
    optimizerFactory: OptimizerFactory = () => new SGD()
  ) {
    if (!isSize(levels)) throw new Error('TCN: levels must be positive (a finite integer)');
    if (!isSize(outputSize)) throw new Error('TCN: outputSize must be positive (a finite integer)');
    // inputChannels, channels and kernelSize are checked by CausalConv1D (all must be positive integers).

    this.inputChannels = inputChannels;
    this.channels      = channels;
    this.kernelSize    = kernelSize;
    this.levels        = levels;
    this.outputSize    = outputSize;

    // Build dilated conv stack
    this.layers = [];
    for (let l = 0; l < levels; l++) {
      const dilation  = Math.pow(2, l);  // 1, 2, 4, 8, ...
      const inCh      = l === 0 ? inputChannels : channels;
      this.layers.push(new CausalConv1D(inCh, channels, kernelSize, dilation, optimizerFactory));
    }

    // Linear output projection: channels → outputSize
    const outLimit = Math.sqrt(2 / channels);
    this._outputW = Array.from({ length: outputSize }, () =>
      Array.from({ length: channels }, () => (random() * 2 - 1) * outLimit)
    );
    this._outputB = new Array(outputSize).fill(0);

    this._outOpts = Array.from({ length: outputSize }, () =>
      Array.from({ length: channels }, () => optimizerFactory())
    );
    this._bOutOpts = Array.from({ length: outputSize }, () => optimizerFactory());
  }

  // ── Receptive field (informational) ──────────────────────────────────────
  // RF = (kernelSize - 1) · (2^levels - 1) + 1
  get receptiveField(): number {
    return (this.kernelSize - 1) * (Math.pow(2, this.levels) - 1) + 1;
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // sequence: [T][inputChannels]
  // Returns:  [T][outputSize]
  forward(sequence: number[][]): number[][] {
    if (sequence.length === 0) {
      throw new Error('TCN.forward: sequence must not be empty');
    }
    checkRows('TCN.forward', 'sequence', sequence, this.inputChannels);

    this._layerInputs  = [];
    this._layerOutputs = [];

    let current = sequence;

    for (let l = 0; l < this.levels; l++) {
      this._layerInputs.push(current.map(row => [...row]));
      const convOut = this.layers[l].forward(current);
      this._layerOutputs.push(convOut);

      // ReLU activation
      const afterRelu = convOut.map(row => row.map(v => reluActivation.fn(v)));

      // Residual connection: add input to output if channel sizes match
      if (current[0].length === afterRelu[0].length) {
        current = afterRelu.map((row, t) => row.map((v, c) => v + current[t][c]));
      } else {
        current = afterRelu;
      }
    }

    this._lastHidden = current;

    // Linear projection at each time step
    const T = current.length;
    this._finalOutputs = Array.from({ length: T }, (_, t) =>
      this._outputB.map((b, i) =>
        b + this._outputW[i].reduce((s, w, j) => s + w * current[t][j], 0)
      )
    );

    return this._finalOutputs.map(row => [...row]);
  }

  // ── Train one step ────────────────────────────────────────────────────────
  // sequence: [T][inputChannels]
  // targets:  [T][outputSize]
  // Returns MSE loss.
  train(sequence: number[][], targets: number[][], lr: number): number {
    if (sequence.length === 0) {
      throw new Error('TCN.train: sequence must not be empty');
    }
    checkRows('TCN.train', 'sequence', sequence, this.inputChannels);
    checkLr('TCN.train', lr);
    checkRows('TCN.train', 'targets', targets, this.outputSize);
    if (targets.length !== sequence.length) {
      throw new Error(`TCN.train: targets must have one row per time step (${sequence.length})`);
    }

    const outputs = this.forward(sequence);
    const T = outputs.length;

    // MSE loss and gradient
    let loss = 0;
    const dOut: number[][] = outputs.map((o, t) =>
      o.map((v, k) => {
        const diff = v - targets[t][k];
        loss += diff * diff;
        // dOut = -dL/dy for L = mean((y - t)^2) over T*outputSize. Optimizers
        // do `w + lr*g`, so they must receive -dL/dw (sign and 1/T folded here).
        return -2 * diff / (T * this.outputSize);
      })
    );
    loss /= (T * this.outputSize);

    // ── Backward through linear projection ───────────────────────────────
    const dWout = Array.from({ length: this.outputSize }, () => new Array(this.channels).fill(0));
    const dBout = new Array(this.outputSize).fill(0);
    const dHidden: number[][] = Array.from({ length: T }, () => new Array(this.channels).fill(0));

    for (let t = 0; t < T; t++) {
      for (let i = 0; i < this.outputSize; i++) {
        const dv = dOut[t][i];
        dBout[i] += dv;
        for (let j = 0; j < this.channels; j++) {
          dWout[i][j] += dv * this._lastHidden[t][j];
          dHidden[t][j] += dv * this._outputW[i][j];
        }
      }
    }

    // Apply output projection updates
    // dOut already carries 1/T: projection and conv layers both use plain lr.
    const scale = lr;
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.channels; j++) {
        this._outputW[i][j] = this._outOpts[i][j].step(this._outputW[i][j], dWout[i][j], scale);
      }
      this._outputB[i] = this._bOutOpts[i].step(this._outputB[i], dBout[i], scale);
    }

    // ── Backward through dilated conv stack (LIFO) ────────────────────────
    let dCurrent = dHidden;

    for (let l = this.levels - 1; l >= 0; l--) {
      const convOut  = this._layerOutputs[l];
      const layerIn  = this._layerInputs[l];

      // Backprop through residual: if shapes match, gradient also flows to skip
      // Backprop through ReLU: d_relu = dCurrent * (convOut[t][c] > 0 ? 1 : 0)
      const dConvOut: number[][] = dCurrent.map((row, t) =>
        row.map((d, c) => d * (convOut[t][c] > 0 ? 1 : 0))
      );

      // Residual adds input to output, so dInput gets dCurrent (pass-through) too
      let dPrevLayer = this.layers[l].backward(dConvOut, lr);

      if (layerIn[0].length === dCurrent[0].length) {
        dPrevLayer = dPrevLayer.map((row, t) =>
          row.map((d, c) => d + dCurrent[t][c])
        );
      }

      dCurrent = dPrevLayer;
    }

    return loss;
  }
}


// ── 0.5 test helpers ──────────────────────────────────────────────────────────

function flat(x: unknown): number[] {
  if (typeof x === 'number') return [x]
  return (x as unknown[]).flatMap(v => flat(v))
}

// Element by element, Object.is (so -0 and 0 differ, and NaN equals NaN).
function expectIdentical(a: unknown, b: unknown): void {
  const fa = flat(a)
  const fb = flat(b)
  expect(fa.length).toBe(fb.length)
  for (let i = 0; i < fa.length; i++) {
    expect(Object.is(fa[i], fb[i])).toBe(true)
  }
}

// Custom optimizer with state: not a built-in, so it takes the scalar fallback.
class Rms implements Optimizer {
  private s = 0
  step(w: number, g: number, lr: number): number {
    this.s = 0.9 * this.s + 0.1 * g * g
    return w + (lr * g) / (Math.sqrt(this.s) + 1e-8)
  }
}

const factories: Array<[string, OptimizerFactory]> = [
  ['SGD', () => new SGD()],
  ['Momentum', () => new Momentum(0.8)],
  ['Adam', () => new Adam()],
  ['Clip(Adam)', ClippedOptimizerFactory(() => new Adam(0.8, 0.99, 1e-7), 0.05)],
  ['custom Rms', () => new Rms()],
]

// Varying lr, always in [0.01, 0.03], so both the lr path and the state are exercised.
function lrAt(step: number): number {
  return 0.02 + 0.01 * Math.sin(step)
}

const STEPS = 100

// ── Conv1D ────────────────────────────────────────────────────────────────────

const conv1dCases = [
  { name: 'valid, stride 1', L: 9, k: 3, F: 4, s: 1, pad: 'valid' as const, C: 2 },
  { name: 'same, stride 2', L: 9, k: 4, F: 3, s: 2, pad: 'same' as const, C: 2 },
]

describe('Conv1D 0.5: exactness against the 0.4 layer', () => {
  for (const cfg of conv1dCases) {
    for (const [name, factory] of factories) {
      it(`${cfg.name} / ${name}: ${STEPS} backward steps give identical kernels, biases and dInput`, () => {
        const fresh = withSeed(7, () => new Conv1DNew(cfg.L, cfg.k, cfg.F, cfg.s, cfg.pad, factory, cfg.C))
        const ref = withSeed(7, () => new Conv1D(cfg.L, cfg.k, cfg.F, cfg.s, cfg.pad, factory, cfg.C))
        ref.setWeights(fresh.getWeights())
        const rnd = seededRandom(31)
        for (let t = 0; t < STEPS; t++) {
          const x = Array.from({ length: cfg.L }, () => Array.from({ length: cfg.C }, () => rnd() * 2 - 1))
          const yF = fresh.forward(x)
          const yR = ref.forward(x)
          expectIdentical(yF, yR)
          const dOut = yF.map(row => row.map(() => rnd() - 0.5))
          expectIdentical(fresh.backward(dOut, lrAt(t)), ref.backward(dOut, lrAt(t)))
        }
        expectIdentical(fresh.getWeights(), ref.getWeights())
      })
    }
  }
})

// ── Conv2D ────────────────────────────────────────────────────────────────────

const conv2dCases = [
  { name: 'same, kernel 3x2, stride 1', H: 6, W: 5, C: 2, k: [3, 2] as [number, number], F: 3, s: 1, pad: 'same' as const },
  { name: 'valid, kernel 2x2, stride 2', H: 6, W: 5, C: 2, k: 2, F: 3, s: 2, pad: 'valid' as const },
]

describe('Conv2D 0.5: exactness against the 0.4 layer', () => {
  for (const cfg of conv2dCases) {
    for (const [name, factory] of factories) {
      it(`${cfg.name} / ${name}: ${STEPS} backward steps give identical kernels, biases and dInput`, () => {
        const opts = { stride: cfg.s, padding: cfg.pad, optimizerFactory: factory }
        const fresh = withSeed(7, () => new Conv2DNew(cfg.H, cfg.W, cfg.C, cfg.k, cfg.F, opts))
        const ref = withSeed(7, () => new Conv2D(cfg.H, cfg.W, cfg.C, cfg.k, cfg.F, opts))
        ref.setWeights(fresh.getWeights())
        const rnd = seededRandom(37)
        for (let t = 0; t < STEPS; t++) {
          const x = Array.from({ length: cfg.H }, () =>
            Array.from({ length: cfg.W }, () => Array.from({ length: cfg.C }, () => rnd() * 2 - 1)))
          const yF = fresh.forward(x)
          const yR = ref.forward(x)
          expectIdentical(yF, yR)
          const dOut = yF.map(row => row.map(cell => cell.map(() => rnd() - 0.5)))
          expectIdentical(fresh.backward(dOut, lrAt(t)), ref.backward(dOut, lrAt(t)))
        }
        expectIdentical(fresh.getWeights(), ref.getWeights())
      })
    }
  }
})

// ── CausalConv1D and TCN ──────────────────────────────────────────────────────

const causalCases = [
  { name: 'kernel 3, dilation 2', C: 2, F: 3, k: 3, d: 2 },
  { name: 'kernel 2, dilation 1', C: 2, F: 2, k: 2, d: 1 },
]

describe('CausalConv1D 0.5: exactness against the 0.4 layer', () => {
  for (const cfg of causalCases) {
    for (const [name, factory] of factories) {
      it(`${cfg.name} / ${name}: ${STEPS} backward steps give identical kernels, biases and dInput`, () => {
        const fresh = withSeed(7, () => new CausalConv1DNew(cfg.C, cfg.F, cfg.k, cfg.d, factory))
        const ref = withSeed(7, () => new CausalConv1D(cfg.C, cfg.F, cfg.k, cfg.d, factory))
        ref.kernels = fresh.kernels.map(f => f.map(k => [...k]))
        ref.biases = [...fresh.biases]
        const rnd = seededRandom(43)
        for (let t = 0; t < STEPS; t++) {
          const x = Array.from({ length: 10 }, () => Array.from({ length: cfg.C }, () => rnd() * 2 - 1))
          const yF = fresh.forward(x)
          const yR = ref.forward(x)
          expectIdentical(yF, yR)
          const dOut = yF.map(row => row.map(() => rnd() - 0.5))
          expectIdentical(fresh.backward(dOut, lrAt(t)), ref.backward(dOut, lrAt(t)))
        }
        expectIdentical(fresh.kernels, ref.kernels)
        expectIdentical(fresh.biases, ref.biases)
      })
    }
  }
})

describe('TCN 0.5: exactness against the 0.4 network', () => {
  for (const [name, factory] of factories) {
    it(`${name}: ${STEPS} train steps give identical losses, outputs and layer weights`, () => {
      // inputChannels 2, channels 3: the first level has no residual, the others do.
      const fresh = withSeed(7, () => new TCNNew(2, 3, 2, 3, 2, factory))
      const ref = withSeed(7, () => new TCN(2, 3, 2, 3, 2, factory))
      const probe = [[0.3, -0.2], [0.1, 0.5], [-0.4, 0.2], [0.6, -0.1]]
      expectIdentical(fresh.forward(probe), ref.forward(probe))
      const rnd = seededRandom(47)
      for (let t = 0; t < STEPS; t++) {
        const seq = Array.from({ length: 8 }, () => [rnd() * 2 - 1, rnd() * 2 - 1])
        const targets = Array.from({ length: 8 }, () => [rnd() * 2 - 1, rnd() * 2 - 1])
        const lossF = fresh.train(seq, targets, lrAt(t))
        const lossR = ref.train(seq, targets, lrAt(t))
        expect(Number.isFinite(lossF)).toBe(true)
        expectIdentical([lossF], [lossR])
      }
      expectIdentical(fresh.forward(probe), ref.forward(probe))
      for (let l = 0; l < fresh.levels; l++) {
        expectIdentical(fresh.layers[l].kernels, ref.layers[l].kernels)
        expectIdentical(fresh.layers[l].biases, ref.layers[l].biases)
      }
    })
  }
})

// ── Memory: one vector optimizer per filter (plus one for the biases) ─────────
// Counted through the factory, which vecFromFactory calls once per vector
// optimizer for a built-in. 0.4 made one scalar optimizer per weight.

describe('Conv 0.5: optimizer memory', () => {
  function counting(): { factory: OptimizerFactory; calls: () => number } {
    let n = 0
    return { factory: () => { n++; return new Adam() }, calls: () => n }
  }

  it('Conv1D: one vector optimizer per filter plus one for the biases', () => {
    const c = counting()
    new Conv1DNew(9, 3, 4, 1, 'valid', c.factory, 2)
    expect(c.calls()).toBe(4 + 1)
  })

  it('Conv2D: one vector optimizer per filter plus one for the biases', () => {
    const c = counting()
    new Conv2DNew(6, 5, 2, [3, 2], 3, { optimizerFactory: c.factory })
    expect(c.calls()).toBe(3 + 1)
  })

  it('CausalConv1D: one vector optimizer per output channel plus one for the biases', () => {
    const c = counting()
    new CausalConv1DNew(2, 3, 3, 2, c.factory)
    expect(c.calls()).toBe(3 + 1)
  })

  it('TCN: levels x (channels + 1) for the conv stack, plus outputSize + 1 for the projection', () => {
    const c = counting()
    new TCNNew(2, 3, 2, 3, 2, c.factory)
    expect(c.calls()).toBe(3 * (3 + 1) + 2 + 1)
  })
})
