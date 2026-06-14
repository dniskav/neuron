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
//   outH = floor((H - kH + 2·padH) / stride) + 1
//   outW = floor((W - kW + 2·padW) / stride) + 1
//
// Padding 'valid': no padding  → padH = padW = 0
// Padding 'same':  zero-pad so output ≈ input size (when stride = 1)
//                  padH = floor(kH / 2),  padW = floor(kW / 2)
//
// Xavier initialization:  limit = sqrt(2 / (kH · kW · channels))
//
// ─────────────────────────────────────────────────────────────────────────────

import { OptimizerFactory, Optimizer, SGD } from "./optimizers";

export class Conv2D {
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
          Array.from({ length: channels }, () => (Math.random() * 2 - 1) * limit)
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

  // ── Padding helper ────────────────────────────────────────────────────────
  private _pad(input: number[][][]): number[][][] {
    if (this.padding === 'valid') return input;

    const padH = Math.floor(this.kH / 2);
    const padW = Math.floor(this.kW / 2);
    const H    = input.length;
    const W    = input[0].length;
    const C    = this.channels;

    const paddedH = H + 2 * padH;
    const paddedW = W + 2 * padW;

    const out: number[][][] = Array.from({ length: paddedH }, () =>
      Array.from({ length: paddedW }, () => new Array(C).fill(0))
    );

    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        for (let c = 0; c < C; c++) {
          out[h + padH][w + padW][c] = input[h][w][c];
        }
      }
    }

    return out;
  }

  // ── Output shape ──────────────────────────────────────────────────────────
  outputShape(): [number, number, number] {
    const padH = this.padding === 'same' ? Math.floor(this.kH / 2) : 0;
    const padW = this.padding === 'same' ? Math.floor(this.kW / 2) : 0;
    const outH = Math.floor((this.inputHeight - this.kH + 2 * padH) / this.stride) + 1;
    const outW = Math.floor((this.inputWidth  - this.kW + 2 * padW) / this.stride) + 1;
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

    const padH = this.padding === 'same' ? Math.floor(this.kH / 2) : 0;
    const padW = this.padding === 'same' ? Math.floor(this.kW / 2) : 0;
    const outH = Math.floor((this.inputHeight - this.kH + 2 * padH) / this.stride) + 1;
    const outW = Math.floor((this.inputWidth  - this.kW + 2 * padW) / this.stride) + 1;

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

    const padded = this._padded;
    const outH   = dOutput.length;
    const outW   = dOutput[0].length;

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
    if (this.padding === 'same') {
      const padH = Math.floor(this.kH / 2);
      const padW = Math.floor(this.kW / 2);
      return dPadded
        .slice(padH, padH + this.inputHeight)
        .map(row => row.slice(padW, padW + this.inputWidth));
    }
    return dPadded
      .slice(0, this.inputHeight)
      .map(row => row.slice(0, this.inputWidth));
  }

  // ── Weight serialization ──────────────────────────────────────────────────
  getWeights(): number[] {
    const w: number[] = [];
    for (const kf of this.kernels)
      for (const kh of kf)
        for (const kw of kh)
          for (const v of kw)
            w.push(v);
    w.push(...this.biases);
    return w;
  }

  setWeights(weights: number[]): void {
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
