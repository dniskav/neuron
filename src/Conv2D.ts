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

import { OptimizerFactory, Optimizer, SGD } from "./optimizers";
import { validateArray } from "./Validation";

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
