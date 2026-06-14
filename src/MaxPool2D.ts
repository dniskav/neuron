// ─── MaxPool2D Layer ──────────────────────────────────────────────────────────
//
// 2D Max Pooling — reduces spatial dimensions by keeping the maximum value
// within each poolSize × poolSize window. Applied independently per channel.
//
// Input / Output: number[][][] of shape [H][W][C]
//
// Output size (with stride = poolSize by default):
//   outH = floor((H - poolSize) / stride) + 1
//   outW = floor((W - poolSize) / stride) + 1
//
// Backward pass:
//   Gradient flows only to the position that held the maximum value in each
//   window. All other positions receive zero gradient.
//   This is equivalent to a binary mask: mask[h][w][c] = 1 iff that position
//   was the max for its window.
//
// ─────────────────────────────────────────────────────────────────────────────

export class MaxPool2D {
  readonly poolSize: number;
  readonly stride: number;

  // Mask stored during forward pass for backprop:
  // _maxMask[h][w][c] = true if input[h][w][c] was the maximum in its window
  private _maxMask: boolean[][][] | null = null;
  private _inputH: number = 0;
  private _inputW: number = 0;
  private _inputC: number = 0;

  constructor(poolSize: number, stride?: number) {
    if (poolSize <= 0) {
      throw new Error('MaxPool2D: poolSize must be positive');
    }
    this.poolSize = poolSize;
    this.stride   = stride ?? poolSize;  // non-overlapping by default
  }

  // ── Output shape ──────────────────────────────────────────────────────────
  outputShape(inputH: number, inputW: number, channels: number): [number, number, number] {
    const outH = Math.floor((inputH - this.poolSize) / this.stride) + 1;
    const outW = Math.floor((inputW - this.poolSize) / this.stride) + 1;
    return [outH, outW, channels];
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // output[oh][ow][c] = max over ph in [0..poolSize), pw in [0..poolSize) of
  //                     input[oh·stride + ph][ow·stride + pw][c]
  forward(input: number[][][]): number[][][] {
    const H = input.length;
    const W = input[0].length;
    const C = input[0][0].length;

    this._inputH = H;
    this._inputW = W;
    this._inputC = C;

    const [outH, outW] = this.outputShape(H, W, C);

    // Allocate output and max-position mask
    const output: number[][][] = Array.from({ length: outH }, () =>
      Array.from({ length: outW }, () => new Array(C).fill(-Infinity))
    );
    this._maxMask = Array.from({ length: H }, () =>
      Array.from({ length: W }, () => new Array(C).fill(false))
    );

    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        for (let c = 0; c < C; c++) {
          let maxVal = -Infinity;
          let maxPH  = 0;
          let maxPW  = 0;

          for (let ph = 0; ph < this.poolSize; ph++) {
            for (let pw = 0; pw < this.poolSize; pw++) {
              const val = input[oh * this.stride + ph][ow * this.stride + pw][c];
              if (val > maxVal) {
                maxVal = val;
                maxPH  = ph;
                maxPW  = pw;
              }
            }
          }

          output[oh][ow][c]                                                     = maxVal;
          this._maxMask[oh * this.stride + maxPH][ow * this.stride + maxPW][c] = true;
        }
      }
    }

    return output;
  }

  // ── Backward pass ─────────────────────────────────────────────────────────
  // dOutput: number[][][] of shape [outH][outW][C]
  // Returns dInput: number[][][] of shape [H][W][C]
  // Gradient is routed only to the max position; all others get 0.
  backward(dOutput: number[][][]): number[][][] {
    if (!this._maxMask) {
      throw new Error('MaxPool2D.backward: call forward() first');
    }

    const dInput: number[][][] = Array.from({ length: this._inputH }, () =>
      Array.from({ length: this._inputW }, () => new Array(this._inputC).fill(0))
    );

    const outH = dOutput.length;
    const outW = dOutput[0].length;
    const C    = this._inputC;

    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        for (let c = 0; c < C; c++) {
          for (let ph = 0; ph < this.poolSize; ph++) {
            for (let pw = 0; pw < this.poolSize; pw++) {
              const ih = oh * this.stride + ph;
              const iw = ow * this.stride + pw;
              if (this._maxMask[ih][iw][c]) {
                dInput[ih][iw][c] += dOutput[oh][ow][c];
              }
            }
          }
        }
      }
    }

    return dInput;
  }
}
