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
//   The argmax of each window is stored per window in forward (not as a
//   per-position mask), so with overlapping windows (stride < poolSize) each
//   window's gradient goes only to its own argmax; a position that is the max
//   of several windows accumulates (+=) their gradients. Ties: first position
//   in scan order (strict `>`).
//
// ─────────────────────────────────────────────────────────────────────────────

export class MaxPool2D {
  readonly poolSize: number;
  readonly stride: number;

  // Argmax per window stored during forward pass for backprop:
  // _argmax[oh][ow][c] = [ih, iw] input position of the max of that window
  private _argmax: [number, number][][][] | null = null;
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
    // Validate before touching any state, so a failed call leaves the
    // previous forward/backward pair intact.
    const H = input.length;
    if (H < this.poolSize) {
      throw new Error(
        `MaxPool2D.forward: input height ${H} is smaller than poolSize ${this.poolSize}`,
      );
    }
    const W = input[0].length;
    if (W < this.poolSize) {
      throw new Error(
        `MaxPool2D.forward: input width ${W} is smaller than poolSize ${this.poolSize}`,
      );
    }
    const C = input[0][0].length;

    this._inputH = H;
    this._inputW = W;
    this._inputC = C;

    const [outH, outW] = this.outputShape(H, W, C);

    // Allocate output and max-position mask
    const output: number[][][] = Array.from({ length: outH }, () =>
      Array.from({ length: outW }, () => new Array(C).fill(-Infinity))
    );
    const argmax: [number, number][][][] = Array.from({ length: outH }, () =>
      Array.from({ length: outW }, () => new Array<[number, number]>(C))
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

          output[oh][ow][c] = maxVal;
          argmax[oh][ow][c] = [oh * this.stride + maxPH, ow * this.stride + maxPW];
        }
      }
    }
    this._argmax = argmax;

    return output;
  }

  // ── Backward pass ─────────────────────────────────────────────────────────
  // dOutput: number[][][] of shape [outH][outW][C]
  // Returns dInput: number[][][] of shape [H][W][C]
  // Gradient is routed only to the max position; all others get 0.
  backward(dOutput: number[][][]): number[][][] {
    if (!this._argmax) {
      throw new Error('MaxPool2D.backward: call forward() first');
    }

    // The gradient must have the output shape of the last forward pass.
    const [outH, outW, C] = this.outputShape(this._inputH, this._inputW, this._inputC);
    if (dOutput.length !== outH) {
      throw new Error(
        `MaxPool2D.backward: expected gradient with ${outH} rows, got ${dOutput.length}`,
      );
    }
    for (let oh = 0; oh < outH; oh++) {
      const row = dOutput[oh] as number[][] | undefined;
      if (row?.length !== outW) {
        throw new Error(
          `MaxPool2D.backward: row ${oh} expected ${outW} columns, got ${row?.length}`,
        );
      }
      for (let ow = 0; ow < outW; ow++) {
        const cell = row[ow] as number[] | undefined;
        if (cell?.length !== C) {
          throw new Error(
            `MaxPool2D.backward: [${oh}][${ow}] expected ${C} channels, got ${cell?.length}`,
          );
        }
      }
    }

    const dInput: number[][][] = Array.from({ length: this._inputH }, () =>
      Array.from({ length: this._inputW }, () => new Array(this._inputC).fill(0))
    );

    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        for (let c = 0; c < C; c++) {
          const [ih, iw] = this._argmax[oh][ow][c];
          dInput[ih][iw][c] += dOutput[oh][ow][c];
        }
      }
    }

    return dInput;
  }
}
