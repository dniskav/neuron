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
// ─────────────────────────────────────────────────────────────────────────────

import { OptimizerFactory, Optimizer, SGD } from "./optimizers";
import { relu as reluActivation } from "./activations";

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
export class CausalConv1D {
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
    if (inputChannels <= 0 || outputChannels <= 0 || kernelSize <= 0 || dilation <= 0) {
      throw new Error('CausalConv1D: all dimensions must be positive');
    }

    this.inputChannels  = inputChannels;
    this.outputChannels = outputChannels;
    this.kernelSize     = kernelSize;
    this.dilation       = dilation;

    // Xavier initialization: fan-in = kernelSize * inputChannels
    const limit = Math.sqrt(2 / (kernelSize * inputChannels));
    this.kernels = Array.from({ length: outputChannels }, () =>
      Array.from({ length: kernelSize }, () =>
        Array.from({ length: inputChannels }, () => (Math.random() * 2 - 1) * limit)
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
// Residual connections: when inputChannels == channels, the input of each block
// is added to the block output before the activation (ResNet-style). This
// further stabilizes gradients in deep stacks.
//
export class TCN {
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
    if (levels <= 0) throw new Error('TCN: levels must be positive');
    if (outputSize <= 0) throw new Error('TCN: outputSize must be positive');

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
      Array.from({ length: channels }, () => (Math.random() * 2 - 1) * outLimit)
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
    const outputs = this.forward(sequence);
    const T = outputs.length;

    // MSE loss and gradient
    let loss = 0;
    const dOut: number[][] = outputs.map((o, t) =>
      o.map((v, k) => {
        const diff = v - targets[t][k];
        loss += diff * diff;
        return 2 * diff / this.outputSize;
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
    const scale = lr / T;
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
