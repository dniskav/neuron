// ─── Seq2Seq (Sequence-to-Sequence) ──────────────────────────────────────────
//
// Two-LSTM architecture for sequence transformation tasks such as translation,
// summarization, or any mapping from one variable-length sequence to another.
//
// Architecture:
//   ┌─────────────────────┐       ┌─────────────────────┐
//   │      ENCODER        │       │      DECODER        │
//   │  LSTMLayer          │──c,h──│  LSTMLayer          │──Linear──▶ output
//   └─────────────────────┘       └─────────────────────┘
//
// Encoder:
//   Processes the full input sequence step by step.
//   Its final (h, c) pair — the "context vector" — summarizes the input.
//
// Decoder:
//   Receives the context vector as its initial (h, c).
//   At each step it takes the previous output (or ground-truth token during
//   teacher forcing) as input and produces the next hidden state.
//   A linear projection W_out maps hidden → output at every step.
//
// Context vector:
//   { h: number[], c: number[] }  — both of shape [hiddenSize]
//
// Training (teacher forcing):
//   During trainStep the decoder is fed the ground-truth token at each step
//   rather than its own previous prediction. This stabilizes early training.
//
// Loss: Mean Squared Error across all decoder timesteps.
//   L = (1 / (T·outputSize)) · Σ_{t,k} (o_{t,k} - y_{t,k})²
//
// Linear output projection (no bias for simplicity):
//   o_t = W_out · h_t,    W_out ∈ R^{outputSize × hiddenSize}
//   dL/dW_out = Σ_t (dL/do_t) ⊗ h_t
//   dL/dh_t   = W_out^T · (dL/do_t)
//
// ─────────────────────────────────────────────────────────────────────────────

import { LSTMLayer } from "./LSTMLayer";
import { OptimizerFactory, Optimizer, SGD } from "./optimizers";

export class Seq2Seq {
  readonly encoder: LSTMLayer;
  readonly decoder: LSTMLayer;

  // Linear output projection: hidden → output space
  // W_out[i][j]: weight from hidden[j] to output[i]
  W_out: number[][];   // [outputSize][hiddenSize]
  b_out: number[];     // [outputSize]

  readonly inputSize:  number;
  readonly hiddenSize: number;
  readonly outputSize: number;

  // Per-scalar optimizers for the output projection
  private _wOutOpts: Optimizer[][];  // [outputSize][hiddenSize]
  private _bOutOpts: Optimizer[];    // [outputSize]

  constructor(
    inputSize:  number,
    hiddenSize: number,
    outputSize: number,
    options?: {
      optimizerFactory?: OptimizerFactory;
    }
  ) {
    if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0) {
      throw new Error('Seq2Seq: all sizes must be positive');
    }

    this.inputSize  = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    const factory: OptimizerFactory = options?.optimizerFactory ?? (() => new SGD());

    this.encoder = new LSTMLayer(inputSize,  hiddenSize, factory);

    // Decoder input at each step is the previous output (outputSize dims)
    this.decoder = new LSTMLayer(outputSize, hiddenSize, factory);

    // Xavier initialization for linear projection
    const limit = Math.sqrt(2 / hiddenSize);
    this.W_out = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * limit)
    );
    this.b_out = new Array(outputSize).fill(0);

    this._wOutOpts = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => factory())
    );
    this._bOutOpts = Array.from({ length: outputSize }, () => factory());
  }

  // ── Linear projection ─────────────────────────────────────────────────────
  private _project(h: number[]): number[] {
    return this.b_out.map((b, i) =>
      b + this.W_out[i].reduce((s, w, j) => s + w * h[j], 0)
    );
  }

  // ── Encode ────────────────────────────────────────────────────────────────
  // Runs the encoder over inputSequence and returns the final (h, c) pair.
  // The context vector summarizes the full input sequence.
  encode(inputSequence: number[][]): { h: number[], c: number[] } {
    this.encoder.reset();
    for (const x of inputSequence) {
      this.encoder.predict(x);
    }
    return {
      h: [...this.encoder.h],
      c: [...this.encoder.c],
    };
  }

  // ── Decode ────────────────────────────────────────────────────────────────
  // Generates `steps` output tokens autoregressively.
  // The decoder starts from contextVector and uses its own previous output
  // as input at each step (greedy / free-running decoding).
  decode(contextVector: { h: number[], c: number[] }, steps: number): number[][] {
    this.decoder.reset();
    this.decoder.h = [...contextVector.h];
    this.decoder.c = [...contextVector.c];

    const results: number[][] = [];
    let prevOutput: number[] = new Array(this.outputSize).fill(0);

    for (let t = 0; t < steps; t++) {
      const hidden = this.decoder.predict(prevOutput);
      const output = this._project(hidden);
      results.push(output);
      prevOutput = output;
    }

    return results;
  }

  // ── Training step (teacher forcing) ──────────────────────────────────────
  // inputSeq:  number[][] of shape [T_in][inputSize]
  // targetSeq: number[][] of shape [T_out][outputSize]
  // Returns the MSE loss for this step.
  trainStep(inputSeq: number[][], targetSeq: number[][], lr: number): number {
    const T = targetSeq.length;
    if (T === 0) return 0;

    // ── Encoder forward ───────────────────────────────────────────────────
    this.encoder.reset();
    for (const x of inputSeq) {
      this.encoder.predict(x);
    }
    const contextH = [...this.encoder.h];
    const contextC = [...this.encoder.c];

    // ── Decoder forward with teacher forcing ─────────────────────────────
    this.decoder.reset();
    this.decoder.h = [...contextH];
    this.decoder.c = [...contextC];

    const hiddens:  number[][] = [];  // decoder hidden states
    const projOuts: number[][] = [];  // projected outputs

    // At t=0 use zero start token; for t>0 use ground-truth from previous step
    let prevTeacher: number[] = new Array(this.outputSize).fill(0);
    for (let t = 0; t < T; t++) {
      const h   = this.decoder.predict(prevTeacher);
      const out = this._project(h);
      hiddens.push(h);
      projOuts.push(out);
      prevTeacher = targetSeq[t];  // teacher forcing: feed ground truth
    }

    // ── MSE loss and dL/dOutput ───────────────────────────────────────────
    let loss = 0;
    const dProjOut: number[][] = projOuts.map((o, t) =>
      o.map((v, k) => {
        const diff = v - targetSeq[t][k];
        loss += diff * diff;
        return 2 * diff / this.outputSize;
      })
    );
    loss /= (T * this.outputSize);

    // ── Backward through linear projection ───────────────────────────────
    // dL/dW_out += (1/T) · Σ_t dProjOut[t] ⊗ hiddens[t]
    // dL/dh_t   = W_out^T · dProjOut[t]
    const dhSeq: number[][] = Array.from({ length: T }, () =>
      new Array(this.hiddenSize).fill(0)
    );

    const dWout = Array.from({ length: this.outputSize }, () =>
      new Array(this.hiddenSize).fill(0)
    );
    const dbOut = new Array(this.outputSize).fill(0);

    for (let t = 0; t < T; t++) {
      for (let i = 0; i < this.outputSize; i++) {
        const dv = dProjOut[t][i];
        dbOut[i] += dv;
        for (let j = 0; j < this.hiddenSize; j++) {
          dWout[i][j]  += dv * hiddens[t][j];
          dhSeq[t][j]  += dv * this.W_out[i][j];
        }
      }
    }

    // Apply output projection updates
    const scale = lr / T;
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.W_out[i][j] = this._wOutOpts[i][j].step(this.W_out[i][j], dWout[i][j], scale);
      }
      this.b_out[i] = this._bOutOpts[i].step(this.b_out[i], dbOut[i], scale);
    }

    // ── Decoder BPTT ─────────────────────────────────────────────────────
    this.decoder.backprop(dhSeq, lr);

    // ── Encoder BPTT: propagate context gradient ──────────────────────────
    // Route gradient only to the last encoder step (the context producer).
    // A full attention-based gradient would require storing all encoder states.
    const dContext = dhSeq[0];
    const encoderDhSeq: number[][] = inputSeq.map((_, t) =>
      t === inputSeq.length - 1 ? [...dContext] : new Array(this.hiddenSize).fill(0)
    );
    this.encoder.backprop(encoderDhSeq, lr);

    return loss;
  }
}
