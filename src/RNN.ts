// ─── Vanilla RNN ──────────────────────────────────────────────────────────────
//
// The simplest recurrent neural network. Introduced here as a conceptual
// baseline before LSTM/GRU — it illustrates why gating mechanisms are needed.
//
// At each time step t:
//   h_t = tanh( W_xh · x_t  +  W_hh · h_{t-1}  +  b_h )   hidden state
//   o_t = W_hy · h_t  +  b_y                                 output logits
//
// Training via BPTT (Backpropagation Through Time):
//   The loss gradient is propagated backwards through all T steps.
//   At each step, dL/dh_t is multiplied by W_hh^T, so after T steps
//   the gradient contains W_hh^T multiplied T times.
//
// ── Vanishing / Exploding Gradient Problem ────────────────────────────────────
//
//   If the largest singular value of W_hh is < 1, gradients shrink
//   exponentially → the network cannot learn long-range dependencies.
//   If > 1, gradients explode → training is unstable.
//
//   LSTM solves this by introducing a cell state c with additive updates
//   (c_t = f⊙c_{t-1} + i⊙g), creating a "gradient highway" where the
//   forget gate keeps the derivative close to 1, so gradients can flow
//   hundreds of steps without vanishing.
//
// Loss function: Mean Squared Error
//   L = (1/T) · Σ_t ||o_t - y_t||²
//
// Xavier initialization:  limit = sqrt(2 / fan_in)
//
// ─────────────────────────────────────────────────────────────────────────────

import { OptimizerFactory, Optimizer, SGD } from "./optimizers";

function tanh(x: number): number {
  const e = Math.exp(2 * x);
  return (e - 1) / (e + 1);
}

// ─── Step record (stored for BPTT) ───────────────────────────────────────────

interface RNNStep {
  x:    number[];  // input at this step
  h:    number[];  // hidden state produced (post-tanh)
  hRaw: number[];  // pre-tanh hidden state
  hPrev: number[]; // hidden state from the previous step
}

// ─── RNN ──────────────────────────────────────────────────────────────────────

export class RNN {
  readonly inputSize:  number;
  readonly hiddenSize: number;
  readonly outputSize: number;

  Wxh: number[][];  // [hiddenSize][inputSize]   — input  → hidden
  Whh: number[][];  // [hiddenSize][hiddenSize]  — hidden → hidden
  Why: number[][];  // [outputSize][hiddenSize]  — hidden → output
  bh:  number[];    // [hiddenSize]
  by:  number[];    // [outputSize]

  // Current hidden state (persists across calls to forward; reset() clears it)
  private _h: number[];

  // Trajectory stored during forward for BPTT
  private _traj: RNNStep[] = [];
  private _outputs: number[][] = [];

  // Per-scalar optimizers
  private _opts: {
    Wxh: Optimizer[][];
    Whh: Optimizer[][];
    Why: Optimizer[][];
    bh:  Optimizer[];
    by:  Optimizer[];
  };

  constructor(
    inputSize:  number,
    hiddenSize: number,
    outputSize: number,
    optimizerFactory: OptimizerFactory = () => new SGD()
  ) {
    if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0) {
      throw new Error('RNN: all sizes must be positive');
    }
    this.inputSize  = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Xavier initialization
    const limXH = Math.sqrt(2 / inputSize);
    const limHH = Math.sqrt(2 / hiddenSize);
    const limHY = Math.sqrt(2 / hiddenSize);

    this.Wxh = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize  }, () => (Math.random() * 2 - 1) * limXH)
    );
    this.Whh = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * limHH)
    );
    this.Why = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => (Math.random() * 2 - 1) * limHY)
    );
    this.bh = new Array(hiddenSize).fill(0);
    this.by = new Array(outputSize).fill(0);

    this._h = new Array(hiddenSize).fill(0);

    // Initialize per-scalar optimizers
    this._opts = {
      Wxh: Array.from({ length: hiddenSize }, () =>
        Array.from({ length: inputSize  }, () => optimizerFactory())
      ),
      Whh: Array.from({ length: hiddenSize }, () =>
        Array.from({ length: hiddenSize }, () => optimizerFactory())
      ),
      Why: Array.from({ length: outputSize }, () =>
        Array.from({ length: hiddenSize }, () => optimizerFactory())
      ),
      bh: Array.from({ length: hiddenSize }, () => optimizerFactory()),
      by: Array.from({ length: outputSize }, () => optimizerFactory()),
    };
  }

  // ── Reset hidden state ────────────────────────────────────────────────────
  // Call at the start of each new sequence / episode.
  reset(): void {
    this._h     = new Array(this.hiddenSize).fill(0);
    this._traj  = [];
    this._outputs = [];
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // sequence: number[][] of shape [T][inputSize]
  // Returns outputs and hidden states for all timesteps.
  forward(sequence: number[][]): { outputs: number[][], hiddens: number[][] } {
    this._traj    = [];
    this._outputs = [];

    const outputs: number[][] = [];
    const hiddens: number[][] = [];

    let hPrev = [...this._h];

    for (const x of sequence) {
      // h_t = tanh( W_xh · x_t + W_hh · h_{t-1} + b_h )
      const hRaw = this.bh.map((b, i) =>
        b
        + this.Wxh[i].reduce((s, w, j) => s + w * x[j], 0)
        + this.Whh[i].reduce((s, w, j) => s + w * hPrev[j], 0)
      );
      const h = hRaw.map(tanh);

      // o_t = W_hy · h_t + b_y
      const o = this.by.map((b, i) =>
        b + this.Why[i].reduce((s, w, j) => s + w * h[j], 0)
      );

      this._traj.push({ x: [...x], h: [...h], hRaw: [...hRaw], hPrev: [...hPrev] });
      this._outputs.push(o);
      outputs.push(o);
      hiddens.push(h);

      hPrev = h;
    }

    this._h = hPrev;
    return { outputs, hiddens };
  }

  // ── BPTT + weight update ──────────────────────────────────────────────────
  // targets: number[][] of shape [T][outputSize], paired with the last forward call.
  // Returns the mean squared error loss.
  backward(sequence: number[][], targets: number[][], lr: number): number {
    // Run forward to populate trajectory (discards previous state so sequence is fresh)
    this.reset();
    const { outputs } = this.forward(sequence);

    const T = this._traj.length;
    if (T === 0) return 0;

    // ── Compute MSE loss ──────────────────────────────────────────────────
    let loss = 0;
    const dOutputs: number[][] = outputs.map((o, t) => {
      return o.map((v, k) => {
        const diff = v - targets[t][k];
        loss += diff * diff;
        return 2 * diff / this.outputSize;  // dL/do_t
      });
    });
    loss /= (T * this.outputSize);

    // ── Gradient accumulators ─────────────────────────────────────────────
    const dWxh = Array.from({ length: this.hiddenSize }, () => new Array(this.inputSize).fill(0));
    const dWhh = Array.from({ length: this.hiddenSize }, () => new Array(this.hiddenSize).fill(0));
    const dWhy = Array.from({ length: this.outputSize }, () => new Array(this.hiddenSize).fill(0));
    const dbh  = new Array(this.hiddenSize).fill(0);
    const dby  = new Array(this.outputSize).fill(0);

    let dhNext: number[] = new Array(this.hiddenSize).fill(0);

    for (let t = T - 1; t >= 0; t--) {
      const s  = this._traj[t];
      const do_ = dOutputs[t];

      // dL/dW_hy and dL/db_y
      for (let i = 0; i < this.outputSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dWhy[i][j] += do_[i] * s.h[j];
        }
        dby[i] += do_[i];
      }

      // dL/dh_t: from output layer + from next timestep (BPTT)
      const dh: number[] = this.hiddenSize > 0
        ? Array.from({ length: this.hiddenSize }, (_, j) =>
            this.Why.reduce((sum, row, i) => sum + row[j] * do_[i], 0) + dhNext[j]
          )
        : [];

      // Backprop through tanh:  d(tanh) = 1 - tanh²
      // NOTE: This is where vanishing gradient occurs over many steps —
      // (1 - tanh²) is at most 1 and approaches 0 when |hRaw| is large.
      const dhRaw: number[] = dh.map((d, k) => d * (1 - s.h[k] ** 2));

      // dL/dW_xh, dL/db_h
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.inputSize; j++) {
          dWxh[i][j] += dhRaw[i] * s.x[j];
        }
        dbh[i] += dhRaw[i];
      }

      // dL/dW_hh
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dWhh[i][j] += dhRaw[i] * s.hPrev[j];
        }
      }

      // Gradient flowing back to h_{t-1} via W_hh
      // ⚠ Vanishing gradient: this multiplication by W_hh^T is applied T times,
      //   causing gradients to shrink or explode exponentially with sequence length.
      dhNext = Array.from({ length: this.hiddenSize }, (_, j) =>
        this.Whh.reduce((sum, row, i) => sum + row[j] * dhRaw[i], 0)
      );
    }

    // ── Apply optimizer updates ───────────────────────────────────────────
    const scale = lr / T;
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.Wxh[i][j] = this._opts.Wxh[i][j].step(this.Wxh[i][j], dWxh[i][j], scale);
      }
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Whh[i][j] = this._opts.Whh[i][j].step(this.Whh[i][j], dWhh[i][j], scale);
      }
      this.bh[i] = this._opts.bh[i].step(this.bh[i], dbh[i], scale);
    }
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Why[i][j] = this._opts.Why[i][j].step(this.Why[i][j], dWhy[i][j], scale);
      }
      this.by[i] = this._opts.by[i].step(this.by[i], dby[i], scale);
    }

    this._traj = [];
    this._outputs = [];
    return loss;
  }
}
