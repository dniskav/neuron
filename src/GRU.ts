// ─── GRU Layer ───────────────────────────────────────────────────────────────
//
// Gated Recurrent Unit — a simplified alternative to LSTM.
// Uses 3 gates instead of 4 (reset, update, new) with no separate cell state.
//
// Architecture:
//   r = sigmoid(Wr · [x, h_prev] + br)     reset gate
//   z = sigmoid(Wz · [x, h_prev] + bz)     update gate
//   n = tanh(Wn · [x, r⊙h_prev] + bn)     new gate (candidate)
//   h = (1 - z) ⊙ n + z ⊙ h_prev          hidden state
//
// Usage pattern (same as LSTM):
//   gru.reset()           ← call at episode start
//   gru.predict(x)        ← each step; returns h
//   gru.backprop(dh, lr)  ← BPTT at episode end
//
// ─────────────────────────────────────────────────────────────────────────────

import { OptimizerFactory, Optimizer, SGD } from "./optimizers";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function tanhFn(x: number): number {
  const e = Math.exp(2 * x);
  return (e - 1) / (e + 1);
}

class Gate {
  W: number[][];
  b: number[];

  constructor(inputSize: number, hSize: number, initBias = 0) {
    const n = inputSize + hSize;
    const limit = Math.sqrt(2 / (n + hSize));  // Xavier fan-in+out for sigmoid/tanh gates
    this.W = Array.from({ length: hSize }, () =>
      Array.from({ length: n }, () => (Math.random() * 2 - 1) * limit)
    );
    this.b = new Array(hSize).fill(initBias);
  }

  linear(combined: number[]): number[] {
    return this.W.map((row, i) =>
      row.reduce((s, w, j) => s + w * combined[j], this.b[i])
    );
  }
}

interface Step {
  combined: number[];
  h_prev: number[];
  r: number[];
  r_a: number[];
  z: number[];
  z_a: number[];
  combined_r: number[];
  n_pre: number[];
  n_a: number[];
  h: number[];
}

export class GRULayer {
  readonly inputSize: number;
  readonly hSize: number;

  h: number[];

  resetGate: Gate;
  updateGate: Gate;
  newGate: Gate;

  // Per-scalar optimizers — one per weight and bias
  private _optimizers: {
    resetW:  Optimizer[][];
    resetB:  Optimizer[];
    updateW: Optimizer[][];
    updateB: Optimizer[];
    newW:    Optimizer[][];
    newB:    Optimizer[];
  };

  private _traj: Step[] = [];

  constructor(inputSize: number, hiddenSize: number, optimizerFactory: OptimizerFactory = () => new SGD()) {
    if (inputSize <= 0 || hiddenSize <= 0) {
      throw new Error(`GRULayer: inputSize and hiddenSize must be positive, got ${inputSize} and ${hiddenSize}`);
    }
    this.inputSize = inputSize;
    this.hSize = hiddenSize;
    this.h = new Array(hiddenSize).fill(0);

    this.resetGate = new Gate(inputSize, hiddenSize);
    this.updateGate = new Gate(inputSize, hiddenSize);
    this.newGate = new Gate(inputSize, hiddenSize);

    // Initialize per-scalar optimizers
    const combSize = inputSize + hiddenSize;
    this._optimizers = {
      resetW:  Array.from({ length: hiddenSize }, () =>
        Array.from({ length: combSize }, () => optimizerFactory())
      ),
      resetB:  Array.from({ length: hiddenSize }, () => optimizerFactory()),
      updateW: Array.from({ length: hiddenSize }, () =>
        Array.from({ length: combSize }, () => optimizerFactory())
      ),
      updateB: Array.from({ length: hiddenSize }, () => optimizerFactory()),
      newW:    Array.from({ length: hiddenSize }, () =>
        Array.from({ length: combSize }, () => optimizerFactory())
      ),
      newB:    Array.from({ length: hiddenSize }, () => optimizerFactory()),
    };
  }

  reset(): void {
    this.h = new Array(this.hSize).fill(0);
    this._traj = [];
  }

  predict(inputs: number[]): number[] {
    if (!Array.isArray(inputs) || inputs.length !== this.inputSize) {
      throw new Error(`GRULayer.predict: expected array of length ${this.inputSize}, got ${inputs?.length}`);
    }

    const combined = [...inputs, ...this.h];
    const h_prev = [...this.h];

    const r_pre = this.resetGate.linear(combined);
    const z_pre = this.updateGate.linear(combined);
    const r_a = r_pre.map(sigmoid);
    const z_a = z_pre.map(sigmoid);

    // New gate uses r ⊙ h_prev
    const combined_r = [...inputs, ...r_a.map((r, i) => r * h_prev[i])];
    const n_pre = this.newGate.linear(combined_r);
    const n_a = n_pre.map(tanhFn);

    // h = (1 - z) ⊙ n + z ⊙ h_prev
    const h = n_a.map((n, i) => (1 - z_a[i]) * n + z_a[i] * h_prev[i]);

    this._traj.push({ combined, h_prev, r: r_pre, r_a, z: z_pre, z_a, combined_r, n_pre, n_a, h });
    this.h = h;
    return h;
  }

  backprop(dh_seq: number[][], lr: number): void {
    const T = this._traj.length;
    if (T === 0 || dh_seq.length !== T) return;

    const hSize = this.hSize;
    const combSize = this.inputSize + hSize;

    const dWr = Array.from({ length: hSize }, () => new Array(combSize).fill(0));
    const dWz = Array.from({ length: hSize }, () => new Array(combSize).fill(0));
    const dWn = Array.from({ length: hSize }, () => new Array(combSize).fill(0));
    const dbr = new Array(hSize).fill(0);
    const dbz = new Array(hSize).fill(0);
    const dbn = new Array(hSize).fill(0);

    let dh_next = new Array(hSize).fill(0);

    for (let t = T - 1; t >= 0; t--) {
      const s = this._traj[t];
      const dh = dh_seq[t].map((d, i) => d + dh_next[i]);

      // h = (1-z)*n + z*h_prev
      // dh/dz = h_prev * dh - n * dh
      // dh/dn = (1-z) * dh
      // dh/dh_prev = z * dh + ... (from recurrence)

      const dz_a = dh.map((d, i) => (s.h_prev[i] - s.n_a[i]) * d);
      const dn_a = dh.map((d, i) => (1 - s.z_a[i]) * d);

      // Backprop through tanh for n
      const dn_pre = dn_a.map((d, i) => d * (1 - s.n_a[i] ** 2));

      // Backprop through sigmoid for z
      const dz_pre = dz_a.map((d, i) => d * s.z_a[i] * (1 - s.z_a[i]));

      // For reset gate, we need gradient from newGate
      // n = tanh(Wn · [x, r⊙h_prev])
      // d(r⊙h_prev) = Wn^T · dn_pre (hSize portion)
      const dr_hprev = Array.from({ length: hSize }, (_, i) =>
        this.newGate.W.reduce((sum, row, k) => sum + dn_pre[k] * row[this.inputSize + i], 0)
      );
      const dr_a = dr_hprev.map((d, i) => d * s.h_prev[i]);

      // Backprop through sigmoid for r
      const dr_pre = dr_a.map((d, i) => d * s.r_a[i] * (1 - s.r_a[i]));

      // Accumulate weight gradients
      for (let k = 0; k < hSize; k++) {
        for (let j = 0; j < combSize; j++) {
          dWr[k][j] += dr_pre[k] * s.combined[j];
          dWz[k][j] += dz_pre[k] * s.combined[j];
          dWn[k][j] += dn_pre[k] * s.combined_r[j];
        }
        dbr[k] += dr_pre[k];
        dbz[k] += dz_pre[k];
        dbn[k] += dn_pre[k];
      }

      // Gradient flowing back to h_{t-1}
      dh_next = new Array(hSize).fill(0);
      for (let k = 0; k < hSize; k++) {
        for (let j = this.inputSize; j < combSize; j++) {
          dh_next[j - this.inputSize] +=
            dr_pre[k] * this.resetGate.W[k][j] +
            dz_pre[k] * this.updateGate.W[k][j];
        }
        // From new gate: through r⊙h_prev
        dh_next[k] += dr_hprev[k] * s.r_a[k];
        // From update gate: z * dh/dh_prev
        dh_next[k] += dh[k] * s.z_a[k];
      }
    }

    // Apply averaged gradient update via per-scalar optimizers
    const scale = lr / T;
    for (let k = 0; k < hSize; k++) {
      for (let j = 0; j < combSize; j++) {
        this.resetGate.W[k][j]  = this._optimizers.resetW[k][j].step(this.resetGate.W[k][j], dWr[k][j], scale);
        this.updateGate.W[k][j] = this._optimizers.updateW[k][j].step(this.updateGate.W[k][j], dWz[k][j], scale);
        this.newGate.W[k][j]    = this._optimizers.newW[k][j].step(this.newGate.W[k][j], dWn[k][j], scale);
      }
      this.resetGate.b[k]  = this._optimizers.resetB[k].step(this.resetGate.b[k], dbr[k], scale);
      this.updateGate.b[k] = this._optimizers.updateB[k].step(this.updateGate.b[k], dbz[k], scale);
      this.newGate.b[k]    = this._optimizers.newB[k].step(this.newGate.b[k], dbn[k], scale);
    }

    this._traj = [];
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: resetGate (W, b), updateGate (W, b), newGate (W, b).
  getWeightsFlat(): number[] {
    const w: number[] = [];
    for (const row of this.resetGate.W) w.push(...row);
    w.push(...this.resetGate.b);
    for (const row of this.updateGate.W) w.push(...row);
    w.push(...this.updateGate.b);
    for (const row of this.newGate.W) w.push(...row);
    w.push(...this.newGate.b);
    return w;
  }

  setWeightsFlat(weights: number[]): void {
    let idx = 0;
    for (let i = 0; i < this.resetGate.W.length; i++)
      for (let j = 0; j < this.resetGate.W[i].length; j++) this.resetGate.W[i][j] = weights[idx++];
    for (let i = 0; i < this.resetGate.b.length; i++) this.resetGate.b[i] = weights[idx++];
    for (let i = 0; i < this.updateGate.W.length; i++)
      for (let j = 0; j < this.updateGate.W[i].length; j++) this.updateGate.W[i][j] = weights[idx++];
    for (let i = 0; i < this.updateGate.b.length; i++) this.updateGate.b[i] = weights[idx++];
    for (let i = 0; i < this.newGate.W.length; i++)
      for (let j = 0; j < this.newGate.W[i].length; j++) this.newGate.W[i][j] = weights[idx++];
    for (let i = 0; i < this.newGate.b.length; i++) this.newGate.b[i] = weights[idx++];
  }

  getWeights() {
    return {
      resetGate: { W: this.resetGate.W, b: this.resetGate.b },
      updateGate: { W: this.updateGate.W, b: this.updateGate.b },
      newGate: { W: this.newGate.W, b: this.newGate.b },
    };
  }

  setWeights(data: ReturnType<GRULayer["getWeights"]>): void {
    this.resetGate.W = data.resetGate.W;
    this.resetGate.b = data.resetGate.b;
    this.updateGate.W = data.updateGate.W;
    this.updateGate.b = data.updateGate.b;
    this.newGate.W = data.newGate.W;
    this.newGate.b = data.newGate.b;
  }
}
