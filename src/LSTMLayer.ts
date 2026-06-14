// ─── LSTM LAYER ────────────────────────────────────────────────────────────────
//
// Recurrent layer that maintains two state vectors across calls:
//   h  — hidden state (output of this layer at the previous step)
//   c  — cell state   (long-term memory)
//
// At each step the combined input is [external_inputs, h_prev].
// Four gates control information flow:
//
//   f = sigmoid( Wf · [x, h] + bf )   forget gate  — what to erase from c
//   i = sigmoid( Wi · [x, h] + bi )   input  gate  — what new info to write
//   g = tanh   ( Wg · [x, h] + bg )   cell   gate  — candidate values to write
//   o = sigmoid( Wo · [x, h] + bo )   output gate  — what part of c to expose
//
//   c  =  f ⊙ c_prev  +  i ⊙ g        new cell state
//   h  =  o ⊙ tanh(c)                  new hidden state  (the layer output)
//
// Usage pattern:
//   layer.reset()          ← call at the start of every episode
//   layer.predict(x)       ← call every step; returns h (size = hiddenSize)
//   layer.backprop(dh, lr) ← BPTT at episode end; clears trajectory
//
// ─────────────────────────────────────────────────────────────────────────────

import { OptimizerFactory, Optimizer, SGD } from "./optimizers";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function tanh(x: number): number {
  const e = Math.exp(2 * x);
  return (e - 1) / (e + 1);
}

// ─── Gate ──────────────────────────────────────────────────────────────────────
// Linear transform: z = W · combined + b
// Activation is applied externally so we can use sigmoid or tanh per gate.

class Gate {
  W: number[][];  // shape: [hSize][inputSize + hSize]
  b: number[];    // shape: [hSize]

  constructor(inputSize: number, hSize: number, initBias = 0) {
    const n     = inputSize + hSize;
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

// ─── Step record (stored for BPTT) ────────────────────────────────────────────

interface Step {
  combined: number[];   // [x, h_prev]  — input to all gates
  c_prev:   number[];
  zf: number[];  zf_a: number[];  // forget  gate: pre / post activation
  zi: number[];  zi_a: number[];  // input   gate
  zg: number[];  zg_a: number[];  // cell    gate
  zo: number[];  zo_a: number[];  // output  gate
  c:  number[];                   // new cell state
  h:  number[];                   // new hidden state
}

// ─── LSTMLayer ─────────────────────────────────────────────────────────────────

export class LSTMLayer {
  readonly inputSize: number;
  readonly hSize:     number;

  h: number[];
  c: number[];

  forgetGate: Gate;
  inputGate:  Gate;
  cellGate:   Gate;
  outputGate: Gate;

  // Per-scalar optimizers — one per weight and bias
  private _optimizers: {
    forgetW: Optimizer[][];
    forgetB: Optimizer[];
    inputW:  Optimizer[][];
    inputB:  Optimizer[];
    cellW:   Optimizer[][];
    cellB:   Optimizer[];
    outputW: Optimizer[][];
    outputB: Optimizer[];
  };

  private _traj: Step[] = [];

  constructor(inputSize: number, hiddenSize: number, optimizerFactory: OptimizerFactory = () => new SGD()) {
    if (inputSize <= 0 || hiddenSize <= 0) {
      throw new Error(`LSTMLayer: inputSize and hiddenSize must be positive, got ${inputSize} and ${hiddenSize}`);
    }
    this.inputSize = inputSize;
    this.hSize     = hiddenSize;
    this.h = new Array(hiddenSize).fill(0);
    this.c = new Array(hiddenSize).fill(0);

    this.forgetGate = new Gate(inputSize, hiddenSize, 1);  // bias=1: remember by default
    this.inputGate  = new Gate(inputSize, hiddenSize);
    this.cellGate   = new Gate(inputSize, hiddenSize);
    this.outputGate = new Gate(inputSize, hiddenSize);

    // Initialize per-scalar optimizers
    const combSize = inputSize + hiddenSize;
    this._optimizers = {
      forgetW: Array.from({ length: hiddenSize }, () =>
        Array.from({ length: combSize }, () => optimizerFactory())
      ),
      forgetB: Array.from({ length: hiddenSize }, () => optimizerFactory()),
      inputW:  Array.from({ length: hiddenSize }, () =>
        Array.from({ length: combSize }, () => optimizerFactory())
      ),
      inputB:  Array.from({ length: hiddenSize }, () => optimizerFactory()),
      cellW:   Array.from({ length: hiddenSize }, () =>
        Array.from({ length: combSize }, () => optimizerFactory())
      ),
      cellB:   Array.from({ length: hiddenSize }, () => optimizerFactory()),
      outputW: Array.from({ length: hiddenSize }, () =>
        Array.from({ length: combSize }, () => optimizerFactory())
      ),
      outputB: Array.from({ length: hiddenSize }, () => optimizerFactory()),
    };
  }

  // ── Reset state and trajectory (call at episode start) ────────────────────
  reset(): void {
    this.h = new Array(this.hSize).fill(0);
    this.c = new Array(this.hSize).fill(0);
    this._traj = [];
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  predict(inputs: number[]): number[] {
    if (!Array.isArray(inputs) || inputs.length !== this.inputSize) {
      throw new Error(`LSTMLayer.predict: expected array of length ${this.inputSize}, got ${inputs?.length}`);
    }
    const combined = [...inputs, ...this.h];
    const c_prev   = [...this.c];

    const zf = this.forgetGate.linear(combined);
    const zi = this.inputGate.linear(combined);
    const zg = this.cellGate.linear(combined);
    const zo = this.outputGate.linear(combined);

    const zf_a = zf.map(sigmoid);
    const zi_a = zi.map(sigmoid);
    const zg_a = zg.map(tanh);
    const zo_a = zo.map(sigmoid);

    const c = c_prev.map((cv, k) => zf_a[k] * cv + zi_a[k] * zg_a[k]);
    const h = zo_a.map((o, k) => o * tanh(c[k]));

    this._traj.push({ combined, c_prev, zf, zf_a, zi, zi_a, zg, zg_a, zo, zo_a, c, h });

    this.h = h;
    this.c = c;
    return h;
  }

  // ── BPTT (Backpropagation Through Time) ────────────────────────────────────
  // dh_seq: dL/dh for each timestep, same length as trajectory.
  // Accumulates gradients across the full sequence, then applies them in one
  // update (batch gradient) scaled by lr / T.
  backprop(dh_seq: number[][], lr: number): void {
    const T = this._traj.length;
    if (T === 0 || dh_seq.length !== T) return;

    const hSize    = this.hSize;
    const combSize = this.inputSize + hSize;

    // Gradient accumulators
    const dWf = Array.from({ length: hSize }, () => new Array(combSize).fill(0));
    const dWi = Array.from({ length: hSize }, () => new Array(combSize).fill(0));
    const dWg = Array.from({ length: hSize }, () => new Array(combSize).fill(0));
    const dWo = Array.from({ length: hSize }, () => new Array(combSize).fill(0));
    const dbf  = new Array(hSize).fill(0);
    const dbi  = new Array(hSize).fill(0);
    const dbg  = new Array(hSize).fill(0);
    const dbo  = new Array(hSize).fill(0);

    let dh_next: number[] = new Array(hSize).fill(0);
    let dc_next: number[] = new Array(hSize).fill(0);

    for (let t = T - 1; t >= 0; t--) {
      const s  = this._traj[t];
      // Total gradient at h_t: from loss + from h_{t+1} via recurrence
      const dh = dh_seq[t].map((d, k) => d + dh_next[k]);

      // h = o * tanh(c)  →  dL/do and dL/dc
      const tanh_c = s.c.map(tanh);
      const do_a   = dh.map((d, k) => d * tanh_c[k]);
      const dc     = dh.map((d, k) =>
        d * s.zo_a[k] * (1 - tanh_c[k] ** 2) + dc_next[k]
      );

      // c = f*c_prev + i*g  →  dL/df, dL/di, dL/dg
      const df_a = dc.map((d, k) => d * s.c_prev[k]);
      const di_a = dc.map((d, k) => d * s.zg_a[k]);
      const dg_a = dc.map((d, k) => d * s.zi_a[k]);

      // Backprop through activations
      const dzo = do_a.map((d, k) => d * s.zo_a[k] * (1 - s.zo_a[k]));   // sigmoid'
      const dzf = df_a.map((d, k) => d * s.zf_a[k] * (1 - s.zf_a[k]));
      const dzi = di_a.map((d, k) => d * s.zi_a[k] * (1 - s.zi_a[k]));
      const dzg = dg_a.map((d, k) => d * (1 - s.zg_a[k] ** 2));           // tanh'

      // Accumulate weight gradients
      for (let k = 0; k < hSize; k++) {
        for (let j = 0; j < combSize; j++) {
          dWf[k][j] += dzf[k] * s.combined[j];
          dWi[k][j] += dzi[k] * s.combined[j];
          dWg[k][j] += dzg[k] * s.combined[j];
          dWo[k][j] += dzo[k] * s.combined[j];
        }
        dbf[k] += dzf[k];
        dbi[k] += dzi[k];
        dbg[k] += dzg[k];
        dbo[k] += dzo[k];
      }

      // Gradient flowing back to h_{t-1} (through the recurrent connection)
      dh_next = new Array(hSize).fill(0);
      for (let k = 0; k < hSize; k++) {
        for (let j = this.inputSize; j < combSize; j++) {
          dh_next[j - this.inputSize] +=
            dzf[k] * this.forgetGate.W[k][j] +
            dzi[k] * this.inputGate.W[k][j]  +
            dzg[k] * this.cellGate.W[k][j]   +
            dzo[k] * this.outputGate.W[k][j];
        }
      }

      // Gradient flowing back to c_{t-1}
      dc_next = dc.map((d, k) => d * s.zf_a[k]);
    }

    // Apply averaged gradient update via per-scalar optimizers
    const scale = lr / T;
    for (let k = 0; k < hSize; k++) {
      for (let j = 0; j < combSize; j++) {
        this.forgetGate.W[k][j] = this._optimizers.forgetW[k][j].step(this.forgetGate.W[k][j], dWf[k][j], scale);
        this.inputGate.W[k][j]  = this._optimizers.inputW[k][j].step(this.inputGate.W[k][j], dWi[k][j], scale);
        this.cellGate.W[k][j]   = this._optimizers.cellW[k][j].step(this.cellGate.W[k][j], dWg[k][j], scale);
        this.outputGate.W[k][j] = this._optimizers.outputW[k][j].step(this.outputGate.W[k][j], dWo[k][j], scale);
      }
      this.forgetGate.b[k] = this._optimizers.forgetB[k].step(this.forgetGate.b[k], dbf[k], scale);
      this.inputGate.b[k]  = this._optimizers.inputB[k].step(this.inputGate.b[k], dbi[k], scale);
      this.cellGate.b[k]   = this._optimizers.cellB[k].step(this.cellGate.b[k], dbg[k], scale);
      this.outputGate.b[k] = this._optimizers.outputB[k].step(this.outputGate.b[k], dbo[k], scale);
    }

    this._traj = [];
  }

  // ── Serialization ─────────────────────────────────────────────────────────
  getWeights() {
    return {
      forgetGate: { W: this.forgetGate.W, b: this.forgetGate.b },
      inputGate:  { W: this.inputGate.W,  b: this.inputGate.b  },
      cellGate:   { W: this.cellGate.W,   b: this.cellGate.b   },
      outputGate: { W: this.outputGate.W, b: this.outputGate.b },
    };
  }

  setWeights(data: ReturnType<LSTMLayer["getWeights"]>): void {
    this.forgetGate.W = data.forgetGate.W;
    this.forgetGate.b = data.forgetGate.b;
    this.inputGate.W  = data.inputGate.W;
    this.inputGate.b  = data.inputGate.b;
    this.cellGate.W   = data.cellGate.W;
    this.cellGate.b   = data.cellGate.b;
    this.outputGate.W = data.outputGate.W;
    this.outputGate.b = data.outputGate.b;
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: forgetGate (W, b), inputGate (W, b), cellGate (W, b), outputGate (W, b).
  getWeightsFlat(): number[] {
    const w: number[] = [];
    for (const row of this.forgetGate.W) w.push(...row);
    w.push(...this.forgetGate.b);
    for (const row of this.inputGate.W) w.push(...row);
    w.push(...this.inputGate.b);
    for (const row of this.cellGate.W) w.push(...row);
    w.push(...this.cellGate.b);
    for (const row of this.outputGate.W) w.push(...row);
    w.push(...this.outputGate.b);
    return w;
  }

  setWeightsFlat(weights: number[]): void {
    let idx = 0;
    for (let i = 0; i < this.forgetGate.W.length; i++)
      for (let j = 0; j < this.forgetGate.W[i].length; j++) this.forgetGate.W[i][j] = weights[idx++];
    for (let i = 0; i < this.forgetGate.b.length; i++) this.forgetGate.b[i] = weights[idx++];
    for (let i = 0; i < this.inputGate.W.length; i++)
      for (let j = 0; j < this.inputGate.W[i].length; j++) this.inputGate.W[i][j] = weights[idx++];
    for (let i = 0; i < this.inputGate.b.length; i++) this.inputGate.b[i] = weights[idx++];
    for (let i = 0; i < this.cellGate.W.length; i++)
      for (let j = 0; j < this.cellGate.W[i].length; j++) this.cellGate.W[i][j] = weights[idx++];
    for (let i = 0; i < this.cellGate.b.length; i++) this.cellGate.b[i] = weights[idx++];
    for (let i = 0; i < this.outputGate.W.length; i++)
      for (let j = 0; j < this.outputGate.W[i].length; j++) this.outputGate.W[i][j] = weights[idx++];
    for (let i = 0; i < this.outputGate.b.length; i++) this.outputGate.b[i] = weights[idx++];
  }
}
