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
    const limit = Math.sqrt(2 / n);
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

  private _traj: Step[] = [];

  constructor(inputSize: number, hiddenSize: number) {
    this.inputSize = inputSize;
    this.hSize     = hiddenSize;
    this.h = new Array(hiddenSize).fill(0);
    this.c = new Array(hiddenSize).fill(0);

    this.forgetGate = new Gate(inputSize, hiddenSize, 1);  // bias=1: remember by default
    this.inputGate  = new Gate(inputSize, hiddenSize);
    this.cellGate   = new Gate(inputSize, hiddenSize);
    this.outputGate = new Gate(inputSize, hiddenSize);
  }

  // ── Reset state and trajectory (call at episode start) ────────────────────
  reset(): void {
    this.h = new Array(this.hSize).fill(0);
    this.c = new Array(this.hSize).fill(0);
    this._traj = [];
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  predict(inputs: number[]): number[] {
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

    // Apply averaged gradient update
    const scale = lr / T;
    for (let k = 0; k < hSize; k++) {
      for (let j = 0; j < combSize; j++) {
        this.forgetGate.W[k][j] += scale * dWf[k][j];
        this.inputGate.W[k][j]  += scale * dWi[k][j];
        this.cellGate.W[k][j]   += scale * dWg[k][j];
        this.outputGate.W[k][j] += scale * dWo[k][j];
      }
      this.forgetGate.b[k] += scale * dbf[k];
      this.inputGate.b[k]  += scale * dbi[k];
      this.cellGate.b[k]   += scale * dbg[k];
      this.outputGate.b[k] += scale * dbo[k];
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
}
