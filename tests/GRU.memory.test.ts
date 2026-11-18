import { describe, it, expect } from 'vitest'
import { GRULayer } from '../src/GRU'
import { Adam, Momentum, SGD, type Optimizer, type OptimizerFactory } from '../src/optimizers'
import { validate2DArray, validateArray } from '../src/Validation'
import { random, seededRandom } from '../src/random'
import { withSeed } from './helpers/seed'

// GRULayer as it was in 0.4 (one scalar optimizer per weight, `_optimizers` as
// nested arrays of `Optimizer`), pasted here as the reference. The only edits
// are mechanical: no `export`, the class renamed to GRULayer04, and the
// imports above instead of the module's own.

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


function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function tanhFn(x: number): number {
  // Math.tanh is stable for large |x|; (e^{2x}-1)/(e^{2x}+1) gives NaN for x >~ 355.
  return Math.tanh(x);
}

class Gate {
  W: number[][];
  b: number[];

  constructor(inputSize: number, hSize: number, initBias = 0) {
    const n = inputSize + hSize;
    const limit = Math.sqrt(2 / (n + hSize));  // Xavier fan-in+out for sigmoid/tanh gates
    this.W = Array.from({ length: hSize }, () =>
      Array.from({ length: n }, () => (random() * 2 - 1) * limit)
    );
    this.b = new Array(hSize).fill(initBias);
  }

  linear(combined: number[]): number[] {
    return this.W.map((row, i) =>
      row.reduce((s, w, j) => s + w * combined[j], this.b[i])
    );
  }
}

// Copia profunda de una puerta (matrices y sesgos nuevos).
function copyGate(g: Gate): { W: number[][]; b: number[] } {
  return { W: g.W.map(row => [...row]), b: [...g.b] };
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

class GRULayer04 {
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

  // Throws (and leaves the trajectory intact) when dh_seq does not match the
  // trajectory or a row does not measure hiddenSize. An empty trajectory with an
  // empty dh_seq is a no-op, as before.
  backprop(dh_seq: number[][], lr: number): void {
    // lr must be finite and >= 0 (0 is a valid no-op); checked before anything is touched.
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`GRULayer.backprop: lr must be a finite number >= 0, got ${lr}`);
    }
    const T = this._traj.length;
    if (!Array.isArray(dh_seq)) {
      throw new Error(`GRULayer.backprop: expected dh_seq array, got ${typeof dh_seq}`);
    }
    if (dh_seq.length !== T) {
      throw new Error(`GRULayer.backprop: expected dh_seq of length ${T}, got ${dh_seq.length}`);
    }
    for (let t = 0; t < T; t++) {
      validateArray(dh_seq[t], this.hSize, `GRULayer.backprop(dh_seq[${t}])`);
    }
    if (T === 0) return;

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
  // Se añade elemento a elemento: pasar una fila grande como argumentos
  // supera el límite de la VM (RangeError).
  getWeightsFlat(): number[] {
    const w: number[] = [];
    for (const row of this.resetGate.W) for (const v of row) w.push(v);
    for (const v of this.resetGate.b) w.push(v);
    for (const row of this.updateGate.W) for (const v of row) w.push(v);
    for (const v of this.updateGate.b) w.push(v);
    for (const row of this.newGate.W) for (const v of row) w.push(v);
    for (const v of this.newGate.b) w.push(v);
    return w;
  }

  setWeightsFlat(weights: number[]): void {
    const total = 3 * (this.hSize * (this.inputSize + this.hSize) + this.hSize);
    validateArray(weights, total, "GRULayer.setWeightsFlat");
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

  // Returns DEEP COPIES: mutating the result never moves the layer's weights.
  getWeights() {
    return {
      resetGate: copyGate(this.resetGate),
      updateGate: copyGate(this.updateGate),
      newGate: copyGate(this.newGate),
    };
  }

  // Validates every gate's shape (W: hSize × (inputSize + hSize), b: hSize)
  // BEFORE writing, then copies element by element into the existing arrays.
  setWeights(data: ReturnType<GRULayer04["getWeights"]>): void {
    const combSize = this.inputSize + this.hSize;
    const gateNames = ["resetGate", "updateGate", "newGate"] as const;
    if (data === null || typeof data !== "object") {
      throw new Error("GRULayer.setWeights: expected object with resetGate, updateGate and newGate");
    }
    for (const name of gateNames) {
      const g = data[name];
      if (g === null || typeof g !== "object") {
        throw new Error(`GRULayer.setWeights: missing gate ${name}`);
      }
      validate2DArray(g.W, this.hSize, combSize, `GRULayer.setWeights(${name}.W)`);
      validateArray(g.b, this.hSize, `GRULayer.setWeights(${name}.b)`);
    }
    for (const name of gateNames) {
      const dst = this[name];
      const src = data[name];
      for (let i = 0; i < this.hSize; i++) {
        for (let j = 0; j < combSize; j++) dst.W[i][j] = src.W[i][j];
        dst.b[i] = src.b[i];
      }
    }
  }
}


// ─── T020: memory (vector optimizers) and exactness against GRULayer 0.4 ────

// Custom optimizer with state (not a built-in, so it takes the scalar fallback).
class Rms implements Optimizer {
  private s = 0
  step(w: number, g: number, lr: number): number {
    this.s = 0.9 * this.s + 0.1 * g * g
    return w + (lr * g) / (Math.sqrt(this.s) + 1e-8)
  }
}

const factories: Array<[string, OptimizerFactory]> = [
  ['SGD', () => new SGD()],
  ['Momentum', () => new Momentum(0.9)],
  ['Adam', () => new Adam()],
  ['custom Rms', () => new Rms()],
]

const INPUT = 3
const HIDDEN = 4
const STEPS = 3
const RUNS = 100
const LR = 0.05
const GATES = ['resetGate', 'updateGate', 'newGate'] as const

// Same initialization for both: the reference copies the weights of the new layer.
function pair(factory: OptimizerFactory) {
  const fresh = withSeed(7, () => new GRULayer(INPUT, HIDDEN, factory))
  const ref = new GRULayer04(INPUT, HIDDEN, factory)
  ref.setWeights(fresh.getWeights())
  return { fresh, ref }
}

function sameWeights(a: ReturnType<GRULayer['getWeights']>, b: ReturnType<GRULayer04['getWeights']>) {
  for (const g of GATES) {
    for (let k = 0; k < a[g].W.length; k++) {
      for (let j = 0; j < a[g].W[k].length; j++) {
        expect(Object.is(a[g].W[k][j], b[g].W[k][j])).toBe(true)
      }
      expect(Object.is(a[g].b[k], b[g].b[k])).toBe(true)
    }
  }
}

describe('GRULayer 0.5: exactness against the 0.4 class', () => {
  for (const [name, factory] of factories) {
    it(`${name}: 100 episodes (predict + backprop) give identical outputs and weights`, () => {
      const { fresh, ref } = pair(factory)
      const rnd = seededRandom(11)
      for (let run = 0; run < RUNS; run++) {
        fresh.reset()
        ref.reset()
        for (let t = 0; t < STEPS; t++) {
          const x = Array.from({ length: INPUT }, () => rnd() * 2 - 1)
          const hA = fresh.predict(x)
          const hB = ref.predict(x)
          for (let k = 0; k < HIDDEN; k++) expect(Object.is(hA[k], hB[k])).toBe(true)
        }
        const dh = Array.from({ length: STEPS }, () =>
          Array.from({ length: HIDDEN }, () => (rnd() - 0.5) * 0.4),
        )
        fresh.backprop(dh, LR)
        ref.backprop(dh, LR)
      }
      sameWeights(fresh.getWeights(), ref.getWeights())
    })
  }
})

describe('GRULayer 0.5: vector optimizers layout', () => {
  it('calls the factory once per weight row and per bias vector (not once per weight)', () => {
    // vecFromFactory calls the factory once to inspect each built-in optimizer:
    // 3 gates × (HIDDEN rows + 1 bias vector). The 0.4 layout made 3 × (HIDDEN × (INPUT + HIDDEN) + HIDDEN) = 96.
    let calls = 0
    new GRULayer(INPUT, HIDDEN, () => {
      calls++
      return new SGD()
    })
    expect(calls).toBe(3 * (HIDDEN + 1))
  })

  it('updates the weight rows and bias vectors in place (same array objects after backprop)', () => {
    const fresh = new GRULayer(INPUT, HIDDEN, () => new Adam())
    const row = fresh.resetGate.W[0]
    const bias = fresh.resetGate.b
    fresh.predict([0.1, 0.2, 0.3])
    fresh.backprop([[0.5, 0.5, 0.5, 0.5]], 0.1)
    expect(fresh.resetGate.W[0]).toBe(row)
    expect(fresh.resetGate.b).toBe(bias)
  })
})
