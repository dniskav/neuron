import { Dropout }                            from "./Dropout";
import { Activation, sigmoid }                from "./activations";
import { OptimizerFactory, defaultOptimizer } from "./optimizers";
import { validateArray }                      from "./Validation";
import { random }                             from "./random";
import { getBackend }                         from "./core/backend";
import { Backend, Vec }                       from "./core/tensor";
import { VecOptimizer, vecFromFactory }       from "./core/optim";
import type { NetworkNOptions }               from "./NetworkN";
import { describeOptimizer, optimizerFromSpec } from "./NetworkN";
import {
  activationFromSpec, activationToSpec, validateNetworkNConfig, type NetworkNConfig,
} from "./core/format2";

// ─── NetworkFast ──────────────────────────────────────────────────────────────
// Same model as NetworkN (see NetworkN.ts for the maths: g, dropout, dfn on the
// output of f, residual that carries g, W^T delta with the weights BEFORE the
// update, step (2/n)(t - p) in train). This file only explains the buffer design.
//
// Design (speed and memory, FR-4):
//   - Per layer: W (nOut*nIn, row-major, W[k*nIn + j]) and b (nOut) in Float64Array,
//     in the order of NetworkN.getWeights() (per neuron: its weights, then its bias).
//   - Per layer, one VecOptimizer for W and another for b (plan, decision 3). The state
//     of each element is independent, so the result equals one optimizer per scalar.
//     The factory is called once per VecOptimizer to inspect it (see vecFromFactory).
//   - Every working buffer (activations, outputs of f, deltas, gradients) is reserved
//     ONCE in the constructor. The hot path allocates only the arrays it returns
//     (and, with dropout, the number[] copies that the Dropout class needs).
//   - Initialisation draws random() in the order of NetworkN: layer by layer, neuron
//     by neuron, weight by weight, limit sqrt(1/nIn), bias 0. Same seed, same weights.
//   - The backend is read once per public call (plan, decision 5). A backend passed in
//     `options.backend` has priority over the global one (setBackend); operations it
//     does not define come from the global backend.
//   - Buffers are shared by all calls: the class is not reentrant (nor is NetworkN
//     usable from two threads).
export interface NetworkFastOptions extends NetworkNOptions {
  // Replaces operations of the numeric backend for THIS instance only.
  backend?: Partial<Backend>;
}

const BACKEND_OPS: (keyof Backend)[] = [
  "matVec", "matTVec", "addOuter", "matMulBatch", "matMulBatchT", "addOuterBatch",
];

export class NetworkFast {
  private readonly _L: number;
  private readonly _acts: Activation[];
  private readonly _residual: boolean | ((layerIndex: number) => boolean);
  private readonly _dropouts: Dropout[];
  private readonly _dropoutRate: number;
  private readonly _optimizerFactory: OptimizerFactory;

  // Parameters.
  private readonly _W: Vec[] = [];
  private readonly _b: Vec[] = [];
  private readonly _wOpt: VecOptimizer[] = [];
  private readonly _bOpt: VecOptimizer[] = [];

  // Working buffers.
  private readonly _a: Vec[] = [];     // a[l] = input of layer l; a[L] = output (after residual and dropout)
  private readonly _pre: Vec[] = [];   // pre[l] = output of f in layer l, BEFORE residual and dropout
  private readonly _delta: Vec[] = []; // delta[l] = -dL/d(pre-activation of layer l)
  private readonly _g: Vec[] = [];     // g[l] = -dL/d(a[l]); g[0] is the input gradient
  private readonly _gW: Vec[] = [];    // gradient of W of each layer
  private readonly _res: Uint8Array;   // residual flag per layer, evaluated once per public call

  // Batch buffers (FR-5). Reserved on the first batch call and regrown only when a
  // larger batch arrives; a batch of n rows uses the first n rows (row-major, so a
  // prefix is a valid smaller batch). _cap = 0 until then.
  private _cap = 0;
  private _bA: Vec[] = [];      // _bA[l] = n x structure[l]; _bA[L] = outputs
  private _bPre: Vec[] = [];    // outputs of f per layer, BEFORE residual and dropout
  private _bDelta: Vec[] = [];  // -dL/d(pre-activation) per layer
  private _bG: Vec[] = [];      // _bG[l] = -dL/d(_bA[l]), l >= 1 (the input gradient is not needed)
  private _bMask: Vec[] = [];   // dropout mask per hidden layer (scale or 0)
  private _gB: Vec[] = [];      // mean bias gradient per layer
  private _ones: number[][] = []; // all-ones vectors fed to Dropout.forward to draw a mask

  private readonly _override: Partial<Backend> | null;
  private _mixOf: Backend | null = null;
  private _mixed: Backend | null = null;

  /** Class name for ModelSaver format 2 (not read from constructor.name: bundlers rename it). */
  static readonly kind = "NetworkFast";
  get kind(): string { return NetworkFast.kind; }

  constructor(readonly structure: number[], options: NetworkFastOptions = {}) {
    if (!Array.isArray(structure) || structure.length < 2 ||
        structure.some(n => !Number.isInteger(n) || n < 1)) {
      throw new Error(
        `NetworkFast: structure must be an array of at least 2 integers >= 1, got ${JSON.stringify(structure)}`
      );
    }
    const nLayers     = structure.length - 1;
    const activations = options.activations ?? Array.from({ length: nLayers }, () => sigmoid);
    const optimizer: OptimizerFactory = options.optimizer ?? defaultOptimizer;
    const dropoutRate = options.dropoutRate ?? 0;

    if (activations.length !== nLayers) {
      throw new Error(`Expected ${nLayers} activations, got ${activations.length}`);
    }

    if (dropoutRate < 0 || dropoutRate >= 1) {
      throw new Error(`Dropout rate must be in [0, 1), got ${dropoutRate}`);
    }

    const ovr = options.backend ?? null;
    if (ovr !== null) {
      for (const op of BACKEND_OPS) {
        if (ovr[op] !== undefined && typeof ovr[op] !== "function") {
          throw new Error(`NetworkFast: backend.${op} must be a function`);
        }
      }
    }
    this._override = ovr;

    this._L        = nLayers;
    this._acts     = [...activations];
    this._residual = options.residual ?? false;
    this._res      = new Uint8Array(nLayers);
    this._dropoutRate     = dropoutRate;
    this._optimizerFactory = optimizer;

    this._a.push(new Float64Array(structure[0]));
    this._g.push(new Float64Array(structure[0]));
    for (let l = 0; l < nLayers; l++) {
      const nIn = structure[l], nOut = structure[l + 1];
      // Same draw order as NetworkN: neuron k, weight j, W[k*nIn + j].
      const limit = Math.sqrt(1 / nIn);
      const W = new Float64Array(nOut * nIn);
      for (let i = 0; i < W.length; i++) W[i] = (random() * 2 - 1) * limit;
      this._W.push(W);
      this._b.push(new Float64Array(nOut)); // bias 0
      this._wOpt.push(vecFromFactory(optimizer, nOut * nIn));
      this._bOpt.push(vecFromFactory(optimizer, nOut));
      this._pre.push(new Float64Array(nOut));
      this._delta.push(new Float64Array(nOut));
      this._gW.push(new Float64Array(nOut * nIn));
      this._a.push(new Float64Array(nOut));
      this._g.push(new Float64Array(nOut));
    }

    // Dropout instances for hidden layers only (not the output layer), as NetworkN.
    this._dropouts = [];
    if (dropoutRate > 0) {
      for (let i = 0; i < nLayers - 1; i++) this._dropouts.push(new Dropout(dropoutRate));
    }
  }

  predict(inputs: number[], training = false): number[] {
    validateArray(inputs, this.structure[0], 'NetworkFast.predict');
    const be = this._backend();
    this._computeResidual();
    this._forward(be, inputs, training, false);
    return Array.from(this._a[this._L]);
  }

  // Step of the gradient of the MEAN squared error that it returns (see NetworkN.train).
  train(inputs: number[], targets: number[], lr: number): number {
    validateArray(inputs, this.structure[0], 'NetworkFast.train');
    validateArray(targets, this.structure[this._L], 'NetworkFast.train');
    // lr must be finite and >= 0 (0 is a valid no-op); checked before anything is touched.
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`NetworkFast.train: lr must be a finite number >= 0, got ${lr}`);
    }

    const be = this._backend();
    this._computeResidual();
    this._forward(be, inputs, true, false);
    const pred = this._a[this._L];
    const n = pred.length;
    const g = this._g[this._L];
    for (let i = 0; i < n; i++) g[i] = (2 / n) * (targets[i] - pred[i]);
    this._backward(be, null, lr);

    let s = 0;
    for (let i = 0; i < n; i++) s += (targets[i] - pred[i]) ** 2;
    return s / n;
  }

  // Inference on a batch (no dropout). Row i equals predict(X[i]); like predict(x, false)
  // it clears the saved dropout masks.
  predictBatch(X: number[][]): number[][] {
    const n = this._checkRows(X, this.structure[0], 'NetworkFast.predictBatch', 'X');
    const be = this._backend();
    this._computeResidual();
    this._ensureBatch(n);
    this._forwardBatch(be, X, n, false);
    const out = this._bA[this._L], m = this.structure[this._L];
    const res: number[][] = new Array(n);
    for (let i = 0; i < n; i++) res[i] = Array.from(out.subarray(i * m, (i + 1) * m));
    return res;
  }

  // ONE optimizer step with the MEAN of the per-example gradients; returns the mean over
  // the batch of the per-example MSE. Per example the gradient is the one of train():
  // (2/m)(t - p) at the output (m outputs). With a batch of 1 it is train() exactly.
  // With dropout every example draws its own masks, in example order (as n train() calls).
  trainBatch(X: number[][], T: number[][], lr: number): number {
    const n = this._checkRows(X, this.structure[0], 'NetworkFast.trainBatch', 'X');
    if (!Array.isArray(T) || T.length !== n) {
      throw new Error(
        `NetworkFast.trainBatch: X and T need the same number of rows, got ${n} and ${Array.isArray(T) ? T.length : typeof T}`
      );
    }
    this._checkRows(T, this.structure[this._L], 'NetworkFast.trainBatch', 'T');
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`NetworkFast.trainBatch: lr must be a finite number >= 0, got ${lr}`);
    }

    const be = this._backend();
    this._computeResidual();
    this._ensureBatch(n);
    this._forwardBatch(be, X, n, true);

    const L = this._L, m = this.structure[L];
    const pred = this._bA[L], g = this._bG[L];
    let total = 0;
    for (let i = 0; i < n; i++) {
      const base = i * m, t = T[i];
      let s = 0;
      for (let k = 0; k < m; k++) {
        const e = t[k] - pred[base + k];
        g[base + k] = (2 / m) * e;
        s += e * e;
      }
      total += s / m;
    }
    this._backwardBatch(be, n, lr);
    return total / n;
  }

  // Deltas are -dL/d(pre-activation of the output layer); not rescaled. See NetworkN.
  trainWithDeltas(inputs: number[], outputDeltas: number[], lr: number, reuseMasks = false): void {
    validateArray(inputs, this.structure[0], 'NetworkFast.trainWithDeltas');
    validateArray(outputDeltas, this.structure[this._L], 'NetworkFast.trainWithDeltas');
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`NetworkFast.trainWithDeltas: lr must be a finite number >= 0, got ${lr}`);
    }
    this._assertDeltasSupported('trainWithDeltas');
    if (reuseMasks) this._assertMasksSaved('trainWithDeltas');
    const be = this._backend();
    this._computeResidual();
    this._forward(be, inputs, true, reuseMasks);
    this._backward(be, outputDeltas, lr);
  }

  // -dL/d(inputs), no weight is touched. See NetworkN.inputGradient.
  inputGradient(inputs: number[], outputDeltas: number[], reuseMasks = false): number[] {
    this._assertDeltasSupported('inputGradient');
    validateArray(inputs, this.structure[0], 'NetworkFast.inputGradient');
    validateArray(outputDeltas, this.structure[this._L], 'NetworkFast.inputGradient');
    if (reuseMasks) this._assertMasksSaved('inputGradient');
    const be = this._backend();
    this._computeResidual();
    this._forward(be, inputs, true, reuseMasks);
    this._backward(be, outputDeltas, null);
    return Array.from(this._g[0]);
  }

  // Order of NetworkN: layer by layer, per neuron its weights and then its bias.
  // Also clears the saved dropout masks (as NetworkN).
  getWeights(): number[] {
    for (const d of this._dropouts) d.resetMask();
    const w: number[] = [];
    for (let l = 0; l < this._L; l++) {
      const nIn = this.structure[l], nOut = this.structure[l + 1];
      const W = this._W[l], b = this._b[l];
      for (let k = 0; k < nOut; k++) {
        const base = k * nIn;
        for (let j = 0; j < nIn; j++) w.push(W[base + j]);
        w.push(b[k]);
      }
    }
    return w;
  }

  // Validates length and finiteness BEFORE resetting masks or writing. Does not reset
  // the optimizer state (as NetworkN).
  setWeights(weights: number[]): void {
    validateArray(weights, this._weightCount(), 'NetworkFast.setWeights');
    for (const d of this._dropouts) d.resetMask();
    let idx = 0;
    for (let l = 0; l < this._L; l++) {
      const nIn = this.structure[l], nOut = this.structure[l + 1];
      const W = this._W[l], b = this._b[l];
      for (let k = 0; k < nOut; k++) {
        const base = k * nIn;
        for (let j = 0; j < nIn; j++) W[base + j] = weights[idx++];
        b[k] = weights[idx++];
      }
    }
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  // ── Format 2 (ModelSaver.save / load) ─────────────────────────────────────
  // Same config as NetworkN (NetworkNConfig). The `backend` option is NOT saved: it is a
  // property of the process (which implementation runs the maths), not of the model, so
  // a loaded network uses the global backend. Throws if an activation has no name.
  toConfig(): NetworkNConfig {
    return {
      structure: [...this.structure],
      activations: this._acts.map((a, i) => activationToSpec(a, `NetworkFast.toConfig: activations[${i}]`)),
      optimizer: describeOptimizer(this._optimizerFactory),
      residual: Array.from({ length: this._L }, (_, l) => {
        const r = this._residual;
        return typeof r === "function" ? r(l) : r;
      }),
      dropoutRate: this._dropoutRate,
    };
  }

  /** New network from a config made by `toConfig` (or by NetworkN). Validates before building. */
  static fromConfig(config: unknown): NetworkFast {
    const c = validateNetworkNConfig(config, "NetworkFast.fromConfig");
    const residual = c.residual;
    return new NetworkFast(c.structure, {
      activations: c.activations.map(activationFromSpec),
      optimizer: optimizerFromSpec(c.optimizer),
      residual: (i: number) => residual[i],
      dropoutRate: c.dropoutRate,
    });
  }

  private _weightCount(): number {
    let total = 0;
    for (let i = 0; i < this._L; i++) total += (this.structure[i] + 1) * this.structure[i + 1];
    return total;
  }

  // Backend of this call: the instance override over the global one, per operation.
  // The mix is cached while the global backend object does not change.
  private _backend(): Backend {
    const g = getBackend();
    const o = this._override;
    if (o === null) return g;
    if (this._mixOf !== g || this._mixed === null) {
      const m: Backend = { ...g };
      for (const op of BACKEND_OPS) {
        const fn = o[op];
        if (fn !== undefined) (m as unknown as Record<string, unknown>)[op] = fn;
      }
      this._mixed = m;
      this._mixOf = g;
    }
    return this._mixed;
  }

  // Residual flag per layer (the function option is evaluated for every layer, as NetworkN does).
  private _computeResidual(): void {
    const r = this._residual;
    for (let l = 0; l < this._L; l++) {
      const want = typeof r === "function" ? r(l) : r;
      this._res[l] = want && this.structure[l] === this.structure[l + 1] ? 1 : 0;
    }
  }

  private _assertDeltasSupported(method: string): void {
    const last = this._L - 1;
    const r = this._residual;
    const want = typeof r === "function" ? r(last) : r;
    if (want && this.structure[last] === this.structure[last + 1]) {
      throw new Error(
        `NetworkFast.${method}: residual on the last layer needs dL/d(output), but only ` +
        `deltas (already multiplied by the activation derivative) are given. ` +
        `Use train(), or disable residual on the last layer.`
      );
    }
  }

  private _assertMasksSaved(method: string): void {
    if (this._dropouts.some(d => !d.hasMask())) {
      throw new Error(
        `NetworkFast.${method}: reuseMasks = true needs the dropout masks of a previous ` +
        `predict(x, true), but none are saved (an inference predict or setWeights/getWeights ` +
        `clears them). Call predict(x, true) first, or use reuseMasks = false.`
      );
    }
  }

  // Validates a non-empty array of rows of `width` finite numbers; touches nothing.
  private _checkRows(rows: unknown, width: number, method: string, name: string): number {
    if (!Array.isArray(rows)) throw new Error(`${method}: ${name} must be an array of rows, got ${typeof rows}`);
    if (rows.length === 0) throw new Error(`${method}: ${name} must have at least one row`);
    for (let i = 0; i < rows.length; i++) validateArray(rows[i], width, `${method} (${name}[${i}])`);
    return rows.length;
  }

  // Reserves the batch buffers for at least n rows (once; grows only for a larger batch).
  private _ensureBatch(n: number): void {
    if (n <= this._cap) return;
    const sz = this.structure, L = this._L;
    this._cap = n;
    this._bA = [new Float64Array(n * sz[0])];
    this._bG = [new Float64Array(0)];
    this._bPre = []; this._bDelta = []; this._bMask = []; this._gB = [];
    for (let l = 0; l < L; l++) {
      const w = sz[l + 1];
      this._bA.push(new Float64Array(n * w));
      this._bG.push(new Float64Array(n * w));
      this._bPre.push(new Float64Array(n * w));
      this._bDelta.push(new Float64Array(n * w));
      this._gB.push(new Float64Array(w));
      if (l < this._dropouts.length) this._bMask.push(new Float64Array(n * w));
    }
    if (this._ones.length === 0) {
      for (let l = 0; l < this._dropouts.length; l++) this._ones.push(new Array(sz[l + 1]).fill(1));
    }
  }

  // Batch forward: same maths as _forward, one matMulBatch per layer. With training and
  // dropout, the masks are drawn FIRST, example by example and layer by layer (the order
  // of n successive train() calls; a mask does not depend on the data), through
  // Dropout.forward on a vector of ones, whose output is the mask itself.
  private _forwardBatch(be: Backend, X: number[][], n: number, training: boolean): void {
    const L = this._L, sz = this.structure;
    const a0 = this._bA[0], w0 = sz[0];
    for (let i = 0; i < n; i++) {
      const row = X[i], base = i * w0;
      for (let j = 0; j < w0; j++) a0[base + j] = row[j];
    }
    const nd = this._dropouts.length;
    if (training) {
      for (let i = 0; i < n; i++) {
        for (let l = 0; l < nd; l++) {
          const w = sz[l + 1];
          const mk = this._dropouts[l].forward(this._ones[l], true);
          const dst = this._bMask[l], base = i * w;
          for (let k = 0; k < w; k++) dst[base + k] = mk[k];
        }
      }
    } else {
      for (const d of this._dropouts) d.resetMask();
    }

    for (let l = 0; l < L; l++) {
      const nIn = sz[l], nOut = sz[l + 1], size = n * nOut;
      const x = this._bA[l], p = this._bPre[l], out = this._bA[l + 1];
      be.matMulBatch(this._W[l], nOut, nIn, x, n, this._b[l], p);
      const f = this._acts[l];
      for (let q = 0; q < size; q++) p[q] = f.fn(p[q]);
      if (this._res[l] === 1) {
        for (let q = 0; q < size; q++) out[q] = p[q] + x[q];
      } else {
        for (let q = 0; q < size; q++) out[q] = p[q];
      }
      if (training && l < nd) {
        const mask = this._bMask[l];
        for (let q = 0; q < size; q++) out[q] = out[q] * mask[q];
      }
    }
  }

  // Batch backward + ONE optimizer step. _bG[L] must hold -dL/d(output) per example
  // (unscaled); the mean over the batch is applied with s = 1/n in the gradients of W
  // (addOuterBatch) and b. W^T delta uses the weights BEFORE the update of that layer.
  private _backwardBatch(be: Backend, n: number, lr: number): void {
    const L = this._L, sz = this.structure, s = 1 / n;
    for (let l = L - 1; l >= 0; l--) {
      const nIn = sz[l], nOut = sz[l + 1], size = n * nOut;
      const gOut = this._bG[l + 1], d = this._bDelta[l];
      if (l < this._dropouts.length) {
        const mask = this._bMask[l];
        for (let q = 0; q < size; q++) gOut[q] = gOut[q] * mask[q];
      }
      const p = this._bPre[l], f = this._acts[l];
      for (let q = 0; q < size; q++) d[q] = gOut[q] * f.dfn(p[q]);
      if (l > 0) {
        const gIn = this._bG[l];
        be.matMulBatchT(this._W[l], nOut, nIn, d, n, gIn);
        if (this._res[l] === 1) {
          const sizeIn = n * nIn;
          for (let q = 0; q < sizeIn; q++) gIn[q] = gIn[q] + gOut[q];
        }
      }
      const G = this._gW[l];
      G.fill(0);
      be.addOuterBatch(G, nOut, nIn, d, this._bA[l], n, s);
      const gb = this._gB[l];
      for (let k = 0; k < nOut; k++) {
        let acc = d[k];
        for (let i = 1; i < n; i++) acc += d[i * nOut + k];
        gb[k] = s * acc;
      }
      this._wOpt[l].step(this._W[l], G, lr);
      this._bOpt[l].step(this._b[l], gb, lr);
    }
  }

  // Forward pass into the buffers. Requires _computeResidual() to be up to date.
  //   a[l+1]  = f(W a[l] + b)  (+ a[l] if residual), then dropout (hidden layers)
  //   pre[l]  = f(W a[l] + b)  before residual and dropout
  private _forward(be: Backend, inputs: number[], training: boolean, reuseMasks: boolean): void {
    const L = this._L, sz = this.structure;
    const a0 = this._a[0];
    for (let i = 0; i < a0.length; i++) a0[i] = inputs[i];

    for (let l = 0; l < L; l++) {
      const nIn = sz[l], nOut = sz[l + 1];
      const x = this._a[l], p = this._pre[l], out = this._a[l + 1];
      be.matVec(this._W[l], nOut, nIn, x, this._b[l], p);
      const f = this._acts[l];
      for (let k = 0; k < nOut; k++) p[k] = f.fn(p[k]);
      if (this._res[l] === 1) {
        for (let k = 0; k < nOut; k++) out[k] = p[k] + x[k];
      } else {
        out.set(p);
      }
      if (l < this._dropouts.length) {
        // Dropout works on number[]: same class, same order of random() draws.
        const d = this._dropouts[l];
        const cur = Array.from(out);
        const masked = reuseMasks ? d.reuseMask(cur) : d.forward(cur, training);
        for (let k = 0; k < nOut; k++) out[k] = masked[k];
      }
    }
  }

  // Backward pass: g[l] = -dL/d(a[l]).
  //   dropout:  g' = mask * g
  //   delta_l = g' * dfn(pre[l])      (the last layer takes `deltas` when given)
  //   g[l-1]  = W^T delta_l (+ g' if residual)   (the skip carries g, not delta)
  // With `deltas` null the output gradient g[L] must be filled. Weights are read
  // before their layer is updated. Updates only when lr !== null.
  private _backward(be: Backend, deltas: number[] | null, lr: number | null): void {
    const L = this._L, sz = this.structure;
    for (let l = L - 1; l >= 0; l--) {
      const nIn = sz[l], nOut = sz[l + 1];
      const gOut = this._g[l + 1];
      if (l < this._dropouts.length) {
        const masked = this._dropouts[l].backward(Array.from(gOut));
        for (let k = 0; k < nOut; k++) gOut[k] = masked[k];
      }
      const d = this._delta[l];
      if (l === L - 1 && deltas !== null) {
        for (let k = 0; k < nOut; k++) d[k] = deltas[k];
      } else {
        const p = this._pre[l], f = this._acts[l];
        for (let k = 0; k < nOut; k++) d[k] = gOut[k] * f.dfn(p[k]);
      }
      // Input of the first layer is only needed for the input gradient, not to train.
      if (l > 0 || lr === null) {
        const gIn = this._g[l];
        be.matTVec(this._W[l], nOut, nIn, d, gIn);
        if (this._res[l] === 1) {
          for (let j = 0; j < nIn; j++) gIn[j] = gIn[j] + gOut[j];
        }
      }
      if (lr !== null) {
        const G = this._gW[l];
        G.fill(0);
        be.addOuter(G, nOut, nIn, d, this._a[l], 1);
        this._wOpt[l].step(this._W[l], G, lr);
        this._bOpt[l].step(this._b[l], d, lr);
      }
    }
  }
}
