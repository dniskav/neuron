// ─── HOPFIELD NETWORK ─────────────────────────────────────────────────────────
//
// A recurrent network of N bipolar neurons (±1) that acts as a content-
// addressable memory. You "store" patterns and then "recall" them from noisy
// or partial inputs — the network converges to the nearest stored pattern.
//
// ─── STORAGE (Hebbian rule) ───────────────────────────────────────────────────
//   W ← W + (1/N) · p · pᵀ     for each pattern p ∈ {-1, +1}^N
//   Wᵢᵢ = 0                    no self-connections (diagonal zeroed)
//
//   The weight matrix encodes the correlations between all pairs of neurons.
//   After storing P patterns: W = (1/N) · Σₚ pₚ · pₚᵀ  (diagonal zeroed)
//
// ─── ENERGY FUNCTION ─────────────────────────────────────────────────────────
//   E(s) = −½ · Σᵢⱼ Wᵢⱼ · sᵢ · sⱼ
//
//   Stored patterns are local minima of E. Asynchronous updates always decrease
//   (or keep constant) the energy, guaranteeing convergence.
//
// ─── RECALL (asynchronous update) ────────────────────────────────────────────
//   For each neuron i (in random order):
//     hᵢ = Σⱼ Wᵢⱼ · sⱼ         local field (weighted sum of neighbors)
//     sᵢ ← sign(hᵢ)            +1 if hᵢ > 0, −1 if hᵢ < 0, unchanged if = 0
//   Repeat until no neuron changes state (convergence).
//
// ─── CAPACITY ────────────────────────────────────────────────────────────────
//   Theoretical maximum before spurious states dominate:
//     P_max ≈ 0.138 · N         (Amit, Gutfreund & Sompolinsky, 1985)
//   Storing more than ~14% of N patterns causes interference and recall errors.
//   Beyond capacity the network may converge to "spurious" states — linear
//   combinations (mixtures) of stored patterns.
//
// ─────────────────────────────────────────────────────────────────────────────

export class HopfieldNetwork {
  /** Symmetric weight matrix, shape [n][n] with zero diagonal. */
  weights: number[][];
  /** Number of neurons. */
  readonly n: number;
  /** Number of patterns stored so far. */
  storedPatterns: number;
  private _capacityWarned = false;

  constructor(n: number) {
    if (!Number.isInteger(n) || n < 1) {
      throw new Error(`HopfieldNetwork: n must be a positive integer, got ${n}`);
    }
    this.n = n;
    this.storedPatterns = 0;
    this.weights = Array.from({ length: n }, () => new Array(n).fill(0));
  }

  // ── store ──────────────────────────────────────────────────────────────────
  // Adds a pattern to the network's memory using the Hebbian learning rule:
  //   W ← W + (1/N) · p · pᵀ   (diagonal stays 0)
  //
  // The pattern must be bipolar: each element must be +1 or −1.
  store(pattern: number[]): void {
    this._assertBipolar('store', 'pattern', pattern);

    const scale = 1 / this.n;
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        if (i !== j) {
          this.weights[i][j] += scale * pattern[i] * pattern[j];
        }
      }
    }

    this.storedPatterns++;

    // Warn once per network when the theoretical capacity is exceeded
    // (~0.138·N patterns). Not thrown: the network keeps working, with errors.
    if (!this._capacityWarned && this.storedPatterns > Math.floor(0.138 * this.n)) {
      this._capacityWarned = true;
      console.warn(
        `HopfieldNetwork.store: ${this.storedPatterns} patterns exceed the capacity ` +
        `~0.138·N = ${Math.floor(0.138 * this.n)} for N = ${this.n}; recall may converge to spurious states.`
      );
    }
  }

  // ── recall ─────────────────────────────────────────────────────────────────
  // Starting from `input` (a noisy/partial copy of a stored pattern), runs
  // asynchronous updates until convergence or maxIter is reached.
  //   hᵢ = Σⱼ Wᵢⱼ · sⱼ
  //   sᵢ ← sign(hᵢ)           (+1, −1; unchanged when hᵢ = 0)
  // Returns the converged state vector.
  recall(input: number[], maxIter = 20 * this.n): number[] {
    this._assertBipolar('recall', 'input', input);

    const s = [...input];

    // Build a shuffled update order (asynchronous = one neuron at a time)
    const order = Array.from({ length: this.n }, (_, i) => i);

    for (let iter = 0; iter < maxIter; iter++) {
      // Random permutation for each sweep
      this._shuffleInPlace(order);

      let changed = false;
      for (const i of order) {
        let h = 0;
        const row = this.weights[i];
        for (let j = 0; j < this.n; j++) h += row[j] * s[j];

        const newSi = h > 0 ? 1 : h < 0 ? -1 : s[i]; // unchanged at exactly 0
        if (newSi !== s[i]) { s[i] = newSi; changed = true; }
      }

      if (!changed) break; // converged — energy minimum reached
    }

    return s;
  }

  // ── energy ─────────────────────────────────────────────────────────────────
  //   E(s) = −½ · Σᵢⱼ Wᵢⱼ · sᵢ · sⱼ
  // Stored patterns are local minima. Updates always push E downward (or keep
  // it constant), so the network is guaranteed to converge.
  energy(state: number[]): number {
    this._assertBipolar('energy', 'state', state);
    let e = 0;
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        e += this.weights[i][j] * state[i] * state[j];
      }
    }
    return -0.5 * e;
  }

  // ── binarize ───────────────────────────────────────────────────────────────
  // Converts a 0/1 array to bipolar −1/+1.
  //   0 → −1,   1 → +1
  static binarize(arr: number[]): number[] {
    HopfieldNetwork._assertFinite('binarize', arr);
    return arr.map(v => (v === 0 ? -1 : 1));
  }

  // ── unbinarize ─────────────────────────────────────────────────────────────
  // Converts a bipolar −1/+1 array back to 0/1.
  //   −1 → 0,   +1 → 1
  static unbinarize(arr: number[]): number[] {
    HopfieldNetwork._assertFinite('unbinarize', arr);
    return arr.map(v => (v === -1 ? 0 : 1));
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private _assertBipolar(method: string, name: string, v: number[]): void {
    if (v.length !== this.n) {
      throw new Error(
        `HopfieldNetwork.${method}: ${name} length ${v.length} does not match network size ${this.n}`
      );
    }
    for (let i = 0; i < this.n; i++) {
      if (v[i] !== 1 && v[i] !== -1) {
        throw new Error(
          `HopfieldNetwork.${method}: ${name} values must be +1 or -1, got ${v[i]} at index ${i}. ` +
          `Use HopfieldNetwork.binarize() to convert 0/1 arrays.`
        );
      }
    }
  }

  private static _assertFinite(method: string, arr: number[]): void {
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) {
        throw new Error(`HopfieldNetwork.${method}: values must be finite, got ${arr[i]} at index ${i}`);
      }
    }
  }

  private _shuffleInPlace(arr: number[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
}
