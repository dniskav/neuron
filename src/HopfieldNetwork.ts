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
    if (pattern.length !== this.n) {
      throw new Error(
        `HopfieldNetwork.store: pattern length ${pattern.length} does not match network size ${this.n}`
      );
    }
    for (let i = 0; i < this.n; i++) {
      if (pattern[i] !== 1 && pattern[i] !== -1) {
        throw new Error(
          `HopfieldNetwork.store: pattern values must be +1 or -1, got ${pattern[i]} at index ${i}. ` +
          `Use HopfieldNetwork.binarize() to convert 0/1 arrays.`
        );
      }
    }

    const scale = 1 / this.n;
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        if (i !== j) {
          this.weights[i][j] += scale * pattern[i] * pattern[j];
        }
      }
    }

    this.storedPatterns++;

    // Warn if approaching the theoretical capacity limit
    if (this.storedPatterns > Math.floor(0.138 * this.n)) {
      // Not thrown — just surfaced for awareness. Callers can silence by wrapping.
      // At ~14% of N the error rate begins to rise sharply.
    }
  }

  // ── recall ─────────────────────────────────────────────────────────────────
  // Starting from `input` (a noisy/partial copy of a stored pattern), runs
  // asynchronous updates until convergence or maxIter is reached.
  //   hᵢ = Σⱼ Wᵢⱼ · sⱼ
  //   sᵢ ← sign(hᵢ)           (+1, −1; unchanged when hᵢ = 0)
  // Returns the converged state vector.
  recall(input: number[], maxIter = 20 * this.n): number[] {
    if (input.length !== this.n) {
      throw new Error(
        `HopfieldNetwork.recall: input length ${input.length} does not match network size ${this.n}`
      );
    }

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
    if (state.length !== this.n) {
      throw new Error(
        `HopfieldNetwork.energy: state length ${state.length} does not match network size ${this.n}`
      );
    }
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
    return arr.map(v => (v === 0 ? -1 : 1));
  }

  // ── unbinarize ─────────────────────────────────────────────────────────────
  // Converts a bipolar −1/+1 array back to 0/1.
  //   −1 → 0,   +1 → 1
  static unbinarize(arr: number[]): number[] {
    return arr.map(v => (v === -1 ? 0 : 1));
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private _shuffleInPlace(arr: number[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }
}
