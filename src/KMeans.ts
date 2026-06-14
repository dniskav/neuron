// ─── K-MEANS CLUSTERING ───────────────────────────────────────────────────────
//
// Lloyd's algorithm with K-Means++ initialization.
//
// ─── DISTANCE ────────────────────────────────────────────────────────────────
//   d(a, b) = √( Σᵢ (aᵢ - bᵢ)² )     Euclidean distance in R^n
//
// ─── ASSIGNMENT STEP ─────────────────────────────────────────────────────────
//   cᵢ = argminₖ d(xᵢ, μₖ)            assign each point to nearest centroid
//
// ─── UPDATE STEP ─────────────────────────────────────────────────────────────
//   μₖ = (1 / |Cₖ|) · Σ_{xᵢ ∈ Cₖ} xᵢ  recompute centroid as cluster mean
//
// ─── INERTIA ─────────────────────────────────────────────────────────────────
//   J = Σᵢ d(xᵢ, μ_{cᵢ})²             sum of squared distances to centroids
//
//   "Elbow method": plot inertia vs K. As K grows, J decreases — but with
//   diminishing returns. The optimal K sits at the "elbow" where the curve
//   bends sharply. Adding more clusters beyond that point offers little gain.
//
// ─── K-MEANS++ INIT ──────────────────────────────────────────────────────────
//   1. Pick first centroid uniformly at random.
//   2. For each remaining centroid: sample a point with probability ∝ D(x)²,
//      where D(x) is the distance to the nearest already-chosen centroid.
//   This initialization reduces the expected inertia and speeds convergence
//   compared to pure random initialization.
//
// ─────────────────────────────────────────────────────────────────────────────

export interface KMeansOptions {
  maxIter?: number;
  seed?: number;
}

export class KMeans {
  /** Cluster centroids, shape [k][features]. Set after fit(). */
  centroids: number[][];

  private readonly _k: number;
  private readonly _maxIter: number;
  private _rng: () => number;

  constructor(k: number, options: KMeansOptions = {}) {
    if (!Number.isInteger(k) || k < 1) {
      throw new Error(`KMeans: k must be a positive integer, got ${k}`);
    }
    this._k = k;
    this._maxIter = options.maxIter ?? 300;
    this.centroids = [];

    // Simple seeded PRNG (mulberry32) so results are reproducible when a seed
    // is provided; falls back to Math.random() otherwise.
    if (options.seed !== undefined) {
      let s = options.seed >>> 0;
      this._rng = () => {
        s += 0x6d2b79f5;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
        return ((t ^ (t >>> 14)) >>> 0) / 0x100000000;
      };
    } else {
      this._rng = () => Math.random();
    }
  }

  // ── fit ────────────────────────────────────────────────────────────────────
  // Runs K-Means++ init then Lloyd iterations until centroids stop moving or
  // maxIter is reached.
  fit(X: number[][]): void {
    if (!X || X.length === 0) {
      throw new Error("KMeans.fit: dataset X must be non-empty");
    }
    const n = X.length;
    const d = X[0].length;
    if (this._k > n) {
      throw new Error(`KMeans.fit: k (${this._k}) cannot exceed number of samples (${n})`);
    }

    // ── K-Means++ initialization ──────────────────────────────────────────
    this.centroids = [];

    // Step 1: pick first centroid uniformly at random
    const firstIdx = Math.floor(this._rng() * n);
    this.centroids.push([...X[firstIdx]]);

    // Step 2: iteratively choose remaining centroids with probability ∝ D²
    for (let c = 1; c < this._k; c++) {
      const dists = X.map(x => this._minDistSq(x));
      const total = dists.reduce((s, v) => s + v, 0);
      let threshold = this._rng() * total;
      let chosen = n - 1;
      for (let i = 0; i < n; i++) {
        threshold -= dists[i];
        if (threshold <= 0) { chosen = i; break; }
      }
      this.centroids.push([...X[chosen]]);
    }

    // ── Lloyd iterations ──────────────────────────────────────────────────
    const assignments = new Int32Array(n);

    for (let iter = 0; iter < this._maxIter; iter++) {
      // Assignment step
      for (let i = 0; i < n; i++) {
        assignments[i] = this._nearestCentroid(X[i]);
      }

      // Update step: recompute centroids as cluster means
      const sums: number[][] = Array.from({ length: this._k }, () => new Array(d).fill(0));
      const counts = new Int32Array(this._k);

      for (let i = 0; i < n; i++) {
        const c = assignments[i];
        counts[c]++;
        for (let j = 0; j < d; j++) sums[c][j] += X[i][j];
      }

      let moved = false;
      for (let c = 0; c < this._k; c++) {
        if (counts[c] === 0) continue; // empty cluster — centroid stays
        for (let j = 0; j < d; j++) {
          const newVal = sums[c][j] / counts[c];
          if (Math.abs(newVal - this.centroids[c][j]) > 1e-10) moved = true;
          this.centroids[c][j] = newVal;
        }
      }

      if (!moved) break; // converged
    }
  }

  // ── predict ────────────────────────────────────────────────────────────────
  // Returns the index of the nearest centroid for a single point.
  predict(x: number[]): number {
    if (this.centroids.length === 0) {
      throw new Error("KMeans.predict: call fit() before predict()");
    }
    return this._nearestCentroid(x);
  }

  // ── predictBatch ──────────────────────────────────────────────────────────
  // Assigns each point in X to a cluster. Returns array of cluster indices.
  predictBatch(X: number[][]): number[] {
    return X.map(x => this.predict(x));
  }

  // ── inertia ───────────────────────────────────────────────────────────────
  //   J = Σᵢ d(xᵢ, μ_{cᵢ})²
  // Lower inertia = tighter clusters. Use the elbow method to pick K:
  // run fit() for K = 1..10 and plot inertia — the elbow is your optimal K.
  inertia(X: number[][]): number {
    if (this.centroids.length === 0) {
      throw new Error("KMeans.inertia: call fit() before inertia()");
    }
    return X.reduce((sum, x) => sum + this._minDistSq(x), 0);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private _euclideanSq(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return s;
  }

  private _minDistSq(x: number[]): number {
    let min = Infinity;
    for (const c of this.centroids) {
      const d = this._euclideanSq(x, c);
      if (d < min) min = d;
    }
    return min;
  }

  private _nearestCentroid(x: number[]): number {
    let best = 0;
    let bestDist = Infinity;
    for (let c = 0; c < this.centroids.length; c++) {
      const d = this._euclideanSq(x, this.centroids[c]);
      if (d < bestDist) { bestDist = d; best = c; }
    }
    return best;
  }
}
