// ─── T-SNE ────────────────────────────────────────────────────────────────────
//
// t-distributed Stochastic Neighbor Embedding (van der Maaten & Hinton, 2008).
//
// Reduces high-dimensional data to a low-dimensional map (usually 2D) while
// preserving local structure: nearby points in the original space remain nearby
// in the embedding. The characteristic cluster visualizations you see everywhere
// are produced by this algorithm.
//
// ─── HIGH-LEVEL INTUITION ────────────────────────────────────────────────────
//
//   1. In the original space, measure how similar each pair of points is using
//      a Gaussian kernel (P matrix). Points close together get high probability.
//
//   2. In the low-dimensional space, measure similarity using a Student-t
//      distribution with 1 degree of freedom (Q matrix). The heavier tails of
//      the t-distribution allow dissimilar points to be placed far apart without
//      exerting a strong attractive force — this prevents the "crowding problem"
//      that occurs with Gaussian kernels in low dimensions.
//
//   3. Minimize KL(P ‖ Q): make Q look as much like P as possible.
//      Gradient descent moves points toward a layout that respects the original
//      neighborhood structure.
//
// ─── ALGORITHM STEPS ─────────────────────────────────────────────────────────
//
//   Step 1 — Conditional probabilities in high-D (Gaussian kernel):
//     P(j|i) = exp(-‖xi - xj‖² / 2σi²) / Σ_{k≠i} exp(-‖xi - xk‖² / 2σi²)
//
//     σi is chosen per-point via binary search to achieve a target perplexity:
//       Perplexity = 2^{H(Pi)}    where    H(Pi) = -Σ_j P(j|i) log₂ P(j|i)
//
//   Step 2 — Symmetrize (joint probability):
//     p_ij = (P(j|i) + P(i|j)) / 2n
//
//   Step 3 — Low-D similarities (Student-t, df=1):
//     q_ij = (1 + ‖yi - yj‖²)^{-1} / Σ_{k≠l} (1 + ‖yk - yl‖²)^{-1}
//
//   Step 4 — KL divergence gradient:
//     ∂KL/∂yi = 4 Σ_j (p_ij - q_ij)(yi - yj)(1 + ‖yi - yj‖²)^{-1}
//
//   Step 5 — Gradient descent with momentum:
//     Y(t) = Y(t-1) + lr · grad + momentum · (Y(t-1) - Y(t-2))
//
// ─── TRICKS ──────────────────────────────────────────────────────────────────
//
//   Early exaggeration: multiply all p_ij by 4 for the first 50 iterations.
//     This makes clusters form faster — the algorithm is free to create large
//     gaps between groups before settling into fine-grained structure.
//
//   Momentum schedule: 0.5 for iterations 0-19, 0.8 thereafter.
//     Low momentum early lets points move freely; higher momentum later dampens
//     oscillations and speeds up convergence.
//
// ─── COMPLEXITY ──────────────────────────────────────────────────────────────
//
//   This implementation is O(n²) per iteration in both time and memory.
//   It works well for n ≲ 1 000 points. For larger datasets consider
//   Barnes-Hut t-SNE (O(n log n)) or UMAP.
//
// ─────────────────────────────────────────────────────────────────────────────

export interface TSNEOptions {
  /** Dimensionality of the output embedding. Default 2. */
  nComponents?: number;
  /**
   * Perplexity — loosely controls the effective number of neighbors considered
   * for each point. Typical values: 5–50. Default 30.
   * Must be less than the number of data points.
   */
  perplexity?: number;
  /** Learning rate for gradient descent. Default 200. */
  lr?: number;
  /** Number of gradient-descent iterations. Default 1000. */
  nIter?: number;
  /**
   * Seed for the pseudo-random number generator.
   * Set to any integer for reproducible results. Default uses Math.random.
   */
  seed?: number;
}

export class TSNE {
  /** Result of the embedding, shape [n][nComponents]. Available after fit(). */
  embedding: number[][];

  private readonly _nComponents: number;
  private readonly _perplexity: number;
  private readonly _lr: number;
  private readonly _nIter: number;
  private readonly _seed: number | undefined;

  // KL divergence tracked during the last fit() call.
  private _klDivergence = 0;
  // P matrix stored for kl() reporting.
  private _P: number[][] = [];

  constructor(options: TSNEOptions = {}) {
    this._nComponents = options.nComponents ?? 2;
    this._perplexity  = options.perplexity  ?? 30;
    this._lr          = options.lr          ?? 200;
    this._nIter       = options.nIter       ?? 1000;
    this._seed        = options.seed;
    this.embedding    = [];
  }

  // ── fit ────────────────────────────────────────────────────────────────────
  // Runs the full t-SNE algorithm on X (shape [n][d]).
  // Stores the result in this.embedding ([n][nComponents]).
  fit(X: number[][]): void {
    const n = X.length;
    if (n < 2) throw new Error('TSNE.fit: need at least 2 data points');
    if (this._perplexity >= n) {
      throw new Error(
        `TSNE.fit: perplexity (${this._perplexity}) must be less than n (${n})`
      );
    }

    const rng = this._seed !== undefined ? _mulberry32(this._seed) : Math.random;

    // ── Step 1: pairwise squared distances in high-D ────────────────────────
    const distSq = _pairwiseDistSq(X, n);

    // ── Step 2: compute conditional probabilities via binary search for σ ───
    const Pcond = this._computePcond(distSq, n);

    // ── Step 3: symmetrize → joint probability p_ij = (P(j|i)+P(i|j)) / 2n ─
    const P = _symmetrize(Pcond, n);
    this._P = P; // keep for kl() reporting

    // ── Step 4: initialize low-D embedding with small Gaussian noise ─────────
    // Box-Muller transform: converts uniform(0,1) pairs to standard normal samples.
    //   Z = sqrt(-2 ln U1) · cos(2π U2)
    // Scale by 0.01 so initial points cluster near the origin.
    let Y: number[][] = Array.from({ length: n }, () => {
      return Array.from({ length: this._nComponents }, () => {
        const u1 = Math.max(rng(), 1e-12);
        const u2 = rng();
        const z  = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return z * 0.01;
      });
    });

    // Y_prev is needed for momentum: Δ(t-1) = Y(t-1) - Y(t-2)
    let Yprev: number[][] = Y.map(row => [...row]);

    // ── Step 5: gradient descent ─────────────────────────────────────────────
    const EXAGGERATION_ITERS = 50;
    const EXAGGERATION_FACTOR = 4;
    const MOMENTUM_SWITCH    = 20;

    for (let iter = 0; iter < this._nIter; iter++) {
      const momentum = iter < MOMENTUM_SWITCH ? 0.5 : 0.8;

      // Early exaggeration: inflate P during the first iterations.
      // This creates strong attractive forces between points that are similar
      // in high-D, causing tight clusters to form quickly.
      const pScale = iter < EXAGGERATION_ITERS ? EXAGGERATION_FACTOR : 1;

      // ── Compute low-D similarities Q ──────────────────────────────────────
      // q_ij = (1 + ‖yi - yj‖²)^{-1} / Z
      // where Z = Σ_{k≠l} (1 + ‖yk - yl‖²)^{-1}
      const { Q, invDist } = _computeQ(Y, n, this._nComponents);

      // ── Gradient ──────────────────────────────────────────────────────────
      // ∂KL/∂yi = 4 Σ_j (p_ij - q_ij)(yi - yj) · invDist_ij
      const grad: number[][] = Array.from({ length: n }, () =>
        new Array(this._nComponents).fill(0)
      );

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const pq = pScale * P[i][j] - Q[i][j];
          const c  = 4 * pq * invDist[i][j];
          for (let d = 0; d < this._nComponents; d++) {
            grad[i][d] += c * (Y[i][d] - Y[j][d]);
          }
        }
      }

      // ── Update with momentum ──────────────────────────────────────────────
      // Y(t) = Y(t-1) - lr · grad + momentum · (Y(t-1) - Y(t-2))
      const Ynext: number[][] = Array.from({ length: n }, (_, i) =>
        Array.from({ length: this._nComponents }, (_, d) =>
          Y[i][d]
          - this._lr * grad[i][d]
          + momentum * (Y[i][d] - Yprev[i][d])
        )
      );

      Yprev = Y;
      Y     = Ynext;
    }

    this.embedding = Y;

    // ── Compute final KL divergence ───────────────────────────────────────────
    // KL(P ‖ Q) = Σ_{i≠j} p_ij · log(p_ij / q_ij)
    // Lower is better — measures how well the low-D structure matches high-D.
    const { Q: Qfinal } = _computeQ(Y, n, this._nComponents);
    let kl = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const p = P[i][j];
        if (p > 1e-12) {
          kl += p * Math.log(p / (Qfinal[i][j] + 1e-12));
        }
      }
    }
    this._klDivergence = kl;
  }

  // ── fitTransform ───────────────────────────────────────────────────────────
  // Convenience: fit() then return this.embedding.
  fitTransform(X: number[][]): number[][] {
    this.fit(X);
    return this.embedding;
  }

  // ── kl ─────────────────────────────────────────────────────────────────────
  // Returns the KL divergence KL(P ‖ Q) from the last fit() call.
  // Lower is better. Useful for comparing perplexity settings or iteration counts.
  kl(): number {
    return this._klDivergence;
  }

  // ── Private: binary search for σi ─────────────────────────────────────────
  // For each point i, find σi such that the Shannon entropy of P(·|i) equals
  // log₂(perplexity). We use binary search on σ².
  private _computePcond(distSq: number[][], n: number): number[][] {
    const targetEntropy = Math.log2(this._perplexity);
    const Pcond: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

    for (let i = 0; i < n; i++) {
      let sigmaLo  = 0;
      let sigmaHi  = 1e10;
      let sigma2   = 1.0;  // current σ²

      for (let attempt = 0; attempt < 50; attempt++) {
        // Compute P(j|i) for current σ²
        const dists = distSq[i];
        let sumExp  = 0;
        const exps  = new Array(n).fill(0);
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const e = Math.exp(-dists[j] / (2 * sigma2));
          exps[j] = e;
          sumExp += e;
        }
        if (sumExp < 1e-12) break; // degenerate — all neighbors too far

        // Normalize to probabilities and compute entropy H = -Σ p·log₂(p)
        let H = 0;
        for (let j = 0; j < n; j++) {
          if (j === i) continue;
          const p = exps[j] / sumExp;
          Pcond[i][j] = p;
          if (p > 1e-12) H -= p * Math.log2(p);
        }

        // Check if perplexity = 2^H is close enough to target
        const delta = H - targetEntropy;
        if (Math.abs(delta) < 1e-5) break;

        // Binary search: if H > target, σ is too large; shrink it
        if (delta > 0) {
          sigmaHi = sigma2;
          sigma2  = (sigmaLo + sigma2) / 2;
        } else {
          sigmaLo = sigma2;
          sigma2  = sigmaHi < 1e9 ? (sigma2 + sigmaHi) / 2 : sigma2 * 2;
        }
      }
    }

    return Pcond;
  }
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────

// Compute pairwise squared Euclidean distances: D[i][j] = ‖xi - xj‖²
function _pairwiseDistSq(X: number[][], n: number): number[][] {
  const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let d = 0;
      for (let k = 0; k < X[i].length; k++) {
        const diff = X[i][k] - X[j][k];
        d += diff * diff;
      }
      D[i][j] = d;
      D[j][i] = d;
    }
  }
  return D;
}

// Symmetrize conditional probabilities and normalize by 2n.
//   p_ij = (P(j|i) + P(i|j)) / 2n
// The 2n denominator ensures Σ_{i,j} p_ij = 1.
function _symmetrize(Pcond: number[][], n: number): number[][] {
  const P: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      P[i][j] = (Pcond[i][j] + Pcond[j][i]) / (2 * n);
    }
  }
  return P;
}

// Compute the low-dimensional joint distribution Q under the Student-t kernel.
//   num_ij = (1 + ‖yi - yj‖²)^{-1}   for i ≠ j, 0 for i = j
//   Z      = Σ_{k≠l} num_kl           normalization constant
//   q_ij   = num_ij / Z
// Also returns the unnormalized inverse-distance matrix (reused in gradient).
function _computeQ(
  Y: number[][],
  n: number,
  nComponents: number
): { Q: number[][]; invDist: number[][] } {
  const num: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  let Z = 0;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let d2 = 0;
      for (let d = 0; d < nComponents; d++) {
        const diff = Y[i][d] - Y[j][d];
        d2 += diff * diff;
      }
      const inv = 1 / (1 + d2);  // (1 + ‖yi - yj‖²)^{-1}
      num[i][j] = inv;
      num[j][i] = inv;
      Z += 2 * inv;               // symmetric pair counted twice
    }
  }

  // Avoid division by zero when all points coincide
  if (Z < 1e-12) Z = 1e-12;

  const Q: number[][] = Array.from({ length: n }, (_, i) =>
    num[i].map(v => v / Z)
  );

  return { Q, invDist: num };
}

// ─── MULBERRY32 PRNG ──────────────────────────────────────────────────────────
// A fast, seedable 32-bit pseudo-random number generator.
// Returns a function that produces values in [0, 1), matching Math.random()'s API.
// Reference: https://gist.github.com/tommyettinger/46a874533244883189143505d203312c
function _mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    z = (z ^ (z >>> 14)) >>> 0;
    return z / 0x100000000;
  };
}
