// ─── PRINCIPAL COMPONENT ANALYSIS ────────────────────────────────────────────
//
// Reduces dimensionality by projecting data onto the axes of greatest variance.
//
// ─── CENTERING ───────────────────────────────────────────────────────────────
//   X̃ = X - μ         subtract the column mean so the origin is at the centroid
//
// ─── COVARIANCE MATRIX ───────────────────────────────────────────────────────
//   Σ = (1 / n) · X̃ᵀ · X̃       shape [features × features], symmetric PSD
//
// ─── EIGENVECTORS ────────────────────────────────────────────────────────────
//   Σ · v = λ · v
//   The principal components are the eigenvectors of Σ sorted by eigenvalue.
//   The first PC explains the most variance; PCs are mutually orthogonal.
//
//   Implementation: power iteration with deflation.
//     Repeat until convergence:  v ← Σ·v / ‖Σ·v‖
//     Then deflate:              Σ ← Σ - λ·v·vᵀ       (Hotelling's deflation)
//     Repeat for the next PC.
//
// ─── PROJECTION ──────────────────────────────────────────────────────────────
//   Z = X̃ · Vᵀ        project centered data onto the PC basis
//                      shape: [n × nComponents]
//
// ─── RECONSTRUCTION ──────────────────────────────────────────────────────────
//   X̂ = Z · V + μ     invert the projection (approximate — not lossless)
//
// ─── EXPLAINED VARIANCE RATIO ────────────────────────────────────────────────
//   rₖ = λₖ / Σⱼ λⱼ   fraction of total variance captured by component k
//
// ─────────────────────────────────────────────────────────────────────────────

export class PCA {
  /** Eigenvectors (principal components), shape [nComponents][nFeatures]. */
  components: number[][];
  /** Eigenvalue for each component — equals the variance along that direction. */
  explainedVariance: number[];
  /** Column mean of the training data. */
  mean: number[];

  private readonly _nComponents: number;

  constructor(nComponents: number) {
    if (!Number.isInteger(nComponents) || nComponents < 1) {
      throw new Error(`PCA: nComponents must be a positive integer, got ${nComponents}`);
    }
    this._nComponents = nComponents;
    this.components = [];
    this.explainedVariance = [];
    this.mean = [];
  }

  // ── fit ────────────────────────────────────────────────────────────────────
  // Computes the mean and the top nComponents principal components from X.
  fit(X: number[][]): void {
    const n = X.length;
    if (n < 2) throw new Error("PCA.fit: need at least 2 samples");
    const p = X[0].length;
    if (this._nComponents > p) {
      throw new Error(
        `PCA: nComponents (${this._nComponents}) cannot exceed number of features (${p})`
      );
    }

    // ── 1. Compute mean ───────────────────────────────────────────────────
    this.mean = new Array(p).fill(0);
    for (const row of X) for (let j = 0; j < p; j++) this.mean[j] += row[j];
    for (let j = 0; j < p; j++) this.mean[j] /= n;

    // ── 2. Center data ────────────────────────────────────────────────────
    const Xc = X.map(row => row.map((v, j) => v - this.mean[j]));

    // ── 3. Covariance matrix: Σ = (1/n) · Xc^T · Xc ─────────────────────
    // Shape [p × p]. We only need to compute Σ once and then deflate it.
    let cov = this._covMatrix(Xc, n, p);

    // ── 4. Power iteration + Hotelling deflation ──────────────────────────
    this.components = [];
    this.explainedVariance = [];

    for (let c = 0; c < this._nComponents; c++) {
      const { eigenvector, eigenvalue } = this._powerIteration(cov, p);
      this.components.push(eigenvector);
      this.explainedVariance.push(eigenvalue);

      // Deflate: Σ ← Σ - λ · v · vᵀ  (remove the found component)
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          cov[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
      }
    }
  }

  // ── transform ──────────────────────────────────────────────────────────────
  //   Z = (X - μ) · Vᵀ       shape [n × nComponents]
  transform(X: number[][]): number[][] {
    if (this.components.length === 0) {
      throw new Error("PCA.transform: call fit() before transform()");
    }
    return X.map(row => {
      const centered = row.map((v, j) => v - this.mean[j]);
      return this.components.map(pc =>
        pc.reduce((s, w, j) => s + w * centered[j], 0)
      );
    });
  }

  // ── fitTransform ───────────────────────────────────────────────────────────
  // Convenience: fit() then transform() in a single call.
  fitTransform(X: number[][]): number[][] {
    this.fit(X);
    return this.transform(X);
  }

  // ── inverseTransform ───────────────────────────────────────────────────────
  //   X̂ = Z · V + μ          shape [n × nFeatures] (approximate reconstruction)
  inverseTransform(Z: number[][]): number[][] {
    if (this.components.length === 0) {
      throw new Error("PCA.inverseTransform: call fit() before inverseTransform()");
    }
    const p = this.mean.length;
    return Z.map(z => {
      const row = new Array(p).fill(0);
      for (let c = 0; c < this._nComponents; c++) {
        for (let j = 0; j < p; j++) {
          row[j] += z[c] * this.components[c][j];
        }
      }
      return row.map((v, j) => v + this.mean[j]);
    });
  }

  // ── explainedVarianceRatio ─────────────────────────────────────────────────
  //   rₖ = λₖ / Σⱼ λⱼ
  // Sum of all ratios ≤ 1. If you chose nComponents = p, the sum is exactly 1.
  explainedVarianceRatio(): number[] {
    const total = this.explainedVariance.reduce((s, v) => s + v, 0);
    if (total === 0) return this.explainedVariance.map(() => 0);
    return this.explainedVariance.map(v => v / total);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  // Build the [p×p] covariance matrix from a centered matrix Xc.
  private _covMatrix(Xc: number[][], n: number, p: number): number[][] {
    const cov: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (const row of Xc) {
      for (let i = 0; i < p; i++) {
        for (let j = i; j < p; j++) {
          cov[i][j] += row[i] * row[j];
        }
      }
    }
    for (let i = 0; i < p; i++) {
      cov[i][i] /= n;
      for (let j = i + 1; j < p; j++) {
        cov[i][j] /= n;
        cov[j][i] = cov[i][j]; // symmetric
      }
    }
    return cov;
  }

  // Power iteration: find the dominant eigenvector of a symmetric matrix.
  //   v ← M·v / ‖M·v‖   (repeated until ‖v_new - v_old‖ < tol)
  // Returns both the eigenvector (unit length) and its eigenvalue λ = vᵀ·M·v.
  private _powerIteration(
    M: number[][],
    p: number,
    maxIter = 1000,
    tol = 1e-10
  ): { eigenvector: number[]; eigenvalue: number } {
    // Initialize with a random unit vector
    let v = Array.from({ length: p }, () => Math.random() - 0.5);
    v = this._normalize(v);

    for (let iter = 0; iter < maxIter; iter++) {
      // Mv = M · v
      const Mv = this._matvec(M, v);
      const vNew = this._normalize(Mv);

      // Check convergence: |dot(v, vNew)| → 1 when they align
      const dot = v.reduce((s, vi, i) => s + vi * vNew[i], 0);
      v = vNew;
      if (Math.abs(Math.abs(dot) - 1) < tol) break;
    }

    // Eigenvalue: λ = vᵀ · M · v  (Rayleigh quotient)
    const Mv = this._matvec(M, v);
    const eigenvalue = v.reduce((s, vi, i) => s + vi * Mv[i], 0);

    return { eigenvector: v, eigenvalue: Math.max(0, eigenvalue) };
  }

  private _matvec(M: number[][], v: number[]): number[] {
    return M.map(row => row.reduce((s, mij, j) => s + mij * v[j], 0));
  }

  private _normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
    if (norm < 1e-14) return v; // zero vector — return as-is
    return v.map(vi => vi / norm);
  }
}
