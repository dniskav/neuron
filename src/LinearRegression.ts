// ─── LINEAR REGRESSION ────────────────────────────────────────────────────────
//
// Models a linear relationship between n input features and a scalar output:
//
//   ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + bias
//     = [X | 1] · W           (augmented matrix form; bias = last element of W)
//
// Two fitting strategies are implemented:
//
// ── Normal Equation (exact, one-shot) ─────────────────────────────────────────
//
//   Minimises ‖y − Xw‖² analytically. Setting the gradient to zero yields:
//
//     ∂L/∂W = −2Xᵀ(y − Xw) = 0
//     →  W = (XᵀX)⁻¹Xᵀy
//
//   Complexity: O(n³) for the matrix inversion.  Best for small feature sets.
//   Fails (singular matrix) when features are linearly dependent or n > m.
//
// ── Gradient Descent (iterative) ─────────────────────────────────────────────
//
//   Loss:  L = (1/m) Σ (ŷᵢ − yᵢ)²         mean squared error
//   Grad:  ∂L/∂W = (2/m) Xᵀ(ŷ − y)
//   Update: W ← W − lr · ∂L/∂W
//
//   Works for any feature count and can be extended to mini-batches.
//
// ─────────────────────────────────────────────────────────────────────────────

import { validateNumber } from "./Validation";

// ─── Internal matrix utilities ────────────────────────────────────────────────
// Pure arithmetic on number[][]; no external dependencies.

/** Matrix multiply: A (m×k) · B (k×n) → (m×n) */
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;
  const C: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += A[i][p] * B[p][j];
      C[i][j] = sum;
    }
  }
  return C;
}

/** Transpose a matrix: (m×n) → (n×m) */
function transpose(A: number[][]): number[][] {
  const m = A.length;
  const n = A[0].length;
  const T: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

/**
 * Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
 * Returns null if the matrix is singular (determinant ≈ 0).
 */
function invertMatrix(M: number[][]): number[][] | null {
  const n = M.length;
  // Augment M with the identity matrix [M | I]
  const aug: number[][] = M.map((row, i) => {
    const id = new Array(n).fill(0);
    id[i] = 1;
    return [...row, ...id];
  });

  for (let col = 0; col < n; col++) {
    // Partial pivot — find row with max absolute value in this column
    let maxRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) return null;  // singular

    // Normalise pivot row
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;

    // Eliminate column in all other rows
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Extract right half (the inverse)
  return aug.map(row => row.slice(n));
}

/** Augment feature matrix X with a bias column of ones: [X | 1] */
function augment(X: number[][]): number[][] {
  return X.map(row => [...row, 1]);
}

// ─── LinearRegression ─────────────────────────────────────────────────────────

export class LinearRegression {
  // weights = [w₁, w₂, ..., wₙ, bias]
  weights: number[] = [];

  private _nFeatures = 0;

  // ─── Normal Equation ───────────────────────────────────────────────────────
  // W = (XᵀX)⁻¹Xᵀy   — exact solution in one matrix operation.
  // Augments X with a bias column so the bias is solved jointly.

  fitNormal(X: number[][], y: number[]): void {
    if (X.length === 0) throw new Error("LinearRegression.fitNormal: X is empty");
    if (X.length !== y.length) {
      throw new Error(
        `LinearRegression.fitNormal: X has ${X.length} rows but y has ${y.length} elements`
      );
    }

    this._nFeatures = X[0].length;
    const Xa = augment(X);                          // [m × (n+1)]
    const XaT = transpose(Xa);                      // [(n+1) × m]
    const XaTXa = matMul(XaT, Xa);                  // [(n+1) × (n+1)]
    const XaTXaInv = invertMatrix(XaTXa);

    if (XaTXaInv === null) {
      throw new Error(
        "LinearRegression.fitNormal: XᵀX is singular — features may be linearly dependent"
      );
    }

    // y as column vector [(m × 1)]
    const yCol = y.map(v => [v]);
    const XaTy = matMul(XaT, yCol);                 // [(n+1) × 1]
    const W = matMul(XaTXaInv, XaTy);               // [(n+1) × 1]

    this.weights = W.map(row => row[0]);
  }

  // ─── Gradient Descent ──────────────────────────────────────────────────────
  // Minimises MSE = (1/m) Σ (ŷᵢ − yᵢ)² iteratively.
  //
  //   ŷ  = Xa · W
  //   dW = (2/m) · Xaᵀ · (ŷ − y)
  //   W  ← W − lr · dW
  //
  // Returns the loss (MSE) at every epoch for convergence diagnostics.

  fitGD(
    X: number[][],
    y: number[],
    lr: number,
    epochs: number,
  ): number[] {
    if (X.length === 0) throw new Error("LinearRegression.fitGD: X is empty");
    if (X.length !== y.length) {
      throw new Error(
        `LinearRegression.fitGD: X has ${X.length} rows but y has ${y.length} elements`
      );
    }
    validateNumber(lr, "LinearRegression.fitGD");
    if (lr <= 0) throw new Error("LinearRegression.fitGD: lr must be positive");
    if (!Number.isInteger(epochs) || epochs <= 0) {
      throw new Error("LinearRegression.fitGD: epochs must be a positive integer");
    }

    this._nFeatures = X[0].length;
    const m = X.length;
    const Xa = augment(X);                          // [m × (n+1)]

    // Initialise weights to zero
    this.weights = new Array(this._nFeatures + 1).fill(0);

    const lossHistory: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      // ŷ = Xa · W
      const yHat = Xa.map(row =>
        row.reduce((s, x, j) => s + x * this.weights[j], 0)
      );

      // residuals: ŷ − y
      const residuals = yHat.map((yh, i) => yh - y[i]);

      // MSE loss: (1/m) Σ residualsᵢ²
      const mse = residuals.reduce((s, r) => s + r * r, 0) / m;
      lossHistory.push(mse);

      // Gradient: (2/m) · Xaᵀ · residuals
      // dW[j] = (2/m) Σᵢ Xa[i][j] · residuals[i]
      for (let j = 0; j < this.weights.length; j++) {
        let grad = 0;
        for (let i = 0; i < m; i++) {
          grad += Xa[i][j] * residuals[i];
        }
        this.weights[j] -= lr * (2 / m) * grad;
      }
    }

    return lossHistory;
  }

  // ─── Inference ─────────────────────────────────────────────────────────────
  // ŷ = Σ wᵢ·xᵢ + bias    (bias = weights[last])

  predict(x: number[]): number {
    if (this.weights.length === 0) {
      throw new Error("LinearRegression.predict: model has not been fitted yet");
    }
    if (x.length !== this._nFeatures) {
      throw new Error(
        `LinearRegression.predict: expected ${this._nFeatures} features, got ${x.length}`
      );
    }
    let out = this.weights[this._nFeatures]; // bias
    for (let i = 0; i < this._nFeatures; i++) {
      out += this.weights[i] * x[i];
    }
    return out;
  }

  // ─── Introspection ─────────────────────────────────────────────────────────
  getCoefficients(): { weights: number[]; bias: number } {
    if (this.weights.length === 0) {
      throw new Error(
        "LinearRegression.getCoefficients: model has not been fitted yet"
      );
    }
    return {
      weights: this.weights.slice(0, this._nFeatures),
      bias: this.weights[this._nFeatures],
    };
  }
}
