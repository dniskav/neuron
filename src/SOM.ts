// ─── SELF-ORGANIZING MAP (Kohonen Map) ───────────────────────────────────────
//
// An unsupervised neural network that learns a low-dimensional (2D) discrete
// representation of high-dimensional input data while preserving topology:
// similar inputs map to nearby neurons on the grid.
//
// ─── ARCHITECTURE ────────────────────────────────────────────────────────────
//   Grid of rows × cols neurons.
//   Each neuron (r, c) has a weight vector w[r][c] ∈ ℝ^inputSize (a prototype).
//
// ─── BMU (Best Matching Unit) ────────────────────────────────────────────────
//   BMU(x) = argmin_{r,c} ‖x − w[r][c]‖    nearest prototype to the input x
//
// ─── GAUSSIAN NEIGHBORHOOD FUNCTION ─────────────────────────────────────────
//   h(BMU, n) = exp( −d²(BMU, n) / (2σ²) )
//   where d(BMU, n) is the Euclidean distance between two neurons on the grid.
//   σ (neighborhood radius) controls how wide the update bubble is.
//
// ─── WEIGHT UPDATE ───────────────────────────────────────────────────────────
//   w[r][c] ← w[r][c] + lr · h(BMU, (r,c)) · (x − w[r][c])
//   The BMU and its neighbors are pulled toward the input; distant neurons
//   receive only a tiny (or zero) update.
//
// ─── LEARNING RATE & SIGMA DECAY ─────────────────────────────────────────────
//   Both lr and σ decay exponentially from their initial to their final values
//   over the course of training:
//
//   lr(t) = lr₀ · (lr_f / lr₀)^(t / T)     exponential decay
//   σ(t)  = σ₀  · (σ_f  / σ₀ )^(t / T)
//
// ─── QUANTIZATION ERROR ──────────────────────────────────────────────────────
//   QE = (1/n) · Σᵢ ‖xᵢ − w[BMU(xᵢ)]‖     mean distance to nearest prototype
//   Lower QE = prototypes are closer to the data they represent.
//
// ─── TOPOLOGY PRESERVATION ───────────────────────────────────────────────────
//   A key property of SOMs: after training, points that are similar in input
//   space will activate neurons that are neighbors on the 2D grid. This makes
//   SOMs useful for visualization and exploratory data analysis.
//
// ─────────────────────────────────────────────────────────────────────────────

export interface SOMOptions {
  initialLr?: number;
  finalLr?: number;
  initialSigma?: number;
  finalSigma?: number;
}

export class SOM {
  /** Weight (prototype) vectors, shape [rows][cols][inputSize]. */
  weights: number[][][];

  private readonly _rows: number;
  private readonly _cols: number;
  private readonly _inputSize: number;
  private readonly _initialLr: number;
  private readonly _finalLr: number;
  private readonly _initialSigma: number;
  private readonly _finalSigma: number;

  constructor(
    rows: number,
    cols: number,
    inputSize: number,
    options: SOMOptions = {}
  ) {
    if (rows < 1 || cols < 1 || inputSize < 1) {
      throw new Error(
        `SOM: rows, cols and inputSize must be positive integers, got ${rows}×${cols}×${inputSize}`
      );
    }

    this._rows = rows;
    this._cols = cols;
    this._inputSize = inputSize;

    this._initialLr    = options.initialLr    ?? 0.5;
    this._finalLr      = options.finalLr      ?? 0.01;
    this._initialSigma = options.initialSigma ?? Math.max(rows, cols) / 2;
    this._finalSigma   = options.finalSigma   ?? 1.0;

    // Initialize weights uniformly in [0, 1)
    this.weights = Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () =>
        Array.from({ length: inputSize }, () => Math.random())
      )
    );
  }

  // ── train ──────────────────────────────────────────────────────────────────
  // Iterates over the dataset `epochs` times, presenting each sample and
  // performing a BMU search + neighborhood weight update.
  train(X: number[][], epochs: number): void {
    if (!X || X.length === 0) {
      throw new Error("SOM.train: dataset X must be non-empty");
    }
    if (X[0].length !== this._inputSize) {
      throw new Error(
        `SOM.train: expected input size ${this._inputSize}, got ${X[0].length}`
      );
    }

    const totalSteps = epochs * X.length;
    let step = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      // Shuffle each epoch for stochastic presentation
      const indices = this._shuffle(X.length);

      for (const idx of indices) {
        const x = X[idx];
        const t = step / totalSteps; // normalized time ∈ [0, 1)

        // Exponential decay of lr and σ
        const lr = this._initialLr * Math.pow(this._finalLr / this._initialLr, t);
        const sigma = this._initialSigma * Math.pow(this._finalSigma / this._initialSigma, t);
        const sigma2 = 2 * sigma * sigma;

        // Find BMU
        const [bmuR, bmuC] = this.getBMU(x);

        // Update all neurons according to their neighborhood distance
        for (let r = 0; r < this._rows; r++) {
          for (let c = 0; c < this._cols; c++) {
            const dr = r - bmuR;
            const dc = c - bmuC;
            const gridDistSq = dr * dr + dc * dc;

            // h = exp(−d² / 2σ²)   — Gaussian neighborhood
            const h = Math.exp(-gridDistSq / sigma2);
            if (h < 1e-6) continue; // negligible influence

            const w = this.weights[r][c];
            for (let i = 0; i < this._inputSize; i++) {
              w[i] += lr * h * (x[i] - w[i]);
            }
          }
        }

        step++;
      }
    }
  }

  // ── getBMU ─────────────────────────────────────────────────────────────────
  // Returns [row, col] of the Best Matching Unit for input x.
  //   BMU = argmin_{r,c} ‖x − w[r][c]‖²
  getBMU(x: number[]): [number, number] {
    if (x.length !== this._inputSize) {
      throw new Error(
        `SOM.getBMU: expected input of length ${this._inputSize}, got ${x.length}`
      );
    }

    let bestR = 0;
    let bestC = 0;
    let bestDist = Infinity;

    for (let r = 0; r < this._rows; r++) {
      for (let c = 0; c < this._cols; c++) {
        const dist = this._distSq(x, this.weights[r][c]);
        if (dist < bestDist) {
          bestDist = dist;
          bestR = r;
          bestC = c;
        }
      }
    }

    return [bestR, bestC];
  }

  // ── predict ────────────────────────────────────────────────────────────────
  // Alias for getBMU — returns [row, col] of the winning neuron.
  predict(x: number[]): [number, number] {
    return this.getBMU(x);
  }

  // ── quantizationError ─────────────────────────────────────────────────────
  //   QE = (1/n) · Σᵢ ‖xᵢ − w[BMU(xᵢ)]‖
  // Measures how well the prototypes represent the data. Lower is better.
  quantizationError(X: number[][]): number {
    let total = 0;
    for (const x of X) {
      const [r, c] = this.getBMU(x);
      total += Math.sqrt(this._distSq(x, this.weights[r][c]));
    }
    return total / X.length;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private _distSq(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return s;
  }

  // Fisher-Yates shuffle — returns an array of shuffled indices.
  private _shuffle(n: number): number[] {
    const arr = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}
