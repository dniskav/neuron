// ─── GAUSSIAN NAIVE BAYES ─────────────────────────────────────────────────────
//
// A generative probabilistic classifier. Instead of learning a decision
// boundary, it models the data distribution for each class and applies Bayes'
// theorem at inference time:
//
//   P(class|x) ∝ P(class) · P(x|class)          Bayes' theorem (∝ = proportional to)
//
// "Naive" assumption: features are conditionally independent given the class.
//   P(x|class) = ∏ᵢ P(xᵢ|class)                 one factor per feature
//
// "Gaussian" variant: each P(xᵢ|class) follows a normal distribution
// characterised by its per-feature, per-class mean (μ) and variance (σ²):
//
//   P(xᵢ | μ, σ²) = 1/√(2πσ²) · exp( −(xᵢ − μ)² / (2σ²) )
//
// ─── LOG-PROBABILITIES ────────────────────────────────────────────────────────
//
// With many features, the product ∏ P(xᵢ|class) underflows to 0 in
// floating-point arithmetic (numbers smaller than ~5×10⁻³²⁴ round to 0).
//
// Solution: work in log-space and ADD instead of multiply:
//   log P(class|x) ∝ log P(class) + Σᵢ log P(xᵢ|class)
//
// The class with the highest log-probability wins. No exp() needed for argmax.
// exp() is applied only in predictProba() to recover actual probabilities
// (using the log-sum-exp trick to avoid overflow there too).
//
// ─── TRAINING (fit) ──────────────────────────────────────────────────────────
//   For each class c and each feature j, compute:
//     μ[c][j]  = mean of Xᵢⱼ for all i where yᵢ = c
//     σ²[c][j] = variance of Xᵢⱼ for all i where yᵢ = c
//     π[c]     = P(class=c) = (count of class c) / m
//
// ─────────────────────────────────────────────────────────────────────────────

export class GaussianNaiveBayes {
  // Per-class, per-feature statistics
  private _means:     Map<number, number[]> = new Map();
  private _variances: Map<number, number[]> = new Map();
  // Log prior: log P(class)
  private _logPriors: Map<number, number>   = new Map();
  private _classes:   number[]              = [];
  private _nFeatures  = 0;

  // ─── Fit ───────────────────────────────────────────────────────────────────
  // Scans the data once to compute μ, σ², and π per class.
  // Variance is clamped to a minimum of 1e-9 to prevent division by zero
  // when a feature is perfectly constant within a class.

  fit(X: number[][], y: number[]): void {
    if (X.length === 0) throw new Error("GaussianNaiveBayes.fit: X is empty");
    if (X.length !== y.length) {
      throw new Error(
        `GaussianNaiveBayes.fit: X has ${X.length} rows but y has ${y.length} labels`
      );
    }

    this._nFeatures = X[0].length;
    const m = X.length;

    // Collect unique classes
    this._classes = [...new Set(y)].sort((a, b) => a - b);

    for (const c of this._classes) {
      // Rows belonging to class c
      const rows = X.filter((_, i) => y[i] === c);
      const count = rows.length;

      if (count === 0) continue;

      // Prior probability (log for numerical stability)
      this._logPriors.set(c, Math.log(count / m));

      // Per-feature mean: μⱼ = (1/n) Σ xᵢⱼ
      const means = new Array(this._nFeatures).fill(0);
      for (const row of rows) {
        for (let j = 0; j < this._nFeatures; j++) means[j] += row[j];
      }
      for (let j = 0; j < this._nFeatures; j++) means[j] /= count;
      this._means.set(c, means);

      // Per-feature variance: σ²ⱼ = (1/n) Σ (xᵢⱼ − μⱼ)²
      const variances = new Array(this._nFeatures).fill(0);
      for (const row of rows) {
        for (let j = 0; j < this._nFeatures; j++) {
          const diff = row[j] - means[j];
          variances[j] += diff * diff;
        }
      }
      for (let j = 0; j < this._nFeatures; j++) {
        // Clamp to 1e-9: avoids log(0) and division by zero in Gaussian PDF
        variances[j] = Math.max(variances[j] / count, 1e-9);
      }
      this._variances.set(c, variances);
    }
  }

  // ─── Log-likelihood of a single feature value under a Gaussian ─────────────
  // log P(x | μ, σ²) = −0.5·log(2πσ²) − (x−μ)² / (2σ²)
  private _logGaussian(x: number, mean: number, variance: number): number {
    return (
      -0.5 * Math.log(2 * Math.PI * variance) -
      ((x - mean) ** 2) / (2 * variance)
    );
  }

  // ─── Log-scores per class ────────────────────────────────────────────────
  // log P(class|x) ∝ log P(class) + Σⱼ log P(xⱼ|class)
  private _logScores(x: number[]): Map<number, number> {
    if (this._classes.length === 0) {
      throw new Error("GaussianNaiveBayes: model has not been fitted yet");
    }
    if (x.length !== this._nFeatures) {
      throw new Error(
        `GaussianNaiveBayes: expected ${this._nFeatures} features, got ${x.length}`
      );
    }

    const scores = new Map<number, number>();
    for (const c of this._classes) {
      const means     = this._means.get(c)!;
      const variances = this._variances.get(c)!;
      const logPrior  = this._logPriors.get(c)!;

      // Σⱼ log P(xⱼ | μ[c][j], σ²[c][j])
      let logLikelihood = 0;
      for (let j = 0; j < this._nFeatures; j++) {
        logLikelihood += this._logGaussian(x[j], means[j], variances[j]);
      }
      scores.set(c, logPrior + logLikelihood);
    }
    return scores;
  }

  // ─── Predict (argmax class) ──────────────────────────────────────────────
  // Returns the class with the highest log-posterior.
  // No exp() needed — argmax is order-preserving.

  predict(x: number[]): number {
    const scores = this._logScores(x);
    let bestClass = this._classes[0];
    let bestScore = -Infinity;
    for (const [c, s] of scores) {
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }
    return bestClass;
  }

  // ─── Predict probabilities ────────────────────────────────────────────────
  // Converts log-scores to actual probabilities using the log-sum-exp trick
  // to avoid numerical overflow:
  //
  //   log Σₖ exp(sₖ) = maxScore + log Σₖ exp(sₖ − maxScore)
  //
  // Then P(c|x) = exp(score[c] − log Σ exp).

  predictProba(x: number[]): Map<number, number> {
    const logScores = this._logScores(x);

    // log-sum-exp for normalisation
    const scores = [...logScores.values()];
    const maxScore = Math.max(...scores);
    const logSumExp = maxScore + Math.log(
      scores.reduce((sum, s) => sum + Math.exp(s - maxScore), 0)
    );

    const proba = new Map<number, number>();
    for (const [c, s] of logScores) {
      proba.set(c, Math.exp(s - logSumExp));
    }
    return proba;
  }
}
