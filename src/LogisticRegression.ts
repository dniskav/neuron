// ─── LOGISTIC REGRESSION ─────────────────────────────────────────────────────
//
// Binary classifier that applies a sigmoid to a linear combination of inputs:
//
//   z   = w · x + bias              pre-activation (linear score)
//   ŷ   = σ(z) = 1 / (1 + e⁻ᶻ)    predicted probability P(y=1|x)
//
// Loss — Binary Cross-Entropy (BCE):
//
//   L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
//
// The gradient of BCE with sigmoid has a beautiful closed form (the sigmoid
// and log cancel out), leaving a simple residual rule:
//
//   ∂L/∂w    = (ŷ − y) · x
//   ∂L/∂bias = (ŷ − y)
//
// Which gives the update:
//   w    ← w + lr · (y − ŷ) · x     (note: +, not −, because we absorb the sign)
//   bias ← bias + lr · (y − ŷ)
//
// Logistic Regression occupies the boundary between classical ML and neural
// networks: it is exactly a single neuron with sigmoid activation and BCE loss,
// without any hidden layers. Adding hidden layers and backpropagation turns it
// into a fully connected neural network.
//
// ─── MULTICLASS EXTENSION (Softmax) ──────────────────────────────────────────
//
// For K classes, each class gets its own weight vector wₖ.
// Softmax converts K raw scores into a probability distribution:
//
//   P(y=k|x) = exp(wₖ·x) / Σⱼ exp(wⱼ·x)
//
// Training uses categorical cross-entropy with the same residual form:
//   ∂L/∂wₖ = (P(y=k|x) − 1{y=k}) · x
//
// ─────────────────────────────────────────────────────────────────────────────

import { validateArray, validateDataset, validateNumber } from "./Validation";

// ─── Internal sigmoid ─────────────────────────────────────────────────────────
function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

// ─── Binary Cross-Entropy (single sample) ─────────────────────────────────────
//   L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
// Clipping ŷ to (eps, 1-eps) prevents log(0) = −Infinity.
function bce(target: number, pred: number): number {
  const eps = 1e-15;
  const p = Math.max(eps, Math.min(1 - eps, pred));
  return -(target * Math.log(p) + (1 - target) * Math.log(1 - p));
}

// ─── LogisticRegression ───────────────────────────────────────────────────────

export class LogisticRegression {
  weights: number[] = [];
  bias = 0;

  private _nFeatures = 0;

  // ─── Train ────────────────────────────────────────────────────────────────
  // Online SGD over the full dataset for `epochs` passes.
  // Updates are applied after each sample (stochastic gradient descent).
  // Continues from the existing weights if their length matches the data.
  //
  // Returns the mean BCE loss per epoch for convergence monitoring.

  train(
    X: number[][],
    y: number[],
    lr: number,
    epochs: number,
  ): number[] {
    if (X.length === 0) throw new Error("LogisticRegression.train: X is empty");
    if (X.length !== y.length) {
      throw new Error(
        `LogisticRegression.train: X has ${X.length} rows but y has ${y.length} labels`
      );
    }
    validateNumber(lr, "LogisticRegression.train");
    if (lr <= 0) throw new Error("LogisticRegression.train: lr must be positive");
    if (!Number.isInteger(epochs) || epochs <= 0) {
      throw new Error("LogisticRegression.train: epochs must be a positive integer");
    }

    // Validates X (rectangular, finite), y (finite) and y in [0, 1]; runs before any weight change.
    validateDataset(X, y, "LogisticRegression.train", { labels: "unit" });

    this._nFeatures = X[0].length;
    // Continue from the existing weights if their length matches; otherwise
    // initialise to small random values for symmetry breaking
    if (this.weights.length !== this._nFeatures) {
      const limit = Math.sqrt(2 / this._nFeatures);
      this.weights = Array.from(
        { length: this._nFeatures },
        () => (Math.random() * 2 - 1) * limit
      );
      this.bias = 0;
    }

    const lossHistory: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;

      for (let i = 0; i < X.length; i++) {
        const xi = X[i];
        const yi = y[i];

        // Forward: z = w·x + b, ŷ = σ(z)
        let z = this.bias;
        for (let j = 0; j < this._nFeatures; j++) z += this.weights[j] * xi[j];
        const yHat = sigmoid(z);

        epochLoss += bce(yi, yHat);

        // Gradient of BCE + sigmoid simplifies to: ∂L/∂z = ŷ − y
        // So update rule:  w ← w + lr·(y − ŷ)·x
        const delta = yi - yHat;   // = −∂L/∂z (gradient ascent form)
        for (let j = 0; j < this._nFeatures; j++) {
          this.weights[j] += lr * delta * xi[j];
        }
        this.bias += lr * delta;
      }

      lossHistory.push(epochLoss / X.length);
    }

    return lossHistory;
  }

  // ─── Predict (probability) ────────────────────────────────────────────────
  // Returns P(y=1|x) ∈ [0, 1]. The number of features is read from `weights`
  // (the public field), so weights assigned by hand without train() work.

  predict(x: number[]): number {
    if (this.weights.length === 0) {
      throw new Error("LogisticRegression.predict: model has not been trained yet");
    }
    validateArray(x, this.weights.length, "LogisticRegression.predict");

    let z = this.bias;
    for (let j = 0; j < this.weights.length; j++) z += this.weights[j] * x[j];
    return sigmoid(z);
  }

  // ─── Classify (hard label) ────────────────────────────────────────────────
  // Returns 0 or 1 using 0.5 as the decision threshold.

  classify(x: number[]): 0 | 1 {
    return this.predict(x) >= 0.5 ? 1 : 0;
  }
}

// ─── SoftmaxRegression (multiclass) ──────────────────────────────────────────
//
// Generalises logistic regression to K > 2 classes.
// Each class k has its own weight vector Wₖ ∈ ℝⁿ and bias bₖ.
//
//   scores[k] = Wₖ · x + bₖ           raw logits
//   P(y=k|x)  = exp(scores[k]) / Σⱼ exp(scores[j])   softmax
//
// Loss — Categorical Cross-Entropy:
//   L = −log P(y=trueClass|x)
//
// Gradient (same residual form as binary case):
//   ∂L/∂Wₖ = (P(y=k|x) − 1{y=k}) · x

export class SoftmaxRegression {
  // weights[k][j] = weight for class k, feature j
  weights: number[][] = [];
  // biases[k] = bias for class k
  biases: number[] = [];

  private _nFeatures = 0;
  private _nClasses = 0;

  // ─── Softmax helper ──────────────────────────────────────────────────────
  private _softmax(scores: number[]): number[] {
    // Subtract max for numerical stability: exp(x − max) avoids overflow
    const maxScore = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxScore));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  // ─── Train ────────────────────────────────────────────────────────────────
  // y must contain integer class labels 0..K-1. Continues from the existing
  // weights when their shape matches the data (a label >= K then throws);
  // otherwise initialises with K = max(y) + 1.
  // Returns mean cross-entropy loss per epoch.

  train(
    X: number[][],
    y: number[],
    lr: number,
    epochs: number,
  ): number[] {
    if (X.length === 0) throw new Error("SoftmaxRegression.train: X is empty");
    if (X.length !== y.length) {
      throw new Error(
        `SoftmaxRegression.train: X has ${X.length} rows but y has ${y.length} labels`
      );
    }
    validateNumber(lr, "SoftmaxRegression.train");
    if (lr <= 0) throw new Error("SoftmaxRegression.train: lr must be positive");
    if (!Number.isInteger(epochs) || epochs <= 0) {
      throw new Error("SoftmaxRegression.train: epochs must be a positive integer");
    }

    // Validates X, y (finite) and the labels (integers >= 0); runs before any weight change.
    validateDataset(X, y, "SoftmaxRegression.train", { labels: "integer" });
    let maxLabel = 0;
    for (let i = 0; i < y.length; i++) {
      if (y[i] > maxLabel) maxLabel = y[i];
    }

    const nFeatures = X[0].length;
    // The existing model is continued when its shape matches the data:
    // K rows of nFeatures weights and K biases.
    const K = this.weights.length;
    const shapeMatches =
      K > 0 &&
      this.biases.length === K &&
      this.weights.every(wk => wk.length === nFeatures);

    if (shapeMatches) {
      // The labels cannot grow an existing model
      for (let i = 0; i < y.length; i++) {
        if (y[i] >= K) {
          throw new Error(
            `SoftmaxRegression.train: label y[${i}] = ${y[i]} is >= ${K} classes of the existing model`
          );
        }
      }
      this._nFeatures = nFeatures;
      this._nClasses = K;
    } else {
      this._nFeatures = nFeatures;
      this._nClasses = maxLabel + 1;

      if (this._nClasses < 2) {
        throw new Error("SoftmaxRegression.train: need at least 2 classes in y");
      }

      // Initialise weights to small random values
      const limit = Math.sqrt(2 / this._nFeatures);
      this.weights = Array.from({ length: this._nClasses }, () =>
        Array.from({ length: this._nFeatures }, () => (Math.random() * 2 - 1) * limit)
      );
      this.biases = new Array(this._nClasses).fill(0);
    }

    const lossHistory: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;

      for (let i = 0; i < X.length; i++) {
        const xi = X[i];
        const trueClass = y[i];

        // Compute scores and softmax probabilities
        const scores = this.weights.map((wk, k) => {
          let s = this.biases[k];
          for (let j = 0; j < this._nFeatures; j++) s += wk[j] * xi[j];
          return s;
        });
        const probs = this._softmax(scores);

        // Cross-entropy loss: −log P(y=trueClass|x)
        epochLoss += -Math.log(Math.max(probs[trueClass], 1e-15));

        // Gradient: ∂L/∂Wₖ = (P(y=k|x) − 1{k=trueClass}) · x
        for (let k = 0; k < this._nClasses; k++) {
          const delta = probs[k] - (k === trueClass ? 1 : 0);
          for (let j = 0; j < this._nFeatures; j++) {
            this.weights[k][j] -= lr * delta * xi[j];
          }
          this.biases[k] -= lr * delta;
        }
      }

      lossHistory.push(epochLoss / X.length);
    }

    return lossHistory;
  }

  // ─── Predict (class probabilities) ───────────────────────────────────────
  // The shape is read from the public `weights` / `biases` (K rows of nFeatures,
  // K biases), so weights assigned by hand without train() work.
  predictProba(x: number[]): number[] {
    if (this.weights.length === 0) {
      throw new Error("SoftmaxRegression.predictProba: model has not been trained yet");
    }
    const nFeatures = this.weights[0].length;
    if (this.biases.length !== this.weights.length) {
      throw new Error(
        `SoftmaxRegression.predictProba: expected ${this.weights.length} biases, got ${this.biases.length}`
      );
    }
    this.weights.forEach((wk, k) => {
      if (wk.length !== nFeatures) {
        throw new Error(
          `SoftmaxRegression.predictProba: weights row ${k} has length ${wk.length}, expected ${nFeatures}`
        );
      }
    });
    validateArray(x, nFeatures, "SoftmaxRegression.predictProba");

    const scores = this.weights.map((wk, k) => {
      let s = this.biases[k];
      for (let j = 0; j < nFeatures; j++) s += wk[j] * x[j];
      return s;
    });
    return this._softmax(scores);
  }

  // ─── Classify (argmax) ────────────────────────────────────────────────────
  predict(x: number[]): number {
    const probs = this.predictProba(x);
    return probs.indexOf(Math.max(...probs));
  }
}
