// ─── DATA AUGMENTATION ────────────────────────────────────────────────────────
//
// Utilities for data preprocessing and augmentation of numeric feature vectors.
// Augmenting the training set helps prevent overfitting by exposing the network
// to slightly varied versions of the same examples, improving generalisation.
//
// ── Noise / Jitter ────────────────────────────────────────────────────────────
//
//   Gaussian noise:  x'ᵢ = xᵢ + εᵢ,  εᵢ ~ N(0, σ²)
//     Good for sensor data, images, and tabular features.
//     σ controls the perturbation magnitude — keep it small (0.01–0.05).
//
//   Jitter (uniform): x'ᵢ = xᵢ + uᵢ,  uᵢ ~ U(-δ, δ)
//     Simpler alternative when the noise distribution is unknown.
//
// ── Normalisation ─────────────────────────────────────────────────────────────
//
//   Min-Max:    x' = (x - min) / (max - min)    →  x' ∈ [0, 1]
//     Preserves the shape of the distribution. Sensitive to outliers.
//
//   Z-Score:    x' = (x - μ) / σ                →  x' ~ N(0, 1)
//     Standard when features have different scales or units.
//     Robust to outliers; recommended before using gradient-based optimisers.
//
// ── Data Split ────────────────────────────────────────────────────────────────
//
//   Standard split: 70% train / 15% val / 15% test (configurable).
//   Always shuffle before splitting to avoid temporal/ordering bias.
//

// ─── DataAugmentation ────────────────────────────────────────────────────────

export class DataAugmentation {
  // ── Noise / Perturbation ─────────────────────────────────────────────────

  // Adds independent Gaussian noise to each feature: x'ᵢ = xᵢ + N(0, σ²).
  // sigma: standard deviation of the noise (default: 0.01).
  static addNoise(x: number[], sigma = 0.01): number[] {
    return x.map((v) => v + _sampleNormal() * sigma);
  }

  // Uniform jitter: x'ᵢ = xᵢ + U(-delta, delta).
  // delta: half-width of the uniform perturbation (default: 0.01).
  static jitter(x: number[], delta = 0.01): number[] {
    return x.map((v) => v + (Math.random() * 2 - 1) * delta);
  }

  // Reverses the order of the feature vector.
  // Useful when features are symmetric or represent a temporal window.
  static flipHorizontal(x: number[]): number[] {
    return [...x].reverse();
  }

  // ── Normalisation ────────────────────────────────────────────────────────

  // Min-Max normalisation fitted on a dataset X.
  // Returns the normalised data and the per-feature min/max arrays.
  // Use normalizePoint() to apply the same transform to new samples.
  static normalize(X: number[][]): {
    normalized: number[][];
    min: number[];
    max: number[];
  } {
    if (X.length === 0) return { normalized: [], min: [], max: [] };
    const d   = X[0].length;
    const min = new Array<number>(d).fill(Infinity);
    const max = new Array<number>(d).fill(-Infinity);

    for (const row of X) {
      for (let j = 0; j < d; j++) {
        if (row[j] < min[j]) min[j] = row[j];
        if (row[j] > max[j]) max[j] = row[j];
      }
    }

    const normalized = X.map((row) => DataAugmentation.normalizePoint(row, min, max));
    return { normalized, min, max };
  }

  // Applies pre-computed min/max normalisation to a single sample.
  // Handles constant features (min === max) by mapping to 0.
  static normalizePoint(x: number[], min: number[], max: number[]): number[] {
    return x.map((v, j) => {
      const range = max[j] - min[j];
      return range === 0 ? 0 : (v - min[j]) / range;
    });
  }

  // Z-Score standardisation fitted on a dataset X.
  // Returns the standardised data and per-feature mean/std arrays.
  // Use standardizePoint() to apply the same transform to new samples.
  static standardize(X: number[][]): {
    standardized: number[][];
    mean: number[];
    std: number[];
  } {
    if (X.length === 0) return { standardized: [], mean: [], std: [] };
    const n = X.length;
    const d = X[0].length;

    const mean = new Array<number>(d).fill(0);
    for (const row of X) {
      for (let j = 0; j < d; j++) mean[j] += row[j];
    }
    for (let j = 0; j < d; j++) mean[j] /= n;

    const variance = new Array<number>(d).fill(0);
    for (const row of X) {
      for (let j = 0; j < d; j++) variance[j] += (row[j] - mean[j]) ** 2;
    }
    const std = variance.map((v) => Math.sqrt(v / n));

    const standardized = X.map((row) =>
      DataAugmentation.standardizePoint(row, mean, std),
    );
    return { standardized, mean, std };
  }

  // Applies pre-computed z-score standardisation to a single sample.
  // Constant features (std === 0) are mapped to 0.
  static standardizePoint(x: number[], mean: number[], std: number[]): number[] {
    return x.map((v, j) => (std[j] === 0 ? 0 : (v - mean[j]) / std[j]));
  }

  // ── Batch Augmentation ───────────────────────────────────────────────────

  // Generates `factor` noisy copies of each sample in X.
  // The original samples are included in the output (at factor = 1 the output
  // equals the input; at factor = 3 the dataset triples in size).
  // sigma: noise std dev (default: 0.01).
  static augmentBatch(
    X: number[][],
    y: number[],
    factor = 2,
    sigma = 0.01,
  ): { X: number[][]; y: number[] } {
    const augX: number[][] = [];
    const augY: number[]   = [];

    for (let i = 0; i < X.length; i++) {
      // Always include the original sample
      augX.push([...X[i]]);
      augY.push(y[i]);
      // Generate (factor - 1) noisy copies
      for (let k = 1; k < factor; k++) {
        augX.push(DataAugmentation.addNoise(X[i], sigma));
        augY.push(y[i]);
      }
    }

    return { X: augX, y: augY };
  }

  // ── Shuffle ──────────────────────────────────────────────────────────────

  // Fisher-Yates shuffle — in-place permutation of indices.
  // Returns new arrays (does not mutate the inputs).
  static shuffle(X: number[][], y: number[]): { X: number[][]; y: number[] } {
    const indices = Array.from({ length: X.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return {
      X: indices.map((i) => X[i]),
      y: indices.map((i) => y[i]),
    };
  }

  // ── Train / Val / Test Split ─────────────────────────────────────────────
  //
  // Splits the dataset into three non-overlapping partitions.
  // trainRatio + valRatio must be < 1.0; the remainder goes to test.
  // Shuffles automatically before splitting.
  //
  // Default split: 70% / 15% / 15%.
  //
  static split(
    X: number[][],
    y: number[],
    trainRatio = 0.7,
    valRatio   = 0.15,
  ): {
    trainX: number[][]; trainY: number[];
    valX:   number[][]; valY:   number[];
    testX:  number[][]; testY:  number[];
  } {
    if (trainRatio + valRatio >= 1) {
      throw new Error(
        `trainRatio (${trainRatio}) + valRatio (${valRatio}) must be < 1`,
      );
    }

    const { X: sX, y: sY } = DataAugmentation.shuffle(X, y);
    const n         = sX.length;
    const trainEnd  = Math.floor(n * trainRatio);
    const valEnd    = trainEnd + Math.floor(n * valRatio);

    return {
      trainX: sX.slice(0, trainEnd),
      trainY: sY.slice(0, trainEnd),
      valX:   sX.slice(trainEnd, valEnd),
      valY:   sY.slice(trainEnd, valEnd),
      testX:  sX.slice(valEnd),
      testY:  sY.slice(valEnd),
    };
  }
}

// ─── Private Helper ───────────────────────────────────────────────────────────

// Box-Muller transform: one sample from N(0, 1).
function _sampleNormal(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
}
