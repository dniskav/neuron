// ─── DECISION TREE (CART) ────────────────────────────────────────────────────
//
// Classification and Regression Trees (Breiman et al., 1984).
// Recursively partitions the feature space with axis-aligned splits.
//
// At each internal node the algorithm:
//   1. Iterates over every feature j and every unique threshold t.
//   2. Splits the samples into left (xⱼ ≤ t) and right (xⱼ > t) subsets.
//   3. Picks the split that minimises the impurity criterion.
//
// ─── GINI IMPURITY (classification) ─────────────────────────────────────────
//
//   G = 1 − Σₖ pₖ²
//
//   where pₖ = fraction of samples belonging to class k in the node.
//
//   G = 0  →  perfect node: all samples belong to one class (leaf is pure).
//   G = 0.5 → maximally impure binary node (50/50 split between 2 classes).
//
//   The weighted impurity of a split is:
//     IG = (|left|/|total|)·G(left) + (|right|/|total|)·G(right)
//
//   We maximise the information gain = G(parent) − IG(split).
//   Equivalently, we minimise IG(split).
//
// ─── VARIANCE / MSE (regression) ─────────────────────────────────────────────
//
//   MSE = (1/n) Σ (yᵢ − ȳ)²
//
//   The prediction at a leaf is ȳ (mean of samples reaching that leaf).
//   We minimise the weighted MSE of the split, same weighted formula as Gini.
//
// ─── STOPPING CONDITIONS ──────────────────────────────────────────────────────
//   - Node depth reaches maxDepth
//   - Node has fewer than minSamplesSplit samples
//   - All samples have the same label (Gini = 0)
//   - No split improves the impurity (pure node or single unique value)
//
// ─────────────────────────────────────────────────────────────────────────────

// ─── Node types ───────────────────────────────────────────────────────────────

interface LeafNode {
  isLeaf: true;
  value: number;    // majority class (classification) or mean (regression)
}

interface SplitNode {
  isLeaf: false;
  featureIndex: number;    // which feature to split on
  threshold: number;       // split point: left if x[j] ≤ threshold
  left:  TreeNode;
  right: TreeNode;
}

type TreeNode = LeafNode | SplitNode;

// ─── DecisionTree ─────────────────────────────────────────────────────────────

export class DecisionTree {
  private _root: TreeNode | null = null;
  private _maxDepth: number;
  private _minSamplesSplit: number;
  private _task: "classification" | "regression";

  constructor(options?: {
    maxDepth?: number;
    minSamplesSplit?: number;
    task?: "classification" | "regression";
  }) {
    this._maxDepth       = options?.maxDepth       ?? 10;
    this._minSamplesSplit = options?.minSamplesSplit ?? 2;
    this._task            = options?.task            ?? "classification";

    if (this._maxDepth <= 0) {
      throw new Error("DecisionTree: maxDepth must be positive");
    }
    if (this._minSamplesSplit < 2) {
      throw new Error("DecisionTree: minSamplesSplit must be at least 2");
    }
  }

  // ─── Gini impurity ─────────────────────────────────────────────────────────
  // G = 1 − Σₖ pₖ²
  // G = 0 when all samples share one class (perfectly pure node).
  // G ≈ 0.5 for a binary node with equal class distribution.
  private _gini(y: number[]): number {
    if (y.length === 0) return 0;
    const counts = new Map<number, number>();
    for (const label of y) counts.set(label, (counts.get(label) ?? 0) + 1);
    let g = 1;
    for (const count of counts.values()) {
      const p = count / y.length;
      g -= p * p;
    }
    return g;
  }

  // ─── Mean Squared Error (regression impurity) ─────────────────────────────
  // MSE = (1/n) Σ (yᵢ − ȳ)²
  private _mse(y: number[]): number {
    if (y.length === 0) return 0;
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    return y.reduce((acc, v) => acc + (v - mean) ** 2, 0) / y.length;
  }

  // ─── Impurity selector ─────────────────────────────────────────────────────
  private _impurity(y: number[]): number {
    return this._task === "classification" ? this._gini(y) : this._mse(y);
  }

  // ─── Leaf value ────────────────────────────────────────────────────────────
  // Classification: majority class. Regression: mean.
  private _leafValue(y: number[]): number {
    if (this._task === "regression") {
      return y.reduce((a, b) => a + b, 0) / y.length;
    }
    // Majority vote
    const counts = new Map<number, number>();
    for (const label of y) counts.set(label, (counts.get(label) ?? 0) + 1);
    let bestClass = y[0];
    let bestCount = 0;
    for (const [cls, cnt] of counts) {
      if (cnt > bestCount) {
        bestCount = cnt;
        bestClass = cls;
      }
    }
    return bestClass;
  }

  // ─── Best split search ─────────────────────────────────────────────────────
  // Brute-force: try every feature × every unique threshold.
  // Returns the split that minimises weighted impurity (or null if none helps).
  private _bestSplit(
    X: number[][],
    y: number[],
  ): { featureIndex: number; threshold: number } | null {
    const nFeatures = X[0].length;
    const n = y.length;
    let bestImpurity = Infinity;
    let bestSplit: { featureIndex: number; threshold: number } | null = null;

    const parentImpurity = this._impurity(y);

    for (let j = 0; j < nFeatures; j++) {
      // Unique values in feature j (sorted) → candidate thresholds are midpoints
      const values = [...new Set(X.map(row => row[j]))].sort((a, b) => a - b);

      for (let vi = 0; vi < values.length - 1; vi++) {
        // Midpoint between consecutive unique values
        const threshold = (values[vi] + values[vi + 1]) / 2;

        const leftY:  number[] = [];
        const rightY: number[] = [];
        for (let i = 0; i < n; i++) {
          if (X[i][j] <= threshold) leftY.push(y[i]);
          else rightY.push(y[i]);
        }

        if (leftY.length === 0 || rightY.length === 0) continue;

        // Weighted impurity of this split
        const weightedImpurity =
          (leftY.length  / n) * this._impurity(leftY) +
          (rightY.length / n) * this._impurity(rightY);

        // Only consider splits that strictly improve over the parent
        if (weightedImpurity < bestImpurity && weightedImpurity < parentImpurity) {
          bestImpurity = weightedImpurity;
          bestSplit = { featureIndex: j, threshold };
        }
      }
    }

    return bestSplit;
  }

  // ─── Recursive tree builder ────────────────────────────────────────────────
  private _buildNode(X: number[][], y: number[], depth: number): TreeNode {
    // Stopping conditions → leaf
    const allSame = y.every(v => v === y[0]);
    if (
      depth >= this._maxDepth ||
      y.length < this._minSamplesSplit ||
      allSame
    ) {
      return { isLeaf: true, value: this._leafValue(y) };
    }

    const split = this._bestSplit(X, y);
    if (split === null) {
      // No beneficial split found → leaf
      return { isLeaf: true, value: this._leafValue(y) };
    }

    const { featureIndex, threshold } = split;

    const leftX:  number[][] = [];
    const leftY:  number[]   = [];
    const rightX: number[][] = [];
    const rightY: number[]   = [];

    for (let i = 0; i < y.length; i++) {
      if (X[i][featureIndex] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    // Recursively build subtrees
    return {
      isLeaf: false,
      featureIndex,
      threshold,
      left:  this._buildNode(leftX,  leftY,  depth + 1),
      right: this._buildNode(rightX, rightY, depth + 1),
    };
  }

  // ─── Fit ──────────────────────────────────────────────────────────────────
  fit(X: number[][], y: number[]): void {
    if (X.length === 0) throw new Error("DecisionTree.fit: X is empty");
    if (X.length !== y.length) {
      throw new Error(
        `DecisionTree.fit: X has ${X.length} rows but y has ${y.length} labels`
      );
    }
    this._root = this._buildNode(X, y, 0);
  }

  // ─── Traverse a single sample ─────────────────────────────────────────────
  private _traverse(node: TreeNode, x: number[]): number {
    if (node.isLeaf) return node.value;
    if (x[node.featureIndex] <= node.threshold) {
      return this._traverse(node.left, x);
    }
    return this._traverse(node.right, x);
  }

  // ─── Predict single sample ────────────────────────────────────────────────
  predict(x: number[]): number {
    if (this._root === null) {
      throw new Error("DecisionTree.predict: model has not been fitted yet");
    }
    return this._traverse(this._root, x);
  }

  // ─── Predict batch ────────────────────────────────────────────────────────
  predictBatch(X: number[][]): number[] {
    return X.map(x => this.predict(x));
  }
}
