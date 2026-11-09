// ─── METRICS ──────────────────────────────────────────────────────────────────
//
// Evaluation metrics for classification, regression, and language models.
//
// ── Classification ────────────────────────────────────────────────────────────
//
//   Confusion Matrix (binary):
//     Rows = actual class, Columns = predicted class.
//     C[i][j] = number of samples with true label i predicted as j.
//
//   Precision = TP / (TP + FP)
//     Of everything predicted positive, how many are actually positive?
//
//   Recall (Sensitivity) = TP / (TP + FN)
//     Of everything actually positive, how many did we catch?
//
//   F1 = 2 · Precision · Recall / (Precision + Recall)
//     Harmonic mean — penalises imbalance between precision and recall.
//
//   Accuracy = (TP + TN) / N
//     Overall fraction correct. Misleading on imbalanced datasets.
//
//   ROC Curve — Receiver Operating Characteristic:
//     Plots TPR vs FPR at every possible classification threshold.
//     TPR = TP / (TP + FN),   FPR = FP / (FP + TN)
//     A perfect classifier: AUC = 1.  Random classifier: AUC = 0.5.
//
//   AUC — Area Under the ROC Curve (trapezoidal rule):
//     AUC = Σ (FPR[k+1] - FPR[k]) · (TPR[k+1] + TPR[k]) / 2
//
// ── Regression ────────────────────────────────────────────────────────────────
//
//   MAE  = (1/n) Σ |yᵢ - ŷᵢ|
//   RMSE = √[ (1/n) Σ (yᵢ - ŷᵢ)² ]
//   R²   = 1 − SS_res / SS_tot
//     SS_res = Σ(yᵢ - ŷᵢ)²,   SS_tot = Σ(yᵢ - ȳ)²
//     R² = 1: perfect fit.  R² = 0: as good as predicting the mean.  R² < 0: worse than mean.
//
// ── Language / Sequences ──────────────────────────────────────────────────────
//
//   Perplexity = exp( -1/T · Σ log P(wₜ | context) )
//     Geometric mean of the inverse probability the model assigned to each
//     true token. Lower is better. PP = 1 means perfect prediction.
//

// ─── Classification ───────────────────────────────────────────────────────────

// Builds a K×K confusion matrix.
// C[i][j] = # samples with true label i predicted as label j.
// Throws if the labels are not non-negative integers, or if numClasses is given
// and a label is ≥ numClasses (the matrix would have no row/column for it).
export function confusionMatrix(
  yTrue: number[],
  yPred: number[],
  numClasses?: number,
): number[][] {
  _assertPair('confusionMatrix', yTrue, yPred);
  _assertLabels('confusionMatrix', yTrue, yPred);
  let K: number;
  if (numClasses === undefined) {
    K = _maxLabel(yTrue, yPred) + 1;
  } else {
    if (!Number.isInteger(numClasses) || numClasses < 1) {
      throw new Error(`Metrics.confusionMatrix: numClasses must be a positive integer (got ${numClasses})`);
    }
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] >= numClasses || yPred[i] >= numClasses) {
        throw new Error(
          `Metrics.confusionMatrix: label at index ${i} is >= numClasses (${numClasses})`,
        );
      }
    }
    K = numClasses;
  }
  const matrix: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < yTrue.length; i++) {
    matrix[yTrue[i]][yPred[i]]++;
  }
  return matrix;
}

// Precision = TP / (TP + FP).
// For multi-class problems, computes macro-average over all classes.
export function precision(
  yTrue: number[],
  yPred: number[],
  positiveClass?: number,
): number {
  _assertPair('precision', yTrue, yPred);
  _assertLabels('precision', yTrue, yPred);
  if (positiveClass !== undefined) {
    _assertPositiveClass('precision', positiveClass);
    return _binaryPrecision(yTrue, yPred, positiveClass);
  }
  const K = _maxLabel(yTrue, yPred) + 1;
  let sum = 0;
  for (let c = 0; c < K; c++) sum += _binaryPrecision(yTrue, yPred, c);
  return sum / K;
}

// Recall = TP / (TP + FN).
// For multi-class problems, computes macro-average over all classes.
export function recall(
  yTrue: number[],
  yPred: number[],
  positiveClass?: number,
): number {
  _assertPair('recall', yTrue, yPred);
  _assertLabels('recall', yTrue, yPred);
  if (positiveClass !== undefined) {
    _assertPositiveClass('recall', positiveClass);
    return _binaryRecall(yTrue, yPred, positiveClass);
  }
  const K = _maxLabel(yTrue, yPred) + 1;
  let sum = 0;
  for (let c = 0; c < K; c++) sum += _binaryRecall(yTrue, yPred, c);
  return sum / K;
}

// F1 = 2·P·R / (P+R) — harmonic mean of precision and recall.
// With positiveClass: F1 of that class.
// Without it (macro): the MEAN of the per-class F1 values, as in sklearn
// average='macro'. Note this is NOT 2·P_macro·R_macro / (P_macro+R_macro).
export function f1Score(
  yTrue: number[],
  yPred: number[],
  positiveClass?: number,
): number {
  _assertPair('f1Score', yTrue, yPred);
  _assertLabels('f1Score', yTrue, yPred);
  if (positiveClass !== undefined) {
    _assertPositiveClass('f1Score', positiveClass);
    return _harmonicMean(
      _binaryPrecision(yTrue, yPred, positiveClass),
      _binaryRecall(yTrue, yPred, positiveClass),
    );
  }
  const K = _maxLabel(yTrue, yPred) + 1;
  let sum = 0;
  for (let c = 0; c < K; c++) {
    sum += _harmonicMean(_binaryPrecision(yTrue, yPred, c), _binaryRecall(yTrue, yPred, c));
  }
  return sum / K;
}

// Accuracy = fraction of correct predictions.
export function accuracy(yTrue: number[], yPred: number[]): number {
  _assertPair('accuracy', yTrue, yPred);
  const correct = yTrue.filter((y, i) => y === yPred[i]).length;
  return correct / yTrue.length;
}

// ROC curve: returns [{threshold, fpr, tpr}] sorted by ascending FPR.
// yScores are continuous scores (e.g. model probabilities for the positive class).
// yTrue should be binary (0 or 1).
export function rocCurve(
  yTrue: number[],
  yScores: number[],
): { fpr: number; tpr: number; threshold: number }[] {
  _assertPair('rocCurve', yTrue, yScores, 'yScores');
  // Collect unique thresholds, sorted descending so we sweep from strict to loose
  const thresholds = [...new Set(yScores)].sort((a, b) => b - a);
  // Add a threshold above max so the first point is (0, 0)
  thresholds.unshift(thresholds[0] + 1);

  const P = yTrue.filter((y) => y === 1).length;
  const N = yTrue.length - P;

  const points: { fpr: number; tpr: number; threshold: number }[] = [];

  for (const t of thresholds) {
    let tp = 0, fp = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const pred = yScores[i] >= t ? 1 : 0;
      if (pred === 1 && yTrue[i] === 1) tp++;
      if (pred === 1 && yTrue[i] === 0) fp++;
    }
    points.push({
      threshold: t,
      fpr: N > 0 ? fp / N : 0,
      tpr: P > 0 ? tp / P : 0,
    });
  }

  return points.sort((a, b) => a.fpr - b.fpr);
}

// AUC — area under the ROC curve via the trapezoidal rule.
// AUC = Σ (fpr[k+1] - fpr[k]) · (tpr[k+1] + tpr[k]) / 2
// Throws if yTrue has a single class: the ROC curve is undefined then (no positives or no negatives).
export function auc(yTrue: number[], yScores: number[]): number {
  _assertPair('auc', yTrue, yScores, 'yScores');
  if (yTrue.every((y) => y === yTrue[0])) {
    throw new Error('Metrics.auc: yTrue contains a single class; AUC needs both classes');
  }
  const curve = rocCurve(yTrue, yScores);
  let area = 0;
  for (let i = 1; i < curve.length; i++) {
    const dx = curve[i].fpr - curve[i - 1].fpr;
    const avgY = (curve[i].tpr + curve[i - 1].tpr) / 2;
    area += dx * avgY;
  }
  return Math.abs(area);  // absolute value in case FPR goes right-to-left
}

// ─── Regression ───────────────────────────────────────────────────────────────

// Mean Absolute Error: (1/n) Σ |yᵢ - ŷᵢ|
export function mae(yTrue: number[], yPred: number[]): number {
  _assertPair('mae', yTrue, yPred);
  return yTrue.reduce((s, y, i) => s + Math.abs(y - yPred[i]), 0) / yTrue.length;
}

// Root Mean Squared Error: √[ (1/n) Σ (yᵢ - ŷᵢ)² ]
export function rmse(yTrue: number[], yPred: number[]): number {
  _assertPair('rmse', yTrue, yPred);
  const mseVal = yTrue.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0) / yTrue.length;
  return Math.sqrt(mseVal);
}

// R² coefficient of determination.
// R² = 1 − SS_res / SS_tot
// SS_res = Σ(yᵢ - ŷᵢ)²,  SS_tot = Σ(yᵢ - ȳ)²
// When yTrue is constant, SS_tot = 0 and R² is undefined. Convention (sklearn):
// 1 if the prediction is perfect (SS_res = 0), 0 otherwise.
export function r2Score(yTrue: number[], yPred: number[]): number {
  _assertPair('r2Score', yTrue, yPred);
  const mean = yTrue.reduce((s, y) => s + y, 0) / yTrue.length;
  const ssTot = yTrue.reduce((s, y) => s + (y - mean) ** 2, 0);
  const ssRes = yTrue.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0);
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  return 1 - ssRes / ssTot;
}

// ─── Language / Sequences ─────────────────────────────────────────────────────

// Perplexity = exp( -1/T · Σ log P(wₜ | context) )
// yTrue: array of true token indices (0-indexed).
// probabilities: T×V matrix where probabilities[t][v] = P(token v at step t).
export function perplexity(yTrue: number[], probabilities: number[][]): number {
  const eps = 1e-15;
  const T = yTrue.length;
  let logSum = 0;
  for (let t = 0; t < T; t++) {
    const p = Math.max(eps, probabilities[t][yTrue[t]]);
    logSum += Math.log(p);
  }
  return Math.exp(-logSum / T);
}

// ─── Visualisation ────────────────────────────────────────────────────────────

// Prints the confusion matrix to the console in a readable grid.
export function printConfusionMatrix(matrix: number[][], labels?: string[]): void {
  const K = matrix.length;
  const lbs = labels ?? Array.from({ length: K }, (_, i) => String(i));
  const colW = Math.max(6, ...lbs.map((l) => l.length));

  const pad = (s: string, w: number): string => s.padStart(w);

  // Header row
  const header = pad('', colW) + '  ' + lbs.map((l) => pad(l, colW)).join('  ');
  console.log('');
  console.log('Confusion Matrix (rows = actual, cols = predicted):');
  console.log(header);
  console.log('─'.repeat(header.length));

  for (let i = 0; i < K; i++) {
    const row = pad(lbs[i], colW) + '  ' + matrix[i].map((v) => pad(String(v), colW)).join('  ');
    console.log(row);
  }
  console.log('');
}

// Prints precision, recall, F1, and support per class, plus macro averages.
export function classificationReport(
  yTrue: number[],
  yPred: number[],
  labels?: string[],
): void {
  _assertPair('classificationReport', yTrue, yPred);
  _assertLabels('classificationReport', yTrue, yPred);
  const K = _maxLabel(yTrue, yPred) + 1;
  const lbs = labels ?? Array.from({ length: K }, (_, i) => `class_${i}`);

  const rows: string[] = [];
  const colW = Math.max(10, ...lbs.map((l) => l.length));

  const fmt = (n: number): string => n.toFixed(4).padStart(10);
  const fmtI = (n: number): string => String(n).padStart(10);

  rows.push(
    'label'.padEnd(colW) + fmt(0).replace(/\d/g, ' ').replace('0.0000', 'precision') +
    fmt(0).replace(/\d/g, ' ').replace('0.0000', '   recall') +
    fmt(0).replace(/\d/g, ' ').replace('0.0000', ' f1-score') +
    fmtI(0).replace(/\d/g, ' ').replace('0', '   support'),
  );
  rows.push('─'.repeat(colW + 44));

  let pSum = 0, rSum = 0, f1Sum = 0;
  for (let c = 0; c < K; c++) {
    const p  = _binaryPrecision(yTrue, yPred, c);
    const r  = _binaryRecall(yTrue, yPred, c);
    const f1 = p + r > 0 ? (2 * p * r) / (p + r) : 0;
    const support = yTrue.filter((y) => y === c).length;
    pSum += p; rSum += r; f1Sum += f1;
    rows.push(lbs[c].padEnd(colW) + fmt(p) + fmt(r) + fmt(f1) + fmtI(support));
  }

  rows.push('─'.repeat(colW + 44));
  rows.push('macro avg'.padEnd(colW) + fmt(pSum / K) + fmt(rSum / K) + fmt(f1Sum / K) + fmtI(yTrue.length));

  console.log('');
  console.log('Classification Report:');
  rows.forEach((r) => console.log(r));
  console.log('');
}

// ─── Private Helpers ──────────────────────────────────────────────────────────

// Constitution V: invalid inputs throw with "Metrics.<function>: <reason>".
// Paired inputs (yTrue with yPred, or with yScores) must be non-empty and of equal length.
function _assertPair(fn: string, a: number[], b: number[], nameB = 'yPred'): void {
  if (a.length === 0) {
    throw new Error(`Metrics.${fn}: inputs must not be empty`);
  }
  if (a.length !== b.length) {
    throw new Error(`Metrics.${fn}: yTrue and ${nameB} must have the same length (${a.length} vs ${b.length})`);
  }
}

// Class labels must be non-negative integers (they index matrix rows and columns).
function _assertLabels(fn: string, yTrue: number[], yPred: number[]): void {
  _assertLabelArray(fn, 'yTrue', yTrue);
  _assertLabelArray(fn, 'yPred', yPred);
}

function _assertLabelArray(fn: string, name: string, labels: number[]): void {
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i];
    if (!Number.isInteger(v) || v < 0) {
      throw new Error(`Metrics.${fn}: ${name}[${i}] = ${v} is not a non-negative integer label`);
    }
  }
}

function _assertPositiveClass(fn: string, positiveClass: number): void {
  if (!Number.isInteger(positiveClass) || positiveClass < 0) {
    throw new Error(`Metrics.${fn}: positiveClass must be a non-negative integer (got ${positiveClass})`);
  }
}

// Largest label over both arrays. A loop, not Math.max(...arr): spreading 200 000
// elements as arguments overflows the call stack.
function _maxLabel(yTrue: number[], yPred: number[]): number {
  let max = 0;
  for (let i = 0; i < yTrue.length; i++) if (yTrue[i] > max) max = yTrue[i];
  for (let i = 0; i < yPred.length; i++) if (yPred[i] > max) max = yPred[i];
  return max;
}

// F1 from precision and recall; 0 when both are 0 (avoids 0/0).
function _harmonicMean(p: number, r: number): number {
  if (p + r === 0) return 0;
  return (2 * p * r) / (p + r);
}

function _binaryPrecision(yTrue: number[], yPred: number[], pos: number): number {
  let tp = 0, fp = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yPred[i] === pos && yTrue[i] === pos) tp++;
    if (yPred[i] === pos && yTrue[i] !== pos) fp++;
  }
  return tp + fp > 0 ? tp / (tp + fp) : 0;
}

function _binaryRecall(yTrue: number[], yPred: number[], pos: number): number {
  let tp = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === pos && yPred[i] === pos) tp++;
    if (yTrue[i] === pos && yPred[i] !== pos) fn++;
  }
  return tp + fn > 0 ? tp / (tp + fn) : 0;
}
