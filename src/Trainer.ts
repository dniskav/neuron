// ─── Trainer ─────────────────────────────────────────────────────────────────
//
// High-level training loop for feedforward + recurrent networks.
// Handles epoch iteration, shuffling, learning rate decay, loss history,
// weight decay (L2), early stopping, optional restoration of the best weights,
// and classification metrics.
//
// Gradient clipping is NOT a Trainer option: Trainer only sees `train()`, so it
// cannot clip the gradient. Configure it where the gradient lives, in the
// optimizer (see ClippedOptimizerFactory).
//
// Options that cannot take effect THROW (they are never ignored silently):
// earlyStopping without setValidationData(), weightDecay/earlyStopping/
// restoreBestWeights on a network without getWeights/setWeights/predict,
// computeMetrics without predict(), an empty dataset, and invalid epochs
// (integer > 0), lr (finite > 0), lrDecay (in (0, 1]), weightDecay (finite >= 0),
// patience (integer >= 1) or minDelta (finite >= 0), and empty or mismatched
// validation data when earlyStopping is set. Errors are thrown before any
// network.train call.
// The only remaining console.warn is computeMetrics with targets that do not
// look like classification (a heuristic over the data).
//
// Notes:
//  - weightDecay multiplies EVERY weight returned by getWeights(), biases
//    included, by (1 - lr * weightDecay) before each sample.
//  - restoreBestWeights (default false) keeps getWeights() at each validation
//    improvement and restores it when training ends, by early stopping or by
//    reaching epochs. Optimizer state (Adam moments, etc.) is NOT restored, only
//    the weights. Restoring goes through setWeights(), which validates: a best
//    snapshot containing NaN/Infinity is not restored and setWeights' error
//    propagates (the network keeps its last-epoch weights). A model that
//    diverges to NaN AFTER its best epoch is restored normally.
//
// Usage:
//   const trainer = new Trainer(network, { epochs: 1000, lr: 0.1 })
//   trainer.train({ inputs, targets })
//   console.log(trainer.getHistory())
//
// With all features:
//   const trainer = new Trainer(network, {
//     epochs: 5000,
//     lr: 0.01,
//     weightDecay: 1e-4,
//     earlyStopping: { patience: 200, minDelta: 1e-4 },
//     computeMetrics: true,
//   })
//   trainer.setValidationData(valData)
//   trainer.train(trainData)
//   console.log(trainer.getMetrics())
//   console.log(trainer.getStopReason())
//
// ─────────────────────────────────────────────────────────────────────────────

import type { DataPair } from "./DataLoader";

// ── Interfaces ──────────────────────────────────────────────────────────────

export interface TrainMetrics {
  accuracy: number;
  precision: number;   // macro-average across classes
  recall: number;      // macro-average across classes
  f1: number;          // macro-average across classes
}

export interface TrainerOptions {
  epochs?: number;
  lr?: number;
  lrDecay?: number;       // multiply lr by this each epoch (e.g. 0.999)
  verbose?: boolean;      // print loss every 100 epochs
  weightDecay?: number;   // L2 regularization coefficient (0 = disabled)
  earlyStopping?: {       // stop early when validation loss stops improving
    patience: number;     // epochs with no improvement before stopping
    minDelta: number;     // minimum improvement required
  };
  computeMetrics?: boolean; // compute classification metrics each epoch
  restoreBestWeights?: boolean; // default false. Requires earlyStopping + validation data:
                                // on finish, restore the weights of the best validation epoch
}

export interface TrainDataset {
  inputs: number[][];
  targets: number[][];
}

export interface TrainableNetwork {
  train(inputs: number[], targets: number[], lr: number): number;
}

/** Extended interface for networks that support weight access and prediction.
 *  Required for weightDecay, earlyStopping, and computeMetrics features. */
export interface TrainableNetworkWithWeights extends TrainableNetwork {
  predict(inputs: number[]): number[];
  getWeights(): number[];
  setWeights(weights: number[]): void;
}

// ── Trainer ─────────────────────────────────────────────────────────────────

export class Trainer {
  readonly network: TrainableNetwork;
  readonly epochs: number;
  readonly lrInitial: number;
  readonly lrDecay: number;
  readonly verbose: boolean;
  readonly weightDecay: number;
  readonly restoreBestWeights: boolean;

  private _history: number[] = [];

  // Early stopping state
  private _validationData?: DataPair;
  private _earlyStopping?: { patience: number; minDelta: number };
  private _bestLoss: number = Infinity;
  private _patienceCounter: number = 0;
  private _stopReason: string = 'maxEpochs';

  // Metrics state
  private _computeMetrics: boolean;
  private _metrics: TrainMetrics[] = [];

  constructor(network: TrainableNetwork, options: TrainerOptions = {}) {
    this.network = network;
    this.epochs = options.epochs ?? 1000;
    this.lrInitial = options.lr ?? 0.1;
    this.lrDecay = options.lrDecay ?? 1.0;
    this.verbose = options.verbose ?? false;
    this.weightDecay = options.weightDecay ?? 0;
    this._earlyStopping = options.earlyStopping;
    this._computeMetrics = options.computeMetrics ?? false;
    this.restoreBestWeights = options.restoreBestWeights ?? false;
  }

  // ── Set external validation data (for early stopping) ────────────────────
  setValidationData(dataset: DataPair): void {
    if (dataset.inputs.length !== dataset.targets.length) {
      throw new Error(
        'Trainer.setValidationData: inputs and targets must have the same length'
      );
    }
    this._validationData = dataset;
  }

  // ── Get best validation loss during training ─────────────────────────────
  getBestLoss(): number {
    return this._bestLoss === Infinity ? -1 : this._bestLoss;
  }

  // ── Why did training stop? ───────────────────────────────────────────────
  getStopReason(): string {
    return this._stopReason;
  }

  // ── Get per-epoch classification metrics ─────────────────────────────────
  getMetrics(): TrainMetrics[] {
    return [...this._metrics];
  }

  // ── Train on dataset ──────────────────────────────────────────────────────
  train(dataset: TrainDataset): number[] {
    const { inputs, targets } = dataset;
    if (inputs.length !== targets.length) {
      throw new Error(
        'Trainer.train: inputs and targets must have the same length'
      );
    }

    this._validateOptions(inputs.length);

    const n = inputs.length;
    let lr = this.lrInitial;
    this._history = [];
    this._bestLoss = Infinity;
    this._patienceCounter = 0;
    this._stopReason = 'maxEpochs';
    this._metrics = [];

    // ── Feature gating (options were validated above; these cannot be null) ──
    const netExt = this._hasWeights(this.network);

    const canDecay = this.weightDecay > 0 && !!netExt;
    const canValidate =
      !!this._earlyStopping && !!netExt && !!this._validationData;
    const canMetric = this._computeMetrics;
    const isClass = canMetric && this._isClassification(targets);
    const predictor = this.network as unknown as { predict(i: number[]): number[] };
    let bestWeights: number[] | null = null;

    if (canMetric && !isClass) {
      console.warn(
        'Trainer: computeMetrics is set but targets do not appear to be ' +
        'one-hot or single-class. Metrics will be skipped.'
      );
    }

    // ── Epoch loop ────────────────────────────────────────────────────────
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Shuffle indices
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Train on each sample
      let epochLoss = 0;
      for (const i of indices) {
        // Apply weight decay before each training step (L2 equivalent)
        if (canDecay) {
          const w = netExt!.getWeights();
          for (let j = 0; j < w.length; j++) {
            w[j] *= 1 - lr * this.weightDecay;
          }
          netExt!.setWeights(w);
        }
        epochLoss += this.network.train(inputs[i], targets[i], lr);
      }
      epochLoss /= n;
      this._history.push(epochLoss);

      // Compute classification metrics on training data
      if (canMetric && isClass) {
        this._metrics.push(this._computeMetricsArray(predictor, inputs, targets));
      }

      // Early stopping: evaluate on validation set
      if (canValidate && this._validationData) {
        const valLoss = this._computeLoss(netExt!, this._validationData);
        const minDelta = this._earlyStopping!.minDelta;

        if (valLoss < this._bestLoss - minDelta) {
          this._bestLoss = valLoss;
          this._patienceCounter = 0;
          if (this.restoreBestWeights) bestWeights = [...netExt!.getWeights()];
        } else {
          this._patienceCounter++;
        }

        if (this._patienceCounter >= this._earlyStopping!.patience) {
          this._stopReason = 'earlyStopping';
          break;
        }
      }

      // Apply learning rate decay
      lr *= this.lrDecay;

      if (this.verbose && (epoch + 1) % 100 === 0) {
        console.log(
          `Epoch ${epoch + 1}/${this.epochs}, loss: ${epochLoss.toFixed(6)}, lr: ${lr.toFixed(6)}`
        );
      }
    }

    // Restore the best weights (also when the loop ended by maxEpochs)
    if (this.restoreBestWeights && bestWeights) {
      netExt!.setWeights(bestWeights);
    }

    return this._history;
  }

  // ── Get loss history ──────────────────────────────────────────────────────
  getHistory(): number[] {
    return [...this._history];
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /** Validates options + dataset. Throws before any network.train call. */
  private _validateOptions(nSamples: number): void {
    if (nSamples === 0) {
      throw new Error('Trainer.train: dataset is empty (inputs and targets have length 0)');
    }
    const fail = (msg: string): never => {
      throw new Error(`Trainer.train: ${msg}`);
    };
    const finite = (v: unknown): v is number =>
      typeof v === 'number' && Number.isFinite(v);

    if (!(Number.isInteger(this.epochs) && this.epochs > 0)) {
      fail(`epochs must be an integer > 0, got ${this.epochs}`);
    }
    if (!(finite(this.lrInitial) && this.lrInitial > 0)) {
      fail(`lr must be a finite number > 0, got ${this.lrInitial}`);
    }
    if (!(finite(this.lrDecay) && this.lrDecay > 0 && this.lrDecay <= 1)) {
      fail(`lrDecay must be in (0, 1], got ${this.lrDecay}`);
    }
    if (!(finite(this.weightDecay) && this.weightDecay >= 0)) {
      fail(`weightDecay must be a finite number >= 0, got ${this.weightDecay}`);
    }
    const es = this._earlyStopping;
    if (es && !(Number.isInteger(es.patience) && es.patience >= 1)) {
      fail(`earlyStopping.patience must be an integer >= 1, got ${es.patience}`);
    }
    if (es && !(finite(es.minDelta) && es.minDelta >= 0)) {
      fail(`earlyStopping.minDelta must be a finite number >= 0, got ${es.minDelta}`);
    }

    const netExt = this._hasWeights(this.network);
    const needs = 'requires a network with getWeights/setWeights/predict';
    if (this.weightDecay > 0 && !netExt) {
      fail(`weightDecay ${needs}`);
    }
    if (es && !netExt) {
      fail(`earlyStopping ${needs}`);
    }
    if (es && !this._validationData) {
      fail('earlyStopping requires validation data: call setValidationData() before train()');
    }
    // Checked here, not in setValidationData, so that a validation set given to a
    // Trainer without earlyStopping keeps its current behaviour. (Mismatched
    // lengths are already rejected by setValidationData itself.)
    if (es && this._validationData && this._validationData.inputs.length === 0) {
      fail('earlyStopping validation data is empty: setValidationData() needs at least one sample');
    }
    if (this.restoreBestWeights && !es) {
      fail('restoreBestWeights requires earlyStopping (and validation data)');
    }
    if (
      this._computeMetrics &&
      typeof (this.network as { predict?: unknown }).predict !== 'function'
    ) {
      fail('computeMetrics requires a network with predict()');
    }
  }

  /** Type guard: does this network support getWeights/setWeights/predict? */
  private _hasWeights(
    network: TrainableNetwork
  ): TrainableNetworkWithWeights | null {
    if (
      'getWeights' in network &&
      'setWeights' in network &&
      'predict' in network &&
      typeof (network as any).getWeights === 'function' &&
      typeof (network as any).setWeights === 'function' &&
      typeof (network as any).predict === 'function'
    ) {
      return network as unknown as TrainableNetworkWithWeights;
    }
    return null;
  }

  /** Mean squared error on a dataset (used for validation loss). */
  private _computeLoss(
    net: TrainableNetworkWithWeights,
    data: DataPair
  ): number {
    let totalLoss = 0;
    for (let i = 0; i < data.inputs.length; i++) {
      const pred = net.predict(data.inputs[i]);
      const target = data.targets[i];
      let sampleLoss = 0;
      for (let j = 0; j < pred.length; j++) {
        sampleLoss += (target[j] - pred[j]) ** 2;
      }
      totalLoss += sampleLoss / pred.length;
    }
    return totalLoss / data.inputs.length;
  }

  /** Heuristic: are targets classification-style (one-hot or single-class)? */
  private _isClassification(targets: number[][]): boolean {
    if (targets.length === 0) return false;
    const first = targets[0];

    // Single-element targets → binary classification
    if (first.length === 1) return true;

    // Multi-element: check one-hot pattern (sum ≈ 1, all values ≈ 0 or 1)
    for (const t of targets) {
      let sum = 0;
      for (const v of t) {
        sum += v;
        // Allow small floating point noise
        if (v < -0.01 || (v > 0.01 && v < 0.99 && Math.abs(v - 1) > 0.01))
          return false;
      }
      if (Math.abs(sum - 1) > 0.01) return false;
    }
    return true;
  }

  /** Compute classification metrics from predictions vs targets. */
  private _computeMetricsArray(
    net: { predict(inputs: number[]): number[] },
    inputs: number[][],
    targets: number[][]
  ): TrainMetrics {
    const targetLen = targets[0].length;
    const nClasses = targetLen === 1 ? 2 : targetLen;

    // Confusion matrix: confusion[trueClass][predClass]
    const confusion: number[][] = Array.from({ length: nClasses }, () =>
      Array(nClasses).fill(0)
    );

    for (let i = 0; i < inputs.length; i++) {
      const pred = net.predict(inputs[i]);
      const target = targets[i];

      let predClass: number;
      let trueClass: number;

      if (targetLen === 1) {
        // Binary classification with single-element targets
        trueClass = target[0] >= 0.5 ? 1 : 0;
        // Output: if single, threshold at 0.5; if multi, argmax
        if (pred.length === 1) {
          predClass = pred[0] >= 0.5 ? 1 : 0;
        } else {
          predClass = pred.indexOf(Math.max(...pred));
        }
      } else {
        // One-hot: argmax for both
        predClass = pred.indexOf(Math.max(...pred));
        trueClass = target.indexOf(Math.max(...target));
      }

      // Clamp to valid range
      predClass = Math.max(0, Math.min(nClasses - 1, predClass));
      trueClass = Math.max(0, Math.min(nClasses - 1, trueClass));

      confusion[trueClass][predClass]++;
    }

    // Compute per-class precision/recall → macro averages
    let totalCorrect = 0;
    let totalSamples = 0;
    const precisions: number[] = [];
    const recalls: number[] = [];

    for (let c = 0; c < nClasses; c++) {
      const tp = confusion[c][c];
      totalCorrect += tp;

      let colSum = 0;
      let rowSum = 0;
      for (let r = 0; r < nClasses; r++) {
        colSum += confusion[r][c];  // predicted as class c
        rowSum += confusion[c][r];  // true class c
      }
      totalSamples += rowSum;

      precisions.push(colSum > 0 ? tp / colSum : 0);
      recalls.push(rowSum > 0 ? tp / rowSum : 0);
    }

    const accuracy = totalSamples > 0 ? totalCorrect / totalSamples : 0;
    const macroPrecision =
      precisions.reduce((a, b) => a + b, 0) / nClasses;
    const macroRecall =
      recalls.reduce((a, b) => a + b, 0) / nClasses;
    const f1 =
      macroPrecision + macroRecall > 0
        ? (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall)
        : 0;

    return {
      accuracy,
      precision: macroPrecision,
      recall: macroRecall,
      f1,
    };
  }
}
