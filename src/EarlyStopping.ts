// ─── EARLY STOPPING ───────────────────────────────────────────────────────────
//
// Monitors a validation metric and stops training when it ceases to improve,
// preventing overfitting and wasted compute.
//
// Algorithm:
//   After each epoch, compare the new metric value to `bestValue`.
//   If the improvement exceeds `minDelta`, reset the patience counter and
//   record the new best. Otherwise, increment the counter.
//   When counter ≥ patience, set `stopped = true` — the caller should break
//   out of its training loop.
//
// Mode:
//   'min' — for metrics that should decrease (loss, MAE, perplexity …)
//   'max' — for metrics that should increase (accuracy, AUC, R² …)
//
// restoreBest:
//   When true, `EarlyStopping` saves the best weights snapshot (as a flat
//   number[] via NetworkN.getWeights / setWeights) and exposes it via
//   `bestWeights`. The caller is responsible for applying them:
//
//     if (es.stopped && es.bestWeights) {
//       net.setWeights(es.bestWeights);
//     }
//
// Usage:
//
//   const es = new EarlyStopping({ patience: 15, mode: 'min', minDelta: 1e-4 });
//
//   for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
//     const valLoss = trainEpoch(...);
//     if (es.update(valLoss, epoch)) break;
//   }
//

// ─── EarlyStopping ────────────────────────────────────────────────────────────

export class EarlyStopping {
  // Current best value of the monitored metric.
  bestValue: number;
  // How many epochs to wait for improvement before stopping.
  readonly patience: number;
  // Minimum change in monitored metric to qualify as an improvement.
  readonly minDelta: number;
  // 'min' for loss-like metrics, 'max' for accuracy-like metrics.
  readonly mode: 'min' | 'max';
  // Whether to save and expose the weights at the best epoch.
  readonly restoreBest: boolean;

  // Number of epochs since the last improvement.
  counter: number;
  // Whether training should stop.
  stopped: boolean;
  // Epoch number at which the best value was observed.
  bestEpoch: number;
  // Saved weight snapshot (only populated when restoreBest = true).
  bestWeights: number[] | null;

  constructor(options?: {
    patience?: number;
    minDelta?: number;
    mode?: 'min' | 'max';
    restoreBest?: boolean;
  }) {
    this.patience    = options?.patience    ?? 10;
    this.minDelta    = options?.minDelta    ?? 1e-4;
    this.mode        = options?.mode        ?? 'min';
    this.restoreBest = options?.restoreBest ?? false;

    this.counter     = 0;
    this.stopped     = false;
    this.bestEpoch   = 0;
    this.bestWeights = null;
    this.bestValue   = this.mode === 'min' ? Infinity : -Infinity;
  }

  // ── update ───────────────────────────────────────────────────────────────
  //
  // Call once per epoch with the current metric value.
  // Returns true when training should stop (patience exhausted).
  //
  // Optionally pass `weights` (from net.getWeights()) to enable weight
  // snapshotting when restoreBest = true.
  //
  update(value: number, epoch: number, weights?: number[]): boolean {
    if (this.stopped) return true;

    const improved = this.mode === 'min'
      ? value < this.bestValue - this.minDelta
      : value > this.bestValue + this.minDelta;

    if (improved) {
      this.bestValue = value;
      this.bestEpoch = epoch;
      this.counter   = 0;
      if (this.restoreBest && weights !== undefined) {
        this.bestWeights = [...weights];
      }
    } else {
      this.counter++;
      if (this.counter >= this.patience) {
        this.stopped = true;
        return true;
      }
    }

    return false;
  }

  // Resets all state — use to re-run training with a fresh early-stop monitor.
  reset(): void {
    this.counter     = 0;
    this.stopped     = false;
    this.bestEpoch   = 0;
    this.bestWeights = null;
    this.bestValue   = this.mode === 'min' ? Infinity : -Infinity;
  }
}
