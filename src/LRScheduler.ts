// ─── Learning Rate Scheduler ─────────────────────────────────────────────────
//
// Provides common learning rate decay strategies:
//   - stepDecay:     drop lr by a factor every N epochs
//   - exponentialDecay: lr *= decayRate^epoch
//   - plateauDecay:  reduce lr when loss plateaus
//
// Usage:
//   const scheduler = new LRScheduler()
//   let lr = 0.1
//   for (let epoch = 0; epoch < 1000; epoch++) {
//     lr = scheduler.stepDecay(lr, epoch, 0.5, 200)
//     // train...
//   }
//
// ─────────────────────────────────────────────────────────────────────────────

export class LRScheduler {
  // ── Step Decay ────────────────────────────────────────────────────────────
  // lr = initialLr * dropRate^floor(epoch / epochsDrop)
  stepDecay(lr: number, epoch: number, dropRate: number, epochsDrop: number): number {
    return lr * Math.pow(dropRate, Math.floor(epoch / epochsDrop))
  }

  // ── Exponential Decay ─────────────────────────────────────────────────────
  // lr = initialLr * decayRate^epoch
  exponentialDecay(lr: number, epoch: number, decayRate: number): number {
    return lr * Math.pow(decayRate, epoch)
  }

  // ── Plateau Decay ─────────────────────────────────────────────────────────
  // If loss hasn't improved for `patience` epochs, multiply lr by `factor`.
  // Returns the new lr. Call this after each epoch with the current loss.
  //
  // Usage:
  //   let patience_counter = 0
  //   let best_loss = Infinity
  //   for (let epoch = 0; epoch < 1000; epoch++) {
  //     const loss = train(...)
  //     lr = scheduler.plateauDecay(lr, loss, history, 10, 0.5)
  //   }
  plateauDecay(
    lr: number,
    currentLoss: number,
    history: number[],
    patience: number,
    factor: number,
  ): number {
    if (history.length < patience) return lr

    // Check if loss has improved in the last `patience` epochs
    const recentLosses = history.slice(-patience)
    const minRecentLoss = Math.min(...recentLosses)

    // If current loss is not better than the minimum of recent losses, decay
    if (currentLoss >= minRecentLoss) {
      return lr * factor
    }

    return lr
  }

  // ── Cosine Annealing ──────────────────────────────────────────────────────
  // lr = minLr + 0.5 * (maxLr - minLr) * (1 + cos(π * epoch / maxEpochs))
  cosineAnnealing(lr: number, epoch: number, maxEpochs: number, minLr = 0): number {
    return minLr + 0.5 * (lr - minLr) * (1 + Math.cos(Math.PI * epoch / maxEpochs))
  }
}
