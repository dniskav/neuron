// ─── LOSS FUNCTIONS ───────────────────────────────────────────────────────────
//
// Scalar losses for evaluation and logging.
// For training with a custom loss, compute output-layer deltas and pass them
// to net.trainWithDeltas(inputs, deltas, lr).
//
// The delta functions below give the negative gradient of each loss
// w.r.t. the network output — what trainWithDeltas expects.
//
// Note: when using cross-entropy with a sigmoid output, the combined delta
// simplifies to (actual - predicted), because the sigmoid derivative cancels
// the cross-entropy gradient. crossEntropyDelta reflects this simplification.

// ── Scalar losses ─────────────────────────────────────────────────────────────

// Mean squared error.
export function mse(predicted: number[], actual: number[]): number {
  return predicted.reduce((sum, p, i) => sum + (actual[i] - p) ** 2, 0) / predicted.length;
}

// Binary cross-entropy (averaged over outputs).
// Assumes predicted values are probabilities in (0, 1).
export function crossEntropy(predicted: number[], actual: number[]): number {
  const eps = 1e-15;
  return -predicted.reduce((sum, p, i) => {
    const clipped = Math.max(eps, Math.min(1 - eps, p));
    return sum + actual[i] * Math.log(clipped) + (1 - actual[i]) * Math.log(1 - clipped);
  }, 0) / predicted.length;
}

// ── Output-layer delta functions ──────────────────────────────────────────────
// These return the error signal at the output neuron (before multiplying by the
// activation derivative). Pass the result array to trainWithDeltas.

// MSE delta: -(∂MSE/∂p) = actual - predicted
export function mseDelta(predicted: number, actual: number): number {
  return actual - predicted;
}

// Cross-entropy delta with sigmoid output.
// The sigmoid derivative cancels out, leaving: actual - predicted
export function crossEntropyDelta(predicted: number, actual: number): number {
  return actual - predicted;
}

// Cross-entropy delta for a raw probability output (no sigmoid).
// Use this when the output neuron has a linear activation.
export function crossEntropyDeltaRaw(predicted: number, actual: number): number {
  const eps = 1e-15;
  const p = Math.max(eps, Math.min(1 - eps, predicted));
  return actual / p - (1 - actual) / (1 - p);
}
