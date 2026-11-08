// ─── Dropout Layer ───────────────────────────────────────────────────────────
//
// Randomly zeroes elements during training to prevent overfitting.
// Uses inverted dropout: scales by 1/(1-rate) during training so that
// at test time the values are unchanged.
//
// Usage:
//   const dropout = new Dropout(0.5)
//   const out = dropout.forward(x, true)   // training mode
//   const out = dropout.forward(x, false)  // inference mode
//
// ─────────────────────────────────────────────────────────────────────────────

export class Dropout {
  readonly rate: number
  private _mask: number[] | null = null

  constructor(rate: number) {
    if (rate < 0 || rate >= 1) {
      throw new Error(`Dropout rate must be in [0, 1), got ${rate}`)
    }
    this.rate = rate
  }

  // ── Forward ───────────────────────────────────────────────────────────────
  // x: number[]  →  number[]
  // If training, applies inverted dropout mask.
  // If not training, returns input unchanged.
  forward(x: number[], training = true): number[] {
    if (!training || this.rate === 0) {
      this._mask = null
      return [...x]
    }

    const scale = 1 / (1 - this.rate)
    this._mask = x.map(() => Math.random() > this.rate ? scale : 0)
    return x.map((v, i) => v * this._mask![i])
  }

  // ── Reuse the saved mask ──────────────────────────────────────────────────
  // Re-applies the mask of the last forward(x, true) WITHOUT drawing from Math.random,
  // so a second pass sees the same units dropped (loss, gradient and update share one
  // mask; Q9). The mask is kept: it can be reused any number of times and backward()
  // still uses it. Throws if there is no saved mask or its length differs from x.
  reuseMask(x: number[]): number[] {
    const mask = this._mask
    if (!mask) {
      throw new Error(
        'Dropout.reuseMask: no saved mask (call forward(x, true) first; an inference ' +
        'forward or resetMask() clears it)'
      )
    }
    if (mask.length !== x.length) {
      throw new Error(
        `Dropout.reuseMask: saved mask has length ${mask.length}, got input of length ${x.length}`
      )
    }
    return x.map((v, i) => v * mask[i])
  }

  // True when a training mask is saved (rate 0 and inference forwards never save one).
  hasMask(): boolean {
    return this._mask !== null
  }

  // ── Backward ──────────────────────────────────────────────────────────────
  // dOut: number[]  →  number[]
  // Applies the same mask (gradient is zeroed where activation was zeroed).
  backward(dOut: number[]): number[] {
    if (!this._mask) return [...dOut]
    return dOut.map((d, i) => d * this._mask![i])
  }

  // ── Reset mask between forward passes ─────────────────────────────────────
  resetMask(): void {
    this._mask = null
  }

  // ── No trainable params ───────────────────────────────────────────────────
  getWeights(): number[] {
    return []
  }

  setWeights(_weights: number[]): void {
    // No-op
  }
}
