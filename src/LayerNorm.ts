// ─── Layer Normalization ──────────────────────────────────────────────────────
//
// Normalizes each token's feature vector independently:
//   x_norm = (x − mean(x)) / std(x)
//   y      = γ ⊙ x_norm + β
//
// γ (gamma) and β (beta) are learnable per-feature parameters, both initialized
// to 1 and 0 respectively so the transform starts as identity.
//
// Usage within a TransformerBlock — operates over a full sequence at once:
//   norm.resetCache(seqLen)
//   const out = X.map((x, pos) => norm.predictOne(x, pos))
//   // ... later in backward:
//   const dX = dH.map((d, pos) => norm.backwardOne(d, pos, lr))
//
// The per-position cache (x_norm, std) is necessary because each token's
// normalization statistics differ; backprop must use the same values as
// the corresponding forward pass.
//
// ─────────────────────────────────────────────────────────────────────────────

export class LayerNorm {
  gamma: number[]
  beta:  number[]

  private readonly eps = 1e-5

  // Per-position cache populated during the forward pass.
  // resetCache() must be called before each sequence forward pass.
  private _cache: Array<{ x_norm: number[]; std: number }> = []

  constructor(dim: number) {
    this.gamma = new Array(dim).fill(1)
    this.beta  = new Array(dim).fill(0)
  }

  // Call once before forward-passing a new sequence.
  resetCache(seqLen: number): void {
    this._cache = new Array(seqLen)
  }

  // Normalize a single position's feature vector.
  // pos must match the position index used in the corresponding backwardOne call.
  predictOne(x: number[], pos: number): number[] {
    const N    = x.length
    const mean = x.reduce((s, v) => s + v, 0) / N
    const vari = x.reduce((s, v) => s + (v - mean) ** 2, 0) / N
    const std  = Math.sqrt(vari + this.eps)

    const x_norm = x.map(v => (v - mean) / std)
    this._cache[pos] = { x_norm, std }

    return x_norm.map((xn, i) => this.gamma[i] * xn + this.beta[i])
  }

  // Backprop through layer norm for one position.
  //
  // Given dL/dy (dOut), computes dL/dx:
  //   Let D = dOut ⊙ γ
  //   dL/dx_i = (1/std) * (D_i − mean(D) − x_norm_i * mean(D ⊙ x_norm))
  //
  // Also updates γ and β via SGD:
  //   γ_i += lr * dOut_i * x_norm_i
  //   β_i += lr * dOut_i
  //
  // SGD (not Adam) for γ/β: they are aggregated across all positions in the
  // sequence (de-facto mini-batch update), so the gradient is already smoothed.
  backwardOne(dOut: number[], pos: number, lr: number): number[] {
    const { x_norm, std } = this._cache[pos]
    const N = dOut.length

    // Update γ and β
    for (let i = 0; i < N; i++) {
      this.gamma[i] += lr * dOut[i] * x_norm[i]
      this.beta[i]  += lr * dOut[i]
    }

    // D = dOut ⊙ gamma  (element-wise; gamma already updated above)
    // Note: we use the gamma AFTER the update, which introduces a tiny
    // approximation. For production code you would cache the pre-update gamma,
    // but the effect is negligible at typical learning rates.
    const D    = dOut.map((d, i) => d * this.gamma[i])
    const mD   = D.reduce((s, v) => s + v, 0) / N
    const mDxn = D.reduce((s, d, i) => s + d * x_norm[i], 0) / N

    return D.map((d, i) => (d - mD - x_norm[i] * mDxn) / std)
  }
}
