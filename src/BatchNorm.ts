import { validateArray } from './Validation';

// ─── Batch Normalization (Online) ────────────────────────────────────────────
//
// Normalizes using running statistics (not mini-batch), suitable for online
// training where one sample is processed at a time.
//
// Forward:
//   x_norm = (x − running_mean) / sqrt(running_var + eps)
//   y      = gamma ⊙ x_norm + beta
//
// Running statistics updated with momentum:
//   running_mean = momentum * running_mean + (1 − momentum) * x
//   running_var  = momentum * running_var  + (1 − momentum) * (x − mean)²
//
// `momentum` (m) is the RETENTION factor of the running statistics (m = 1 keeps
// them frozen; m = 0 uses only the current sample), as in the formulas above.
//
// Gamma and beta are learnable parameters.
//
// Input gradient (exact derivative of forward() w.r.t. x, feature 004 / N2):
// forward() updates the statistics with x and normalizes with the UPDATED ones.
// With u = x − mean_old:  x − mean_new = m·u  and
// var_new = m·var_old + (1−m)·m²·u², s = sqrt(var_new + eps), y = γ·m·u/s + β, so
//   dy/dx = γ · ( m/s − (1−m)·m³·u² / s³ )
// (the second term is the dependence of the variance on x). Behavior change vs
// earlier versions, which treated the statistics as constants and returned
// γ/s: the returned dX is now about m times smaller (≈10× with m = 0.1).
// With m = 1 both coincide.
//
// ─────────────────────────────────────────────────────────────────────────────

export class BatchNorm {
  readonly dim: number
  readonly momentum: number

  gamma: number[]
  beta: number[]

  runningMean: number[]
  runningVar: number[]

  private _xNorm: number[] | null = null
  private _std: number[] | null = null
  private _u: number[] | null = null // x − mean_old, saved by forward()

  constructor(dim: number, momentum = 0.1) {
    this.dim = dim
    this.momentum = momentum
    this.gamma = new Array(dim).fill(1)
    this.beta = new Array(dim).fill(0)
    this.runningMean = new Array(dim).fill(0)
    this.runningVar = new Array(dim).fill(1)
  }

  // ── Forward ───────────────────────────────────────────────────────────────
  forward(x: number[]): number[] {
    if (x.length !== this.dim) {
      throw new Error(`BatchNorm.forward: expected array of length ${this.dim}, got ${x.length}`)
    }

    const eps = 1e-5

    // u = x − mean_old (before the update), needed for the exact backward
    this._u = x.map((v, i) => v - this.runningMean[i])

    // Update running statistics
    for (let i = 0; i < this.dim; i++) {
      this.runningMean[i] = this.momentum * this.runningMean[i] + (1 - this.momentum) * x[i]
      const diff = x[i] - this.runningMean[i]
      this.runningVar[i] = this.momentum * this.runningVar[i] + (1 - this.momentum) * diff * diff
    }

    // Normalize using running statistics
    this._std = this.runningVar.map(v => Math.sqrt(v + eps))
    this._xNorm = x.map((v, i) => (v - this.runningMean[i]) / this._std![i])

    // Scale and shift
    return this._xNorm.map((xn, i) => this.gamma[i] * xn + this.beta[i])
  }

  // ── Backward ──────────────────────────────────────────────────────────────
  backward(dOut: number[]): number[] {
    if (!this._xNorm || !this._std || !this._u) {
      throw new Error('BatchNorm.backward: call forward() first')
    }

    // dx = dOut · γ · (m/s − (1−m)·m³·u²/s³)  (see header). The caller applies
    // gamma/beta updates via trainParams().
    const m = this.momentum
    return dOut.map((d, i) => {
      const s = this._std![i]
      const u = this._u![i]
      return d * this.gamma[i] * (m / s - ((1 - m) * m ** 3 * u * u) / (s * s * s))
    })
  }

  // ── Train gamma and beta (call after backward) ────────────────────────────
  trainParams(dOut: number[], lr: number): void {
    if (!this._xNorm) return
    for (let i = 0; i < this.dim; i++) {
      this.gamma[i] += lr * dOut[i] * this._xNorm[i]
      this.beta[i] += lr * dOut[i]
    }
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: gamma, beta.
  getWeights(): number[] {
    return [...this.gamma, ...this.beta]
  }

  setWeights(weights: number[]): void {
    validateArray(weights, 2 * this.dim, 'BatchNorm.setWeights')
    for (let i = 0; i < this.dim; i++) this.gamma[i] = weights[i]
    for (let i = 0; i < this.dim; i++) this.beta[i] = weights[this.dim + i]
  }
}
