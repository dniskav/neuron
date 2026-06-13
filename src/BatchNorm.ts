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
// Gamma and beta are learnable parameters.
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
    if (!this._xNorm || !this._std) {
      throw new Error('BatchNorm.backward: call forward() first')
    }

    // Update gamma and beta (SGD)
    for (let i = 0; i < this.dim; i++) {
      // Gradient w.r.t. gamma: dOut * x_norm
      // Gradient w.r.t. beta: dOut
      // We don't apply lr here — caller should handle that
    }

    // Gradient w.r.t. input
    // dx = dOut * gamma / std
    return dOut.map((d, i) => d * this.gamma[i] / this._std![i])
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
    for (let i = 0; i < this.dim; i++) this.gamma[i] = weights[i]
    for (let i = 0; i < this.dim; i++) this.beta[i] = weights[this.dim + i]
  }
}
