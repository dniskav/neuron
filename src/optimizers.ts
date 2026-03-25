// ─── OPTIMIZERS ───────────────────────────────────────────────────────────────
//
// Each Optimizer is a stateful object responsible for one scalar weight.
// NeuronN creates one optimizer instance per weight (plus one for the bias),
// so state (velocity, moments) is tracked independently for every parameter.
//
// Usage — pass a factory function when building a network:
//
//   const net = new NetworkN([2, 8, 1], {
//     optimizer: () => new Adam(),
//   });
//
// The factory is called once per weight, ensuring isolated state across neurons.
//
export interface Optimizer {
  step(weight: number, gradient: number, lr: number): number;
}

// A factory function that produces one fresh Optimizer per weight.
export type OptimizerFactory = () => Optimizer;

// ── SGD ───────────────────────────────────────────────────────────────────────
// Vanilla stochastic gradient descent. Stateless.
//   w ← w + lr·g
export class SGD implements Optimizer {
  step(weight: number, gradient: number, lr: number): number {
    return weight + lr * gradient;
  }
}

// ── Momentum ──────────────────────────────────────────────────────────────────
// Accumulates a velocity vector in the gradient direction.
// Dampens oscillations and accelerates convergence along consistent directions.
//   v ← β·v + lr·g
//   w ← w + v
export class Momentum implements Optimizer {
  private v = 0;

  constructor(readonly beta = 0.9) {}

  step(weight: number, gradient: number, lr: number): number {
    this.v = this.beta * this.v + lr * gradient;
    return weight + this.v;
  }
}

// ── Adam ──────────────────────────────────────────────────────────────────────
// Adaptive moment estimation. Maintains per-parameter first and second moment
// estimates with bias correction. Works well across a wide range of problems.
//   m ← β₁·m + (1-β₁)·g          (first moment)
//   v ← β₂·v + (1-β₂)·g²         (second moment)
//   m̂ = m / (1-β₁ᵗ)              (bias-corrected)
//   v̂ = v / (1-β₂ᵗ)
//   w ← w + lr·m̂ / (√v̂ + ε)
export class Adam implements Optimizer {
  private m = 0;
  private v = 0;
  private t = 0;

  constructor(
    readonly beta1   = 0.9,
    readonly beta2   = 0.999,
    readonly epsilon = 1e-8,
  ) {}

  step(weight: number, gradient: number, lr: number): number {
    this.t++;
    this.m = this.beta1 * this.m + (1 - this.beta1) * gradient;
    this.v = this.beta2 * this.v + (1 - this.beta2) * gradient * gradient;
    const mHat = this.m / (1 - Math.pow(this.beta1, this.t));
    const vHat = this.v / (1 - Math.pow(this.beta2, this.t));
    return weight + lr * mHat / (Math.sqrt(vHat) + this.epsilon);
  }
}
