// ─── PERCEPTRON (Rosenblatt, 1957) ────────────────────────────────────────────
//
// The original model of a biological neuron. Historically the first trainable
// machine-learning model with a formal learning rule.
//
// Architecture:
//   z      = w · x + bias          linear combination (dot product)
//   output = z > 0 ? 1 : 0         Heaviside step function — NOT differentiable
//
// Learning rule (Perceptron Convergence Theorem):
//   error  = target − output        ∈ {-1, 0, +1}
//   w      ← w + lr · error · x    only updates on misclassification
//   bias   ← bias + lr · error
//
// The theorem guarantees convergence IF AND ONLY IF the data is linearly
// separable — i.e. a single hyperplane can divide the two classes.
//
// ─── WHY XOR CANNOT BE LEARNED ───────────────────────────────────────────────
//
// XOR is not linearly separable. Its four points:
//
//   (0,0) → 0      (0,1) → 1
//   (1,0) → 1      (1,1) → 0
//
// No single straight line can separate the 0s from the 1s. Because the
// Perceptron's decision boundary is always a hyperplane (w·x = 0), it cannot
// represent a non-convex region. This fundamental limitation, documented by
// Minsky & Papert (1969), led to the first "AI winter" and directly motivated
// the development of multi-layer networks and backpropagation.
//
// ─────────────────────────────────────────────────────────────────────────────

import { validateArray, validateNumber } from "./Validation";

export class Perceptron {
  weights: number[];
  bias: number;

  // ─── Constructor ─────────────────────────────────────────────────────────────
  // All weights and bias start at 0. The perceptron learning rule does not
  // require random initialization because the step function already breaks
  // symmetry when any misclassification occurs.

  constructor(nInputs: number) {
    if (!Number.isInteger(nInputs) || nInputs <= 0) {
      throw new Error(
        `Perceptron: nInputs must be a positive integer, got ${nInputs}`
      );
    }
    this.weights = new Array(nInputs).fill(0);
    this.bias = 0;
  }

  // ─── Forward pass ────────────────────────────────────────────────────────────
  // Computes z = Σ(wᵢ·xᵢ) + bias, then applies the Heaviside step function.
  // Returns 1 if z > 0, else 0.

  predict(inputs: number[]): 0 | 1 {
    validateArray(inputs, this.weights.length, "Perceptron.predict");
    let z = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      z += this.weights[i] * inputs[i];
    }
    return z > 0 ? 1 : 0;
  }

  // ─── Training step ───────────────────────────────────────────────────────────
  // Applies the perceptron update rule for a single (input, target) pair.
  //
  //   error = target − output   (0 on correct prediction → no update)
  //   wᵢ   ← wᵢ + lr · error · xᵢ
  //   bias ← bias + lr · error
  //
  // Returns the error (useful for tracking convergence).

  train(inputs: number[], target: 0 | 1, lr: number): number {
    validateArray(inputs, this.weights.length, "Perceptron.train");
    validateNumber(target, "Perceptron.train");
    validateNumber(lr, "Perceptron.train");

    if (target !== 0 && target !== 1) {
      throw new Error(
        `Perceptron.train: target must be 0 or 1, got ${target}`
      );
    }
    if (lr <= 0) {
      throw new Error(
        `Perceptron.train: learning rate must be positive, got ${lr}`
      );
    }

    const output = this.predict(inputs);
    const error = target - output;   // ∈ {-1, 0, +1}

    if (error !== 0) {
      // Only update on misclassification (error = 0 → no-op)
      for (let i = 0; i < this.weights.length; i++) {
        this.weights[i] += lr * error * inputs[i];
      }
      this.bias += lr * error;
    }

    return error;
  }
}
