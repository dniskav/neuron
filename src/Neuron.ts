// ─── ACTIVATION FUNCTION ─────────────────────────────────────────────────────
// Squashes any number into a value between 0 and 1
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

import { validateNumber } from "./Validation";

// ─── NEURON (single input) ────────────────────────────────────────────────────
// Educational version: one input, one weight.
// See NeuronN for the N-input version with Xavier initialization.
export class Neuron {
  weight: number;
  bias: number;

  constructor() {
    this.weight = Math.random() * 0.1;
    this.bias   = Math.random() * 0.1;
  }

  predict(input: number): number {
    validateNumber(input, 'Neuron.predict');
    return sigmoid(input * this.weight + this.bias);
  }

  train(input: number, target: number, lr: number): void {
    validateNumber(input, 'Neuron.train');
    validateNumber(target, 'Neuron.train');
    // lr must be finite and >= 0 (0 is a valid no-op); checked before anything is touched.
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`Neuron.train: lr must be a finite number >= 0, got ${lr}`);
    }
    const prediction = this.predict(input);
    const error = target - prediction;
    const grad = error * prediction * (1 - prediction); // sigmoid derivative
    this.weight += lr * grad * input;
    this.bias   += lr * grad;
  }
}
