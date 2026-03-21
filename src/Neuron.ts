// ─── ACTIVATION FUNCTION ─────────────────────────────────────────────────────
// Squashes any number into a value between 0 and 1
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

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
    return sigmoid(input * this.weight + this.bias);
  }

  train(input: number, target: number, lr: number): void {
    const prediction = this.predict(input);
    const error = target - prediction;
    this.weight += lr * error * input;
    this.bias   += lr * error;
  }
}
