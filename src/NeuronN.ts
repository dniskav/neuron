// ─── ACTIVATION FUNCTION ─────────────────────────────────────────────────────
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── N-INPUT NEURON ───────────────────────────────────────────────────────────
// Generalized version of Neuron: accepts any number of inputs.
// Uses simplified Xavier initialization: weights in [-√(1/n), +√(1/n)].
// This ensures gradients flow well from the start of training.
export class NeuronN {
  weights: number[];
  bias: number;

  constructor(nInputs: number) {
    const limit = Math.sqrt(1 / nInputs);
    this.weights = Array.from({ length: nInputs }, () => (Math.random() * 2 - 1) * limit);
    this.bias = 0;
  }

  predict(inputs: number[]): number {
    const sum = inputs.reduce((acc, e, i) => acc + e * this.weights[i], this.bias);
    return sigmoid(sum);
  }

  train(inputs: number[], target: number, lr: number): void {
    const prediction = this.predict(inputs);
    const error = target - prediction;
    this.weights = this.weights.map((w, i) => w + lr * error * inputs[i]);
    this.bias += lr * error;
  }
}
