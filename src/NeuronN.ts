import { Activation, sigmoid }          from "./activations";
import { Optimizer, OptimizerFactory, SGD } from "./optimizers";

const defaultOptimizer: OptimizerFactory = () => new SGD();

// ─── N-INPUT NEURON ───────────────────────────────────────────────────────────
// Generalized version of Neuron: accepts any number of inputs.
// Uses simplified Xavier initialization: weights in [-√(1/n), +√(1/n)].
// This ensures gradients flow well from the start of training.
//
// Optional activation and optimizer can be provided at construction time.
// Default: sigmoid activation, SGD optimizer.
export class NeuronN {
  weights: number[];
  bias: number;
  readonly activation: Activation;

  // One optimizer instance per weight plus one for the bias.
  // Each instance maintains its own state (velocity, moments, etc.).
  private _opts: Optimizer[];

  constructor(
    nInputs: number,
    activation: Activation = sigmoid,
    optimizerFactory: OptimizerFactory = defaultOptimizer,
  ) {
    const limit = Math.sqrt(1 / nInputs);
    this.weights = Array.from({ length: nInputs }, () => (Math.random() * 2 - 1) * limit);
    this.bias = 0;
    this.activation = activation;
    this._opts = Array.from({ length: nInputs + 1 }, optimizerFactory);
  }

  predict(inputs: number[]): number {
    const sum = inputs.reduce((acc, e, i) => acc + e * this.weights[i], this.bias);
    return this.activation.fn(sum);
  }

  // Apply pre-computed gradients via the optimizer.
  // Called internally by Layer / NetworkN / NetworkLSTM during backprop.
  _update(weightGrads: number[], biasGrad: number, lr: number): void {
    this.weights = this.weights.map((w, i) => this._opts[i].step(w, weightGrads[i], lr));
    this.bias = this._opts[this.weights.length].step(this.bias, biasGrad, lr);
  }

  train(inputs: number[], target: number, lr: number): void {
    const prediction = this.predict(inputs);
    const error = target - prediction;
    this._update(inputs.map(inp => error * inp), error, lr);
  }
}
