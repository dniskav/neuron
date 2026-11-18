import { Activation, sigmoid }          from "./activations";
import { OptimizerFactory, defaultOptimizer } from "./optimizers";
import { VecOptimizer, vecFromFactory } from "./core/optim";
import { validateArray }                from "./Validation";
import { random }                       from "./random";

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

  // One vector optimizer for the weights and one of size 1 for the bias; they
  // keep the state (velocity, moments, etc.) in typed arrays, not one object
  // per weight. The weights are updated IN PLACE: `weights` keeps its identity.
  private _wOpt: VecOptimizer;
  private _bOpt: VecOptimizer;
  private _bBuf = new Float64Array(1);
  private _bGrad = new Float64Array(1);

  constructor(
    nInputs: number,
    activation: Activation = sigmoid,
    optimizerFactory: OptimizerFactory = defaultOptimizer,
  ) {
    const limit = Math.sqrt(1 / nInputs);
    this.weights = Array.from({ length: nInputs }, () => (random() * 2 - 1) * limit);
    this.bias = 0;
    this.activation = activation;
    this._wOpt = vecFromFactory(optimizerFactory, nInputs);
    this._bOpt = vecFromFactory(optimizerFactory, 1);
  }

  predict(inputs: number[]): number {
    validateArray(inputs, this.weights.length, 'NeuronN.predict');
    const sum = inputs.reduce((acc, e, i) => acc + e * this.weights[i], this.bias);
    return this.activation.fn(sum);
  }

  // Apply pre-computed gradients via the optimizer.
  // Called internally by Layer / NetworkN / NetworkLSTM during backprop.
  _update(weightGrads: number[], biasGrad: number, lr: number): void {
    this._wOpt.step(this.weights, weightGrads, lr);
    this._bBuf[0] = this.bias;
    this._bGrad[0] = biasGrad;
    this._bOpt.step(this._bBuf, this._bGrad, lr);
    this.bias = this._bBuf[0];
  }

  train(inputs: number[], target: number, lr: number): void {
    // lr must be finite and >= 0 (0 is a valid no-op); checked before anything is touched.
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`NeuronN.train: lr must be a finite number >= 0, got ${lr}`);
    }
    const prediction = this.predict(inputs);
    const error = target - prediction;
    const grad = error * this.activation.dfn(prediction);
    this._update(inputs.map(inp => grad * inp), grad, lr);
  }
}
