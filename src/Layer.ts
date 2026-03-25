import { NeuronN }                           from "./NeuronN";
import { Activation, sigmoid }               from "./activations";
import { OptimizerFactory, SGD }             from "./optimizers";

const defaultOptimizer: OptimizerFactory = () => new SGD();

// ─── LAYER ────────────────────────────────────────────────────────────────────
// A group of neurons that share the same inputs.
// Each neuron produces its own output independently.
// The layer output is an array of each neuron's output,
// which becomes the input for the next layer.
export class Layer {
  neurons: NeuronN[];

  constructor(
    nNeurons: number,
    nInputs: number,
    activation: Activation = sigmoid,
    optimizerFactory: OptimizerFactory = defaultOptimizer,
  ) {
    this.neurons = Array.from(
      { length: nNeurons },
      () => new NeuronN(nInputs, activation, optimizerFactory),
    );
  }

  predict(inputs: number[]): number[] {
    return this.neurons.map(n => n.predict(inputs));
  }
}
