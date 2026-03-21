import { NeuronN } from "./NeuronN";

// ─── LAYER ────────────────────────────────────────────────────────────────────
// A group of neurons that share the same inputs.
// Each neuron produces its own output independently.
// The layer output is an array of each neuron's output,
// which becomes the input for the next layer.
export class Layer {
  neurons: NeuronN[];

  constructor(nNeurons: number, nInputs: number) {
    this.neurons = Array.from({ length: nNeurons }, () => new NeuronN(nInputs));
  }

  predict(inputs: number[]): number[] {
    return this.neurons.map(n => n.predict(inputs));
  }
}
