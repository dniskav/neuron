import { Layer }                            from "./Layer";
import { Activation, sigmoid }              from "./activations";
import { OptimizerFactory, SGD }            from "./optimizers";

const defaultOptimizer: OptimizerFactory = () => new SGD();

export interface NetworkNOptions {
  // One activation per layer (hidden layers + output layer).
  // Defaults to sigmoid for all layers.
  activations?: Activation[];
  // Optimizer factory shared across all weights in the network.
  // Defaults to SGD.
  optimizer?: OptimizerFactory;
}

// ─── N-LAYER NETWORK ──────────────────────────────────────────────────────────
// Generalization of Network for arbitrary depth.
//
//   new NetworkN([3, 16, 8, 1])
//   → 3 inputs → layer 16 → layer 8 → 1 output
//
//   new NetworkN([3, 16, 8, 1], {
//     activations: [relu, relu, sigmoid],
//     optimizer: () => new Adam(),
//   })
//
export class NetworkN {
  layers: Layer[];

  constructor(readonly structure: number[], options: NetworkNOptions = {}) {
    const nLayers    = structure.length - 1;
    const activations = options.activations ?? Array.from({ length: nLayers }, () => sigmoid);
    const optimizer   = options.optimizer   ?? defaultOptimizer;

    this.layers = [];
    for (let i = 1; i < structure.length; i++) {
      this.layers.push(new Layer(structure[i], structure[i - 1], activations[i - 1], optimizer));
    }
  }

  predict(inputs: number[]): number[] {
    return this.layers.reduce((acc, layer) => layer.predict(acc), inputs);
  }

  // Generalized backpropagation across L layers.
  // Returns the mean squared error for the example.
  train(inputs: number[], targets: number[], lr: number): number {
    // Forward pass — store activations at every layer
    const act: number[][] = [inputs];
    for (const layer of this.layers) act.push(layer.predict(act[act.length - 1]));

    const pred = act[act.length - 1];

    // Output layer deltas — use the output layer's activation derivative
    const outAct = this.layers[this.layers.length - 1].neurons[0].activation;
    let deltas = pred.map((p, i) => (targets[i] - p) * outAct.dfn(p));

    // Backprop from last layer to first
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer   = this.layers[l];
      const layerIn = act[l];

      // Activation of the previous layer (null when l === 0: raw inputs, no activation).
      const prevAct = l > 0 ? this.layers[l - 1].neurons[0].activation : null;

      // Error reaching each neuron in the previous layer (computed before updating weights)
      const prevDeltas = layerIn.map((out, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
        return prevAct ? errProp * prevAct.dfn(out) : errProp;
      });

      layer.neurons.forEach((n, k) => {
        n._update(layerIn.map(inp => deltas[k] * inp), deltas[k], lr);
      });

      deltas = prevDeltas;
    }

    return pred.reduce((s, p, i) => s + (targets[i] - p) ** 2, 0) / pred.length;
  }

  // Backprop with externally provided output-layer deltas.
  // Useful for custom loss functions (e.g. physics-based gradients).
  trainWithDeltas(inputs: number[], outputDeltas: number[], lr: number): void {
    const act: number[][] = [inputs];
    for (const layer of this.layers) act.push(layer.predict(act[act.length - 1]));

    let deltas = outputDeltas;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer   = this.layers[l];
      const layerIn = act[l];
      const prevAct = l > 0 ? this.layers[l - 1].neurons[0].activation : null;
      const prevDeltas = layerIn.map((out, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
        return prevAct ? errProp * prevAct.dfn(out) : errProp;
      });
      layer.neurons.forEach((n, k) => {
        n._update(layerIn.map(inp => deltas[k] * inp), deltas[k], lr);
      });
      deltas = prevDeltas;
    }
  }
}
