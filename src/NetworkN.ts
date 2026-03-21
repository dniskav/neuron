import { Layer } from "./Layer";

// ─── N-LAYER NETWORK ──────────────────────────────────────────────────────────
// Generalization of Network for arbitrary depth.
//
//   new NetworkN([3, 16, 8, 1])
//   → 3 inputs → layer 16 → layer 8 → 1 output
//
export class NetworkN {
  layers: Layer[];

  constructor(readonly structure: number[]) {
    this.layers = [];
    for (let i = 1; i < structure.length; i++) {
      this.layers.push(new Layer(structure[i], structure[i - 1]));
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

    // Output layer deltas
    let deltas = pred.map((p, i) => (targets[i] - p) * p * (1 - p));

    // Backprop from last layer to first
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer   = this.layers[l];
      const layerIn = act[l];

      // Error reaching each neuron in the previous layer (computed before updating weights)
      const prevDeltas = layerIn.map((out, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
        return errProp * out * (1 - out);
      });

      layer.neurons.forEach((n, k) => {
        n.weights = n.weights.map((w, j) => w + lr * deltas[k] * layerIn[j]);
        n.bias += lr * deltas[k];
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
      const prevDeltas = layerIn.map((out, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
        return errProp * out * (1 - out);
      });
      layer.neurons.forEach((n, k) => {
        n.weights = n.weights.map((w, j) => w + lr * deltas[k] * layerIn[j]);
        n.bias += lr * deltas[k];
      });
      deltas = prevDeltas;
    }
  }
}
