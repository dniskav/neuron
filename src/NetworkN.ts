import { Layer }                            from "./Layer";
import { Dropout }                          from "./Dropout";
import { Activation, sigmoid }              from "./activations";
import { OptimizerFactory, SGD }            from "./optimizers";
import { validateArray }                    from "./Validation";

const defaultOptimizer: OptimizerFactory = () => new SGD();

export interface NetworkNOptions {
  // One activation per layer (hidden layers + output layer).
  // Defaults to sigmoid for all layers.
  activations?: Activation[];
  // Optimizer factory shared across all weights in the network.
  // Defaults to SGD.
  optimizer?: OptimizerFactory;
  // Residual (skip) connections. When true, adds input to output for layers
  // where the number of inputs equals the number of outputs.
  // Can also be a function that returns true/false per layer index.
  residual?: boolean | ((layerIndex: number) => boolean);
  // Dropout rate for hidden layers (0 = disabled). Applied during training only.
  dropoutRate?: number;
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
  private _dropouts: Dropout[];
  private _residual: boolean | ((layerIndex: number) => boolean);

  constructor(readonly structure: number[], options: NetworkNOptions = {}) {
    const nLayers    = structure.length - 1;
    const activations = options.activations ?? Array.from({ length: nLayers }, () => sigmoid);
    const optimizer   = options.optimizer   ?? defaultOptimizer;
    const dropoutRate = options.dropoutRate ?? 0;

    if (activations.length !== nLayers) {
      throw new Error(`Expected ${nLayers} activations, got ${activations.length}`);
    }

    if (dropoutRate < 0 || dropoutRate >= 1) {
      throw new Error(`Dropout rate must be in [0, 1), got ${dropoutRate}`);
    }

    this._residual = options.residual ?? false;

    this.layers = [];
    for (let i = 1; i < structure.length; i++) {
      this.layers.push(new Layer(structure[i], structure[i - 1], activations[i - 1], optimizer));
    }

    // Create dropout instances for hidden layers only (not output layer)
    this._dropouts = [];
    if (dropoutRate > 0) {
      for (let i = 0; i < nLayers - 1; i++) {
        this._dropouts.push(new Dropout(dropoutRate));
      }
    }

    // Validate all output neurons share the same activation
    const outputLayer = this.layers[this.layers.length - 1];
    const outputActivation = outputLayer.neurons[0].activation;
    for (let i = 1; i < outputLayer.neurons.length; i++) {
      if (outputLayer.neurons[i].activation !== outputActivation) {
        throw new Error('All output neurons must share the same activation function');
      }
    }
  }

  predict(inputs: number[], training = false): number[] {
    validateArray(inputs, this.structure[0], 'NetworkN.predict');

    let current = [...inputs];
    for (let i = 0; i < this.layers.length; i++) {
      const layerInput = [...current];
      const layerOutput = this.layers[i].predict(current);

      // Residual connection: add input to output when sizes match
      if (this._shouldResidual(i)) {
        if (this.structure[i] === this.structure[i + 1]) {
          current = layerOutput.map((v, j) => v + layerInput[j]);
        } else {
          current = [...layerOutput];
        }
      } else {
        current = [...layerOutput];
      }

      // Apply dropout to hidden layers only (not output layer)
      if (i < this._dropouts.length) {
        current = this._dropouts[i].forward(current, training);
      }
    }
    return current;
  }

  // Generalized backpropagation across L layers.
  // Returns the mean squared error for the example.
  train(inputs: number[], targets: number[], lr: number): number {
    validateArray(inputs, this.structure[0], 'NetworkN.train');
    validateArray(targets, this.structure[this.structure.length - 1], 'NetworkN.train');
    // Forward pass — store activations at every layer (with dropout during training)
    const act: number[][] = [inputs];
    for (let i = 0; i < this.layers.length; i++) {
      const layerInput = act[act.length - 1];
      const layerOutput = this.layers[i].predict(layerInput);

      let current: number[];
      // Residual connection: add input to output when sizes match
      if (this._shouldResidual(i)) {
        if (this.structure[i] === this.structure[i + 1]) {
          current = layerOutput.map((v, j) => v + layerInput[j]);
        } else {
          current = [...layerOutput];
        }
      } else {
        current = [...layerOutput];
      }

      // Apply dropout to hidden layers only (not output layer)
      if (i < this._dropouts.length) {
        current = this._dropouts[i].forward(current, true);
      }
      act.push(current);
    }

    const pred = act[act.length - 1];

    // Output layer deltas — use the output layer's activation derivative
    const outAct = this.layers[this.layers.length - 1].neurons[0].activation;
    let deltas = pred.map((p, i) => (targets[i] - p) * outAct.dfn(p));

    // Backprop from last layer to first
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer   = this.layers[l];

      // Apply dropout backward to hidden layers (not output layer)
      if (l < this._dropouts.length) {
        deltas = this._dropouts[l].backward(deltas);
      }

      const layerIn = act[l];

      // Activation of the previous layer (null when l === 0: raw inputs, no activation).
      const prevAct = l > 0 ? this.layers[l - 1].neurons[0].activation : null;

      // Error reaching each neuron in the previous layer (computed before updating weights)
      const prevDeltas = layerIn.map((out, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
        return prevAct ? errProp * prevAct.dfn(out) : errProp;
      });

      // Residual backprop: gradient also flows through skip connection
      if (this._shouldResidual(l) && this.structure[l] === this.structure[l + 1]) {
        for (let j = 0; j < prevDeltas.length; j++) {
          prevDeltas[j] += deltas[j];
        }
      }

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
    for (let i = 0; i < this.layers.length; i++) {
      const layerInput = act[act.length - 1];
      const layerOutput = this.layers[i].predict(layerInput);

      let current: number[];
      if (this._shouldResidual(i)) {
        if (this.structure[i] === this.structure[i + 1]) {
          current = layerOutput.map((v, j) => v + layerInput[j]);
        } else {
          current = [...layerOutput];
        }
      } else {
        current = [...layerOutput];
      }

      if (i < this._dropouts.length) {
        current = this._dropouts[i].forward(current, true);
      }
      act.push(current);
    }

    let deltas = outputDeltas;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      const layer   = this.layers[l];

      if (l < this._dropouts.length) {
        deltas = this._dropouts[l].backward(deltas);
      }

      const layerIn = act[l];
      const prevAct = l > 0 ? this.layers[l - 1].neurons[0].activation : null;
      const prevDeltas = layerIn.map((out, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
        return prevAct ? errProp * prevAct.dfn(out) : errProp;
      });

      if (this._shouldResidual(l) && this.structure[l] === this.structure[l + 1]) {
        for (let j = 0; j < prevDeltas.length; j++) {
          prevDeltas[j] += deltas[j];
        }
      }

      layer.neurons.forEach((n, k) => {
        n._update(layerIn.map(inp => deltas[k] * inp), deltas[k], lr);
      });
      deltas = prevDeltas;
    }
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: layer 0 (all neurons), layer 1, ..., layer N.
  getWeights(): number[] {
    // Reset dropout masks to avoid stale state
    for (const d of this._dropouts) d.resetMask();
    const w: number[] = [];
    for (const layer of this.layers) {
      for (const n of layer.neurons) {
        w.push(...n.weights, n.bias);
      }
    }
    return w;
  }

  setWeights(weights: number[]): void {
    // Reset dropout masks to avoid stale state
    for (const d of this._dropouts) d.resetMask();
    let idx = 0;
    for (const layer of this.layers) {
      for (const n of layer.neurons) {
        for (let j = 0; j < n.weights.length; j++) n.weights[j] = weights[idx++];
        n.bias = weights[idx++];
      }
    }
  }

  // ── Helper ───────────────────────────────────────────────────────────────
  private _shouldResidual(layerIndex: number): boolean {
    if (typeof this._residual === 'function') return this._residual(layerIndex);
    return this._residual;
  }
}
