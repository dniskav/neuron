import { Layer } from "./Layer";
import { validateArray, validateNumber } from "./Validation";

// ─── NETWORK (2 layers) ───────────────────────────────────────────────────────
// Neural network with one hidden layer and one output layer.
// Uses backpropagation to adjust weights.
// For arbitrary depth see NetworkN.
export class Network {
  hiddenLayer: Layer;
  outputLayer: Layer;

  constructor(nInputs: number, nHidden: number, nOutputs: number) {
    this.hiddenLayer = new Layer(nHidden, nInputs);
    this.outputLayer = new Layer(nOutputs, nHidden);
  }

  predict(inputs: number[]): number[] {
    validateArray(inputs, this.hiddenLayer.neurons[0].weights.length, 'Network.predict');
    const hiddenOut = this.hiddenLayer.predict(inputs);
    return this.outputLayer.predict(hiddenOut);
  }

  // Trains on a single example. `target` is a number (only if nOutputs === 1)
  // or one value per output neuron. Every output neuron is updated.
  // Returns the MEAN squared error over the outputs (the one-output case
  // gives the squared error). The step is the gradient of that returned loss:
  // the output delta is scaled by 2 / nOutputs (N10, Q8), so with one output the
  // step is twice the one of versions before 0.4.0 (which trained 0.5 sum (t - p)^2).
  train(inputs: number[], target: number | number[], lr: number): number {
    validateArray(inputs, this.hiddenLayer.neurons[0].weights.length, 'Network.train');
    const targets = this._targetVector(target);
    validateNumber(lr, 'Network.train');
    const hiddenOut = this.hiddenLayer.predict(inputs);
    const outputNeurons = this.outputLayer.neurons;
    // All predictions and deltas use the ORIGINAL weights (before any update)
    const predictions = this.outputLayer.predict(hiddenOut);
    const outputDeltas = outputNeurons.map((neuron, k) => {
      const outputError = targets[k] - predictions[k];
      return (2 / outputNeurons.length) * outputError * neuron.activation.dfn(predictions[k]);
    });

    // Hidden deltas: sum over ALL outputs, using ORIGINAL output weights
    const hiddenDeltas = this.hiddenLayer.neurons.map((neuron, i) => {
      let hiddenError = outputDeltas[0] * outputNeurons[0].weights[i];
      for (let k = 1; k < outputNeurons.length; k++) {
        hiddenError += outputDeltas[k] * outputNeurons[k].weights[i];
      }
      return hiddenError * neuron.activation.dfn(hiddenOut[i]);
    });

    // Update hidden layer via optimizer
    this.hiddenLayer.neurons.forEach((neuron, i) => {
      neuron._update(inputs.map(inp => hiddenDeltas[i] * inp), hiddenDeltas[i], lr);
    });

    // Update every output neuron via optimizer
    outputNeurons.forEach((neuron, k) => {
      neuron._update(hiddenOut.map(h => outputDeltas[k] * h), outputDeltas[k], lr);
    });

    let sumSquared = 0;
    for (let k = 0; k < targets.length; k++) {
      const error = targets[k] - predictions[k];
      sumSquared += error * error;
    }
    return sumSquared / targets.length;
  }

  /** Normalizes the target to one value per output neuron, or throws. */
  private _targetVector(target: number | number[]): number[] {
    const nOutputs = this.outputLayer.neurons.length;
    if (typeof target === 'number') {
      if (nOutputs !== 1) {
        throw new Error(
          `Network.train: expected ${nOutputs} targets (number[]), got a single number`
        );
      }
      validateNumber(target, 'Network.train');
      return [target];
    }
    if (!Array.isArray(target)) {
      throw new Error(
        `Network.train: expected ${nOutputs} targets (number[]), got ${typeof target}`
      );
    }
    if (target.length !== nOutputs) {
      throw new Error(
        `Network.train: expected ${nOutputs} targets, got ${target.length}`
      );
    }
    for (let k = 0; k < target.length; k++) {
      if (typeof target[k] !== 'number' || !isFinite(target[k])) {
        throw new Error(`Network.train: invalid target at index ${k}: ${target[k]}`);
      }
    }
    return target;
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: hidden layer (all neurons: weights then bias), then output layer.
  getWeights(): number[] {
    const w: number[] = [];
    // Loops, not a push spread: a spread of >~125 000 arguments overflows the stack.
    for (const n of this.hiddenLayer.neurons) {
      for (const v of n.weights) w.push(v);
      w.push(n.bias);
    }
    for (const n of this.outputLayer.neurons) {
      for (const v of n.weights) w.push(v);
      w.push(n.bias);
    }
    return w;
  }

  setWeights(weights: number[]): void {
    // Validate the total length (structure, not `weights`) before writing anything
    validateArray(weights, this.getWeights().length, 'Network.setWeights');
    let idx = 0;
    for (const n of this.hiddenLayer.neurons) {
      for (let j = 0; j < n.weights.length; j++) n.weights[j] = weights[idx++];
      n.bias = weights[idx++];
    }
    for (const n of this.outputLayer.neurons) {
      for (let j = 0; j < n.weights.length; j++) n.weights[j] = weights[idx++];
      n.bias = weights[idx++];
    }
  }
}
