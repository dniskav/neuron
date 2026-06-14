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

  // Trains on a single example. Returns the squared error.
  train(inputs: number[], target: number, lr: number): number {
    validateArray(inputs, this.hiddenLayer.neurons[0].weights.length, 'Network.train');
    validateNumber(target, 'Network.train');
    validateNumber(lr, 'Network.train');
    const hiddenOut  = this.hiddenLayer.predict(inputs);
    const prediction = this.outputLayer.predict(hiddenOut)[0];

    const outputNeuron = this.outputLayer.neurons[0];
    const outputError  = target - prediction;
    const outputDelta  = outputError * outputNeuron.activation.dfn(prediction);

    // Compute hidden deltas using ORIGINAL output weights (before update)
    const hiddenDeltas = this.hiddenLayer.neurons.map((neuron, i) => {
      const hiddenError = outputDelta * outputNeuron.weights[i];
      return hiddenError * neuron.activation.dfn(hiddenOut[i]);
    });

    // Update hidden layer via optimizer
    this.hiddenLayer.neurons.forEach((neuron, i) => {
      neuron._update(inputs.map(inp => hiddenDeltas[i] * inp), hiddenDeltas[i], lr);
    });

    // Update output layer via optimizer
    outputNeuron._update(hiddenOut.map(h => outputDelta * h), outputDelta, lr);

    return outputError * outputError;
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: hidden layer (all neurons: weights then bias), then output layer.
  getWeights(): number[] {
    const w: number[] = [];
    for (const n of this.hiddenLayer.neurons) {
      w.push(...n.weights, n.bias);
    }
    for (const n of this.outputLayer.neurons) {
      w.push(...n.weights, n.bias);
    }
    return w;
  }

  setWeights(weights: number[]): void {
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
