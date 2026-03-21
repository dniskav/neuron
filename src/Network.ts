import { Layer } from "./Layer";

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

  predict(inputs: number[]): number {
    const hiddenOut = this.hiddenLayer.predict(inputs);
    return this.outputLayer.predict(hiddenOut)[0];
  }

  // Trains on a single example. Returns the squared error.
  train(inputs: number[], target: number, lr: number): number {
    const hiddenOut  = this.hiddenLayer.predict(inputs);
    const prediction = this.outputLayer.predict(hiddenOut)[0];

    const outputError = target - prediction;
    const outputDelta = outputError * prediction * (1 - prediction);

    const outputNeuron = this.outputLayer.neurons[0];
    outputNeuron.weights = outputNeuron.weights.map(
      (w, i) => w + lr * outputDelta * hiddenOut[i]
    );
    outputNeuron.bias += lr * outputDelta;

    // Backpropagate error to hidden layer
    this.hiddenLayer.neurons.forEach((neuron, i) => {
      const hiddenOut_i  = hiddenOut[i];
      const hiddenError  = outputDelta * outputNeuron.weights[i];
      const hiddenDelta  = hiddenError * hiddenOut_i * (1 - hiddenOut_i);
      neuron.weights = neuron.weights.map((w, j) => w + lr * hiddenDelta * inputs[j]);
      neuron.bias += lr * hiddenDelta;
    });

    return outputError * outputError;
  }
}
