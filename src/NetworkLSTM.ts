import { LSTMLayer }  from "./LSTMLayer";
import { Layer }       from "./Layer";

// ─── LSTM NETWORK ─────────────────────────────────────────────────────────────
//
// A network where the first layer is an LSTMLayer (recurrent, with memory)
// followed by one or more standard dense layers.
//
// Architecture example:
//   new NetworkLSTM(7, 16, [16, 3])
//   → 7 external inputs
//   → LSTMLayer(7 → 16)   recurrent, maintains state between steps
//   → Layer(16 → 16)      dense
//   → Layer(16 → 3)       dense output
//
// Usage pattern (RL episode loop):
//
//   net.resetState()              ← start of every episode
//
//   for each step:
//     const out = net.predict(inputs)   ← stores step in internal trajectory
//     ... take action, collect reward ...
//
//   net.train(targetsPerStep, lr)  ← BPTT at episode end; clears trajectory
//
// ─────────────────────────────────────────────────────────────────────────────

export class NetworkLSTM {
  readonly inputSize:  number;
  readonly hiddenSize: number;

  lstm:        LSTMLayer;
  denseLayers: Layer[];

  // Dense layer activations stored per step for backprop
  private _acts: number[][][];  // [T][layer+1][neuron]

  constructor(inputSize: number, hiddenSize: number, denseStructure: number[]) {
    this.inputSize  = inputSize;
    this.hiddenSize = hiddenSize;

    this.lstm = new LSTMLayer(inputSize, hiddenSize);

    this.denseLayers = [];
    const sizes = [hiddenSize, ...denseStructure];
    for (let i = 1; i < sizes.length; i++) {
      this.denseLayers.push(new Layer(sizes[i], sizes[i - 1]));
    }

    this._acts = [];
  }

  // ── Reset recurrent state (call at episode start) ─────────────────────────
  resetState(): void {
    this.lstm.reset();
    this._acts = [];
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  predict(inputs: number[]): number[] {
    const h = this.lstm.predict(inputs);  // advances LSTM state, stores step

    // Forward through dense layers, recording activations for backprop
    const acts: number[][] = [h];
    for (const layer of this.denseLayers) {
      acts.push(layer.predict(acts[acts.length - 1]));
    }
    this._acts.push(acts);

    return acts[acts.length - 1];
  }

  // ── Train on a full episode ────────────────────────────────────────────────
  // targets: one target vector per step (same order as predict() calls).
  // Accumulates gradients across all T steps before applying (batch update).
  train(targets: number[][], lr: number): void {
    const T = this._acts.length;
    if (T === 0 || targets.length !== T) return;

    // Gradient accumulators for each dense layer
    const denseGrads = this.denseLayers.map(layer => ({
      dW: layer.neurons.map(n  => new Array(n.weights.length).fill(0)),
      db: new Array(layer.neurons.length).fill(0),
    }));

    // dL/dh for each step — will be passed to LSTM BPTT
    const dh_seq: number[][] = [];

    for (let t = 0; t < T; t++) {
      const acts = this._acts[t];
      const pred = acts[acts.length - 1];

      // Output-layer error deltas (sigmoid cross-entropy style)
      let deltas = pred.map((p, i) => (targets[t][i] - p) * p * (1 - p));

      // Backprop through dense layers (accumulate, don't apply yet)
      for (let l = this.denseLayers.length - 1; l >= 0; l--) {
        const layer   = this.denseLayers[l];
        const layerIn = acts[l];
        const grad    = denseGrads[l];

        // Gradient to previous layer (using current weights, before update)
        const prevDeltas = layerIn.map((out, j) => {
          const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
          return errProp * out * (1 - out);
        });

        layer.neurons.forEach((n, k) => {
          n.weights.forEach((_, j) => { grad.dW[k][j] += deltas[k] * layerIn[j]; });
          grad.db[k] += deltas[k];
        });

        deltas = prevDeltas;
      }

      // deltas is now dL/dh (gradient w.r.t. the LSTM output at step t)
      dh_seq.push(deltas);
    }

    // Apply averaged dense layer gradients
    for (let l = 0; l < this.denseLayers.length; l++) {
      const layer = this.denseLayers[l];
      const grad  = denseGrads[l];
      layer.neurons.forEach((n, k) => {
        n.weights = n.weights.map((w, j) => w + (lr / T) * grad.dW[k][j]);
        n.bias   += (lr / T) * grad.db[k];
      });
    }

    // BPTT through the LSTM (clears trajectory internally)
    this.lstm.backprop(dh_seq, lr);
    this._acts = [];
  }

  // ── Serialization ─────────────────────────────────────────────────────────
  getWeights() {
    return {
      lstm: this.lstm.getWeights(),
      dense: this.denseLayers.map(layer =>
        layer.neurons.map(n => ({ weights: [...n.weights], bias: n.bias }))
      ),
    };
  }

  setWeights(data: ReturnType<NetworkLSTM["getWeights"]>): void {
    this.lstm.setWeights(data.lstm);
    data.dense.forEach((layerData, l) => {
      layerData.forEach((neuronData, k) => {
        this.denseLayers[l].neurons[k].weights = [...neuronData.weights];
        this.denseLayers[l].neurons[k].bias    = neuronData.bias;
      });
    });
  }
}
