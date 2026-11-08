import { LSTMLayer }                         from "./LSTMLayer";
import { Layer }                             from "./Layer";
import { Activation, sigmoid }               from "./activations";
import { OptimizerFactory, defaultOptimizer } from "./optimizers";
import { validateArray, validateNumber }     from "./Validation";

export interface NetworkLSTMOptions {
  // Activation for the dense layers that follow the LSTM.
  // Defaults to sigmoid.
  denseActivation?: Activation;
  // Optimizer factory for the dense layers.
  // Defaults to SGD. (The LSTM gates always use plain SGD internally.)
  optimizer?: OptimizerFactory;
}

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

  constructor(
    inputSize: number,
    hiddenSize: number,
    denseStructure: number[],
    options: NetworkLSTMOptions = {},
  ) {
    this.inputSize  = inputSize;
    this.hiddenSize = hiddenSize;

    this.lstm = new LSTMLayer(inputSize, hiddenSize);

    const activation = options.denseActivation ?? sigmoid;
    const optimizer  = options.optimizer       ?? defaultOptimizer;

    this.denseLayers = [];
    const sizes = [hiddenSize, ...denseStructure];
    for (let i = 1; i < sizes.length; i++) {
      this.denseLayers.push(new Layer(sizes[i], sizes[i - 1], activation, optimizer));
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
    validateArray(inputs, this.inputSize, 'NetworkLSTM.predict');
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
  // Throws (and leaves the episode intact) when targets does not match the
  // trajectory or a row does not measure the output size. An empty episode with
  // empty targets is a no-op, as before.
  train(targets: number[][], lr: number): void {
    const T = this._acts.length;
    if (!Array.isArray(targets)) {
      throw new Error(`NetworkLSTM.train: expected targets array, got ${typeof targets}`);
    }
    if (targets.length !== T) {
      throw new Error(`NetworkLSTM.train: expected targets of length ${T}, got ${targets.length}`);
    }
    const outSize = this.denseLayers.length > 0
      ? this.denseLayers[this.denseLayers.length - 1].neurons.length
      : this.hiddenSize;
    for (let t = 0; t < T; t++) {
      validateArray(targets[t], outSize, `NetworkLSTM.train(targets[${t}])`);
    }
    if (T === 0) return;

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

      // Output-layer deltas — use the output dense layer's activation derivative
      const outAct = this.denseLayers[this.denseLayers.length - 1].neurons[0].activation;
      let deltas = pred.map((p, i) => (targets[t][i] - p) * outAct.dfn(p));

      // Backprop through dense layers (accumulate, don't apply yet)
      for (let l = this.denseLayers.length - 1; l >= 0; l--) {
        const layer   = this.denseLayers[l];
        const layerIn = acts[l];
        const grad    = denseGrads[l];

        // Gradient to previous layer (using current weights, before update).
        // When l === 0, layerIn is the raw LSTM output h = o·tanh(c) ∈ (-1,1).
        // h is NOT produced by sigmoid, so we must NOT apply the sigmoid derivative
        // h·(1-h) — doing so would flip the gradient sign for negative h values,
        // causing the network to learn the opposite of the correct direction.
        // For l > 0, layerIn is the output of a dense layer: activation derivative applies.
        const prevAct = l > 0 ? this.denseLayers[l - 1].neurons[0].activation : null;
        const prevDeltas = layerIn.map((out, j) => {
          const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
          return prevAct ? errProp * prevAct.dfn(out) : errProp;
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

    // BPTT through the LSTM first: if its trajectory does not match, it throws
    // before any dense weight changes. dh_seq is already computed above, so the
    // order does not change the numbers.
    this.lstm.backprop(dh_seq, lr);

    // Apply averaged dense layer gradients via optimizer
    for (let l = 0; l < this.denseLayers.length; l++) {
      const layer = this.denseLayers[l];
      const grad  = denseGrads[l];
      layer.neurons.forEach((n, k) => {
        n._update(
          grad.dW[k].map(g => g / T),
          grad.db[k] / T,
          lr,
        );
      });
    }

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

  // Validates the dense structure BEFORE writing; LSTMLayer.setWeights validates
  // its own gates before writing, so a bad input leaves the whole network intact.
  // Values are copied element by element into the existing arrays (no aliasing).
  setWeights(data: ReturnType<NetworkLSTM["getWeights"]>): void {
    const M = "NetworkLSTM.setWeights";
    if (data === null || typeof data !== "object") {
      throw new Error(`${M}: expected object with lstm and dense`);
    }
    if (data.lstm === null || typeof data.lstm !== "object") {
      throw new Error(`${M}: missing lstm`);
    }
    if (!Array.isArray(data.dense)) {
      throw new Error(`${M}: expected dense array, got ${typeof data.dense}`);
    }
    if (data.dense.length !== this.denseLayers.length) {
      throw new Error(`${M}: expected ${this.denseLayers.length} dense layers, got ${data.dense.length}`);
    }
    this.denseLayers.forEach((layer, l) => {
      const layerData = data.dense[l];
      if (!Array.isArray(layerData)) {
        throw new Error(`${M}: dense layer ${l} is not an array`);
      }
      if (layerData.length !== layer.neurons.length) {
        throw new Error(`${M}: dense layer ${l}: expected ${layer.neurons.length} neurons, got ${layerData.length}`);
      }
      layer.neurons.forEach((n, k) => {
        const nd = layerData[k];
        if (nd === null || typeof nd !== "object") {
          throw new Error(`${M}: dense[${l}][${k}] is not an object`);
        }
        validateArray(nd.weights, n.weights.length, `${M}(dense[${l}][${k}].weights)`);
        validateNumber(nd.bias, `${M}(dense[${l}][${k}].bias)`);
      });
    });

    this.lstm.setWeights(data.lstm);
    this.denseLayers.forEach((layer, l) => {
      layer.neurons.forEach((n, k) => {
        const nd = data.dense[l][k];
        for (let j = 0; j < n.weights.length; j++) n.weights[j] = nd.weights[j];
        n.bias = nd.bias;
      });
    });
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: LSTM (flat), then dense layer 0, dense layer 1, ..., dense layer N.
  getWeightsFlat(): number[] {
    const w: number[] = [];
    w.push(...this.lstm.getWeightsFlat());
    for (const layer of this.denseLayers) {
      for (const n of layer.neurons) {
        w.push(...n.weights, n.bias);
      }
    }
    return w;
  }

  setWeightsFlat(weights: number[]): void {
    let idx = 0;
    const lstmLen = this.lstm.getWeightsFlat().length;
    const denseLen = this.denseLayers.reduce(
      (s, layer) => s + layer.neurons.reduce((t, n) => t + n.weights.length + 1, 0),
      0,
    );
    validateArray(weights, lstmLen + denseLen, "NetworkLSTM.setWeightsFlat");
    this.lstm.setWeightsFlat(weights.slice(idx, idx + lstmLen));
    idx += lstmLen;
    for (const layer of this.denseLayers) {
      for (const n of layer.neurons) {
        for (let j = 0; j < n.weights.length; j++) n.weights[j] = weights[idx++];
        n.bias = weights[idx++];
      }
    }
  }
}
