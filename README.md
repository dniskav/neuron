[![npm](https://img.shields.io/npm/v/@dniskav/neuron)](https://www.npmjs.com/package/@dniskav/neuron)
[![license](https://img.shields.io/npm/l/@dniskav/neuron)](LICENSE)

A minimal, dependency-free neural network library built from scratch in TypeScript. Designed for learning and experimentation — every line of math is readable.

## What's inside

| Export | Description |
|--------|-------------|
| `Neuron` | Single-input neuron. The simplest possible unit: one weight, one bias. |
| `NeuronN` | N-input neuron with Xavier initialization and configurable activation. |
| `Layer` | A group of `NeuronN` neurons that share the same inputs. |
| `Network` | Two-layer network (hidden + output) with backpropagation. |
| `NetworkN` | Deep network of arbitrary depth. Define your architecture as `[inputs, ...hidden, outputs]`. |
| `LSTMLayer` | Recurrent layer with persistent hidden and cell state. Learns sequences via BPTT. |
| `NetworkLSTM` | Wraps an `LSTMLayer` + dense layers. Maintains memory across steps within an episode. |
| `NetworkTransformer` | Full token-classification Transformer: embeddings → N blocks → per-token logits. |
| `TransformerBlock` | One Transformer block: multi-head attention + FFN + LayerNorm × 2 with residuals. |
| `MultiHeadAttention` | N parallel attention heads concatenated and projected to `d_model`. |
| `AttentionHead` | Single scaled dot-product self-attention head (Q / K / V projections + backprop). |
| `LayerNorm` | Layer normalization with learnable γ / β per feature. |
| `WeightMatrix` | 2D weight matrix with per-scalar Adam optimizers. |
| `EmbeddingMatrix` | Lookup-table embedding matrix with SGD updates. |
| `sigmoid` `relu` `tanh` `linear` | Built-in activation functions. |
| `SGD` `Momentum` `Adam` | Optimizers. Each instance tracks its own state per weight. |
| `mse` `crossEntropy` | Loss functions for evaluation and logging. |
| `mseDelta` `crossEntropyDelta` | Output-layer delta functions for use with `trainWithDeltas`. |

## Install

```bash
npm install @dniskav/neuron
```

## Usage

### Single neuron — learn a threshold

```ts
import { Neuron } from "@dniskav/neuron";

const neuron = new Neuron();

// Train: output 1 if input >= 18, else 0
for (let epoch = 0; epoch < 1000; epoch++) {
  neuron.train(20, 1, 0.1); // adult
  neuron.train(15, 0, 0.1); // minor
}

console.log(neuron.predict(17)); // ~0.1 (minor)
console.log(neuron.predict(25)); // ~0.9 (adult)
```

### N-input neuron — multi-feature classification

```ts
import { NeuronN } from "@dniskav/neuron";

const neuron = new NeuronN(3); // 3 inputs: R, G, B

// Teach it to detect bright colors (luminance > 0.65)
neuron.train([1, 1, 1], 1, 0.05); // white → bright
neuron.train([0, 0, 0], 0, 0.05); // black → dark

console.log(neuron.predict([0.9, 0.9, 0.9])); // close to 1
```

### Network — non-linear classification

```ts
import { Network } from "@dniskav/neuron";

// 2 inputs → 8 hidden neurons → 1 output
const net = new Network(2, 8, 1);

// Train on XOR (not linearly separable — needs hidden layer)
const data = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]];

for (let epoch = 0; epoch < 5000; epoch++) {
  for (const [x, y, t] of data) {
    net.train([x, y], t, 0.3);
  }
}

console.log(net.predict([0, 1])); // ~0.97
console.log(net.predict([1, 1])); // ~0.03
```

### NetworkN — deep network with custom architecture

```ts
import { NetworkN } from "@dniskav/neuron";

// 3 inputs → 24 hidden → 16 hidden → 2 outputs
const net = new NetworkN([3, 24, 16, 2]);

// Train with multiple targets
net.train([0.5, 0.3, 0.8], [1, 0], 0.05);

// Predict returns an array — one value per output neuron
const [out1, out2] = net.predict([0.5, 0.3, 0.8]);
```

### Activations — ReLU, tanh, and more

Pass an activation per layer. The last layer typically uses `sigmoid` for binary output or `linear` for regression.

```ts
import { NetworkN, relu, sigmoid } from "@dniskav/neuron";

const net = new NetworkN([3, 64, 32, 1], {
  activations: [relu, relu, sigmoid],
});
```

Available: `sigmoid`, `relu`, `tanh`, `linear`.

### Optimizers — Adam, Momentum, SGD

Pass an optimizer factory. Each weight gets its own instance with independent state.

```ts
import { NetworkN, relu, sigmoid, Adam } from "@dniskav/neuron";

const net = new NetworkN([2, 64, 1], {
  activations: [relu, sigmoid],
  optimizer: () => new Adam(),          // default: beta1=0.9, beta2=0.999
});

// Momentum example
import { Momentum } from "@dniskav/neuron";
const net2 = new NetworkN([2, 32, 1], {
  optimizer: () => new Momentum(0.9),
});
```

Optimizers also work in `NetworkLSTM` (applied to the dense layers):

```ts
import { NetworkLSTM, relu, Adam } from "@dniskav/neuron";

const net = new NetworkLSTM(1, 8, [4, 1], {
  denseActivation: relu,
  optimizer: () => new Adam(0.001),
});
```

### Loss utilities

```ts
import { mse, crossEntropy } from "@dniskav/neuron";

const predicted = net.predict([0.5, 0.3]);
console.log(mse(predicted, [1, 0]));
console.log(crossEntropy(predicted, [1, 0]));
```

### trainWithDeltas — custom loss / physics-based gradients

`NetworkN` also exposes `trainWithDeltas` for when you compute your own output-layer deltas (e.g., from a physics simulation or a custom loss function):

```ts
import { NetworkN, mseDelta } from "@dniskav/neuron";

const net = new NetworkN([3, 16, 2]);
const pred = net.predict(inputs);

// Compute deltas manually using a helper, or from any external signal
const deltas = pred.map((p, i) => mseDelta(p, targets[i]));
net.trainWithDeltas(inputs, deltas, 0.01);
```

### NetworkLSTM — recurrent network with memory

`NetworkLSTM` adds within-episode memory: the network can remember what happened in previous steps of the same sequence.

```ts
import { NetworkLSTM } from "@dniskav/neuron";

// 1 input → LSTM(8 hidden) → Dense(4) → 1 output
const net = new NetworkLSTM(1, 8, [4, 1]);

// Task: predict 1 if we're past step 3 in the episode, else 0
// A feedforward net can't do this — it has no memory of step count.

for (let epoch = 0; epoch < 300; epoch++) {
  net.resetState();             // clear memory at episode start

  const targets: number[][] = [];
  for (let step = 0; step < 6; step++) {
    net.predict([1]);           // same input every step
    targets.push([step >= 3 ? 1 : 0]);
  }

  net.train(targets, 0.05);    // BPTT across the full episode
}

// Run a fresh episode and check predictions
net.resetState();
for (let step = 0; step < 6; step++) {
  const [out] = net.predict([1]);
  console.log(`step ${step}: ${out.toFixed(2)}  (expected: ${step >= 3 ? 1 : 0})`);
}
// step 0: 0.07  (expected: 0)
// step 1: 0.11  (expected: 0)
// step 2: 0.18  (expected: 0)
// step 3: 0.81  (expected: 1)
// step 4: 0.89  (expected: 1)
// step 5: 0.93  (expected: 1)
```

The network learns to count steps using its hidden state — no external counter needed.

## How it works

Each class applies an **activation function** to the weighted sum of inputs and uses **gradient descent** to update weights:

```
weight += lr × delta × input
bias   += lr × delta
```

`NetworkN` implements full **backpropagation** across all layers, propagating deltas from the output back to the first layer using the chain rule. The derivative of the chosen activation is applied at each layer.

`NeuronN` uses simplified **Xavier initialization** — weights start in `[-√(1/n), +√(1/n)]` — so gradients flow well from the start of training.

When an **optimizer** is used (e.g., Adam), the raw gradient is passed to the optimizer instead of being applied directly. Each weight maintains its own optimizer state (velocity, moments).

## Build

```bash
npm run build   # outputs CJS + ESM + type declarations to dist/
npm run dev     # watch mode
```

## For AI agents

If you are an AI agent or LLM working with this codebase, read [AGENTS.md](AGENTS.md) first. It contains the full class hierarchy, design constraints, and what this library does not do.

### NetworkTransformer — self-attention over sequences

```ts
import { NetworkTransformer } from "@dniskav/neuron";

// Sudoku solver: 81 cells (tokens), values 0–9, predict digit 1–9 per cell
const net = new NetworkTransformer(81, {
  vocabSize: 10,   // digits 0–9
  d_model:   64,   // embedding / hidden dimension
  nHeads:    4,    // attention heads (d_k = d_model / nHeads = 16)
  d_ff:      128,  // FFN hidden size
  nBlocks:   4,    // number of transformer blocks
  nClasses:  9,    // output classes per token (digits 1–9)
});

// tokens: 81 cell values (0 = empty)
const puzzle   = [5,3,0, 0,7,0, 0,0,0, ...];
const targets  = [...];   // 81*9 one-hot values
const mask     = puzzle.map(v => v === 0);   // only train on empty cells

const loss = net.train(puzzle, targets, 0.001, mask);
const logits = net.predict(puzzle);   // 729 logits (81 × 9)

// Attention weights from all blocks for visualization
const weights = net.getAttentionWeights();
// weights[blockIdx][headIdx]  → seqLen × seqLen matrix
```

Each head in each block learns a different type of relationship (row, column,
3×3 box). The network figures this out by itself through training.

## Possible improvements

1. **Support for batches** in training to improve efficiency.
2. **Improve documentation** with more advanced examples and use cases.

## License

MIT
