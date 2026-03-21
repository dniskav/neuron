# @dniskav/neuron

A minimal, dependency-free neural network library built from scratch in TypeScript. Designed for learning and experimentation — every line of math is readable.

## What's inside

| Class | Description |
|-------|-------------|
| `Neuron` | Single-input neuron. The simplest possible unit: one weight, one bias. |
| `NeuronN` | N-input neuron with Xavier initialization and sigmoid activation. |
| `Layer` | A group of `NeuronN` neurons that share the same inputs. |
| `Network` | Two-layer network (hidden + output) with backpropagation. |
| `NetworkN` | Deep network of arbitrary depth. Define your architecture as `[inputs, ...hidden, outputs]`. |

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

### trainWithDeltas — custom loss / physics-based gradients

`NetworkN` also exposes `trainWithDeltas` for when you compute your own output-layer deltas (e.g., from a physics simulation or a custom loss function):

```ts
net.trainWithDeltas(inputs, [0.4, -0.2], 0.05);
```

## How it works

Every class uses **sigmoid** as its activation function and **gradient descent** to update weights:

```
weight += lr × error × input
bias   += lr × error
```

`NetworkN` implements full **backpropagation** across all layers, propagating deltas from the output back to the first layer using the chain rule.

`NeuronN` uses simplified **Xavier initialization** — weights start in `[-√(1/n), +√(1/n)]` — so gradients flow well from the start of training.

## Build

```bash
npm run build   # outputs CJS + ESM + type declarations to dist/
npm run dev     # watch mode
```

## License

MIT
