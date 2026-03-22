# AGENTS.md — @dniskav/neuron

This file is intended for AI agents and LLMs working with this codebase.

## What this library is

`@dniskav/neuron` is a **minimal, dependency-free neural network library** built from scratch in TypeScript. It is educational by design: every class is self-contained, the math is readable, and each abstraction builds on the previous one. There are no external dependencies.

## Class hierarchy (build on each other in order)

| Class | File | Purpose |
|-------|------|---------|
| `Neuron` | `src/Neuron.ts` | Single-input neuron. One weight, one bias. Simplest possible unit. |
| `NeuronN` | `src/NeuronN.ts` | N-input neuron. Xavier init, sigmoid activation. |
| `Layer` | `src/Layer.ts` | Array of `NeuronN` sharing the same inputs. |
| `Network` | `src/Network.ts` | Two-layer network (hidden + output) with backprop. |
| `NetworkN` | `src/NetworkN.ts` | Deep network of arbitrary depth `[inputs, ...hidden, outputs]`. |
| `LSTMLayer` | `src/LSTMLayer.ts` | Recurrent layer with `h` (hidden) and `c` (cell) state. BPTT. |
| `NetworkLSTM` | `src/NetworkLSTM.ts` | `LSTMLayer` + dense layers. Episode-level memory. |

All classes are exported from `src/index.ts`.

## When to use each class

- **Neuron** — single scalar input, learning a threshold or a simple linear relationship.
- **NeuronN** — single neuron with multiple inputs, binary classification.
- **Layer** — parallel neurons for a single layer of a network.
- **Network** — fixed two-layer architecture, good for simple non-linear problems (e.g. XOR).
- **NetworkN** — use this for most feedforward tasks. Flexible depth and width.
- **LSTMLayer** — use when you need raw access to the recurrent layer (e.g. custom architectures).
- **NetworkLSTM** — use when inputs arrive as a sequence and the network must remember previous steps within the same episode (e.g. RL agents, time series).

## Key design constraints

- **All activations are sigmoid** (except the LSTM cell gate which uses tanh). There is no softmax, ReLU, or other activation — do not suggest adding them unless the user asks.
- **No batching** — training is online (one sample at a time) for feedforward classes. `NetworkLSTM` accumulates gradients across an episode but applies them once at the end.
- **No optimizer** — plain gradient descent only. No Adam, no momentum.
- **No automatic differentiation** — all gradients are hand-coded.
- **Outputs are always in (0, 1)** due to sigmoid. Do not use this library for regression tasks that require unbounded outputs without modification.

## LSTM-specific details

`LSTMLayer` uses standard 4-gate LSTM:
- `f` = forget gate (sigmoid) — initialized with bias=1 so network remembers by default
- `i` = input gate (sigmoid)
- `g` = cell gate (tanh)
- `o` = output gate (sigmoid)

**Episode pattern (mandatory):**
```ts
net.resetState()           // call at the START of every episode
for each step:
  net.predict(inputs)      // advances state, stores trajectory
net.train(targets, lr)     // BPTT across full episode, clears trajectory
```

Forgetting to call `resetState()` between episodes will bleed state from one episode into the next.

Gradients are averaged over episode length (`lr / T`) to prevent explosion over long episodes (up to ~600 steps in known use cases).

## What this library does NOT do

- No GPU acceleration
- No automatic differentiation
- No convolutional layers
- No attention / transformers
- No data loaders or batching utilities
- No loss functions (MSE, cross-entropy) as standalone utilities — loss is implicit in the delta computation inside each class

## Build

```bash
npm run build   # tsup → CJS + ESM + .d.ts in dist/
```

Output: `dist/index.js` (CJS), `dist/index.mjs` (ESM), `dist/index.d.ts` + `dist/index.d.mts` (types).

## Known usage context

This library is used alongside a demo app (`neuron_app`) where RL agents are trained. `NetworkLSTM` was added specifically for a maze-navigation agent that needed within-episode memory to avoid revisiting dead ends.
