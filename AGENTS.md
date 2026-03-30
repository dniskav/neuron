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
| `WeightMatrix` | `src/MatMul.ts` | 2D weight matrix with per-scalar Adam optimizers. Used by Transformer. |
| `EmbeddingMatrix` | `src/MatMul.ts` | Lookup-table embedding with SGD updates. |
| `LayerNorm` | `src/LayerNorm.ts` | Layer normalization with learnable γ / β. Per-position caching for backprop. |
| `AttentionHead` | `src/AttentionHead.ts` | Single scaled dot-product self-attention head (Q/K/V + full backprop). |
| `MultiHeadAttention` | `src/MultiHeadAttention.ts` | N parallel heads + output projection Wo. |
| `TransformerBlock` | `src/TransformerBlock.ts` | MHA + FFN + LayerNorm × 2 + residual connections. Full backprop. |
| `NetworkTransformer` | `src/NetworkTransformer.ts` | Full token-classification Transformer: embeddings → blocks → per-token logits. |

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

- **All activations are sigmoid** for `Neuron`/`NeuronN`/`Network`/`NetworkN` by default. The Transformer classes use ReLU (FFN) and softmax (attention). Do not suggest adding activations to the feedforward classes unless the user asks.
- **No batching** — training is online (one sample at a time). `NetworkLSTM` accumulates gradients across an episode; `NetworkTransformer.train` processes one sequence per call.
- **No automatic differentiation** — all gradients are hand-coded.
- **Outputs are unbounded for Transformer** — `NetworkTransformer.predict` returns raw logits; apply softmax externally if probabilities are needed.
- **`NetworkTransformer.train` uses cross-entropy + softmax** (not MSE). The combined softmax+CE gradient at the logit level is `prob_c − target_c`, which is naturally bounded and ~10× smaller than the equivalent MSE gradient. This prevents gradient explosion without requiring gradient clipping.

## Transformer design decisions

- **Post-norm** (LayerNorm after residual add) — original "Attention Is All You Need" style.
- **d_k = d_v = d_model / nHeads** — equal capacity split across heads.
- **Adam for all weight matrices** (`WeightMatrix`), SGD for embedding lookups (`EmbeddingMatrix`).
- **`WeightMatrix.update(dW, lr, clipValue?)` accepts an optional per-element gradient clip.** Pass e.g. `1.0` to clip gradients to `[-1, 1]` before the Adam step. Default is `Infinity` (no clipping) — existing callers are unaffected.
- **LayerNorm uses per-position caching** — `resetCache(seqLen)` must be called before each forward pass, then `predictOne(x, pos)` per token. This is handled internally by `TransformerBlock`.
- **`getAttentionWeights()`** is available on `AttentionHead`, `MultiHeadAttention`, `TransformerBlock`, and `NetworkTransformer` — returns the softmax attention matrix from the last forward pass, useful for visualization.

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
- No data loaders or batching utilities
- No loss functions (MSE, cross-entropy) as standalone utilities — loss is implicit in the delta computation inside each class

## Build

```bash
npm run build   # tsup → CJS + ESM + .d.ts in dist/
```

Output: `dist/index.js` (CJS), `dist/index.mjs` (ESM), `dist/index.d.ts` + `dist/index.d.mts` (types).

## Known usage context

This library is used alongside a demo app (`neuron_app`) where RL agents are trained. `NetworkLSTM` was added for a maze-navigation agent that needed within-episode memory. `NetworkTransformer` was added for a Sudoku solver card in the "Razonamiento" section of the app, where the goal is to demonstrate that self-attention learns constraint-aware reasoning (row, column, box relationships) without explicit rule encoding.
