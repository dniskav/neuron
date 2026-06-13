# Notepad — @dniskav/neuron Full Implementation

## Project Structure
- package.json: no test framework, only tsup + typescript
- tsconfig.json: strict=true, ES2020, ESNext, bundler resolution, rootDir=src
- src/: 17 TypeScript files
- dist/ exists (built output)
- No test directory, no vitest, no jest

## Files to know
- src/index.ts (exports)
- src/Neuron.ts, src/NeuronN.ts, src/Layer.ts, src/Network.ts, src/NetworkN.ts
- src/LSTMLayer.ts, src/NetworkLSTM.ts
- src/MatMul.ts (WeightMatrix + EmbeddingMatrix), src/LayerNorm.ts
- src/AttentionHead.ts, src/MultiHeadAttention.ts, src/TransformerBlock.ts
- src/NetworkTransformer.ts, src/NetworkTransformerRL.ts
- src/Utils.ts, src/Activations.ts, src/Optimizers.ts, src/Initializers.ts

## Critical Bugs

### Bug 1: Network.ts backprop order wrong
Line ~55: output layer is updated BEFORE error is propagated to hidden layer.
This means hidden deltas use already-modified output weights.
Fix: compute hidden deltas first using the ORIGINAL output weights, then update both layers.

### Bug 2: Neuron.ts train() missing sigmoid derivative
Line ~25: train() does `this.weight += error * input` and `this.bias += error`.
This is perceptron rule, not gradient descent. For sigmoid, it needs `error * input * sigmoid'(output)`.
sigmoid'(x) = x * (1 - x) when x is the activated output.
Fix: multiply by `output * (1 - output)`.

### Bug 3: NetworkTransformerRL.ts missing causal mask
The class claims causal attention but there is no causal mask in AttentionHead or MultiHeadAttention.
Fix: add `causal` parameter to AttentionHead, mask out future positions with -Infinity before softmax.

### Bug 4: NetworkTransformerRL.ts pooling gradient wrong
Pooling in predict uses weighted average (last step 2x weight), but train() backprop divides gradient uniformly by seqLen.
Fix: backprop must use the same weighted distribution as forward.

### Bug 5: NetworkN.ts hardcoded neurons[0].activation
Line ~55: uses `this.output.neurons[0].activation` for output layer gradient. If output layer has different activations per neuron, this is wrong. Should verify all neurons share same activation, or use per-neuron activation.

### Bug 6: MatMul.ts no dimension validation
matMul should validate that dimensions match.

### Bug 7: No input validation anywhere
Every predict/train should validate array lengths, number types, etc.

## Missing Features
- Tests: zero. Need vitest + comprehensive suite.
- getWeights/setWeights: missing on NetworkTransformer, NetworkN, Network, TransformerBlock, MultiHeadAttention.
- Serialization: no generic save/load.
- Layers: Dropout, GRU, BatchNorm, Conv1D.
- Utilities: Trainer, DataLoader, LRScheduler, ModelSaver.

## Conventions
- All activations are sigmoid for feedforward classes.
- No batching — online training only.
- No autograd — hand-coded gradients.
- Transformer uses ReLU (FFN) and softmax (attention).
- Post-norm LayerNorm.
- Adam for WeightMatrix, SGD for EmbeddingMatrix.
- WeightMatrix.update(dW, lr, clipValue?) with optional clip.
- LayerNorm uses per-position caching: resetCache(seqLen), predictOne(x, pos).
- getAttentionWeights() available on AttentionHead, MHA, TransformerBlock, NetworkTransformer.
- LSTM episode pattern: resetState() → predict() loop → train(targets, lr).
- NetworkTransformerRL uses sliding window of last N states.

## Implementation Status
✅ ALL COMPLETED — Sat Jun 13 2026

### Verified
- `npm run build` passes (CJS + ESM + .d.ts)
- `npm test` passes — 26 test files, 229 tests
- Zero TODOs, FIXMEs, or stubs in codebase

### Bugs Fixed
1. Network.ts backprop order — hidden deltas computed before output weights updated
2. Neuron.ts sigmoid derivative — added `output * (1 - output)` gradient
3. AttentionHead causal mask — added `causal` parameter, masks future with -Infinity
4. NetworkTransformerRL pooling gradient — backprop uses same weighted distribution as forward
5. NetworkN.ts activation validation — constructor checks all output neurons share same activation
6. MatMul.ts dimension validation — throws on incompatible dimensions
7. Input validation everywhere — Validation.ts helpers used across all classes

### New Files Created
- src/Validation.ts — input validation helpers
- src/Dropout.ts — inverted dropout layer
- src/GRU.ts — gated recurrent unit with BPTT
- src/BatchNorm.ts — online batch normalization with running stats
- src/Conv1D.ts — 1D convolution with valid/same padding
- src/Trainer.ts — high-level training loop with lr decay and history
- src/DataLoader.ts — batch iteration, shuffling, sequence windowing
- src/LRScheduler.ts — step decay, exponential, plateau, cosine annealing
- src/ModelSaver.ts — JSON serialization/deserialization
- vitest.config.ts — vitest configuration
- tests/ — 26 test files covering all classes

### Updated Files
- src/Network.ts — fixed backprop, added getWeights/setWeights
- src/Neuron.ts — fixed sigmoid derivative, added validation
- src/NetworkN.ts — added activation validation, getWeights/setWeights
- src/AttentionHead.ts — added causal mask, getWeights/setWeights
- src/MultiHeadAttention.ts — passes causal parameter
- src/TransformerBlock.ts — accepts causal parameter
- src/NetworkTransformerRL.ts — fixed pooling gradient, added getWeights/setWeights
- src/NetworkTransformer.ts — added getWeights/setWeights
- src/NetworkLSTM.ts — added getWeights/setWeights
- src/LSTMLayer.ts — added getWeights/setWeights
- src/MatMul.ts — added dimension validation
- src/index.ts — exported all new modules
- package.json — added vitest devDependency, test scripts
