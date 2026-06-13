# @dniskav/neuron v0.2.5 — Implementation Plan

**Version:** 0.2.5
**Date:** Sat Jun 13 2026
**Total Improvements:** 12
**Files in scope:** 26 source files, 26 test files

---

## Dependency Graph

```
Phase 1 (Foundation)
├── P1.1  Unify optimizers for LSTMLayer
├── P1.2  Unify optimizers for GRULayer
├── P1.3  Unify optimizers for Conv1D
├── P1.4  EmbeddingMatrix already has flat getWeights/setWeights (no code change needed)
└── P1.5  Fix NetworkTransformerRL to implement flat Serializable
    └── (P1.5 unblocks ModelSaver universal compatibility)

Phase 2 (Trainer Enhancements)
├── P2.1  Add weight decay to Trainer
├── P2.2  Add early stopping to Trainer
│   └── (uses DataLoader's validation split from P4.2 — NOT its own internal split)
├── P2.3  Add metrics (accuracy, precision, recall, F1) to Trainer
│   └── (only for classification tasks with discrete targets)
└── P2.4  Add gradient clipping to Trainer (via ClipOptimizer wrapper)
    └── (P2.4 depends on P1.1-P1.3 — ClipOptimizer wraps per-scalar optimizers)

Phase 3 (NetworkN + Conv1D)
├── P3.1  Add residual connections to NetworkN
├── P3.2  Integrate Dropout into NetworkN (between layers)
└── P3.3  Make Conv1D multi-channel (kernel: [filters][kernelSize][inChannels])
    └── (P3.1, P3.2, P3.3 are independent)

Phase 4 (Remaining Features)
├── P4.1  Add configurable pooling to NetworkTransformerRL (avg, max, last, weighted)
├── P4.2  Add validation split to DataLoader
│   └── (Trainer/P2.2 uses DataLoader's split — single source of truth)
└── P4.3  Add gradient checks (finite differences) to test suite

Cross-cutting
└── P5.1  Update index.ts exports for all new/changed APIs
```

---

## Phase 1 — Foundation: Optimizer Unification + Serializable Fixes

### P1.1 — Unify optimizers for LSTMLayer

**File:** `src/LSTMLayer.ts`

**Problem:** Lines 214–227 use manual SGD: `this.forgetGate.W[k][j] += scale * dWf[k][j]`. No optimizer factory support.

**Action:**
- Add `optimizerFactory?: OptimizerFactory` parameter to `LSTMLayer` constructor (default `() => new SGD()`).
- Replace the manual SGD updates in `backprop()` with the per-scalar optimizer pattern.
- Store one optimizer per gate weight and per gate bias: 4 gates × (hSize × combSize + hSize) optimizers.
- Use the same `scale = lr / T` averaging already in the code; pass it as `lr` to each `step()` call.
- The `Gate` class (private) does not need to change — its `W` and `b` remain plain number arrays. Optimizers live in `LSTMLayer`.

**Verification:**
- `npm test` passes — LSTMLayer tests unchanged in behavior
- `LSTMLayer` instance can be constructed with `optimizer: () => new Adam()`
- Weights change after `backprop()` with Adam optimizer

---

### P1.2 — Unify optimizers for GRULayer

**File:** `src/GRU.ts`

**Problem:** Lines 189–200 use manual SGD: `this.resetGate.W[k][j] += scale * dWr[k][j]`.

**Action:** Same pattern as P1.1.
- Add `optimizerFactory?: OptimizerFactory` to `GRULayer` constructor.
- Replace lines 189–200 with per-scalar optimizer updates.
- 3 gates × (hSize × combSize + hSize) optimizers.

**Verification:** Same as P1.1 but for GRULayer.

---

### P1.3 — Unify optimizers for Conv1D

**File:** `src/Conv1D.ts`

**Problem:** Lines 131–137 use hardcoded `lr = 0.001`: `this.kernels[f][k][0] += dKernels[f][k][0] * 0.001`. This is not configurable.

**⚠️ Breaking Change:** `backward()` currently updates weights as a side effect. This plan refactors `backward()` to **only compute and return gradients** — weight updates are applied externally by the caller using the optimizer factory. This matches the pattern in `AttentionHead.backward()`. Any direct callers of `conv.backward()` (existing tests, user code) will need to apply the returned gradients manually. Tests that relied on `backward()` updating weights internally must be updated to call `forward()` then manually invoke the optimizer with the returned `dX`.

**Action:**
- Add `optimizerFactory?: OptimizerFactory` to `Conv1D` constructor.
- Store one optimizer per kernel scalar (`filters × kernelSize × inputChannels`) plus one per bias (`filters`).
- The `_input` and `_paddedInput` caches remain.
- Refactor `backward()` to:
  1. Compute `dKernels`, `dBiases`, `dPadded` (gradient accumulators)
  2. Apply optimizer updates to `this.kernels` and `this.biases` using the stored optimizers
  3. Return `dPadded` (gradient w.r.t. input) to the caller
- This makes `backward()` consistent with `AttentionHead.backward()` which also updates weights internally using the optimizer.

**Verification:**
- `Conv1D` instance can be constructed with `optimizerFactory: () => new Adam()`.
- `conv.backward(dOut)` returns `dX` (gradient w.r.t. input) and updates weights internally.
- Weights change after `backward()` with Adam optimizer.

---

### P1.4 — EmbeddingMatrix already implements flat Serializable (no code change)

**File:** `src/MatMul.ts`

**Clarification:** `EmbeddingMatrix` already has `getWeights(): number[]` (flatten `this.W` row-major) and `setWeights(weights: number[]): void` (restore from flat array). It already satisfies the `Serializable` interface.

The real issue is that `NetworkTransformer.setWeights()` currently accesses `tokenEmb.W` and `posEmb.W` directly instead of using `tokenEmb.setWeights()`. This is a bug in the caller, not in `EmbeddingMatrix`.

**Action:** No code changes needed. Verify `EmbeddingMatrix` already satisfies `Serializable`:
- `getWeights()` returns flat number[] (row-major flatten of `this.W`)
- `setWeights(weights: number[])` restores `this.W` from flat array
- Both methods already exist in the current codebase.

**Verification:**
- `ModelSaver.toJSON(embeddingMatrix)` / `ModelSaver.fromJSON(embeddingMatrix, json)` already works.
- `NetworkTransformer.setWeights()` uses `tokenEmb.setWeights()` and `posEmb.setWeights()` instead of direct `.W` mutation.

---

### P1.5 — Fix NetworkTransformerRL to implement flat Serializable

**File:** `src/NetworkTransformerRL.ts`

**Problem:** `NetworkTransformerRL` has BOTH structured `getWeights()`/`setWeights()` (returning/accepting nested objects) AND flat `getWeightsFlat()`/`setWeightsFlat()` (returning/accepting number[]). The structured pair does NOT satisfy the `Serializable` interface (expects flat number[]). The flat pair exists but `ModelSaver` calls `getWeights()/setWeights()`, not `getWeightsFlat()/setWeightsFlat()`.

**Action:**
- Keep existing structured `getWeights()`/`setWeights()` for backward compatibility.
- Add alias methods: `getWeightsFlat()` already exists, `setWeightsFlat(weights)` already exists.
- Add `getWeights(): number[]` that delegates to `getWeightsFlat()`.
- Add `setWeights(weights: number[]): void` that delegates to `setWeightsFlat(weights)`.
- The structured methods remain as `getWeightsStructured()` and `setWeightsStructured()`.

**Verification:**
- `NetworkTransformerRL` now satisfies `Serializable` interface (flat `getWeights/setWeights`).
- `ModelSaver.toJSON(net)` / `ModelSaver.fromJSON(net, json)` works.
- Existing structured API is preserved as `getWeightsStructured()`.

---

## Phase 2 — Trainer Enhancements

### P2.1 — Add weight decay (L2 regularization) to Trainer

**File:** `src/Trainer.ts`

**Problem:** `Trainer` has no L2 regularization. Weight decay helps prevent overfitting.

**Correct L2 Weight Decay Approach:**
Applying L2 penalty `w -= lr * wd * w` **after** `network.train()` is mathematically equivalent to scaling the weight by `(1 - lr*wd)` — this is not the standard L2 regularization formulation. Standard L2 adds `wd * w` to the gradient, so the update becomes `w -= lr * (∇L + wd * w) = w - lr*∇L - lr*wd*w`.

The correct implementation: since we cannot intercept the gradient inside `network.train()`, apply weight decay **before** each `train()` call by scaling the weights:
```
w[i] *= (1 - lr * weightDecay)  // apply before train()
```
This is mathematically equivalent to L2 regularization with the gradient bonus absorbed into the weight scaling. The effective learning rate for the decay term is `lr * weightDecay`, matching the standard formulation.

**Action:**
- Add `weightDecay?: number` to `TrainerOptions` (default `0`, meaning disabled).
- In the `train()` loop, before each `network.train()` call, apply `w[i] *= (1 - lr * weightDecay)` to all network weights using `network.getWeights()/setWeights()`.
- Add `TrainableNetworkWithWeights` interface that extends `TrainableNetwork` with `getWeights/setWeights`.
- **Note:** `weightDecay` is applied at the current learning rate `lr` (which decays over epochs), so the effective decay strength automatically decreases as lr decays — this is a standard property of weight decay with learning rate scheduling.

**Verification:**
- Training with `weightDecay > 0` produces smaller final weights than without.
- Weight norm decreases monotonically (or stays stable) over training with decay.
- `npm test` passes.

---

### P2.2 — Add early stopping to Trainer

**File:** `src/Trainer.ts`

**Problem:** `Trainer` always runs all epochs regardless of convergence.

**Action:**
- Add `earlyStopping?: { patience: number; minDelta: number }` to `TrainerOptions`.
- Add internal `_bestLoss` and `_patienceCounter` state.
- The Trainer does NOT do its own train/validation split — it uses the validation dataset provided externally (e.g. via `DataLoader.getValidationData()` from P4.2).
- Add `setValidationData(dataset: DataPair): void` method to allow passing a separate validation set.
- After each epoch, compute validation loss if validation data was provided.
- If validation loss hasn't improved by `minDelta` for `patience` epochs, break the training loop.
- Expose `getBestLoss(): number` and `getStopReason(): string`.
- **Note:** `validationSplit` belongs in DataLoader (P4.2), not in Trainer. Trainer accepts a pre-split validation dataset. This avoids double-splitting if both DataLoader and Trainer had their own split logic.

**Verification:**
- Early stopping triggers when validation loss plateaus.
- Training history reflects the early exit.

---

### P2.3 — Add metrics (accuracy, precision, recall, F1) to Trainer

**File:** `src/Trainer.ts`

**Problem:** `Trainer` only tracks loss. No classification metrics.

**⚠️ Scope Limitation:** Metrics are only meaningful for **classification** tasks (discrete class labels). For regression/continuous targets (e.g. MSE loss), "accuracy" and "precision" are undefined. The Trainer will only compute metrics when `computeMetrics: true` AND targets appear to be one-hot vectors or single-class indices. The library uses MSE loss for continuous outputs — metrics are opt-in for users who construct classification-style targets.

**Action:**
- Add `computeMetrics?: boolean` to `TrainerOptions`.
- Add `TrainMetrics` interface with `accuracy`, `precision`, `recall`, `f1` (per-class and macro-average).
- After each epoch, if `computeMetrics` is true, evaluate on the full dataset.
- **Classification detection:** If targets are one-hot (each row sums to 1) or single-element arrays, treat as classification. Convert argmax of network output to predicted class and argmax of targets to true class, then build a confusion matrix.
- Add `getMetrics(): TrainMetrics | null` to retrieve metrics history.
- If targets appear continuous (not one-hot), metrics are skipped and a warning is logged.

**Verification:**
- `getMetrics()` returns valid metrics after training on one-hot classification targets.
- Metrics update meaningfully as training progresses (accuracy increases for a solvable problem).
- `npm test` passes — metrics tests use one-hot targets.

---

### P2.4 — Add gradient clipping to Trainer (via ClipOptimizer wrapper)

**File:** `src/Trainer.ts` + `src/optimizers.ts`

**Problem:** No gradient clipping at Trainer level. The `Optimizer.step(weight, gradient, lr)` interface has no `clipValue` parameter — we cannot change this without breaking all existing `Optimizer` implementations.

**Correct approach: Clip before step(), not inside step().**
The `clipValue` is applied **in `backprop()` before calling each optimizer's `step()`**, not passed into `step()`. Each per-scalar optimizer is already instantiated in `LSTMLayer`/`GRULayer`/`Conv1D` with no clip parameter. The clipping happens at the call site.

**Action:**
- Create a `ClippedOptimizerFactory` in `src/optimizers.ts`:
  ```ts
  export function ClippedOptimizerFactory(
    inner: OptimizerFactory,
    clipValue: number,
  ): OptimizerFactory {
    return () => new ClipOptimizer(inner(), clipValue);
  }
  ```
  Where `ClipOptimizer` wraps an existing `Optimizer` and clips the gradient before forwarding:
  ```ts
  class ClipOptimizer implements Optimizer {
    constructor(private inner: Optimizer, private clipValue: number) {}
    step(w: number, g: number, lr: number): number {
      const clipped = Math.max(-this.clipValue, Math.min(this.clipValue, g));
      return this.inner.step(w, clipped, lr);
    }
  }
  ```
- Add `clipValue?: number` to `TrainerOptions`.
- When `clipValue > 0`, wrap the network's optimizer factory with `ClippedOptimizerFactory`.
- **For LSTMLayer/GRULayer/Conv1D** (P1.1–P1.3): their constructors accept `optimizerFactory`. Pass `ClippedOptimizerFactory(userFactory, clipValue)` instead of raw `userFactory` when `clipValue > 0`.
- The `clipValue` is stored in the layer/Conv1D constructor and used to wrap the optimizer factory at construction time.

**Verification:**
- Training with `clipValue: 1.0` prevents gradient explosion in deep networks.
- No `Optimizer` interface changes — backward compatible.
- `npm test` passes.

---

## Phase 3 — NetworkN + Conv1D

### P3.1 — Add residual connections to NetworkN

**File:** `src/NetworkN.ts`

**Problem:** `NetworkN` has no skip/residual connections. Deep networks suffer from vanishing gradients.

**Action:**
- Add `residual?: boolean | ((layerIndex: number) => boolean)` to `NetworkNOptions`. Default `false`.
- When `residual = true`, each layer's output is added to its input (residual connection) before passing to the next layer — but **only when** the layer dimensions match (same number of neurons as previous layer). If sizes differ, skip the residual connection for that layer.
- Forward pass cache: store the pre-residual input per layer in `act[]` (the `act` array already stores pre-activation values; use `act[l]` as the residual input for layer `l`).
- Backward pass: after computing `deltas` for layer `l` (i.e., after calling `layer.neurons[k]._update(...)`), the gradient `dX` flowing to the previous layer must include the skip connection gradient. The residual input `act[l]` has the same shape as `layerIn`. The skip gradient `dSkip = dY` (gradient from downstream) flows to `act[l]` unchanged (identity connection). So the combined upstream gradient is:
  ```
  dAct[l] = dY_from_current_layer + dY_from_skip  // element-wise addition
  ```
  Then propagate `dAct[l]` to compute `prevDeltas` for layer `l-1`.
- Implementation detail: in the current `train()` backprop loop, after `layer.neurons.forEach(...)` updates weights, the `prevDeltas` are computed. For residual layers, add `deltas` to `prevDeltas` element-wise before they are used:
  ```
  // After computing prevDeltas for layer l, before assigning to deltas for layer l-1:
  if (isResidualLayer(l)) {
    prevDeltas = prevDeltas.map((pd, i) => pd + deltas[i]);  // element-wise add
  }
  deltas = prevDeltas;
  ```

**Verification:**
- `NetworkN` with `residual: true` and matching layer sizes trains deeper networks faster.
- Residual gradient correctly flows through skip connection (gradient check in P4.3).
- `npm test` passes — existing NetworkN tests still work.

---

### P3.2 — Integrate Dropout into NetworkN

**File:** `src/NetworkN.ts` + `src/Dropout.ts`

**Problem:** `Dropout` exists as a standalone class but is not integrated into `NetworkN`.

**Action:**
- Add `dropoutRate?: number` to `NetworkNOptions`. Default `0` (disabled).
- Insert `Dropout` layers after each hidden layer (not after the output layer). Store array of `Dropout` instances parallel to `this.layers`.
- `NetworkN.predict(inputs: number[], training = false): number[]` — add optional `training` parameter (default `false` for backward compatibility).
- During `predict()` with `training=true`: apply each Dropout layer with `forward(x, true)`.
- During `predict()` with `training=false`: apply each Dropout layer with `forward(x, false)` (identity).
- `NetworkN.train()` internally calls `this.predict(inputs, true)` to enable dropout during training.
- The `Dropout` mask is stored per-layer and reset via `resetMasks()` at the start of each `predict()` call (both training and inference — ensures clean state).
- `getWeights()/setWeights()` — Dropout has no trainable weights, but call `dropout.resetMask()` during `setWeights()` to clear any stale masks.

**Verification:**
- Network with `dropoutRate: 0.5` shows regularization effect (higher training loss but better generalization on held-out data).
- `predict(x, true)` applies dropout mask; `predict(x, false)` returns deterministic output.
- `npm test` passes.

---

### P3.3 — Make Conv1D multi-channel

**File:** `src/Conv1D.ts`

**Problem:** Current kernel shape is `[filters][kernelSize][1]` — only 1 input channel. Real sequences often have multiple channels (e.g. stereo audio, multi-sensor data).

**Action:**
- Add `inputChannels?: number` parameter to `Conv1D` constructor (default `1` for backward compatibility).
- Change kernel shape from `[filters][kernelSize][1]` to `[filters][kernelSize][inputChannels]`.
- Update `forward()`: input is now `[inputLength][inputChannels]` (2D array). For each filter `f` and each output position `pos`:
  ```
  output[f][pos] = Σ_{k=0}^{kernelSize-1} Σ_{c=0}^{inputChannels-1} kernels[f][k][c] * padded[start+k][c] + bias[f]
  ```
  Where `start = pos * stride`. This is a 2D convolution: each kernel position `k` is a vector of `inputChannels` weights, and it dot-products with the input patch's `inputChannels` values.
- Update `backward()`: gradients w.r.t. kernels now have shape `[filters][kernelSize][inputChannels]`. The gradient accumulation mirrors the forward computation.
- Update `getWeights()/setWeights()` serialization: kernel count becomes `filters * kernelSize * inputChannels` (plus biases).
- Update `getOutputLength()` — output shape is `[filters][outputLength]` (unchanged).
- Update `constructor` validation: reject `inputChannels < 1`.

**Verification:**
- `Conv1D` with `inputChannels: 3` correctly processes 3-channel input (verified by checking output shape and finite gradient).
- Serialization round-trips correctly with multi-channel kernels.
- `npm test` passes — existing Conv1D tests still work (single-channel default).

---

## Phase 4 — Remaining Features

### P4.1 — Configurable pooling in NetworkTransformerRL

**File:** `src/NetworkTransformerRL.ts`

**Problem:** Pooling is hardcoded as weighted average (last step 2× weight). No option for avg, max, or last-only pooling.

**Action:**
- Add `pooling?: 'avg' | 'max' | 'last' | 'weighted'` to `NetworkTransformerRLOptions`. Default `'weighted'`.
- Implement `_pool(h: number[][]): number[]` with a switch:
  - `'avg'`: unweighted mean over all seqLen positions
  - `'max'`: element-wise max over all positions
  - `'last'`: return `h[seqLen - 1]` directly
  - `'weighted'`: current implementation (last step 2× weight)
- **Max pooling backward (argmax-based):** During `_pool(h)`, also track `argmax[m] = argmax_i h[i][m]` for each dimension `m`. In `train()`, the backward gradient for `'max'` pooling is routed only to the argmax position for each dimension:
  ```
  dH[i][m] = (i === argmax[m]) ? dPooled[m] : 0
  ```
  This is the standard max pooling gradient. Ties are broken by selecting the first occurrence (standard behavior).
- Add `getPoolingType(): string` to query current pooling mode.
- **Important:** `'max'` pooling requires storing argmax during `_pool()`. Add a private cache `_argmax: number[] | null` that is populated during forward and used during backward.

**Verification:**
- Switching pooling type changes `predict()` output.
- Training with different pooling types converges (verified by P4.3 gradient check).
- `'max'` pooling backward correctly routes gradient to argmax positions.
- `npm test` passes.

---

### P4.2 — Add validation split to DataLoader

**File:** `src/DataLoader.ts`

**Problem:** `DataLoader` has no train/validation split support.

**Action:**
- Add optional `validationSplit?: number` parameter to `DataLoader` constructor (e.g. `0.2`).
- When `validationSplit > 0`, split the data into training and validation sets.
- Add `getValidationData(): DataPair` method to retrieve the validation set.
- The training set uses the remaining `1 - validationSplit` of data.
- Ensure shuffling is consistent: shuffle the full dataset first, then split.

**Verification:**
- `DataLoader` with `validationSplit: 0.2` returns 80% data via `next()` and 20% via `getValidationData()`.
- Total samples equal original dataset size.

---

### P4.3 — Add proper gradient checks (finite differences) to test suite

**Files:** New file `tests/GradientCheck.test.ts`

**Problem:** The existing "gradient check" in `Integration.test.ts` only verifies loss decreases after training — it does NOT verify analytical gradients match numerical gradients using finite differences.

**Action:**
- Create `tests/GradientCheck.test.ts`.
- Implement finite difference gradient check for:
  - `Network` (single output, MSE loss)
  - `NetworkN` (multi-output, MSE loss)
  - `LSTMLayer` (BPTT, one step at a time)
  - `GRULayer` (BPTT)
  - `Conv1D` (single-step forward/backward)
  - `AttentionHead` (single sequence)
  - `NetworkTransformer` (single sequence, cross-entropy loss)
- For each class, the test:
  1. Creates a small network (2–4 units)
  2. Computes analytical gradient `∂L/∂w` via `backprop()` or `train()`
  3. Computes numerical gradient: `∂L/∂w ≈ (L(w+ε) - L(w-ε)) / (2ε)` with ε = 1e-5
  4. Asserts `|analytical - numerical| < 1e-4` (relative tolerance)
- Run this as part of `npm test` (not skipped).

**Verification:**
- All gradient checks pass (within tolerance).
- Catches any gradient computation bugs introduced in Phase 1 (optimizer changes).

---

## Breaking Changes (v0.2.5)

The following changes are **intentional breaking changes** that require migration notes:

| Change | Description | Migration |
|--------|-------------|-----------|
| P1.3 `Conv1D.backward()` | `backward()` no longer updates weights as a side effect. It now uses the optimizer factory (passed at construction time) internally. Existing code calling `conv.backward()` directly will observe weight updates as before, but the learning rate behavior changes (from hardcoded 0.001 to the configured optimizer). | If you were using `Conv1D` standalone with a custom training loop, ensure you pass an `optimizerFactory` to the constructor. The `backward()` method still returns `dX` and updates weights internally — behavior is preserved, just with configurable lr. |
| P1.3 `Conv1D.backward()` return | `backward()` signature does not change — it still returns `number[]`. | No caller signature changes. |
| P3.2 `NetworkN.predict(training)` | `predict()` gains an optional second parameter `training: boolean = false`. Existing calls without the second argument continue to work (inference mode). | Add `training: true` when calling `predict()` during training loops that need dropout. |
| P4.1 `NetworkTransformerRL` pooling | Default pooling remains `'weighted'` (backward compatible). `'max'` pooling backward is now correctly implemented (argmax routing). | No migration needed for existing code. |

---

## Out of Scope

- **GPU acceleration** — not planned for v0.2.5
- **Automatic differentiation** — hand-coded gradients remain
- **Batch normalization integration into NetworkN** — `BatchNorm` exists but is not integrated into `NetworkN` (separate feature)
- **Convolutional layers beyond Conv1D** — Conv2D, pooling layers not planned
- **Learning rate scheduler integration into Trainer** — `LRScheduler` exists but is not wired into `Trainer` automatically; users call it manually
- **Serialization of optimizer state** — saving/loading optimizer momentum/state is not in scope for v0.2.5

---

## Risk Analysis

### Phase 1 — Optimizer Unification
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking Conv1D backward behavior (hardcoded 0.001 lr → configurable optimizer) | Medium | Medium | Conv1D is standalone (no internal library callers). The lr=0.001 was undocumented and incorrect for any real task. Tests verify Adam/SGD/Momentum work. |
| Per-scalar optimizer memory overhead | Low | Low | LSTMLayer with hiddenSize=128 and inputSize=64: 4 gates × (128 × 192 + 128) ≈ 100K optimizers × 2 (m,v) = 200K objects. Acceptable for educational use. |
| EmbeddingMatrix already implements Serializable — no code change needed | N/A | N/A | P1.4 is a clarification only; no implementation risk. |

### Phase 2 — Trainer Enhancements
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Weight decay applied at wrong phase (post-update instead of gradient-level) | High (fixed) | High | The corrected approach applies `w *= (1 - lr*wd)` BEFORE each `train()` call, which is mathematically equivalent to L2 regularization. Document clearly in migration notes. |
| Early stopping triggering too early on noisy validation | Medium | Low | Require `minDelta` threshold so small fluctuations don't trigger early stop. Document tuning guidance. |
| Metrics computation slowing down training | Low | Low | Metrics are optional (`computeMetrics: false` by default). When enabled, overhead is O(N) per epoch. |
| ClipOptimizer wrapping per-scalar optimizers adds indirection | Low | Low | The wrapping happens once at construction time. No runtime overhead in the `step()` call path. |

### Phase 3 — NetworkN + Conv1D
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Residual connections with mismatched layer sizes | Medium | Medium | Only apply residual when `layerSizes[i] === layerSizes[i-1]`. Skip residual for size mismatches. |
| Dropout `predict(training)` parameter breaks backward compatibility | Medium | Medium | `training` parameter defaults to `false`, so existing `predict(x)` calls work unchanged. `train()` internally calls `predict(x, true)` — behavior preserved. |
| Dropout integration breaking NetworkN backward pass | Medium | High | Dropout backward passes gradient through the same mask. No weight changes. Test with and without dropout to verify equivalence. |
| Conv1D multi-channel breaking existing tests | Low | Low | `inputChannels` defaults to 1, so all existing single-channel tests remain valid. |

### Phase 4 — Remaining Features
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Max pooling gradient non-differentiability at ties | Medium | Low | Argmax-based routing with first-occurrence tie-breaking is the standard approach. Document this behavior. Gradient check in P4.3 verifies correctness. |
| Gradient check tests being brittle | Medium | Low | Use relative tolerance (1e-4) and ε = 1e-5, which is the standard approach. Some variance is expected. |
| DataLoader validation split + Trainer early stopping coordination | Low | Low | Trainer accepts validation data from DataLoader via `setValidationData()`. No overlap in split logic. |

---

## Verification Checklist

After all phases are implemented, all of the following must be true:

- [ ] `npm run build` passes (CJS + ESM + .d.ts)
- [ ] `npm test` passes — all 26+ test files, 229+ tests
- [ ] `LSTMLayer` accepts `optimizerFactory` parameter and uses per-scalar optimizers
- [ ] `GRULayer` accepts `optimizerFactory` parameter and uses per-scalar optimizers
- [ ] `Conv1D` accepts `optimizerFactory` parameter and uses per-scalar optimizers
- [ ] `EmbeddingMatrix` already satisfies `Serializable` — verified by `ModelSaver` round-trip
- [ ] `NetworkTransformerRL` implements `Serializable` (flat `getWeights/setWeights` via `getWeightsFlat/setWeightsFlat`)
- [ ] `ModelSaver.toJSON(net)` / `ModelSaver.fromJSON(net, json)` works for ALL network classes
- [ ] `Trainer` supports `weightDecay` option — weight norm decreases measurably with decay > 0
- [ ] `Trainer` supports `earlyStopping` with `patience` and `minDelta` — early exit verified
- [ ] `Trainer` supports `computeMetrics` with accuracy, precision, recall, F1 (only for classification targets)
- [ ] `Trainer` supports `clipValue` via `ClippedOptimizerFactory` wrapping
- [ ] `Trainer` accepts external validation data via `setValidationData()` (no internal split)
- [ ] `NetworkN.predict(x)` and `NetworkN.predict(x, true)` both work (training parameter)
- [ ] `NetworkN` supports `residual: true` with skip connections (gradient check passes)
- [ ] `NetworkN` supports `dropoutRate` with Dropout layers between hidden layers
- [ ] `Conv1D` supports `inputChannels > 1` with correct multi-channel forward/backward
- [ ] `NetworkTransformerRL` supports `pooling: 'avg' | 'max' | 'last' | 'weighted'`
- [ ] `NetworkTransformerRL` `'max'` pooling backward routes gradient to argmax positions
- [ ] `DataLoader` supports `validationSplit` with `getValidationData()` (80/20 split verified)
- [ ] `tests/GradientCheck.test.ts` exists with proper finite-difference gradient verification for all major classes
- [ ] `src/index.ts` exports all new/changed APIs correctly
- [ ] Zero `TODO` or `FIXME` markers introduced in changed files
- [ ] All breaking changes from the Breaking Changes section are documented and tested

---

## File Manifest

### New Files
| File | Purpose |
|------|---------|
| `tests/GradientCheck.test.ts` | Finite-difference gradient verification for all major classes |

### Modified Files
| File | Changes |
|------|---------|
| `src/LSTMLayer.ts` | P1.1 — optimizer factory support, per-scalar optimizers, clipValue support |
| `src/GRU.ts` | P1.2 — optimizer factory support, per-scalar optimizers, clipValue support |
| `src/Conv1D.ts` | P1.3 — optimizer factory (breaking change: lr behavior), P3.3 — multi-channel |
| `src/NetworkTransformerRL.ts` | P1.5 — flat Serializable, P4.1 — configurable pooling (max with argmax backward) |
| `src/Trainer.ts` | P2.1 — weight decay, P2.2 — early stopping, P2.3 — classification metrics, P2.4 — gradient clipping |
| `src/DataLoader.ts` | P4.2 — validation split |
| `src/NetworkN.ts` | P3.1 — residual connections, P3.2 — Dropout integration (predict gains training parameter) |
| `src/optimizers.ts` | P2.4 — ClipOptimizer wrapper + ClippedOptimizerFactory |
| `src/index.ts` | Export all new APIs |