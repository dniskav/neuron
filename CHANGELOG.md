# Changelog

All notable changes to `@dniskav/neuron` are documented here. The format is loosely based on [Keep a Changelog](https://keepachangelog.com/), without following its schema strictly. The project is 0.x: minor versions may contain incompatible changes.

Most numeric claims in the 0.4.0 section were checked by running the same case against 0.3.2 and the current code.

## 0.4.0 — 2026-11-13

### Read this first

If you are upgrading from 0.3.2, these are the points that matter most:

1. **Several models were training in the wrong direction in 0.3.2.** The loss went up instead of down in `RNN`, `TCN`, `Seq2Seq`, `NetworkTransformer` and `NetworkTransformerRL`. Other models (`Autoencoder`, `VAE`, `GAN`, `ContrastiveLearning`, `Word2Vec`, `NetworkN` with `residual`) trained with incorrect gradients. In 0.4.0 the same calls lower the loss.
2. **Weights saved with 0.3.2 and learning rates tuned against 0.3.2 are not comparable.** A weights file for any of the models above can still be loaded if the shape matches, but it is not a good starting point for the corrected training. Retrain, and re-tune `lr`.
3. **The step size of `NetworkN.train`, `Network.train` and `Autoencoder.train` changed.** They now take the gradient step of the mean squared error that they return. Scale recipes:

   ```ts
   // NetworkN.train with n outputs, and Network.train with ONE output:
   //   lr_new = lr_old * n / 2
   // Exact with the default optimizer (SGD) or Momentum, without `residual` and with
   // dropoutRate 0. With Adam keep the same lr (Adam is insensitive to the gradient scale).
   // With `residual` there is no recipe, and with dropout it is only approximate.
   // n = 1 doubles the effective step, so lr_new = lr_old / 2 gives the old dynamics.
   const lrNew = lrOld * nOutputs / 2

   // Autoencoder.train / trainBatch, d = inputSize (APPROXIMATE):
   const lrNewApprox = lrOld * inputSize / 2
   ```

   For `Network.train` with several outputs there is no recipe: 0.3.2 only trained output 0. Retrain. For `Autoencoder` the recipe reproduces the first step of the decoder but not that of the encoder, because the encoder gradient also changed; after a few steps the decoder differs too. Retraining is recommended.
4. **`lr = 0` is a valid no-op** in the networks and in the generative and sequence models, but it **throws** in `Trainer` (0.3.2 accepted it) and in `Perceptron.train`, `LinearRegression.fitGD`, `LogisticRegression.train` and `SoftmaxRegression.train` (they already rejected it). NaN, Infinity and negative `lr` throw in all of them.
5. **Many methods now throw on invalid input** (empty data, wrong lengths, NaN, bad options) before touching any state. Code that relied on silent behavior must now handle or avoid the error.

### Breaking changes

#### Training and gradients

The following changes alter the numbers a training run produces. There is no `lr` factor that reproduces the old trajectories unless stated. Retrain.

**Training metrics**

- **`Trainer` with `computeMetrics`: `f1`.**
  - Before: `f1` was `2PR / (P + R)` computed from the macro precision and the macro recall.
  - Now: `f1` is the mean of the per-class F1, the same definition as `Metrics.f1Score` (scikit-learn `average='macro'`). A class with precision + recall = 0 contributes 0. `accuracy`, `precision` and `recall` are unchanged.
  - Effect: the new value is never higher than the old one, and equal when precision and recall are the same for every class. Example with confusion matrix `[[1,2,0],[0,1,0],[0,0,4]]`: 7/9 before, 2/3 now.
  - Migration: recalibrate any threshold or comparison that used the `f1` reported by `Trainer`.
  - Note: `Trainer` averages over as many classes as the network has outputs; `Metrics` averages up to the highest label present in the data. They differ only when the last classes have no samples and no predictions.

**Sequence and attention models**

- **`RNN.backward`, `TCN.train`, `Seq2Seq.trainStep`, `NetworkTransformer.train`, `NetworkTransformerRL.train`.**
  - Before: the loss went up while training. Same case measured on 0.3.2: `RNN(1,4,1)`, 60 steps at `lr` 0.05, 0.5701 to 6.6e14; `TCN(1,4,2,2,1)` at `lr` 0.3, NaN; `Seq2Seq(1,4,1)` at `lr` 0.1, 0.6106 to 9.0e27.
  - Now: the same calls lower the loss. `RNN` 0.5701 to 0.0290; `TCN` 0.6199 to 0.0369; `Seq2Seq` 0.6106 to 0.1253.
  - Migrate: retrain. `lr` values tuned against the broken training and weights saved with it are not comparable (see "Saved weights").
- **`TCN.train`.**
  - Before: the output gradient had the wrong sign, and the output projection moved with `lr/T` while the convolution layers used `lr` (they ran T times faster).
  - Now: correct sign; projection and convolutions both use `lr`.
  - Migrate: retrain. A stable `lr` is different now (see "`TCN` stable `lr`").
- **`NetworkTransformer.train`.**
  - Before: the cross-entropy gradient was not divided by the number of positions.
  - Now: it is divided by `count`, so the direct SGD step (embeddings, LayerNorm, biases) is `count` times smaller. Measured with `count` = 4: the change in `tokenEmb` after one step went from 4.162e-2 to 1.038e-2. Adam on the weight matrices is invariant to this scale.
  - Migrate: with a mask (for example Sudoku), raise `lr` by about the number of masked positions to keep the SIZE of the direct SGD step. The direction is not comparable: in 0.3.2 that step went the wrong way (see "`RNN.backward`, `TCN.train`, `Seq2Seq.trainStep`, `NetworkTransformer.train`, `NetworkTransformerRL.train`").
- **`Seq2Seq.trainStep`, `LSTMLayer.backprop`.**
  - Before: the encoder did not receive the exact decoder gradient. `LSTMLayer.backprop(dh_seq, lr)` returned `undefined`.
  - Now: the encoder receives the exact gradient (full BPTT) with scale `lr * T_in / T_out`. `LSTMLayer.backprop(dh_seq, lr, dcFinal?)` returns `{ dh0, dc0 }`.
  - Migrate: retrain. Code that used the (undefined) return value of `backprop` now receives an object.
- **`AttentionHead`, `MultiHeadAttention`, `TransformerBlock`, `LayerNorm` backward.**
  - Before: the `dX` propagated backward depended on `lr`, because weights were updated before `dX` was computed. `LayerNorm.backward` used a `gamma` already updated by earlier positions. Measured on `AttentionHead(4,2,2)`, `dX[0][0]` with `lr` 0, 0.05 and 0.5: 0.012523, 0.018809, 0.075385.
  - Now: `dX` is the same for any `lr` (0.012523 in the three cases). `LayerNorm` keeps the `gamma` of the forward pass.
  - Migrate: no action. It only affects models with several stacked layers; retrain them.
- **`NetworkTransformerRL.train`.**
  - Before: several Adam steps per call on `inputProj`.
  - Now: one Adam step per call on `inputProj`, with the summed gradient.
  - Migrate: retrain.
- **`RNN`, `GRULayer`, `LSTMLayer`, `activations.tanh`.**
  - Before: `tanh` was computed as `(e^{2x}-1)/(e^{2x}+1)`, which gives NaN for x above about 355 (`tanh.fn(355)` was NaN).
  - Now: `Math.tanh`, which returns 1 there.
  - Migrate: none.

**Feed-forward and unsupervised models**

- **`NetworkN` with `residual`.**
  - Before: the gradient of residual networks differed from the true gradient.
  - Now: the gradient is correct. `trainWithDeltas` and `inputGradient` throw if the LAST layer has `residual`: `NetworkN.trainWithDeltas: residual on the last layer needs dL/d(output), but only deltas (already multiplied by the activation derivative) are given. Use train(), or disable residual on the last layer.` This reaches `Autoencoder`, `VAE` and `GAN`: `new Autoencoder(4, [4], 4, [4], { residual: true }).train(x, lr)` trained in 0.3.2 and now throws.
  - Migrate: remove `residual` from the last layer (or from the output layer of `Autoencoder`, `VAE`, `GAN`) and retrain the rest.
- **`BatchNorm.backward`.**
  - Before: `dX` was wrong. `BatchNorm(2, m)`, `forward([1,2])`, `backward([1,1])` gave `[3.03, 2.71]` with `m` 0.1.
  - Now: exact derivative: `[0.278, 0.199]` with `m` 0.1. With `momentum` 0 the backward gradient is now 0.
  - Migrate: retrain networks that contain `BatchNorm`.
- **`Autoencoder.train`.**
  - Before: the encoder gradient was computed without the derivative of the encoder output activation, and the decoder was traversed by hand (it ignored `residual` and dropout).
  - Now: the encoder gradient includes the activation derivative and comes from `decoder.inputGradient(...)`.
  - Migrate: retrain. See "Step size and learning rate" below for the scale of the step.
- **`VAE` encoder.**
  - Before: the encoder output used the activation from the options (or sigmoid), so `mu` was in (0, 1) and the standard deviation was at least 1. The encoder did not learn from the reconstruction, only from the KL term.
  - Now: the encoder output is linear and the encoder learns from the reconstruction. `options.activations` no longer applies to the encoder output.
  - Migrate: retrain. See "`VAE` `klWeight`" below, and "`VAE` constructor" under "Generative models" for `options.activations`.
- **`VAE.train` decoder.**
  - Before: the decoder delta was `(x - x̂)/n`, without the derivative of the output activation and without the factor 2.
  - Now: the decoder delta is `2(x - x̂)/n * dfn(x̂)`.
  - Migrate: retrain. No single `lr` factor reproduces the old step, because it depends on `dfn(x̂)`.
- **`Autoencoder`, `VAE`, `GAN` with `dropoutRate > 0`.**
  - Before: each sub-network drew a new mask in every forward pass, and the update used a different mask from the one used for the loss. `VAE.train` computed the loss without dropout.
  - Now: one mask per sub-network and step, reused with `reuseMasks = true`. `VAE.train` applies dropout in the training forward pass.
  - Migrate: trajectories with dropout change even with the same seed. With `dropoutRate` 0 nothing changes because of this.
- **`GAN.trainStep` generator.**
  - Before: the generator received a uniform delta `gError/n` per output, without going through the discriminator.
  - Now: the generator receives the real gradient of `-log D(G(z))` through the frozen discriminator. The discriminator must end in a sigmoid (see "`GAN` constructor").
  - Migrate: retrain and re-check the generator `lr`.
- **`ContrastiveLearning`.**
  - Before: the projection head was `[relu, relu]` (`project()` was never negative and could collapse to all zeros), the step used finite differences with `eps` 1e-4, and `cosineSimilarity` was `dot/denom` with `denom < 1e-10` returning 0.
  - Now: the head is `[relu, linear]` (`project()` can be negative), the step is simultaneous (all gradients at the original weights) with `eps` 1e-6, and `cosineSimilarity = dot / (‖a‖‖b‖ + 1e-8)`.
  - Migrate: retrain. Do not assume `project() >= 0`.
- **`Word2Vec.train`, skip-gram and CBOW.**
  - Before: approximate gradient. Skip-gram diverged at `lr` 0.1 (8 dimensions, 20 repeated sentences, 10 epochs, loss 1.821 to 25.631).
  - Now: exact gradient of the pair; the same case goes from 1.814 to 1.681.
  - Migrate: retrain. Low `lr` values chosen to avoid the divergence can be raised.
- **`MaxPool2D.backward`.**
  - Before: with overlapping windows (`stride < poolSize`) it added gradient to positions that were not the maximum of the window. `MaxPool2D(2, 1)` on a 3x3 input with an all-ones output gradient gave a gradient sum above the number of windows (4), depending on the input (9 for the input 1..9).
  - Now: the gradient sum equals the number of windows (4 in that case).
  - Migrate: retrain networks with overlapping `MaxPool2D`. No change when `stride = poolSize`.
- **`PCA`.**
  - Before: `explainedVarianceRatio` divided by the sum of the retained components, so it always added up to 1. For four points in 3-D with `new PCA(1)` it returned `[1]`.
  - Now: it divides by the total variance (`[0.6798]` in that case). The sum is no longer 1 when you keep fewer components than dimensions.
  - Migrate: update any code that assumed the ratios add up to 1. Do not use axes of zero variance as an orthonormal basis (see Known limitations).
- **`DecisionTree.fit`.**
  - Before: it required a gain greater than 0, so it could not split XOR (predictions `[0,0,0,0]`).
  - Now: it accepts zero-gain splits and learns XOR (`[0,1,1,0]` with `maxDepth` of at least 2).
  - Migrate: none; trees may be deeper.
- **`Metrics.f1Score` and `Metrics.r2Score`.**
  - Before: macro `f1Score` (no `positiveClass`) was the harmonic mean of macro precision and macro recall (0.6933 for `[0,0,1,1,2,2]` against `[0,1,1,1,2,0]`). `r2Score` with a constant `yTrue` always returned 1.
  - Now: macro `f1Score` is the mean of the per-class F1 (0.6556, the scikit-learn convention). `r2Score` with a constant `yTrue` returns 1 only if the residual sum of squares is 0, otherwise 0.
  - Migrate: recompute any tables and thresholds that used macro F1 or degenerate R².
- **`Network.train` with several outputs.**
  - Before: with more than one output it only trained output 0 and returned the squared error of that output. `target` had to be a `number`.
  - Now: it trains all outputs. `target` can be a `number` or a `number[]`; with more than one output a single `number` throws `Network.train: expected 2 targets (number[]), got a single number`. It returns the mean MSE. `Network` now satisfies `TrainableNetwork` (`new Trainer(new Network(...))` threw in 0.3.2 and works now).
  - Migrate: with one output only the scale changes (see "`Network.train`"). With several outputs there is no recipe, because the other outputs did not learn before: retrain.
- **`LinearRegression.fitGD`, `SoftmaxRegression.train`.**
  - Before: every call reset the weights. (`LogisticRegression.train` was already incremental.)
  - Now: both are incremental and continue from the current weights when the shape matches (two calls of 5 epochs match one call of 10). With a different shape they reset; in `SoftmaxRegression` a label greater than or equal to K throws.
  - Migrate: create a new instance to start from scratch.
- **`LinearRegression.predict`, `getCoefficients` with weights set by hand.**
  - Before: with `weights = [2, 1]` assigned by hand, `predict([3])` threw `expected 0 features` and `getCoefficients()` returned `{ weights: [], bias: 2 }`.
  - Now: `predict([3])` is 7 and `getCoefficients()` is `{ weights: [2], bias: 1 }`.
  - Migrate: none (bug fix).
- **`Trainer` with `restoreBestWeights`.** New option (see "New (additive)"). It restores the weights of the best validation epoch; it does not restore the optimizer state.

#### Step size and learning rate

- **`NetworkN.train`.**
  - Before: output delta `(t - p) * dfn`, the gradient of `0.5 * Σ(t - p)²`.
  - Now: output delta `(2/n)(t - p) * dfn`, the gradient of the mean squared error it returns (n = number of outputs). The step is `2/n` times the old one; with one output the effective `lr` doubles. `trainWithDeltas` and `inputGradient` do NOT change scale.
  - Migrate: `lr_new = lr_old * n / 2` reproduces the old dynamics with the default optimizer (SGD) or `Momentum`, without `residual` and with `dropoutRate` 0. With `Adam` keep the same `lr`; with `residual` there is no recipe (the gradient changed, see "`NetworkN` with `residual`"); with dropout the recipe is approximate. Checked on `[2,3,n]` with n = 1, 2, 3, 4 (20 epochs of XOR, `lr` 0.2, fixed seed): maximum weight difference 0, 0, 2.2e-16 and 0 with the recipe.

    ```ts
    // before (0.3.2)
    net.train(x, target, 0.2)
    // now, same dynamics for a network with 1 output
    net.train(x, target, 0.2 * 1 / 2) // 0.1
    ```
- **`Network.train`.**
  - Before: with one output, delta `(t - p) * dfn`.
  - Now: delta `(2/n)(t - p) * dfn`.
  - Migrate: `lr_new = lr_old * n / 2`. With n = 1 this is `lr / 2`; checked on `Network(2,3,1)`, 20 epochs of XOR, `lr` 0.2, maximum difference 0. With n > 1 there is no recipe (see "`Network.train` with several outputs").
- **`Autoencoder.train` and `trainBatch`.**
  - Before: output delta `(x - x̂) * dfn`.
  - Now: output delta `(2/d)(x - x̂) * dfn`, where `d` is `inputSize` (the length of `x`, not `latentSize`). The step is `d/2` times smaller.
  - Migrate: `lr * d / 2` reproduces the first step of the decoder but it is only APPROXIMATE, because the encoder gradient also changed. Measured with `lr_old` 0.1 and `lr_new = 0.1 * d / 2`, after 30 epochs the weights still differ (up to 1.3e-1 in the encoder and 4.8e-3 in the decoder in the smallest case). Retraining is recommended.
- **`TCN` stable `lr`.** Because training was broken in 0.3.2, there was no valid `lr` to compare with. In one measured case (`TCN(1,4,2,2,1)`, `x = sin(1.3t + 0.4)`, 16 steps, 300 updates, 5 seeds) the loss went down for `lr` between 0.21 and 0.65, went up at 1.0 and diverged from 2.0. Use `lr` around 0.2 to 0.4 for this kind of task. This is one measured case, not a law.
- **`VAE` `klWeight`.**
  - Before: the KL term had a fixed weight of 1.
  - Now: a new sixth constructor parameter `klWeight` (default 1; the default behavior does not change). In one measured case (`VAE(4,[6],2,[6])`, 6 points in [0,1], 3000 epochs, `lr` 0.1) `klWeight` 1 collapsed (the spread of `mu` was 0.003) and `klWeight` 0.003 learned.
  - Migrate: for small datasets in [0,1] try `klWeight` around 0.003. See also Known limitations.
- **`TSNE` stable `lr`.** No code change. In one measured case (n = 30, two clusters in 5-D, `perplexity` 5, 300 iterations) the default `lr` of 200 diverged and values of 5 to 10 worked. For small n use `lr` of about n/12 to n/4; 200 is the right order for large n. The default `perplexity` (30) throws when n is 30 or less (this was already true in 0.3.2).

#### Validation that now throws

Every new validation throws `Error('Class.method: ...')` BEFORE touching any state. If you compare error messages by text, update them. For example: `DecisionTree` `maxDepth must be positive` is now `DecisionTree: maxDepth must be positive integer or Infinity (no depth limit), got X`.

**Weights and serialization inputs**

- **`setWeights` of `WeightMatrix`, `BiasVector`, `EmbeddingMatrix`, `LayerNorm`, `BatchNorm`, `AttentionHead`, `MultiHeadAttention`, `TransformerBlock`, `NetworkTransformer`, `NetworkTransformerRL`, plus `setWeightsFlat` and `setWeightsStructured`.**
  - Before: any length and NaN/Infinity were accepted (a short array left `undefined`, a long one was cut).
  - Now: they throw (`Class.setWeights: expected array of length N, got M` or `invalid value at index i: NaN`) before writing anything.
  - Migrate: a model that diverged to NaN cannot be restored with `setWeights(getWeights())`. Keep the last valid weights instead.
- **`NetworkN.setWeights`, `Network.setWeights`, `Conv1D.setWeights`, `Conv2D.setWeights`, `LSTMLayer.setWeightsFlat`, `GRULayer.setWeightsFlat`, `NetworkLSTM.setWeightsFlat`.** Same as the previous entry: before they accepted a wrong length and NaN (the state became NaN); now they throw and leave the state intact.
- **`LSTMLayer.setWeights`, `GRULayer.setWeights`, `NetworkLSTM.setWeights` (structured).** Before they accepted NaN, missing rows and missing dense layers. Now they throw, for example `LSTMLayer.setWeights(forgetGate.W): invalid value at [0][0]: NaN`, `expected 3 rows, got 2`, `NetworkLSTM.setWeights: expected 2 dense layers, got 1`.
- **`NetworkN.getWeights` and `setWeights` with dropout.** Both now clear the saved dropout masks. A `getWeights()` call between `predict(x, true)` and `trainWithDeltas` or `inputGradient` with `reuseMasks = true` throws `NetworkN.trainWithDeltas: reuseMasks = true needs the dropout masks of a previous predict(x, true), but none are saved ...`. Migrate: do not call `getWeights()` between the forward pass and the update. With `dropoutRate` 0 nothing changes.
- **`ModelSaver.toJSON` and `fromJSON`.**
  - Before: the JSON was `{ weights, timestamp }` and was not validated on load (it accepted a wrong `format`, a wrong length and `null`).
  - Now: the JSON is `{ format: 1, length, structure?, weights, timestamp }` (`structure` only if the model has one: `NetworkN` does, `Network` does not). `fromJSON` validates everything before writing: `ModelSaver.fromJSON: unsupported format 2 (expected 1)`, `ModelSaver.fromJSON: the model expects 9 weights and the file has 3`, `ModelSaver.fromJSON: weights[0] is not a finite number (value: null); NaN and Infinity are saved as null`, and a mismatch in `structure`.
  - Migrate: an OLD JSON without `format` still loads if the length matches. Save again to get `format: 1`. NaN and Infinity are saved as `null` and now prevent loading.

**Training and sequence inputs**

- **`LSTMLayer.backprop`, `GRULayer.backprop`, `NetworkLSTM.train`.** Before, a wrong sequence length or NaN gave a silent no-op or zeros. Now they throw (`expected dh_seq of length 2, got 1`, `expected targets of length 2, got 1`, `invalid value at index 0: NaN`). An error does not clear the stored trajectory, so you can retry with the right call. Migrate: a trajectory of T steps needs T rows.
- **`NetworkLSTM` constructor.** `denseStructure = []` used to build the object and fail later with a `TypeError`. It now throws in the constructor: `NetworkLSTM: denseStructure must have at least one dense layer (got []), e.g. new NetworkLSTM(inputSize, hiddenSize, [outputSize])`.
- **`Trainer`.**
  - Before: it accepted `clipValue` (stored and ignored), `epochs` 0 or 1.5, `lr` 0, `lr` NaN (it failed later), an empty dataset, `earlyStopping` without validation data and `patience` 0.
  - Now: `clipValue` is REMOVED from `TrainerOptions` (it does not compile in TypeScript, and a JavaScript caller that passes it is ignored). `Trainer.train` throws `Trainer.train: ...` for `epochs` (integer greater than 0), `lr` (finite and greater than 0, so `lr = 0` throws), `lrDecay` (in (0, 1]), `weightDecay` (at least 0), `patience` (integer at least 1), `minDelta` (at least 0), an empty dataset, `earlyStopping` without `setValidationData()` or with empty data, and `restoreBestWeights` without `earlyStopping`. `setValidationData` without `earlyStopping` is stored and ignored.
  - Migrate: for gradient clipping use `ClippedOptimizerFactory`. Fix the options.
- **`NetworkN`, `Network`, `Neuron`, `NeuronN`, `NetworkLSTM`, `LSTMLayer`, `GRULayer`, `NetworkTransformer`, `NetworkTransformerRL`.**
  - Before: `NetworkN.train`, `NetworkN.trainWithDeltas`, `NeuronN.train`, `NetworkLSTM.train`, `LSTMLayer.backprop`, `GRULayer.backprop`, `NetworkTransformer.train` and `NetworkTransformerRL.train` did not validate `lr`. `lr = NaN` did not throw and left every weight as NaN, and a negative `lr` trained in the wrong direction. `Network.train` and `Neuron.train` rejected NaN but accepted a negative `lr`.
  - Now: all of them throw `Class.method: lr must be a finite number >= 0, got X` for NaN, Infinity, a negative or a non-numeric `lr`, before any weight, state or stored trajectory is touched. `lr = 0` is still a valid no-op.
  - Migration: pass a finite `lr >= 0`. `Network.train` and `Neuron.train` used to throw `expected finite number, got X`; update code that matches the old message.
  - Not validated: the low-level building blocks (`Conv1D`/`Conv2D` `backward`, `BatchNorm`, `LayerNorm`, the attention classes, `WeightMatrix`/`BiasVector`/`EmbeddingMatrix` `update`, the optimizers) and the `lr` options of `SOM` and `TSNE`.
- **`RNN`, `TCN`/`CausalConv1D`, `Seq2Seq`.** Before: `lr` NaN or -1 was accepted and corrupted weights, an empty sequence returned silently (`RNN.backward([])` gave 0, `Seq2Seq.encode([])` a zero context), NaN in rows passed, and constructors accepted NaN or 1.5. Now they throw before touching state: `lr` must be finite and at least 0 (**`lr = 0` is a valid no-op**), empty sequences (including `RNN.backward([])`, `Seq2Seq.encode([])`, `TCN.forward([])`), wrong row width or target length, NaN or Infinity in rows, targets and the context (`ctx.h`, `ctx.c`), and constructors need finite integers greater than 0. Migrate: validate inputs, or use `lr = 0` for a no-op.
- **`Autoencoder`, `VAE`, `GAN`, `ContrastiveLearning`.** Before: empty batches and `lr` NaN or negative gave NaN and corrupted weights (`GAN.trainStep([])` returned `{ dLoss: NaN, gLoss: NaN }`), and one invalid row in `Autoencoder.trainBatch` left the earlier rows trained. Now they throw on empty batches or pairs, wrong width, non-finite values and `lr` NaN or negative, before touching weights. `Autoencoder.trainBatch` validates `lr` and ALL rows up front. `lr = 0` is a valid no-op. Migrate: clean the batches.
- **`EarlyStopping`, `LRScheduler`, `DataLoader`, `Word2Vec.train`.** Before they accepted `patience` 0 or 1.5, negative `minDelta`, `update(NaN)`, `stepDecay` with `epochsDrop` 0, `exponentialDecay` with NaN, `cosineAnnealing` with `maxEpochs` 0, `batchSize` 0 or 1.5, and `Word2Vec.train` with `epochs` 0 or negative `lr`. Now they throw: `patience` integer at least 1, `minDelta` finite and at least 0, `epochsDrop > 0`, `decayRate` finite, `maxEpochs > 0`, `batchSize` integer at least 1, `epochs` integer at least 1, `lr` finite and at least 0 (`Word2Vec.train` with `lr = 0` is still valid). Migrate: fix the options.

**Convolution and shape**

- **`Conv1D`, `Conv2D` constructors.** Before: `stride` 0, 1.5 or NaN was accepted, and `Conv2D` with `valid` padding and a kernel larger than the input gave a `[0,0,1]` shape. Now: `Conv1D: stride must be a finite integer >= 1, got 0` and `Conv2D: kernel 3x3 cannot exceed input 2x2 with valid padding`. (`Conv1D` with `valid` and `k > L` already threw in 0.3.2.) Migrate: fix `stride` and sizes.
- **`Conv1D`, `Conv2D` with `padding: 'same'` and an even kernel.**
  - Before: the output size was wrong. `Conv1D` with `L` = 8, `k` = 4, `stride` 1 gave 7. `Conv2D` with `H` = 5 and `k` = 2 or 4, `stride` 1 gave 6x6 (`H + 1`). With an odd kernel it was always right.
  - Now: the output is `ceil(L / stride)` with asymmetric padding (the extra zero goes on the right): `Conv1D` gives 8 and 4 in those cases, `Conv2D` gives 5x5 and 3x3. It is not identical to TensorFlow when `stride > 1`.
  - Migrate: resize the downstream layers (`Flatten`, dense layers) that depend on the output shape with an even kernel. Saved weights of those layers are not compatible.

**Classical ML, unsupervised and utilities**

- **`LogisticRegression.train`, `SoftmaxRegression.train`, `LinearRegression.fitGD`.** Before, data was not validated (`y` = 2 in logistic was accepted, `y` = 1.5 or NaN in softmax gave `Invalid array length`, irregular rows or NaN passed). Now they validate before touching weights: constant row length, finite values, `y` in [0, 1] for logistic (`y[1] = 2 is outside [0, 1]`) and integer labels of at least 0 for softmax (`label y[1] = 1.5 must be an integer >= 0`). `lr = 0` throws in these three (they already rejected `lr` less than or equal to 0). Migrate: clean the data.
- **`LinearRegression.fitNormal`.** Before, irregular rows or NaN returned NaN weights without throwing. Now it throws (`X[1]: expected array of length 2, got 1`) and does not touch the weights.
- **`Metrics`.**
  - Before: `accuracy([], [])` was 0, different lengths gave 0.5, `mae([], [])` was NaN, `precision(..., -1)` was 0, `confusionMatrix` with a label of at least `numClasses` failed with a `TypeError`, and `auc` with one class gave 0.
  - Now these throw `Metrics.<fn>: ...`:
    - empty inputs, including `accuracy([], [])`;
    - different lengths in `accuracy`, `precision`, `recall`, `f1Score`, `confusionMatrix`, `rocCurve`, `auc`, `mae`, `rmse`, `r2Score` and `classificationReport`;
    - a `positiveClass` that is not an integer of at least 0, and an invalid `numClasses`;
    - a label of at least `numClasses`;
    - `auc` with a single class (`Metrics.auc: yTrue contains a single class; AUC needs both classes`).
  - Not everything is validated. Negative or non-integer labels throw ONLY in `confusionMatrix`, `precision`, `recall`, `f1Score` and `classificationReport` (for example `Metrics.precision: yTrue[0] = -1 is not a non-negative integer label`); `accuracy([-1], [0])` does NOT throw. `mae`, `rmse` and `r2Score` with NaN still return NaN, and `perplexity` does not validate.
  - Migrate: handle or clean the inputs; do not expect 0 from empty inputs.
- **`SOM`.** Before: non-integer sizes were truncated silently, `quantizationError([])` was NaN, and `train` accepted NaN or `epochs` 1.5. Now it throws (`SOM: rows, cols and inputSize must be positive integers, got 2.5×2×2`, `dataset X must be non-empty`, `epochs must be an integer >= 1, got 1.5`, and NaN in the data).
- **`KMeans`, `PCA`.** Before: `KMeans.predict` with the wrong dimension returned a cluster, `PCA.transform` returned `[[NaN]]`, and `fit` accepted irregular or NaN data. Now `predict` and `transform` throw on a wrong dimension and on NaN or Infinity, and `fit` throws on irregular or non-finite `X` before touching state.
- **`GaussianNaiveBayes`, `DecisionTree`.** Before: `fit` accepted irregular rows or NaN, `predict` with a wrong dimension or NaN returned 0, and `DecisionTree({ maxDepth: 2.5 })` was accepted. Now `fit` validates, `predict` validates dimension and finiteness, and `DecisionTree` requires `maxDepth` to be an integer of at least 1 or `Infinity` (`DecisionTree: maxDepth must be positive integer or Infinity (no depth limit), got 2.5`) and `minSamplesSplit` to be an integer of at least 2.
- **`PositionalEncoding.encode` and `encodeSequence`.** Before: a negative or non-integer `pos` and `dModel` 0 or 2.5 were accepted (`encode(-1, 4)` returned 4 values). Now: `pos must be an integer >= 0` and `dModel must be an integer >= 1, got 0`. An odd `dModel` is still allowed.
- **`DataAugmentation`, `MaxPool2D`, `Flatten`.** Before: `shuffle` and `split` with different lengths returned `y` with `null`, `normalize([])` and `standardize([])` returned empty, `MaxPool2D.forward` with an input smaller than `poolSize` returned `[]`, and `Flatten.forward([])` threw a `TypeError`. Now they throw clear messages, for example `DataAugmentation.shuffle: X has 2 rows but y has 1 labels`, `normalize: X is empty`, `MaxPool2D.forward: input height 2 is smaller than poolSize 3`.
- **`Tokenizer`, `DatasetLoader.fromCSV`.** Before: `maxVocab` smaller than the number of special tokens was accepted, empty numeric cells became NaN and irregular rows were encoded as categories. Now: `Tokenizer: maxVocab must be >= number of special tokens (4)`, `DatasetLoader.fromCSV: empty value in numeric column "b" at row 1.` and `row 2 has 1 fields, expected 2.` Migrate: clean the CSV.
- **`HopfieldNetwork`.** Before: `recall` and `energy` accepted non-bipolar input or NaN, and `binarize([NaN])` and `unbinarize([NaN])` returned `[1]`. Now they throw (`input values must be +1 or -1, got 2 at index 1`). `store` already threw on non-bipolar input and now prints a console warning ONCE per network when the stored patterns exceed `max(1, floor(0.138 * N))`.
- **`LossPlotter`.** Before: `add(NaN)` and `width` 0 or 1.5 were accepted, and a constant series had a wrong middle label on the Y axis. Now it throws on NaN or Infinity and on `width` or `height` that are not positive integers, and the label is correct for a constant series.
- **`WeightInspector`.** Before: `deadThreshold` NaN or -1 and `countDeadReLUs` with irregular rows or NaN were accepted. Now they throw (`threshold must be a non-negative number, got NaN`, `row 1 has length 1, expected 2`). With NaN in the weights the statistics are still NaN and do not throw.

**Generative models**

- **`GAN` constructor.** Before: any output activation of the discriminator was accepted. Now it throws unless the output is a sigmoid by value: `GAN: the discriminator output must use the sigmoid activation (...)`. `relu`, `tanh`, `linear` and `leakyRelu` throw. Migrate: end `discriminatorOptions.activations` with sigmoid.
- **`VAE` constructor.** Before: `klWeight` was ignored, and an `options.activations` of the wrong length made `NetworkN` throw `Expected 2 activations, got 1`. Now `klWeight` must be finite and at least 0, and `options.activations` needs `decoderHidden.length + 1` entries and must cover the hidden layers of the encoder (`VAE: options.activations needs 2 entries ...`).

**Large inputs**

- **`Value.backward`, `getWeights` and `getWeightsFlat`, `softmax`, `Trainer` metrics, `confusionMatrix`, `WeightInspector` with vectors above about 125,000 elements.** Before they failed with `RangeError: Maximum call stack size exceeded` (for example `Value.backward` with 100,000 nodes, or `NetworkN([150000, 1]).getWeights()`). Now they work with the same numeric output. No action needed.

#### Serialization and packaging

- **`ModelSaver` and dropout masks.** See "Weights and serialization inputs" above.
- **`LSTMLayer`, `GRULayer`, `NetworkLSTM` `getWeights` and `setWeights`.** Before they returned references to internal arrays (changing the result of `getWeights()` changed the model) and `setWeights` kept aliases to its input. Now both make deep copies. Migrate: none, unless you relied on the alias; in that case call `setWeights` explicitly.
- **`package.json`.**
  - Before: `exports` had `types` LAST, and there were no `engines`, `sideEffects`, `repository`, `homepage` or `bugs` fields.
  - Now: `exports["."]` lists `types` first, then `import` and `require`; `engines.node` is `>= 20`; `sideEffects` is `false`; `repository`, `homepage` and `bugs` are set. `dependencies` is still empty. The published tarball contains `dist`, `LICENSE`, `README.md` and `package.json`.
  - Migrate: Node older than 20 is no longer supported (npm shows an `engines` warning). Types now resolve under strict `moduleResolution`.
- **Repository files.** Not visible to people who consume the package: `LICENSE` (MIT) added, CI workflow added, publish workflow updated, `vitest.config.mts` replaces `vitest.config.ts`, `bun.lock` and an old `.tgz` removed, `typecheck` and `coverage` scripts added.
- **Saved weights.** Weights saved with 0.3.2 for `RNN`, `TCN`, `Seq2Seq`, `NetworkTransformer`, `NetworkTransformerRL`, `VAE`, `GAN`, `Autoencoder`, `ContrastiveLearning`, `Word2Vec` and `NetworkN` with `residual` came from training that raised the loss or used wrong gradients (see "Training and gradients"). They can be loaded when the shape matches, but they are not comparable with the corrected training. Retrain.

### New (additive)

- `NetworkN.inputGradient(inputs, outputDeltas, reuseMasks = false)` returns -dL/dx without changing weights.
- `NetworkN.trainWithDeltas(inputs, outputDeltas, lr, reuseMasks = false)` and `Dropout.reuseMask` / `Dropout.hasMask`. With `reuseMasks = true` it reuses the masks of `predict(x, true)`, and it throws if `dropoutRate > 0` and no masks are saved.
- `Trainer` option `restoreBestWeights`. It needs `earlyStopping` and validation data (otherwise it throws) and restores the weights of the best epoch.
- `LSTMLayer.backprop(dh_seq, lr, dcFinal?)` returns `{ dh0, dc0 }` (it returned nothing before). `GRULayer.backprop` still returns nothing.
- `VAE` takes a sixth constructor parameter `klWeight`, and `forward()` returns the extra key `eps`. `train` returns `{ totalLoss, reconLoss, klLoss }` as before.
- `Tokenizer.toJSON()` includes `specialTokens`. `fromJSON` still accepts older snapshots without it.
- `Network.train(inputs, number | number[], lr)`, and `Network` satisfies `TrainableNetwork`.
- `NetworkTransformerRL.setWeights`, `getWeights`, `setWeightsFlat` and `setWeightsStructured` are validated.
- `GAN` accepts any discriminator output activation that is equivalent to sigmoid by value, not only the exported `sigmoid` object (this is the same behavior as 0.3.2 and is not a breaking change).

### Fixed

- **`LossPlotter.render()` frame alignment.** The top border was one character longer than the rows, and Y-axis labels longer than 6 characters (for example `0.01417`) pushed the right border out. Now every line of the frame has the same length, equal to `width`: long labels widen the label column and the plot area gives up those columns. A title longer than the frame is cut. The footer and the options are unchanged. Code that compares the rendered string character by character will see the difference.

### Migration guide

1. **Retrain** every model listed under "Training and gradients". Do not reuse 0.3.2 weights as a starting point and do not carry over the old `lr`.
2. **Rescale `lr`** where you keep an old setting:
   - `NetworkN.train` with n outputs, and `Network.train` with one output: `lr_new = lr_old * n / 2` (exact with SGD or `Momentum`, no `residual`, no dropout; with `Adam` keep `lr`).
   - `Network.train` with several outputs: no recipe; retrain.
   - `Autoencoder`: `lr_new = lr_old * inputSize / 2` is approximate (it preserves the decoder step, not the encoder step); retrain.
   - `NetworkTransformer.train` with masks: raise `lr` by about the number of masked positions to keep the size of the direct SGD step.
   - `VAE` decoder: no factor exists.
3. **Remove `residual` from the last layer** of `NetworkN`, and from the output layer of `Autoencoder`, `VAE` and `GAN`.
4. **End the `GAN` discriminator with sigmoid**.
5. **Handle new errors.** Clean empty batches, NaN values, wrong lengths and invalid options (see "Validation that now throws"). Use `lr = 0` only where it is valid: it is a no-op in networks and in generative and sequence models, but `Trainer`, `LinearRegression.fitGD`, `LogisticRegression.train` and `SoftmaxRegression.train` throw.
6. **Replace `Trainer` `clipValue`** with `ClippedOptimizerFactory`.
7. **Resize downstream layers** after `Conv1D` or `Conv2D` with `padding: 'same'` and an even kernel.
8. **Update code that compares error messages** by text (see the note at the start of "Validation that now throws").
9. **Save your models again** with `ModelSaver.toJSON` to get `format: 1`. Use Node 20 or newer.

### Known limitations

- **`ContrastiveLearning` with raw initialization.** Some random seeds can start with a dead projection head, because the weights use a raw initialization.
- **`VAE` with `klWeight` 1 on small datasets.** The model can collapse (the latent code stops carrying information). A value around 0.003 learned in the measured case.
- **`PCA` with rank-deficient data.** The axes with variance 0 can have a norm close to 0 (about 1e-111 in the minimal case with 8 points in 8-D and `new PCA(8)`), so they are not an orthonormal basis. Do not use them as one.
- **`TSNE` with a small number of points.** The default `lr` of 200 diverges. Use a smaller `lr` (about n/12 to n/4).

## 0.3.2

- **New — NLP:** `Tokenizer` (char / word / whitespace modes, special tokens PAD/UNK/BOS/EOS, one-hot encoding, `fit` / `encode` / `decode` / `encodeBatch`, JSON serialization)
- **New — Data:** `DatasetLoader` (parse CSV and JSON into `DataPair`; auto one-hot encoding for string columns; returns `categoricalMaps` for decoding predictions)

## 0.3.1

- **New — Embeddings:** `Word2Vec` (Skip-gram + CBOW, full-softmax, cosine similarity, analogies), `TSNE` (binary-search perplexity, Student-t kernel, KL gradient, early exaggeration, seeded PRNG), `PositionalEncoding` (sinusoidal, Vaswani et al.), `LearnedPositionalEncoding` (trainable), `ContrastiveLearning` (NT-Xent, SimCLR encoder + projection head), `Augmenter` (noise, feature dropout, `makePair`)

## 0.3.0

- **New — Classical ML:** `Perceptron`, `LinearRegression` (normal equation + GD), `LogisticRegression`, `SoftmaxRegression`, `GaussianNaiveBayes`, `DecisionTree` (CART, Gini/MSE)
- **New — Unsupervised:** `KMeans` (K-Means++ init), `PCA` (power iteration + Hotelling deflation), `SOM` (Kohonen map), `HopfieldNetwork` (Hebbian storage + energy), `Autoencoder`
- **New — Deep Learning:** `Conv2D` (full forward/backward), `MaxPool2D` (position mask for exact backprop), `Flatten`, `RNN` (BPTT, documents vanishing gradient), `Seq2Seq` (encoder-decoder LSTM), `CausalConv1D`, `TCN` (dilated temporal convolutions)
- **New — Generative:** `GAN` (min-max game, Box-Muller sampling), `VAE` (reparametrization trick, ELBO = MSE + KL)
- **New — Autograd:** `Value` / `Tape` — scalar reverse-mode AD with topological backprop (micrograd-style)
- **New — Metrics:** `confusionMatrix`, `accuracy`, `precision`, `recall`, `f1Score`, `rocCurve`, `auc`, `mae`, `rmse`, `r2Score`, `perplexity`, `printConfusionMatrix`, `classificationReport`
- **New — Utilities:** `EarlyStopping` (patience + best-weight restore), `LossPlotter` (ASCII terminal curve), `WeightInspector` (per-layer stats, dead ReLU detection), `DataAugmentation` (noise, normalize, z-score, shuffle, split)

## 0.2.7

- **Docs:** Added architecture diagram to README

## 0.2.6

- **Fix:** `Network.predict` now returns `number[]` (consistent with all other network classes)
- **Fix:** `Network.train` now uses the configured optimizer and `activation.dfn()`
- **Fix:** `LayerNorm.backwardOne` correctly uses pre-update γ
- **Fix:** LSTM and GRU gate initialization corrected to Xavier fan-in+out
- **New:** `BiasVector` — 1D counterpart to `WeightMatrix`
- **New:** `defaultOptimizer` — shared default factory
- **Refactor:** `NetworkN` extracts `_forwardAll()` and `_backpropLayers()`

## 0.2.5

- Unified optimizer factories for `LSTMLayer`, `GRULayer`, `Conv1D`
- `NetworkN`: residual connections and dropout
- `Conv1D`: multi-channel input
- `Trainer`: weight decay, early stopping, classification metrics
- `DataLoader`: validation split
- `ModelSaver`: universal serialization
