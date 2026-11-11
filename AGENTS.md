# AGENTS.md — @dniskav/neuron

This file is for AI agents and LLMs working with this codebase. Every claim below was checked against `src/` (file names are given; line numbers drift). The `specs/` directory holds the history of the 0.4 repair (spec, tasks, handoff); you do not need it to work on the library. `CHANGELOG.md` lists what changed since 0.3.2 and how to migrate.

## What this library is

`@dniskav/neuron` is a **minimal, dependency-free neural network library** built from scratch in TypeScript. It is educational by design: every class is self-contained, the math is readable, gradients are written by hand, and each abstraction builds on the previous one. `dependencies` in `package.json` is empty. Besides the hand-coded backprop of every model, `Value` (`src/Tape.ts`) is a separate scalar autograd module (reverse-mode, micrograd style); no network class uses it.

Everything is exported from `src/index.ts` (126 names: 97 values, 29 types). The map below lists all of them.

## Module map

### Feedforward

| Export | File | Purpose |
|--------|------|---------|
| `Neuron` | `Neuron.ts` | One input, one weight, one bias, fixed sigmoid. |
| `NeuronN` | `NeuronN.ts` | N inputs, Xavier-style init, configurable activation and optimizer. |
| `Layer` | `Layer.ts` | Array of `NeuronN` sharing the same inputs. |
| `Network` | `Network.ts` | Fixed two-layer network (hidden + output), sigmoid, backprop. |
| `NetworkN` | `NetworkN.ts` | Any depth `[inputs, ...hidden, outputs]`; per-layer activations, optimizer factory, `residual`, `dropoutRate`; `trainWithDeltas`, `inputGradient`. |

### Recurrent

| Export | File | Purpose |
|--------|------|---------|
| `LSTMLayer` | `LSTMLayer.ts` | 4-gate LSTM with `h` and `c` state; stores a trajectory; BPTT in `backprop`. |
| `NetworkLSTM` | `NetworkLSTM.ts` | `LSTMLayer` + dense layers; episode-level memory. |
| `GRULayer` | `GRU.ts` | Gated recurrent unit (reset and update gates), no cell state. |
| `RNN` | `RNN.ts` | Vanilla recurrent network with BPTT (`backward` returns the MSE). |
| `Seq2Seq` | `Seq2Seq.ts` | LSTM encoder-decoder; final encoder state feeds the decoder; teacher forcing. |

### Transformers and their parts

| Export | File | Purpose |
|--------|------|---------|
| `NetworkTransformer` | `NetworkTransformer.ts` | Token classifier: embeddings, blocks, per-token logits. `new NetworkTransformer(seqLen, options)`. |
| `NetworkTransformerRL` | `NetworkTransformerRL.ts` | Continuous input projection, causal blocks, pooling (`'avg'`, `'max'`, `'last'`, `'weighted'`; default `'weighted'`), Q-values. |
| `TransformerBlock` | `TransformerBlock.ts` | Multi-head attention + FFN (ReLU) + two `LayerNorm` + residuals; full backprop. |
| `MultiHeadAttention` | `MultiHeadAttention.ts` | N heads in parallel + output projection `Wo`. |
| `AttentionHead` | `AttentionHead.ts` | One scaled dot-product head (Q/K/V, backprop). |
| `LayerNorm` | `LayerNorm.ts` | Layer norm with learnable gamma and beta; per-position cache. |
| `WeightMatrix` | `MatMul.ts` | 2D weights, one `Adam` per element; `update(dW, lr, clipValue?)`. |
| `BiasVector` | `MatMul.ts` | 1D bias, one `Adam` per element. |
| `EmbeddingMatrix` | `MatMul.ts` | Lookup table; SGD on the rows used. |
| `matMul` `transpose` `softmax` `softmaxBackward` | `MatMul.ts` | Matrix and softmax helpers. |

### Layers and convolutions

| Export | File | Purpose |
|--------|------|---------|
| `Dropout` | `Dropout.ts` | Inverted dropout; `reuseMask` / `hasMask` replay a saved mask. |
| `BatchNorm` | `BatchNorm.ts` | Batch norm with running statistics. |
| `Conv1D` | `Conv1D.ts` | 1D convolution, multi-channel, `stride`, `padding: 'valid' \| 'same'`. |
| `Conv2D` | `Conv2D.ts` | 2D convolution over `[H][W][C]`; `padding` as above. |
| `MaxPool2D` | `MaxPool2D.ts` | Max pooling; routes the gradient to each window's argmax. |
| `Flatten` | `Flatten.ts` | `forward` flattens `[H][W][C]` to a vector; `backward` restores the shape. |
| `CausalConv1D` | `TCN.ts` | Causal dilated 1D convolution (left padding). |
| `TCN` | `TCN.ts` | Stack of `CausalConv1D` for sequences. |

### Activations, optimizers, losses

| Export | File | Purpose |
|--------|------|---------|
| `sigmoid` `relu` `tanh` `linear` `leakyRelu` `elu` | `activations.ts` | Objects with `fn(x)` and `dfn(out)`; the derivative takes the OUTPUT. |
| `makeLeakyRelu` `makeElu` | `activations.ts` | Factories with a custom alpha. |
| `SGD` `Momentum` `Adam` | `optimizers.ts` | One instance per scalar weight; `step(weight, gradient, lr)` returns `w + lr*g`. |
| `ClipOptimizer` `ClippedOptimizerFactory` | `optimizers.ts` | Clip each gradient to `[-clipValue, clipValue]` before delegating. |
| `defaultOptimizer` | `optimizers.ts` | Factory `() => new SGD()`; the default of `NeuronN`, `Layer`, `NetworkN`, `NetworkLSTM`. |
| `mse` `crossEntropy` | `losses.ts` | Scalar losses for evaluation (`crossEntropy` is binary and averaged). |
| `mseDelta` `crossEntropyDelta` `crossEntropyDeltaRaw` | `losses.ts` | Output deltas for `trainWithDeltas`; NOT rescaled (`mseDelta` is `actual - predicted`). |

### Training utilities and data

| Export | File | Purpose |
|--------|------|---------|
| `Trainer` | `Trainer.ts` | Epoch loop: `lrDecay`, `weightDecay`, `earlyStopping`, `restoreBestWeights`, `computeMetrics`. |
| `EarlyStopping` | `EarlyStopping.ts` | Patience-based stop on a monitored value. |
| `LRScheduler` | `LRScheduler.ts` | `stepDecay`, `exponentialDecay`, `plateauDecay`, `cosineAnnealing`. |
| `DataLoader` | `DataLoader.ts` | Batching, shuffling, validation split; `DataLoader.sequences`. |
| `DatasetLoader` | `DatasetLoader.ts` | `fromCSV`, `fromJSON`; one-hot encodes string columns. |
| `DataAugmentation` | `DataAugmentation.ts` | Noise, jitter, flip, normalize, standardize, `shuffle`, `split`. |
| `LossPlotter` | `LossPlotter.ts` | ASCII loss curve. |
| `ModelSaver` | `ModelSaver.ts` | JSON save/load through `getWeights()` / `setWeights()`. |

### Classical ML, unsupervised, generative

| Export | File | Purpose |
|--------|------|---------|
| `Perceptron` | `Perceptron.ts` | Rosenblatt perceptron (step function). |
| `LinearRegression` | `LinearRegression.ts` | `fitNormal` (closed form) and `fitGD`. |
| `LogisticRegression` `SoftmaxRegression` | `LogisticRegression.ts` | Binary and multinomial, trained by gradient descent. |
| `GaussianNaiveBayes` | `NaiveBayes.ts` | Per-feature Gaussians in log-space. |
| `DecisionTree` | `DecisionTree.ts` | CART (Gini or weighted MSE); `maxDepth` integer >= 1 or `Infinity`. |
| `KMeans` | `KMeans.ts` | K-Means++ and Lloyd; `seed` option. |
| `PCA` | `PCA.ts` | Power iteration with deflation. |
| `SOM` | `SOM.ts` | Kohonen map. |
| `HopfieldNetwork` | `HopfieldNetwork.ts` | Bipolar associative memory; warns once past about `0.138·N` patterns. |
| `Autoencoder` | `Autoencoder.ts` | Encoder and decoder, two `NetworkN`. |
| `GAN` | `GAN.ts` | Generator vs discriminator; D must end in a sigmoid. |
| `VAE` | `VAE.ts` | Reparametrization; loss = MSE + `klWeight`·KL (sixth constructor argument). |

### Embeddings and self-supervised

| Export | File | Purpose |
|--------|------|---------|
| `Word2Vec` | `Word2Vec.ts` | Skip-gram or CBOW, full softmax. |
| `TSNE` | `TSNE.ts` | t-SNE; `seed` option. |
| `PositionalEncoding` `LearnedPositionalEncoding` | `PositionalEncoding.ts` | Sinusoidal (no parameters) and trainable table. |
| `ContrastiveLearning` `Augmenter` | `ContrastiveLearning.ts` | SimCLR-style NT-Xent; `Augmenter` builds the views. |

### Autograd, metrics, tokenizer, validators

| Export | File | Purpose |
|--------|------|---------|
| `Value` | `Tape.ts` | Scalar reverse-mode autograd node (`backward()`). |
| `WeightInspector` | `WeightInspector.ts` | Per-layer weight statistics, dead-ReLU counts. |
| `confusionMatrix` `precision` `recall` `f1Score` `accuracy` `rocCurve` `auc` `mae` `rmse` `r2Score` `perplexity` `printConfusionMatrix` `classificationReport` | `Metrics.ts` | Evaluation functions. Multiclass `precision`/`recall`/`f1Score` are macro averages. |
| `Tokenizer` | `Tokenizer.ts` | `char`, `word` or `whitespace` vocabulary; `encode`, `decode`, `oneHot`, snapshots. |
| `validateArray` `validateArrayMinLength` `validate2DArray` `validateNumber` | `Validation.ts` | Throw `Error('<method>: ...')` on bad arrays or numbers. |

### Types (all in `src/index.ts` via `export type`)

- Activations and optimizers (`activations.ts`, `optimizers.ts`): `Activation`, `Optimizer`, `OptimizerFactory`.
- Model options: `NetworkNOptions`, `NetworkLSTMOptions`, `NetworkTransformerOptions`, `NetworkTransformerRLOptions`, `TransformerBlockOptions`, `KMeansOptions`, `SOMOptions`, `Word2VecOptions`, `Word2VecModel`, `TSNEOptions`.
- Training (`Trainer.ts`, `DataLoader.ts`, `DatasetLoader.ts`, `ModelSaver.ts`): `TrainerOptions`, `TrainDataset`, `TrainableNetwork`, `TrainableNetworkWithWeights`, `TrainMetrics`, `DataPair`, `DatasetLoaderOptions`, `DatasetLoaderResult`, `CategoricalMap`, `Serializable`.
- Diagnostics and tokenizer: `WeightStats`, `TokenizerMode`, `TokenizerOptions`, `EncodeOptions`, `EncodeBatchOptions`, `TokenizerSnapshot`.

## When to use each class

- **Neuron** — single scalar input, learning a threshold or a simple linear relationship.
- **NeuronN** — single neuron with multiple inputs, binary classification.
- **Layer** — parallel neurons for a single layer of a network.
- **Network** — fixed two-layer architecture, good for simple non-linear problems (e.g. XOR).
- **NetworkN** — use this for most feedforward tasks. Flexible depth, width, activations and optimizer. Use `trainWithDeltas` / `inputGradient` when the loss is computed outside (GAN, custom losses).
- **LSTMLayer** — raw access to the recurrent layer (custom architectures).
- **NetworkLSTM** — inputs arrive as a sequence and the network must remember previous steps within the same episode (RL agents, time series).
- **GRULayer** — same role as `LSTMLayer` with fewer parameters and no cell state. `GRULayer.backprop` returns nothing.
- **RNN**, **TCN**, **Seq2Seq** — small educational sequence models. `TCN` is convolutional (no recurrence); `Seq2Seq` maps a sequence to a sequence.
- **NetworkTransformer** — token classification where every position attends to every other (e.g. Sudoku).
- **NetworkTransformerRL** — the agent needs to attend to specific past moments, not just a compressed hidden state. It takes a sliding window of the last N states and is causal. Heavier than `NetworkLSTM`.
- **Trainer** — a ready epoch loop for anything with `train(inputs, targets, lr)` that returns a loss. For early stopping with restored weights the model also needs `predict`, `getWeights` and `setWeights`.
- **Classical ML** (`Perceptron`, `LinearRegression`, `LogisticRegression`, `SoftmaxRegression`, `GaussianNaiveBayes`, `DecisionTree`) — baselines and teaching; no hidden layers.
- **Unsupervised** (`KMeans`, `PCA`, `SOM`, `HopfieldNetwork`) — clustering, projection, topographic maps, associative memory. `Autoencoder` learns a compressed code with two `NetworkN`.
- **GAN**, **VAE** — generative models built on `NetworkN`.
- **Word2Vec**, **TSNE**, **PositionalEncoding**, **ContrastiveLearning** — embeddings, visualization, sequence position, self-supervised pretraining.
- **Value** — to see or test reverse-mode autograd on scalars; use the hand-coded classes for training real models.

## Key design constraints

- **Default activation is sigmoid** in `Neuron`, `NeuronN`, `Layer`, `Network` and `NetworkN`. Available: `sigmoid`, `relu`, `tanh`, `linear`, `leakyRelu`, `elu`, plus `makeLeakyRelu(alpha)` and `makeElu(alpha)`. `NeuronN` and `Layer` take one activation; `NetworkN` takes one per layer (`NetworkNOptions.activations`, length = number of non-input layers, otherwise it throws); `Neuron` and `Network` do not take an activation. Any object with `fn` and `dfn(out)` works (`dfn` receives the output, not the pre-activation). The Transformer classes use ReLU in the FFN (`Math.max(0, v)` in `TransformerBlock.ts`) and softmax in attention.
- **No batching in the models.** Training is online: one sample (or one sequence or episode) per call. `NetworkLSTM` accumulates gradients across an episode; `NetworkTransformer.train` processes one sequence per call. `DataLoader` only slices data into batches and `Autoencoder.trainBatch` trains on the rows of a batch one after another (it returns the mean MSE); neither vectorizes.
- **Gradients are hand-coded**, one backward pass per class. `Value` is the only autograd and it is standalone.
- **No GPU, no native code, no dependencies.**
- **There is no global random seed.** Initialization and sampling use `Math.random`. Only `TSNE` (`options.seed`, mulberry32 in `TSNE.ts`) and `KMeans` (`options.seed`, `KMeans.ts`) accept a `seed`. To make anything else reproducible, replace `Math.random` before constructing and training (the tests do this with `tests/helpers/seed.ts`).
- **Optimizers use the ascent convention.** `Optimizer.step` returns `w + lr * g`, so models pass `g = -dL/dw`. Keep this when adding a model or a custom delta.
- **Transformer outputs are unbounded.** `NetworkTransformer.predict` returns raw logits (`seqLen * nClasses`, flattened); apply `softmax` externally for probabilities.
- **`NetworkTransformer.train` uses softmax + cross-entropy**, not MSE. The logit gradient is `prob_c - target_c`, divided by the number of positions that count (`mask`), so it is the gradient of the MEAN cross-entropy that the method returns.
- **Step scale of `NetworkN.train` and `Network.train`.** They give the step of the gradient of the MEAN squared error they return, `(2/n)(t - p)` with `n` outputs, so with one output the step is twice that of 0.3.2 (`lr_new = lr_old * n / 2` reproduces the old dynamics). `trainWithDeltas` and `inputGradient` take caller-computed deltas and do not rescale. `Autoencoder.train` uses `(2/d)` with `d = inputSize`.

## Validation contract

- Invalid inputs throw `Error('Class.method: ...')` BEFORE any state is touched: wrong length, empty data, NaN or Infinity, bad options. A rejected call leaves the object as it was. Many constructors validate their sizes and options the same way (e.g. `LSTMLayer`, `NetworkLSTM`, `Conv1D`, `Conv2D`, `DecisionTree`, `SOM`, `VAE`, `GAN`).
- `setWeights` and `setWeightsFlat` validate length and finiteness before writing anything. A model that diverged to NaN cannot be restored with `setWeights(getWeights())`.
- `getWeights` returns copies. In the recurrent classes `setWeights` copies its input too, so neither side aliases the model.
- **`lr` is validated per class, not globally.**
  - `lr = 0` is accepted (a no-op for the default SGD) in `NetworkN`, `Network`, `NetworkLSTM`, `RNN`, `TCN`, `CausalConv1D`, `Seq2Seq`, `Autoencoder`, `VAE`, `GAN`, `ContrastiveLearning` and `Word2Vec`.
  - `lr = 0` THROWS in `Trainer` (needs finite `lr > 0`) and in `LinearRegression.fitGD`, `LogisticRegression.train` and `SoftmaxRegression.train` (need `lr > 0`).
  - NaN or negative `lr` throws in `RNN`, `TCN`, `CausalConv1D`, `Seq2Seq`, `Autoencoder`, `VAE`, `GAN`, `ContrastiveLearning` and `Word2Vec` (finite and `>= 0`), and in the regressions and `Trainer` (above). `Network.train`, `Neuron.train` and `Perceptron.train` reject NaN and Infinity but accept negative values. `NetworkN.train`, `NetworkLSTM.train`, `LSTMLayer.backprop` and the Transformer classes do not validate `lr` at all: do not pass NaN or a negative value.
- `NetworkLSTM.train` and `LSTMLayer.backprop` check the whole input (length against the trajectory, row widths, finiteness) before changing weights, and a rejected call does NOT clear the stored trajectory, so you can retry with the right input.
- `Trainer` rejects options that cannot take effect instead of ignoring them (`epochs` integer > 0, `lrDecay` in (0, 1], `weightDecay >= 0`, `patience` integer >= 1, empty dataset, `earlyStopping` without validation data, `restoreBestWeights` without `earlyStopping`). `clipValue` is not a `Trainer` option; use `ClippedOptimizerFactory`.
- `GAN` throws unless the discriminator output is a sigmoid (checked by value). `NetworkN.trainWithDeltas` and `inputGradient` throw when the last layer has `residual`.
- `Metrics` validates empty inputs and length mismatches everywhere, but label values only in `confusionMatrix`, `precision`, `recall`, `f1Score` and `classificationReport`; `accuracy([-1], [0])` does not throw, and `mae`, `rmse`, `r2Score` return NaN for NaN input.

## Serialization (`ModelSaver`)

- `ModelSaver.toJSON(model)` writes `{ format: 1, length, structure?, weights, timestamp }`. `structure` is present only if the model exposes a numeric `structure` array (`NetworkN` does, `Network` does not). The class name is not saved.
- `ModelSaver.fromJSON(model, json)` validates everything before writing: valid JSON, an object, `format` absent or equal to 1 (older `{ weights, timestamp }` files still load), `weights` an array of finite numbers (NaN and Infinity are saved as `null` and fail on load), `length` equal to `weights.length`, the weight count the model expects, and `structure` equal to the model's when both exist. Messages start with `ModelSaver.fromJSON:`.
- `saveToFile` / `loadFromFile` take the write/read function as an argument (no `fs` import).
- Weights saved with 0.3.2 for models whose training was fixed in 0.4 load if the shape matches, but are not a good starting point; see `CHANGELOG.md`.

## Transformer design decisions

- **Post-norm**: `LayerNorm` after each residual add (`TransformerBlock.ts`).
- **d_k = d_v = d_model / nHeads** (`MultiHeadAttention.ts`).
- **Adam on every `WeightMatrix` and `BiasVector`, SGD on `EmbeddingMatrix` rows.** `WeightMatrix.update(dW, lr, clipValue?)` clips each element to `[-clipValue, clipValue]` before the Adam step; the default `Infinity` does not clip.
- **The input gradient `dX` returned by the attention, block and `LayerNorm` backward passes does not depend on `lr`** (`LayerNorm.backwardOne` uses the gamma of the forward pass).
- **`LayerNorm` caches per position**: call `resetCache(seqLen)` before each forward pass, then `predictOne(x, pos)` per token. `TransformerBlock` does this internally.
- **`getAttentionWeights()`** exists on `AttentionHead`, `MultiHeadAttention`, `TransformerBlock`, `NetworkTransformer` and `NetworkTransformerRL` and returns the softmax attention of the last forward pass.
- `NetworkTransformerRL.train` takes one Adam step per call on `inputProj`, with the summed gradient.

## LSTM-specific details

`LSTMLayer` is a standard 4-gate LSTM: `f` (forget, sigmoid, bias initialized to 1 so it remembers by default), `i` (input, sigmoid), `g` (cell, tanh), `o` (output, sigmoid).

**Episode pattern (mandatory):**
```ts
net.resetState()           // call at the START of every episode
for each step:
  net.predict(inputs)      // advances state, stores trajectory
net.train(targets, lr)     // BPTT across the full episode, clears the trajectory
```

- Forgetting `resetState()` bleeds state from one episode into the next.
- `targets` needs one row per `predict` call, each as wide as the last dense layer; otherwise `train` throws and the episode stays intact. An empty episode with empty targets is a no-op.
- Gradients are averaged over the episode length: the LSTM step uses `lr / T` and the dense gradients are divided by `T`.
- `NetworkLSTM` needs at least one dense layer (`denseStructure = []` throws in the constructor).
- `LSTMLayer.backprop(dh_seq, lr, dcFinal?)` returns `{ dh0, dc0 }` (the gradient to the initial state, unscaled by `T`) and throws if `dh_seq` does not match the trajectory length or a row is not `hiddenSize` wide. `dh_seq[t]` is `-dL/dh_t`.

## Known limitations

These are documented behaviors, not bugs to fix silently (also listed in `CHANGELOG.md`).

- **`ContrastiveLearning`** uses a raw initialization: some seeds start with a dead projection head and the loss does not go down. `Augmenter.dropoutFeatures` zeroes values with probability `rate` and does NOT rescale the kept ones (it is not inverted dropout).
- **`VAE`** with the default `klWeight = 1` can collapse on small datasets in [0, 1] (the latent code carries no information); `klWeight` around 0.003 learned in the measured case.
- **`PCA`** on rank-deficient data: components for axes of variance 0 can have norm close to 0 (about 1e-111 in the minimal case), so they are not an orthonormal basis. `transform` and the reconstruction are correct.
- **`TSNE` and `TCN` need a suitable `lr`.** `TSNE` default `lr = 200` diverges with few points (n = 30, perplexity 5: `lr` 5-20 converges); from n = 100 the default works (comment on `TSNEOptions.lr`). `TCN(1,4,2,2,1)` on one measured task: `lr` 0.1-0.39 converges, 1.0 ends higher than it started, 1.5 or more diverges (header of `TCN.ts`). One case each, not a law.
- **`Trainer.setValidationData`** is used only with `earlyStopping` (and `restoreBestWeights`); without `earlyStopping` the data is stored and ignored.
- **`NetworkN.getWeights()` and `setWeights()` clear the saved dropout masks.** A call between `predict(x, true)` and `trainWithDeltas` / `inputGradient` with `reuseMasks = true` makes that call throw ("needs the dropout masks").
- **`Autoencoder`** accepts `latentSize >= inputSize` without error or warning (an over-complete autoencoder is valid, e.g. for denoising).
- **`Autoencoder.train` step scale**: `lr * inputSize / 2` only approximates the 0.3.2 step (the encoder gradient also changed); retrain.

## What this library does NOT do

- No GPU acceleration and no native or WASM code.
- No vectorized batch training: models train one sample, sequence or episode per call (`DataLoader`, `DataAugmentation` and `Trainer` loop over data).
- No global random seed or injectable RNG (only `TSNE` and `KMeans` take `seed`).
- No automatic differentiation inside the models: `Value` is a separate scalar tool; every other class has its own hand-coded backward pass.
- No file I/O: `ModelSaver` and `DatasetLoader` work on strings (`saveToFile` / `loadFromFile` receive the read/write function).
- No gradient clipping inside `Trainer`; clip through the optimizer (`ClippedOptimizerFactory`).

Convolutions, data loaders, loss functions, optimizers (`SGD`, `Momentum`, `Adam`), metrics and a tokenizer do exist; see the map above.

## Working on this codebase

- Scripts (`package.json`): `npm run build` (tsup, CJS + ESM + types into `dist/`), `npm run dev` (watch), `npm test` (`vitest run`), `npm run coverage`, `npm run typecheck` (`tsc --noEmit -p tsconfig.test.json`). Current version in `package.json`: 0.3.2; the 0.4.0 changes are listed as unreleased in `CHANGELOG.md`.
- Tests live in `tests/` (vitest). `tests/setup.ts` seeds `Math.random` before every test (`tests/helpers/seed.ts`).
- Gradient oracles: `tests/oracle/` compares each model's real update with finite differences (helpers in `tests/helpers/gradientOracle.ts`). `tests/helpers/learns.ts` is the helper for "the loss goes down" tests.
- `tests/coverage.modules.test.ts` requires a test file per `src/` module (case-sensitive name, searched in `tests/`, `tests/oracle/` and `tests/api/`, with documented exceptions).
- Coverage thresholds are in `vitest.config.mts` (`npm run coverage` fails below them; CI runs it).
- CI (`.github/workflows/ci.yml`) runs on Node 20: `npm ci`, `typecheck`, `build`, `coverage`. `engines.node` is `>=20`. Golden tests that use transcendental functions (`Math.exp`, `Math.tanh`) compare with a tolerance (1e-12 in `tests/Activations.test.ts` and `tests/GRU.test.ts`), not exact equality.
- Adding or removing an export in `src/index.ts` means updating this file's map as well.

## Known usage context

This library has been used alongside a demo app (`neuron_app`, a separate project) where RL agents are trained. `NetworkLSTM` was added for a maze-navigation agent that needed within-episode memory. `NetworkTransformer` was added for a Sudoku solver card, to show self-attention learning row, column and box constraints without explicit rules. `NetworkTransformerRL` was added as an alternative RL backbone for agents that benefit from attending to specific past steps. The app updates to new library versions separately; this file makes no claim about its current state.
