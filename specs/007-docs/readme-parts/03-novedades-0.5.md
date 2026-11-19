## What's new in 0.5

0.5 is additive: your code and your saved weights keep working. The full list is in [CHANGELOG.md](CHANGELOG.md), and the long examples are in [docs/fast-core.md](https://github.com/dniskav/neuron/blob/master/docs/fast-core.md).

- **`NetworkFast`** has the same constructor, options, methods and weight order as `NetworkN`, on `Float64Array` buffers with the vectorized optimizers (`VecSGD`, `VecMomentum`, `VecAdam`, `VecClip`, the `VecOptimizer` interface, `vecFromFactory`, `ScalarArrayOptimizer`). With the same seed it reaches the same weights, and it trains about 10 times faster. `NetworkFastOptions` adds `backend`.
- **Batches.** `NetworkFast.trainBatch(X, T, lr)` takes one step with the mean gradient of the batch and returns the mean squared error; `predictBatch(X)` predicts a list. `Trainer` takes `batchSize`.
- **Seeds.** `setRandom(seededRandom(seed))` makes construction and training reproducible. `getRandom`, `withRandom`, `random` and the type `RandomFn` complete the API.
- **Format 2.** `ModelSaver.save(model)` writes the class, its configuration and the weights, and `ModelSaver.load(json)` returns a new instance ready to predict and to train. `ModelSaver.register` adds your own classes (types `Saveable`, `Registrable`, `SaveOptions`, `NetworkNConfig`, `ActivationSpec`, `OptimizerSpec`, `SavedModel`).
- **Inference.** The `@dniskav/neuron/infer` entry exports `loadForInference`, which reads format 2 of dense models and only predicts: no optimizers, backpropagation, `Trainer` or random generator. `@dniskav/neuron/infer/min` is a minified ESM build of about 7 KB, for pages without a bundler.
- **Backend.** The numeric kernels (`matVec`, `matTVec`, `addOuter` and their batch versions) sit behind `setBackend`, `getBackend`, `defaultBackend` and `checkBackend`, with the types `Backend` and `BackendReport`. Check your implementation with `checkBackend` before you plug it in.
- **Your own embeddings.** `NetworkTransformer.setTokenEmbeddings(rows, { freeze })` loads vectors computed elsewhere, and `freezeEmbeddings` does the same at construction.

### Same network, faster engine

`NetworkFast` and `NetworkN` built with the same seed train to the same weights.

```ts
import { NetworkN, NetworkFast, Adam, relu, sigmoid, setRandom, seededRandom } from "@dniskav/neuron";

const options = { activations: [relu, sigmoid], optimizer: () => new Adam() };
setRandom(seededRandom(1));
const slow = new NetworkN([2, 8, 1], options);
setRandom(seededRandom(1));
const fast = new NetworkFast([2, 8, 1], options);
setRandom(null);

const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const T = [[0], [1], [1], [0]];
for (let epoch = 0; epoch < 200; epoch++)
  X.forEach((x, i) => { slow.train(x, T[i], 0.05); fast.train(x, T[i], 0.05); });

const a = slow.getWeights();
const b = fast.getWeights();
const maxDiff = Math.max(...a.map((w, i) => Math.abs(w - b[i])));
const batchLoss = fast.trainBatch(X, T, 0.05); // a batch is one step with the mean gradient
console.log(maxDiff < 1e-12, fast.predictBatch(X).length, batchLoss.toFixed(4));
if (!(maxDiff < 1e-12) || fast.predictBatch(X).length !== 4) throw new Error(`NetworkFast drifted: ${maxDiff}`);
```

### Save a model and predict without the training code

`ModelSaver.save` writes format 2, which keeps the class and its configuration, so `loadForInference` can rebuild a dense model that only predicts.

```ts
import { NetworkN, ModelSaver } from "@dniskav/neuron";
import { loadForInference } from "@dniskav/neuron/infer";

const net = new NetworkN([2, 4, 1]);
const model = loadForInference(ModelSaver.save(net));
const x = [0.3, 0.7];
console.log(model.kind, model.predict(x)[0] === net.predict(x)[0], model.predictBatch([x, [1, 0]]).length);
if (model.kind !== "NetworkN" || model.predict(x)[0] !== net.predict(x)[0]) throw new Error("infer and the model disagree");
```

### Your own embeddings in NetworkTransformer

`setTokenEmbeddings` checks that there is one row per token, `d_model` values per row, all finite, before it writes anything. With `freeze: true`, training leaves the table unchanged.

```ts
import { NetworkTransformer } from "@dniskav/neuron";

const net = new NetworkTransformer(3, { vocabSize: 4, d_model: 8, nHeads: 2, nClasses: 2 });
const mine = Array.from({ length: 4 }, (_, i) => Array.from({ length: 8 }, (_, j) => Math.sin(i + j)));
net.setTokenEmbeddings(mine, { freeze: true });

const before = net.tokenEmb.getWeights();
for (let i = 0; i < 20; i++) net.train([1, 2, 3], [0, 1, 0], 0.01);
const same = net.tokenEmb.getWeights().every((w, k) => w === before[k]);
console.log(same); // true
if (!same) throw new Error("the frozen embeddings changed");
```
