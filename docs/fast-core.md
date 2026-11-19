# Fast core, seeds and extending the library (0.5)

These pieces are new in 0.5. They are exported and tested. `NetworkFast` also trains on batches (see "Batches" below).

### setRandom — a global, injectable random generator

`random()` calls `Math.random` at call time, so nothing changes unless you install a generator. `setRandom(fn)` installs one, `setRandom(null)` goes back to `Math.random` and `getRandom()` returns the one installed (or `null`). `seededRandom(seed)` is the mulberry32 generator; `withRandom(fn, body)` runs `body` with `fn` installed and restores the previous one, even if `body` throws. The type of a generator is `RandomFn`.

```ts
import { random, setRandom, getRandom, seededRandom, withRandom } from "@dniskav/neuron";

const a = withRandom(seededRandom(42), () => [random(), random()]);
const b = withRandom(seededRandom(42), () => [random(), random()]);
console.log(a[0] === b[0] && a[1] === b[1]); // true
if (a[0] !== b[0] || a[1] !== b[1]) throw new Error("same seed, different numbers");

setRandom(seededRandom(7));
console.log(getRandom() !== null); // true
setRandom(null);
if (getRandom() !== null) throw new Error("setRandom(null) did not restore Math.random");
```

### VecAdam and friends — one optimizer per buffer

`VecSGD`, `VecMomentum`, `VecAdam` and `VecClip` update a whole buffer in one `step(params, grads, lr)` and give exactly the same numbers as one scalar optimizer per element (the `VecOptimizer` interface). `vecFromFactory(factory, size)` recognizes the built-in optimizers and copies their settings; for a custom optimizer it falls back to `ScalarArrayOptimizer`, which creates one instance per element as before.

```ts
import { Adam, VecAdam, vecFromFactory, ScalarArrayOptimizer, VecSGD, VecMomentum, VecClip } from "@dniskav/neuron";

const w = new Float64Array([0.5, -0.25]);
const scalar = [new Adam(), new Adam()];
const expected = [scalar[0].step(0.5, 0.1, 0.01), scalar[1].step(-0.25, -0.2, 0.01)];
new VecAdam(2).step(w, new Float64Array([0.1, -0.2]), 0.01);
console.log(w[0] === expected[0] && w[1] === expected[1]); // true
if (w[0] !== expected[0] || w[1] !== expected[1]) throw new Error("VecAdam differs from Adam");

console.log(vecFromFactory(() => new Adam(), 3) instanceof VecAdam); // true
console.log([VecSGD, VecMomentum, VecClip, ScalarArrayOptimizer].every(c => typeof c === "function")); // true
```

### setBackend — swap the numeric kernels

The new core works on row-major `Float64Array` matrices through a `Backend` (`matVec`, `matTVec`, `addOuter`, `matMulBatch`, `matMulBatchT`, `addOuterBatch`). `defaultBackend` is the plain JavaScript one. `setBackend(partial)` replaces only the functions you pass (the rest stay the default), `setBackend(null)` restores the default and `getBackend()` returns the active one. This is the extension point for your own implementation (WASM, a paid library…).

```ts
import { defaultBackend, setBackend, getBackend } from "@dniskav/neuron";

// W is 2×3 (row-major), x has 3 values: out = W·x + b
const W = new Float64Array([1, 2, 3, 4, 5, 6]);
const out = defaultBackend.matVec(W, 2, 3, new Float64Array([1, 0, -1]), new Float64Array([0.5, 0]), new Float64Array(2));
console.log(Array.from(out)); // [ -1.5, -2 ]
if (out[0] !== -1.5 || out[1] !== -2) throw new Error("matVec is wrong");

let calls = 0;
setBackend({ matVec: (...args) => { calls++; return defaultBackend.matVec(...args); } });
getBackend().matVec(W, 2, 3, new Float64Array(3), null, new Float64Array(2));
setBackend(null);
console.log(calls, getBackend() === defaultBackend || getBackend().matVec === defaultBackend.matVec); // 1 true
if (calls !== 1) throw new Error("the custom matVec was not used");
```

`checkBackend(backend)` runs the conformance cases (several sizes and batches, zeros, negatives, very small and very large values) against the default backend and returns a `BackendReport` (`{ ok, failures }`). Use it before plugging your own implementation in.

```ts
import { checkBackend, defaultBackend } from "@dniskav/neuron";

console.log(checkBackend(defaultBackend).ok); // true
if (!checkBackend(defaultBackend).ok) throw new Error("the default backend failed its own checks");

// A matVec that is slightly off is caught.
const wrong = checkBackend({
  matVec: (W, rows, cols, x, b, out) => {
    defaultBackend.matVec(W, rows, cols, x, b, out);
    for (let k = 0; k < rows; k++) out[k] += 1e-6;
    return out;
  },
});
console.log(wrong.ok, wrong.failures[0].op); // false matVec
if (wrong.ok || wrong.failures[0].op !== "matVec") throw new Error("checkBackend missed a wrong matVec");
```

### NetworkFast — the same network, about 10 times faster

`NetworkFast` has the same constructor, options (`NetworkFastOptions` adds `backend`), methods and weight order as `NetworkN`, and with the same seed it builds and trains to the same weights. It keeps the weights in `Float64Array` buffers and updates them with the vectorized optimizers. Measured on an Apple M3 with Node 26, it trains about 10.7 times faster than `NetworkN`, and with Adam it uses about 32 bytes per parameter (heap plus typed arrays), against about 137 for `NetworkN` in 0.4. `NetworkN` stays as the readable version; in 0.5 it uses the same buffers and needs about 31 bytes per parameter with Adam.

```ts
import { NetworkN, NetworkFast, relu, sigmoid, Adam, setRandom, seededRandom } from "@dniskav/neuron";

const opts = { activations: [relu, sigmoid], optimizer: () => new Adam() };
setRandom(seededRandom(1));
const slow = new NetworkN([2, 8, 1], opts);
setRandom(seededRandom(1));
const fast = new NetworkFast([2, 8, 1], opts);
setRandom(null);

const data: [number[], number[]][] = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]];
for (let epoch = 0; epoch < 200; epoch++)
  for (const [x, t] of data) { slow.train(x, t, 0.05); fast.train(x, t, 0.05); }

const a = slow.getWeights(), b = fast.getWeights();
const maxDiff = Math.max(...a.map((w, i) => Math.abs(w - b[i])));
console.log(maxDiff < 1e-12); // true
if (!(maxDiff < 1e-12)) throw new Error(`NetworkFast drifted from NetworkN: ${maxDiff}`);
```

### Batches: `trainBatch`, `predictBatch` and `Trainer` `batchSize`

`NetworkFast.trainBatch(X, T, lr)` makes one optimizer step with the mean of the gradients of the batch, and returns the mean squared error of the batch. With a batch of one example it is the same as `train`. `predictBatch(X)` predicts a list of inputs. A batch step is one step, not one per example, so tune `lr` again when you switch from `train`. `Trainer` takes `batchSize` (an integer of at least 1, default 1); above 1 it needs a model with `trainBatch` and throws otherwise. The other models train one example (or one sequence or episode) per call.

```ts
import { NetworkFast, Adam, Trainer, setRandom, seededRandom } from "@dniskav/neuron";

const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const T = [[0], [1], [1], [0]];

setRandom(seededRandom(5));
const net = new NetworkFast([2, 8, 1], { optimizer: () => new Adam() });
setRandom(null);

const first = net.trainBatch(X, T, 0.05);
let last = first;
for (let i = 0; i < 1500; i++) last = net.trainBatch(X, T, 0.05);
console.log(last < first / 10); // true
if (!(last < first / 10)) throw new Error(`trainBatch did not learn: ${first} -> ${last}`);

const history = new Trainer(net, { epochs: 30, lr: 0.05, batchSize: 4 }).train({ inputs: X, targets: T });
console.log(history.length, net.predictBatch(X).length); // 30 4
```

### ModelSaver.save and load — the whole model, not only the weights

Format 1 (`toJSON` / `fromJSON`) saves only the weights, so you must rebuild the same model before loading. Format 2 saves the class (`kind`), its configuration and the weights: `ModelSaver.save(model)` writes it and `ModelSaver.load(json)` returns a new instance ready to predict and to keep training (the optimizer state is not saved; it starts from zero). Activations are saved by name (`ActivationSpec`), the optimizer with its settings (`OptimizerSpec`) and the configuration of `NetworkN` as a `NetworkNConfig`; the parsed file is a `SavedModel`. A model with a custom activation cannot be saved this way, because the file has to name the activation (see "Extending the library"). A model you can save is a `Saveable`; `ModelSaver.register(kind, entry)` adds your own class (a `Registrable`). A custom optimizer only saves with `SaveOptions` `{ allowCustomOptimizer: true }` and loads as SGD.

```ts
import { ModelSaver, NetworkN, makeLeakyRelu, linear, Adam } from "@dniskav/neuron";

const net = new NetworkN([3, 6, 2], { activations: [makeLeakyRelu(0.2), linear], optimizer: () => new Adam() });
for (let i = 0; i < 50; i++) net.train([0.1, 0.5, -0.3], [1, -1], 0.01);

const json = ModelSaver.save(net);
const copy = ModelSaver.load(json) as NetworkN;
const a = net.predict([0.2, 0.1, 0.4]), b = copy.predict([0.2, 0.1, 0.4]);
console.log(copy instanceof NetworkN, a[0] === b[0] && a[1] === b[1]); // true true
if (!(copy instanceof NetworkN) || a[0] !== b[0] || a[1] !== b[1]) throw new Error("load did not restore the model");
console.log(JSON.parse(json).kind); // NetworkN
```

### @dniskav/neuron/infer — predict only, without the training code

For a server or a page that only predicts, import the `infer` entry: it reads a format 2 file of a dense model (`NetworkN`, `NetworkFast` or `Network`) and gives `predict` and `predictBatch`, without optimizers, backprop, `Trainer` or the random generator. Other kinds throw and point to `ModelSaver.load`. For a page without a bundler there is a minified ESM build, `@dniskav/neuron/infer/min` (about 7 KB). The package also ships `src/` and sourcemaps, so "go to definition" in your editor opens the commented TypeScript.

```ts
import { NetworkN, ModelSaver } from "@dniskav/neuron";
import { loadForInference } from "@dniskav/neuron/infer";

const net = new NetworkN([2, 4, 1]);
const model = loadForInference(ModelSaver.save(net));
const x = [0.3, 0.7];
console.log(model.kind, model.predict(x)[0] === net.predict(x)[0]); // NetworkN true
if (model.predict(x)[0] !== net.predict(x)[0]) throw new Error("infer and the model disagree");
console.log(model.predictBatch([x, [1, 0]]).length); // 2
```

### Your own embeddings in NetworkTransformer

Vectors computed elsewhere (another model, yours or a paid one) can go into the token table: `setTokenEmbeddings(rows, { freeze: true })` checks that there is one row per token, `d_model` wide and finite, before writing anything, and `freeze` keeps training from changing it. The option `freezeEmbeddings` (`true`, or `{ token, position }`) does the same at construction, and format 2 saves it. If your vectors have another size, use `NetworkTransformerRL`, which projects any `inputDim`, or reduce them first with `PCA`; a plain `NetworkN` can also take each vector as its input.

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

## Extending the library

Every extension point is a small interface. The library calls the methods it needs and does not check which class you pass in.

| Contract | What you provide | Used by |
|---|---|---|
| `Activation` | `fn(x)` and `dfn(out)`; `dfn` takes the output of `fn`. Optional `name` and `alpha`, which saving needs for built-ins. | `NeuronN`, `NetworkN`, `NetworkFast` |
| `Optimizer` and `OptimizerFactory` | `step(weight, gradient, lr)` returns the new weight. The gradient is the ascent direction, so return `weight + lr * gradient`. The factory makes one instance per weight. | `NeuronN`, `NetworkN`, `NetworkFast`, the layers |
| `trainWithDeltas` and `inputGradient` | Output deltas that already include the activation derivative. Use them when the loss is computed outside the library. | `NetworkN`, `NetworkFast`; `GAN` and `VAE` use them internally |
| `TrainableNetwork` | `train(inputs, targets, lr)` returns the loss. Optional `trainBatch(X, T, lr)` for `batchSize` above 1. | `Trainer` |
| `Serializable` and `Saveable` | `getWeights()` and `setWeights()`. For format 2 also `kind` and `toConfig()`, which returns plain JSON data. | `ModelSaver` |
| `Registrable` | A static `fromConfig(config)` that rebuilds the model. Add it with `ModelSaver.register(kind, Class)`. | `ModelSaver.load` |
| `RandomFn` | A function that returns a number in [0, 1). Install it with `setRandom`. | every random draw |
| `Backend` | The kernels you replace with `setBackend`. Check them with `checkBackend` first. | `NetworkFast`, `loadForInference` |

### A custom activation

A custom activation trains like a built-in one. The `dfn` of this example takes the output: for softplus, `ln(1 + e^x)`, the derivative is `1 - e^(-y)`.

```ts
import { NetworkN, sigmoid, setRandom, seededRandom } from "@dniskav/neuron";

const softplus = {
  fn: (x: number) => Math.log(1 + Math.exp(x)),
  dfn: (out: number) => 1 - Math.exp(-out),
};

setRandom(seededRandom(3));
const net = new NetworkN([2, 6, 1], { activations: [softplus, sigmoid] });
setRandom(null);

const data: [number[], number[]][] = [[[0, 0], [0]], [[0, 1], [0.5]], [[1, 0], [0.5]], [[1, 1], [1]]];
const loss = () => data.reduce((s, [x, t]) => s + (net.predict(x)[0] - t[0]) ** 2, 0) / data.length;
const before = loss();
for (let epoch = 0; epoch < 300; epoch++)
  for (const [x, t] of data) net.train(x, t, 0.5);
const after = loss();
console.log(after < before / 2); // true
if (!(after < before / 2)) throw new Error(`the softplus network did not learn: ${before} -> ${after}`);
```

### A custom optimizer

The factory builds one instance per weight, so each instance keeps its own state. The library also calls the factory once more per buffer to check whether it is a built-in optimizer, so a factory that counts its calls sees extra calls.

```ts
import { NetworkN, sigmoid, setRandom, seededRandom } from "@dniskav/neuron";
import type { Optimizer } from "@dniskav/neuron";

// AdaGrad: each step is scaled by the accumulated size of the gradients.
class AdaGrad implements Optimizer {
  private sumSquares = 0;
  step(weight: number, gradient: number, lr: number): number {
    this.sumSquares += gradient * gradient;
    return weight + (lr * gradient) / (Math.sqrt(this.sumSquares) + 1e-8);
  }
}

setRandom(seededRandom(4));
const net = new NetworkN([2, 6, 1], { activations: [sigmoid, sigmoid], optimizer: () => new AdaGrad() });
setRandom(null);

const data: [number[], number[]][] = [[[0, 0], [0]], [[0, 1], [0.5]], [[1, 0], [0.5]], [[1, 1], [1]]];
const loss = () => data.reduce((s, [x, t]) => s + (net.predict(x)[0] - t[0]) ** 2, 0) / data.length;
const before = loss();
for (let epoch = 0; epoch < 300; epoch++)
  for (const [x, t] of data) net.train(x, t, 0.2);
const after = loss();
console.log(after < before / 2); // true
if (!(after < before / 2)) throw new Error(`the AdaGrad network did not learn: ${before} -> ${after}`);
```

### A custom model in format 2

`ModelSaver.save` needs `kind` and `toConfig()`. `ModelSaver.load` calls the static `fromConfig` of the class registered under that kind, and sets the weights with `setWeights`.

```ts
import { ModelSaver, NetworkN } from "@dniskav/neuron";
import type { NetworkNConfig } from "@dniskav/neuron";

// A NetworkN whose output is multiplied by a scale.
class Scaled {
  readonly kind = "Scaled";
  constructor(readonly inner: NetworkN, readonly scale: number) {}
  predict(x: number[]): number[] { return this.inner.predict(x).map(v => v * this.scale); }
  getWeights(): number[] { return this.inner.getWeights(); }
  setWeights(weights: number[]): void { this.inner.setWeights(weights); }
  toConfig(): { scale: number; net: NetworkNConfig } { return { scale: this.scale, net: this.inner.toConfig() }; }
  static fromConfig(config: unknown): Scaled {
    const c = config as { scale: number; net: NetworkNConfig };
    return new Scaled(NetworkN.fromConfig(c.net), c.scale);
  }
}
ModelSaver.register("Scaled", Scaled);

const model = new Scaled(new NetworkN([2, 3, 1]), 10);
const copy = ModelSaver.load(ModelSaver.save(model)) as Scaled;
const same = copy instanceof Scaled && copy.scale === 10 && copy.predict([0.2, 0.7])[0] === model.predict([0.2, 0.7])[0];
console.log(same); // true
if (!same) throw new Error("the registered class did not round-trip");
```
