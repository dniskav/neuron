## Seeds, vectorized optimizers and the numeric core (0.5, in progress)

These pieces are part of the 0.5 work (`specs/010-fast-core/`). They are exported and tested; batch training and the inference-only entry are still in progress.

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

`NetworkFast` has the same constructor, options (`NetworkFastOptions` adds `backend`), methods and weight order as `NetworkN`, and with the same seed it builds and trains to the same weights. It keeps the weights in `Float64Array` buffers and updates them with the vectorized optimizers, so it trains about 10 times faster and uses about a quarter of the memory with Adam. `NetworkN` stays as the readable version.

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

### ModelSaver.save and load — the whole model, not only the weights

Format 1 (`toJSON` / `fromJSON`) saves only the weights, so you must rebuild the same model before loading. Format 2 saves the class (`kind`), its configuration and the weights: `ModelSaver.save(model)` writes it and `ModelSaver.load(json)` returns a new instance ready to predict and to keep training (the optimizer state is not saved; it starts from zero). Activations are saved by name (`ActivationSpec`), the optimizer with its settings (`OptimizerSpec`) and the configuration of `NetworkN` as a `NetworkNConfig`; the parsed file is a `SavedModel`. A model you can save is a `Saveable`; `ModelSaver.register(kind, entry)` adds your own class (a `Registrable`). A custom optimizer only saves with `SaveOptions` `{ allowCustomOptimizer: true }` and loads as SGD.

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
