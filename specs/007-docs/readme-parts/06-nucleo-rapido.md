## Seeds, vectorized optimizers and the numeric core (0.5, in progress)

These pieces are part of the 0.5 work (`specs/010-fast-core/`). They are exported and tested, but the models do not use them all yet.

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
