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

A custom optimizer and a custom model saved in format 2 (with `ModelSaver.register`) are in the [extending section of docs/fast-core.md](https://github.com/dniskav/neuron/blob/master/docs/fast-core.md#extending-the-library).
