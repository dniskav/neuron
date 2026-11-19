## Quick start

Build a network, train it one sample at a time, then predict. This example learns XOR, which a single neuron cannot represent.

```ts
import { NetworkN, Adam, relu, sigmoid, seededRandom, setRandom } from "@dniskav/neuron";

setRandom(seededRandom(7)); // optional: reproducible initial weights
const net = new NetworkN([2, 8, 1], { activations: [relu, sigmoid], optimizer: () => new Adam() });
setRandom(null);            // back to Math.random

const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [1], [1], [0]];
for (let epoch = 0; epoch < 2000; epoch++)
  inputs.forEach((x, i) => net.train(x, targets[i], 0.01));

const out = inputs.map((x) => net.predict(x)[0]);
console.log(out.map((p) => p.toFixed(2)));
if (!(out[0] < 0.5 && out[1] > 0.5 && out[2] > 0.5 && out[3] < 0.5)) throw new Error(`XOR not learned: ${out}`);
```

Every model follows the same pattern: construct it, call `train` (or `trainBatch` in `NetworkFast`), then `predict`. The guides below cover each class with examples that are run and checked.
