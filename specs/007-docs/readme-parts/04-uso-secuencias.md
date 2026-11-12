## Autograd, convolutions and sequences

The examples in this part cover the autograd class `Value`, a convolutional pipeline with a pooling and a flattening step, and the sequence models `RNN`, `TCN` and `NetworkLSTM`. They follow the same rules as the examples in Usage: each one is self-contained, fixes its own randomness when it uses any, and ends with a check that throws if the result is not the expected one.

### Value / Tape — automatic differentiation

`Value` is a scalar node in a computation graph. `backward()` fills `.grad` for every node that contributed to the result. The check compares the gradients with the analytic values.

```ts
import { Value } from "@dniskav/neuron";

// Build a computation graph: n = x·w + b, o = tanh(n)
const x = new Value(2.0);
const w = new Value(-3.0);
const b = new Value(6.7);
const n = x.mul(w).add(b);
const o = n.tanh();

// Backward pass: fills .grad for every node in the graph
o.backward();

console.log(o.data.toFixed(4), x.grad.toFixed(4), w.grad.toFixed(4), b.grad.toFixed(4));
// 0.6044 -1.9042 1.2695 0.6347

// Analytic values at n = 0.7: do/dn = 1 − tanh(n)², dn/dx = w, dn/dw = x, dn/db = 1
const dn = 1 - Math.tanh(0.7) ** 2;
const expected = [Math.tanh(0.7), dn * -3, dn * 2, dn];
const got = [o.data, x.grad, w.grad, b.grad];
if (got.some((v, i) => Math.abs(v - expected[i]) > 1e-9)) throw new Error(`unexpected gradients: ${got}`);
```

### Conv2D + MaxPool2D + Flatten — CNN pipeline

Shapes are `[H][W][C]`. With `padding: "same"` and stride 1 the convolution keeps the height and width. `MaxPool2D.backward` sends each window's gradient only to its maximum, so a gradient of 1 on every pooled cell gives a total equal to the number of windows.

```ts
import { Conv2D, MaxPool2D, Flatten, NetworkN } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 5;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// 8×8 grayscale image: a vertical bar in columns 3 and 4.
const image = Array.from({ length: 8 }, (_, h) =>
  Array.from({ length: 8 }, (_, w) => [w === 3 || w === 4 ? 1 : 0]),
);

const conv    = new Conv2D(8, 8, 1, 3, 2, { padding: "same" }); // 8×8×1 → 8×8×2
const pool    = new MaxPool2D(2);                                // 8×8×2 → 4×4×2
const flatten = new Flatten();                                   // 4×4×2 → 32 values
const dense   = new NetworkN([32, 4, 1]);                        // 32 → 4 → 1

const featureMaps = conv.forward(image);
const pooled      = pool.forward(featureMaps);
const flat        = flatten.forward(pooled);
const output      = dense.predict(flat)[0];
console.log(featureMaps.length, featureMaps[0].length, pooled.length, flat.length, output.toFixed(3));
// 8 8 4 32 0.527

// Gradient of 1 on every pooled cell: each window sends it to its maximum, so the total is 4×4×2 = 32.
const grad = pool.backward(pooled.map((row) => row.map((cell) => cell.map(() => 1))));
const total = grad.flat(2).reduce((s, v) => s + v, 0);
console.log("pool gradient total:", total); // 32

let rejected = false;
try {
  new MaxPool2D(2).forward([[[1]]]); // input height 1 is smaller than the pool
} catch {
  rejected = true;
}

const shapesOk =
  featureMaps.length === 8 && featureMaps[0].length === 8 && featureMaps[0][0].length === 2 &&
  pooled.length === 4 && pooled[0].length === 4 && flat.length === 32;
if (!shapesOk || Math.abs(total - 32) > 1e-9 || !(output > 0 && output < 1) || !rejected) {
  throw new Error(`CNN pipeline check failed: ${featureMaps.length}, ${pooled.length}, ${flat.length}, ${total}`);
}
```

### Conv1D — `same` padding with an even kernel

With `padding: "same"` the output length is `ceil(L / stride)`, also when the kernel is even. `forward` returns `[filters][outputLength]`. Version 0.3.2 returned `L − 1` for an even kernel with stride 1.

```ts
import { Conv1D } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 9;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Input of 8 steps with 1 channel, kernel 4, 2 filters, stride 1.
const stride1 = new Conv1D(8, 4, 2, 1, "same");
const out1 = stride1.forward([[0], [1], [0], [1], [0], [1], [0], [1]]);

// Input of 7 steps, kernel 4, 1 filter, stride 2.
const stride2 = new Conv1D(7, 4, 1, 2, "same");
const out2 = stride2.forward([[0], [1], [0], [1], [0], [1], [0]]);

console.log(out1.length, out1[0].length, out2[0].length); // 2 8 4

if (out1.length !== 2 || out1[0].length !== 8 || out2[0].length !== 4) {
  throw new Error(`unexpected Conv1D shapes: ${out1.length}×${out1[0].length}, ${out2[0].length}`);
}
```

### RNN — vanilla recurrent network

`RNN.backward` runs backpropagation through time on one sequence, applies the update and returns the mean squared error measured before the update. An empty sequence throws. A learning rate of 0 is valid and leaves the weights unchanged, so the first call below only measures the loss.

```ts
import { RNN } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 2;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// 1 input → 8 hidden → 1 output, over a sequence of 4 steps
const rnn = new RNN(1, 8, 1);
const sequence = [[0.1], [0.3], [0.7], [0.9]];
const targets  = [[0.2], [0.5], [0.8], [1.0]];

const initialLoss = rnn.backward(sequence, targets, 0); // lr 0: loss only
let loss = initialLoss;
for (let epoch = 0; epoch < 300; epoch++) loss = rnn.backward(sequence, targets, 0.05);

console.log(initialLoss.toFixed(3), loss.toExponential(1)); // 0.543 1.9e-5
if (!(loss < 1e-3 && loss < initialLoss / 10)) throw new Error(`RNN did not learn: ${initialLoss} -> ${loss}`);

let rejected = false;
try {
  rnn.backward([], [], 0.05);
} catch {
  rejected = true;
}
if (!rejected) throw new Error("an empty sequence was accepted");
```

### TCN — Temporal Convolutional Network

`TCN` stacks causal dilated convolutions with residual connections. Its receptive field is `(kernelSize − 1)·(2^levels − 1) + 1`. The learning rate matters: the example uses 0.2, which is inside the range measured for this case in the notes at the top of `src/TCN.ts`. A much larger value can make the loss rise.

```ts
import { TCN } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 2;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// 1 input channel → 4 channels, kernel 2, 2 levels (dilations 1 and 2) → 1 output
const tcn = new TCN(1, 4, 2, 2, 1);

// Learn to repeat the signal one step late: target[t] = x[t − 1], and 0 for t = 0.
const T = 16;
const x = Array.from({ length: T }, (_, t) => Math.sin(1.3 * t + 0.4));
const sequence = x.map((v) => [v]);
const targets  = x.map((_, t) => [t === 0 ? 0 : x[t - 1]]);

const initialLoss = tcn.train(sequence, targets, 0);
let loss = initialLoss;
for (let epoch = 0; epoch < 300; epoch++) loss = tcn.train(sequence, targets, 0.2);
console.log(initialLoss.toFixed(3), loss.toExponential(1)); // 0.523 2.6e-4

// Receptive field of a 3-tap, 4-level stack: (3 − 1)·(2⁴ − 1) + 1
const wide = new TCN(3, 32, 3, 4, 1);
console.log("receptive field:", wide.receptiveField); // 31

if (!(loss < 0.1 * initialLoss && loss < 0.03)) throw new Error(`TCN did not learn: ${initialLoss} -> ${loss}`);
if (wide.receptiveField !== 31) throw new Error(`unexpected receptive field: ${wide.receptiveField}`);
```

### NetworkLSTM — recurrent memory

`NetworkLSTM` keeps its state within an episode. Call `resetState()` at the start of every episode, or the state of the previous episode carries over. Gradients are averaged over the episode length, so a longer episode needs a larger learning rate: in this example, rates of 0.05–0.2 did not learn the task in 3000 episodes.

The task: the bit is shown at the first of 5 steps, and the output at the last step must repeat it.

```ts
import { NetworkLSTM } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 2;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

const T = 5;
const net = new NetworkLSTM(1, 8, [1]); // 1 input → LSTM with 8 units → 1 sigmoid output

// One episode: the bit is shown at step 0. With train = true the episode is trained,
// with the target 0 at every step except the last one, where it is the bit.
function run(bit: number, train: boolean): number {
  net.resetState();
  let out = 0;
  for (let t = 0; t < T; t++) out = net.predict([t === 0 ? bit : 0])[0];
  if (train) net.train(Array.from({ length: T }, (_, t) => [t === T - 1 ? bit : 0]), 0.6);
  return out;
}

for (let episodeIndex = 0; episodeIndex < 3000; episodeIndex++) run(episodeIndex % 2, true);

const outFor0 = run(0, false);
const outFor1 = run(1, false);
console.log(outFor0.toFixed(2), outFor1.toFixed(2)); // 0.01 0.90

if (!(outFor0 < 0.2 && outFor1 > 0.5)) throw new Error(`memory task not learned: ${outFor0}, ${outFor1}`);

// Invalid use throws: a model without dense layers, and a targets array of the wrong length.
let rejectedStructure = false;
try {
  new NetworkLSTM(1, 8, []);
} catch {
  rejectedStructure = true;
}
net.resetState();
net.predict([1]);
net.predict([0]);
let rejectedTargets = false;
try {
  net.train([[1]], 0.6); // 1 target for 2 steps
} catch {
  rejectedTargets = true;
}
if (!rejectedStructure || !rejectedTargets) throw new Error("invalid NetworkLSTM use was accepted");
```
