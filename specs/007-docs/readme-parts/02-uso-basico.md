## Usage

Each example is self-contained: it imports from `@dniskav/neuron`, defines its own data and runs in a few seconds, with no network and no files. Examples that use randomness fix it, so the printed values are reproducible: most replace `Math.random` with a small seeded generator, and `KMeans` takes a `seed` option. Each example ends with a check that throws if the result is not the expected one.

### Single neuron — learn a threshold

```ts
import { Neuron } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 42;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Minors (10 to 16) are labelled 0 and adults (20 to 30) are labelled 1.
const data: [number, number][] = [[10, 0], [30, 1], [12, 0], [28, 1], [14, 0], [25, 1], [16, 0], [20, 1]];
const neuron = new Neuron();

for (let epoch = 0; epoch < 1000; epoch++) {
  for (const [age, label] of data) neuron.train(age, label, 0.1);
}

const minor = neuron.predict(10);
const adult = neuron.predict(30);
console.log(minor.toFixed(3), adult.toFixed(3)); // 0.094 1.000
if (!(minor < 0.2 && adult > 0.8)) throw new Error(`unexpected output: ${minor}, ${adult}`);
```

### NetworkN — deep network with custom architecture

`NetworkN` takes the layer sizes, one activation per layer and an optimizer factory. This example learns XOR, which a single neuron cannot represent, with one ReLU hidden layer and a sigmoid output trained with Adam.

`train` returns the mean squared error of the example. The step it applies is the gradient of that mean error: with one output, the error term is `2 × (target − prediction)`, multiplied by the activation derivative and by the learning rate.

```ts
import { NetworkN, relu, sigmoid, Adam } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 7;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

const inputs  = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [1], [1], [0]];

const net = new NetworkN([2, 8, 1], {
  activations: [relu, sigmoid],
  optimizer: () => new Adam(),
});

let meanLoss = 0;
for (let epoch = 0; epoch < 2000; epoch++) {
  let sum = 0;
  for (let i = 0; i < inputs.length; i++) {
    sum += net.train(inputs[i], targets[i], 0.01);
  }
  meanLoss = sum / inputs.length;
}

const predictions = inputs.map((x) => net.predict(x)[0]);
console.log(predictions.map((p) => p.toFixed(2)), meanLoss.toFixed(4)); // [ '0.00', '1.00', '1.00', '0.00' ] 0.0000
const correct = predictions.every((p, i) => (targets[i][0] === 1 ? p > 0.5 : p < 0.5));
if (!correct || meanLoss > 0.05) throw new Error(`XOR not learned: ${predictions}, loss ${meanLoss}`);
```

### Historical Perceptron — step function, no hidden layers

```ts
import { Perceptron } from "@dniskav/neuron";

const p = new Perceptron(2);

// Learns AND gate (linearly separable). Each row is [a, b, target].
const data: [number, number, 0 | 1][] = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]];
for (let e = 0; e < 100; e++)
  for (const [a, b, t] of data) p.train([a, b], t, 0.1);

const and11 = p.predict([1, 1]);
const and01 = p.predict([0, 1]);
console.log(and11, and01); // 1 0
if (and11 !== 1 || and01 !== 0) throw new Error(`AND not learned: ${and11}, ${and01}`);
// XOR cannot be learned: it is not linearly separable.
```

### Linear Regression — normal equation

```ts
import { LinearRegression } from "@dniskav/neuron";

const model = new LinearRegression();

// Exact closed-form solution in one call
model.fitNormal(
  [[1], [2], [3], [4]],  // X
  [2, 4, 6, 8]           // y = 2x
);

const y5 = model.predict([5]);
const { weights, bias } = model.getCoefficients();
console.log(y5, weights, bias); // ~10, [~2], ~0
if (Math.abs(y5 - 10) > 1e-9 || Math.abs(weights[0] - 2) > 1e-9 || Math.abs(bias) > 1e-9)
  throw new Error(`unexpected fit: ${y5}, ${weights}, ${bias}`);
```

### Logistic Regression — sigmoid + BCE

`train` runs online SGD for the given number of epochs and returns the mean binary cross-entropy of each epoch. Labels must be numbers in [0, 1].

```ts
import { LogisticRegression } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 11;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

const clf = new LogisticRegression();
const lossHistory = clf.train(
  [[0, 0], [1, 1], [1, 0], [0, 1]],
  [0, 1, 1, 0],
  0.1, 500
);

const first = lossHistory[0];
const last = lossHistory[lossHistory.length - 1];
const high = clf.classify([0.9, 0.9]);
const low = clf.classify([0.1, 0.1]);
console.log(first.toFixed(3), last.toFixed(3), high, low); // 1.022 0.027 1 0
if (!(last < first && high === 1 && low === 0)) throw new Error("logistic regression did not learn");
```

### Gaussian Naive Bayes — zero gradient descent

```ts
import { GaussianNaiveBayes } from "@dniskav/neuron";

const nb = new GaussianNaiveBayes();
nb.fit(
  [[1.2, 0.5], [1.4, 0.7], [5.0, 4.5], [5.2, 4.8]],
  [0, 0, 1, 1]
);

const a = nb.predict([1.3, 0.6]);
const b = nb.predict([5.1, 4.6]);
console.log(a, b); // 0 1
if (a !== 0 || b !== 1) throw new Error(`unexpected classes: ${a}, ${b}`);
```

### Decision Tree — Gini split

`maxDepth` limits how deep the tree grows (an integer ≥ 1, or `Infinity` for no limit). `predict` rejects inputs with the wrong number of features.

```ts
import { DecisionTree } from "@dniskav/neuron";

// XOR: no single split helps at the root, but a depth-2 tree separates it.
const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [0, 1, 1, 0];

const tree = new DecisionTree({ maxDepth: 2, task: "classification" });
tree.fit(X, y);
const predictions = tree.predictBatch(X);
console.log(predictions); // [0, 1, 1, 0]
if (JSON.stringify(predictions) !== "[0,1,1,0]") throw new Error(`unexpected: ${predictions}`);

let rejected = false;
try {
  tree.predict([1]); // one feature, the tree was fitted with two
} catch {
  rejected = true;
}
if (!rejected) throw new Error("a 1-feature input was not rejected");
```

### K-Means — unsupervised clustering

```ts
import { KMeans } from "@dniskav/neuron";

// Three small groups of 2-D points, around (0, 0), (5, 5) and (0, 5).
const points = [
  [0.1, 0.2], [0.0, 0.4], [0.3, 0.1], [0.2, 0.3],
  [5.1, 5.0], [4.9, 5.2], [5.2, 4.8], [5.0, 5.1],
  [0.1, 5.1], [-0.1, 4.9], [0.2, 5.2], [0.0, 5.0],
];

const km = new KMeans(3, { seed: 42 }); // 3 clusters; the seed fixes the initial centroids
km.fit(points);

const labels = km.predictBatch(points);
const groups = [labels.slice(0, 4), labels.slice(4, 8), labels.slice(8, 12)];
const uniform = groups.every((g) => g.every((l) => l === g[0]));
const distinct = new Set(groups.map((g) => g[0])).size === 3;

const cluster = km.predict([1.2, 0.5]); // index 0, 1 or 2: the cluster of the first group
const inertia = km.inertia(points);     // lower = better fit
console.log(labels, cluster, inertia.toFixed(3)); // [1,1,1,1,0,0,0,0,2,2,2,2] 1 0.337
if (!uniform || !distinct || cluster !== labels[0] || inertia > 1)
  throw new Error(`unexpected clustering: ${labels}, ${cluster}, ${inertia}`);

let rejected = false;
try {
  km.predict([1, 2, 3]); // 3 features, the model was fitted with 2
} catch {
  rejected = true;
}
if (!rejected) throw new Error("a 3-feature input was not rejected");
```

### PCA — dimensionality reduction

`explainedVarianceRatio()` divides each eigenvalue by the total variance of the data (all features), not by the variance of the kept components. So with fewer components than features the ratios sum to less than 1, unless the dropped components have zero variance.

```ts
import { PCA } from "@dniskav/neuron";

// Seeded generator (Park-Miller): PCA starts its power iteration from a random vector.
let seed = 5;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// 100 samples × 10 features built from three signals plus a little deterministic noise.
const X = Array.from({ length: 100 }, (_, i) => {
  const a = Math.sin(0.1 * i);
  const b = Math.cos(0.37 * i);
  const c = Math.sin(0.23 * i + 1);
  return Array.from({ length: 10 }, (_, j) =>
    (j + 1) * a + (10 - j) * 0.5 * b + Math.cos((j + 2) * 0.9) * c + 0.05 * Math.sin(1.7 * i * (j + 1))
  );
});

const pca = new PCA(2); // keep top 2 components
pca.fit(X);

const Z = pca.transform(X);         // 100 × 2
const X2 = pca.inverseTransform(Z); // reconstructed 100 × 10

const ratios = pca.explainedVarianceRatio();
const total = ratios.reduce((s, r) => s + r, 0);
let maxError = 0;
for (let i = 0; i < X.length; i++)
  for (let j = 0; j < X[i].length; j++)
    maxError = Math.max(maxError, Math.abs(X[i][j] - X2[i][j]));

console.log(Z.length, Z[0].length, ratios.map((r) => r.toFixed(4)), maxError.toFixed(4));
// 100 2 [ '0.8581', '0.1307' ] 1.2585
// the ratios sum to ~0.99, below 1 because 8 of the 10 components are dropped
if (Z.length !== 100 || Z[0].length !== 2 || !(total > 0.95 && total < 1) || maxError > 2)
  throw new Error(`unexpected PCA: sum ${total}, error ${maxError}`);
```

### Self-Organizing Map

```ts
import { SOM } from "@dniskav/neuron";

// Seeded generator (Park-Miller) for the initial prototypes and the sample order.
let seed = 3;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// 80 colours on a grid, normalised to [0, 1] (RGB).
const colors: number[][] = [];
for (let r = 0; r < 4; r++)
  for (let g = 0; g < 4; g++)
    for (let b = 0; b < 5; b++) colors.push([r / 3, g / 3, b / 4]);

const som = new SOM(10, 10, 3); // 10×10 grid, 3-dimensional inputs (RGB)
som.train(colors, 200);

const [redRow, redCol] = som.getBMU([1, 0, 0]);   // best matching unit for red
const [blueRow, blueCol] = som.getBMU([0, 0, 1]); // and for blue
const qe = som.quantizationError(colors);         // lower = better fit
console.log([redRow, redCol], [blueRow, blueCol], qe.toFixed(3)); // [0, 0] [9, 0] 0.187

// Red and blue end up in different parts of the grid (9 cells apart with this seed), and the
// prototypes stay close to the data.
const gridDistance = Math.hypot(redRow - blueRow, redCol - blueCol);
if (!(gridDistance >= 3 && qe < 0.25)) throw new Error(`unexpected map: distance ${gridDistance}, qe ${qe}`);
```

### Hopfield Network — associative memory

Inputs must be bipolar (+1/−1). `HopfieldNetwork.binarize` converts a 0/1 array to bipolar; `recall` returns the converged state. Capacity is about 0.138 × N patterns: `store` logs a warning once when the number of stored patterns exceeds `max(1, floor(0.138 × N))`, which is 8 for `N = 64`.

```ts
import { HopfieldNetwork } from "@dniskav/neuron";

// Seeded generator (Park-Miller) for the order of the asynchronous updates.
let seed = 9;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Two 8×8 patterns (64 bits): a checkerboard and vertical stripes.
const pattern1 = Array.from({ length: 64 }, (_, i) => ((i % 8) + Math.floor(i / 8)) % 2);
const pattern2 = Array.from({ length: 64 }, (_, i) => (i % 8 < 4 ? 1 : 0));

const net = new HopfieldNetwork(64); // 64 binary neurons
net.store(HopfieldNetwork.binarize(pattern1)); // converts 0/1 → -1/+1
net.store(HopfieldNetwork.binarize(pattern2));

// Flip 5 bits of the first pattern and recall it from the noisy copy.
const noisy = pattern1.slice();
for (const i of [0, 9, 18, 27, 36]) noisy[i] = 1 - noisy[i];
const noisyBipolar = HopfieldNetwork.binarize(noisy);
const recovered = net.recall(noisyBipolar);

const same = recovered.every((v, i) => v === HopfieldNetwork.binarize(pattern1)[i]);
console.log(same, net.energy(noisyBipolar).toFixed(1), net.energy(recovered).toFixed(1)); // true -22.1 -31.0
// the recovered state has lower energy than the noisy input
if (!same || net.energy(recovered) >= net.energy(noisyBipolar))
  throw new Error("the stored pattern was not recalled");
```
