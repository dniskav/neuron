## Training and evaluation tools

These helpers sit around the training loop: metrics, early stopping, the `Trainer` loop, model serialization, data loading and preparation, the tokenizer, and diagnostics for weights and loss curves. Each example is self-contained, fixes its randomness when it uses any, and ends with a check that throws if the result is not the expected one.

### Metrics — evaluate your model

Classification metrics take the true and the predicted labels. `f1Score` without `positiveClass` is the mean of the per-class F1 values (the macro average), not the harmonic mean of the macro precision and recall.

```ts
import { accuracy, f1Score, precision, recall, confusionMatrix, auc } from "@dniskav/neuron";

const yTrue = [0, 1, 1, 0, 1, 2, 2, 2];
const yPred = [0, 1, 0, 0, 1, 2, 0, 2];

const acc = accuracy(yTrue, yPred);       // 6 of 8 correct
const f1 = f1Score(yTrue, yPred);         // macro: mean of the three per-class F1 scores
const p = precision(yTrue, yPred);        // macro precision
const r = recall(yTrue, yPred);           // macro recall
const cm = confusionMatrix(yTrue, yPred); // rows = true class, columns = predicted class
console.log(acc, f1.toFixed(4), p.toFixed(4), r.toFixed(4), JSON.stringify(cm));
// 0.75 0.7556 0.8333 0.7778 [[2,0,0],[1,2,0],[1,0,2]]
// The per-class F1 values are 0.6667, 0.8 and 0.8; the harmonic mean of the macro precision and recall would be 0.8046.
const expectedF1 = (2 / 3 + 0.8 + 0.8) / 3;
if (Math.abs(acc - 0.75) > 1e-12 || Math.abs(f1 - expectedF1) > 1e-12 || JSON.stringify(cm) !== "[[2,0,0],[1,2,0],[1,0,2]]")
  throw new Error(`unexpected metrics: ${acc}, ${f1}, ${JSON.stringify(cm)}`);

// AUC from continuous scores for a binary problem: 1 means every positive ranks above every negative.
const yBin = [0, 1, 1, 0, 1];
const scores = [0.1, 0.9, 0.4, 0.2, 0.8];
const area = auc(yBin, scores);
console.log(area.toFixed(3)); // 1.000
if (Math.abs(area - 1) > 1e-12) throw new Error(`unexpected AUC: ${area}`);

// Invalid inputs throw with the function name. Negative labels are rejected by f1Score,
// but accuracy does not check its labels (a negative label is just a wrong prediction).
const errorOf = (fn: () => unknown): string | null => {
  try {
    fn();
    return null;
  } catch (e) {
    return (e as Error).message;
  }
};
const empty = errorOf(() => accuracy([], []));
const mismatch = errorOf(() => f1Score([0, 1], [0]));
const negative = errorOf(() => f1Score([0, -1], [0, 1]));
console.log(empty);    // Metrics.accuracy: inputs must not be empty
console.log(mismatch); // Metrics.f1Score: yTrue and yPred must have the same length (2 vs 1)
console.log(negative); // Metrics.f1Score: yTrue[1] = -1 is not a non-negative integer label
const acceptsNegative = accuracy([-1], [0]) === 0;
if (!empty || !mismatch || !negative || !acceptsNegative) throw new Error("metrics did not reject invalid input");
```

### EarlyStopping — stop when the validation loss stops improving

`update` takes the monitored value once per epoch and returns `true` when the patience is exhausted. The loss values below are fixed, so the example is deterministic.

```ts
import { EarlyStopping } from "@dniskav/neuron";

// Validation loss per epoch (fixed values): it improves until epoch 2, then gets worse.
const valLoss = [1.0, 0.8, 0.7, 0.71, 0.72, 0.705, 0.9];
const stopper = new EarlyStopping({ patience: 2, minDelta: 0, mode: "min" });

let stoppedAt = -1;
for (let epoch = 0; epoch < valLoss.length; epoch++) {
  if (stopper.update(valLoss[epoch], epoch)) {
    stoppedAt = epoch;
    break;
  }
}
console.log(stoppedAt, stopper.bestValue, stopper.bestEpoch); // 4 0.7 2
if (stoppedAt !== 4 || stopper.bestValue !== 0.7 || stopper.bestEpoch !== 2)
  throw new Error(`unexpected stop: epoch ${stoppedAt}, best ${stopper.bestValue} at ${stopper.bestEpoch}`);

// Invalid settings throw: patience must be an integer >= 1, and a NaN value is rejected.
const errorOf = (fn: () => unknown): string | null => {
  try {
    fn();
    return null;
  } catch (e) {
    return (e as Error).message;
  }
};
const badPatience = errorOf(() => new EarlyStopping({ patience: 0 }));
const nanValue = errorOf(() => new EarlyStopping().update(NaN, 0));
console.log(badPatience); // EarlyStopping.constructor: patience must be an integer >= 1, got 0
console.log(nanValue);    // EarlyStopping.update: value must not be NaN, got NaN
if (!badPatience || !nanValue) throw new Error("EarlyStopping did not reject invalid input");
```

### Trainer — the training loop

`Trainer` runs the epochs, shuffles the samples and records the mean loss of each epoch. It needs a network with a `train(inputs, targets, lr)` method, such as `NetworkN`. An `lr` of 0 is rejected when `train` starts.

```ts
import { NetworkN, Trainer, sigmoid } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights and the shuffling are reproducible.
let seed = 21;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Logical AND.
const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [0], [0], [1]];

const net = new NetworkN([2, 3, 1], { activations: [sigmoid, sigmoid] });
const history = new Trainer(net, { epochs: 300, lr: 0.5 }).train({ inputs, targets });

console.log(history.length, history[0].toFixed(3), history[history.length - 1].toFixed(3)); // 300 0.253 0.006
if (history.length !== 300 || !(history[299] < history[0] / 2))
  throw new Error(`Trainer did not learn: ${history[0]} -> ${history[299]}`);

// lr = 0 is rejected by Trainer (it is a no-op in NetworkN itself).
let message: string | null = null;
try {
  new Trainer(net, { lr: 0 }).train({ inputs, targets });
} catch (e) {
  message = (e as Error).message;
}
console.log(message); // Trainer.train: lr must be a finite number > 0, got 0
if (message === null) throw new Error("lr = 0 was not rejected");
```

### Trainer — early stopping and the best weights

Early stopping needs validation data (`setValidationData`). With `restoreBestWeights`, the network gets back the weights of the best validation epoch when training ends. Without `earlyStopping`, the validation data is stored and ignored.

```ts
import { NetworkN, Trainer, sigmoid } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights and the shuffling are reproducible.
let seed = 8;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Training samples of y = 0.2 + 0.6x with a small deterministic disturbance; clean validation samples.
const train = { inputs: [] as number[][], targets: [] as number[][] };
for (let i = 0; i < 20; i++) {
  const x = i / 19;
  train.inputs.push([x]);
  train.targets.push([0.2 + 0.6 * x + 0.05 * Math.sin(7.3 * i)]);
}
const valid = {
  inputs: Array.from({ length: 10 }, (_, i) => [i / 9]),
  targets: Array.from({ length: 10 }, (_, i) => [0.2 + 0.6 * (i / 9)]),
};

const net = new NetworkN([1, 4, 1], { activations: [sigmoid, sigmoid] });
const trainer = new Trainer(net, {
  epochs: 3000,
  lr: 0.5,
  earlyStopping: { patience: 20, minDelta: 1e-5 },
  restoreBestWeights: true,
});
trainer.setValidationData(valid);
trainer.train(train);

// Mean squared error on the validation set, computed the same way as the Trainer does.
const validationMse = valid.inputs.reduce(
  (s, x, i) => s + (net.predict(x)[0] - valid.targets[i][0]) ** 2, 0,
) / valid.inputs.length;
console.log(trainer.getStopReason(), trainer.getBestLoss().toFixed(6), validationMse.toFixed(6));
// earlyStopping 0.000271 0.000271
if (trainer.getStopReason() !== "earlyStopping" || Math.abs(validationMse - trainer.getBestLoss()) > 1e-9)
  throw new Error(`unexpected stop: ${trainer.getStopReason()}, ${validationMse} vs ${trainer.getBestLoss()}`);

// Without earlyStopping the validation data is ignored and training runs all the epochs.
const plain = new NetworkN([1, 4, 1], { activations: [sigmoid, sigmoid] });
const plainTrainer = new Trainer(plain, { epochs: 5, lr: 0.5 });
plainTrainer.setValidationData(valid);
plainTrainer.train(train);
console.log(plainTrainer.getStopReason(), plainTrainer.getHistory().length); // maxEpochs 5
if (plainTrainer.getStopReason() !== "maxEpochs" || plainTrainer.getHistory().length !== 5)
  throw new Error("validation data changed a run without earlyStopping");
```

### ClippedOptimizerFactory — clip the gradients

`Trainer` has no `clipValue` option. The gradient lives in the optimizer, so the clipping wraps the optimizer factory: `ClippedOptimizerFactory(inner, clipValue)` clamps each gradient to `[-clipValue, +clipValue]` before the inner optimizer runs.

```ts
import { NetworkN, Trainer, Adam, ClippedOptimizerFactory, sigmoid } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights and the shuffling are reproducible.
let seed = 4;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
const targets = [[0], [1], [1], [0]];

const net = new NetworkN([2, 4, 1], {
  activations: [sigmoid, sigmoid],
  optimizer: ClippedOptimizerFactory(() => new Adam(), 1), // every gradient in [-1, 1]
});
const history = new Trainer(net, { epochs: 500, lr: 0.05 }).train({ inputs, targets });

const finite = history.every(Number.isFinite);
console.log(history[0].toFixed(3), history[history.length - 1].toFixed(3), finite); // 0.266 0.001 true
if (!finite || !(history[history.length - 1] < history[0]))
  throw new Error(`clipped training failed: ${history[0]} -> ${history[history.length - 1]}`);
```

### ModelSaver — save and load weights

`toJSON` writes `{ format, length, structure?, weights, timestamp }`. `fromJSON` checks the whole text before it writes anything: a `format` other than 1 throws, and so does a number of weights that does not match the model. Files without `format` (the layout of 0.3.2) still load when the length matches.

```ts
import { NetworkN, ModelSaver, sigmoid } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 2;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const make = (structure: number[]) => new NetworkN(structure, { activations: [sigmoid, sigmoid] });

const original = make([2, 3, 1]);
for (let i = 0; i < 50; i++) {
  original.train([0, 1], [1], 0.5);
  original.train([1, 1], [0], 0.5);
}

const json = ModelSaver.toJSON(original);
const saved = JSON.parse(json);
console.log(saved.format, saved.length, saved.structure); // 1 13 [ 2, 3, 1 ]
if (saved.format !== 1 || saved.length !== 13 || JSON.stringify(saved.structure) !== "[2,3,1]")
  throw new Error(`unexpected header: ${json.slice(0, 80)}`);

// Same structure, different initial weights: loading the JSON makes the predictions identical.
const copy = make([2, 3, 1]);
ModelSaver.fromJSON(copy, json);
const same = original.predict([0, 1])[0] === copy.predict([0, 1])[0];
console.log(same); // true
if (!same) throw new Error("the loaded model does not reproduce the saved one");

// Saving and loading through callbacks, with an in-memory store (no file system).
const files = new Map<string, string>();
ModelSaver.saveToFile(original, "model.json", (path, data) => files.set(path, data));
const fromFile = make([2, 3, 1]);
ModelSaver.loadFromFile(fromFile, "model.json", (path) => files.get(path) ?? "");
if (fromFile.predict([0, 1])[0] !== original.predict([0, 1])[0])
  throw new Error("saveToFile/loadFromFile did not round-trip");

// Invalid files throw before any weight is written.
const errorOf = (fn: () => unknown): string | null => {
  try {
    fn();
    return null;
  } catch (e) {
    return (e as Error).message;
  }
};
const wrongFormat = errorOf(() => ModelSaver.fromJSON(make([2, 3, 1]), JSON.stringify({ ...saved, format: 2 })));
const wrongSize = errorOf(() => ModelSaver.fromJSON(make([2, 4, 1]), json));
console.log(wrongFormat); // ModelSaver.fromJSON: unsupported format 2 (expected 1)
console.log(wrongSize);   // ModelSaver.fromJSON: the model expects 17 weights and the file has 13
if (!wrongFormat || !wrongSize) throw new Error("ModelSaver accepted an invalid file");
```

### DataLoader — batches, validation split and sequences

`DataLoader` yields the training samples in batches of `batchSize` and holds out a fraction of the data for validation. `DataLoader.sequences` turns a time series into windows of `seqLen` values, each with the next value as its target.

```ts
import { DataLoader } from "@dniskav/neuron";

// Seeded generator (Park-Miller) for the shuffling.
let seed = 13;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

const data = {
  inputs: Array.from({ length: 10 }, (_, i) => [i]),
  targets: Array.from({ length: 10 }, (_, i) => [i * 2]),
};
const loader = new DataLoader(data, 4, 0.2); // batches of 4; 20% of the samples held out

let batches = 0;
let seen = 0;
while (loader.hasNext()) {
  const batch = loader.next();
  batches++;
  seen += batch.inputs.length;
}
console.log(loader.length, loader.validationLength, batches, seen); // 8 2 2 8
if (loader.length !== 8 || loader.validationLength !== 2 || batches !== 2 || seen !== 8)
  throw new Error(`unexpected split: ${loader.length} ${loader.validationLength} ${batches} ${seen}`);

// Sliding windows over a series: 6 values with seqLen 3 give 3 windows (the order is shuffled).
const windows = DataLoader.sequences([[1], [2], [3], [4], [5], [6]], 3);
const pairs: string[] = [];
while (windows.hasNext()) {
  const batch = windows.next();
  pairs.push(`${batch.inputs[0].join(",")}>${batch.targets[0][0]}`);
}
pairs.sort();
console.log(windows.length, pairs); // 3 [ '1,2,3>4', '2,3,4>5', '3,4,5>6' ]
if (windows.length !== 3 || pairs.join(" ") !== "1,2,3>4 2,3,4>5 3,4,5>6")
  throw new Error(`unexpected sequence windows: ${pairs}`);

// batchSize must be an integer >= 1.
let rejected = false;
try {
  new DataLoader(data, 0);
} catch {
  rejected = true;
}
if (!rejected) throw new Error("batchSize 0 was not rejected");
```

### Tokenizer — text to ids and back

`Tokenizer` builds a vocabulary with `fit`. Tokens not seen during `fit` are encoded as `<UNK>`, and `decode` writes an unknown id as `<UNK>` too.

```ts
import { Tokenizer } from "@dniskav/neuron";

const tok = new Tokenizer({ mode: "word", lowercase: true });
tok.fit(["the cat sat on the mat", "the cat ate the rat"]);

const ids = tok.encode("the cat", { addBOS: true, addEOS: true });
console.log(ids, JSON.stringify(tok.decode(ids))); // [ 2, 4, 5, 3 ] "the cat"
if (JSON.stringify(tok.decode(ids)) !== '"the cat"') throw new Error(`unexpected decode: ${tok.decode(ids)}`);

// Sequences padded to the same length; <PAD> has id 0.
const batch = tok.encodeBatch(["the cat", "the mat on the mat"], { padTo: 5 });
console.log(batch.map((row) => row.length), batch[0]); // [ 5, 5 ] [ 4, 5, 0, 0, 0 ]
if (batch.some((row) => row.length !== 5) || batch[0][2] !== 0) throw new Error("padding failed");

// Unknown words and unknown ids both come back as <UNK>.
const unknownId = tok.encode("dog")[0];
const unknownText = tok.decode([999]);
console.log(unknownId === tok.tokenToId(Tokenizer.UNK), unknownText); // true <UNK>
if (unknownId !== tok.tokenToId(Tokenizer.UNK) || unknownText !== "<UNK>")
  throw new Error(`unexpected unknown token: ${unknownId}, ${unknownText}`);
```

### DatasetLoader — CSV to a DataPair

`DatasetLoader.fromCSV` parses CSV text (no file access). Numeric columns are parsed as they are, and string columns are one-hot encoded; `targetNames` says what each target position means.

```ts
import { DatasetLoader, DataLoader } from "@dniskav/neuron";

const csv = `sepal,petal,species
5.1,1.4,setosa
6.9,4.9,versicolor
6.3,6.0,virginica
4.9,1.5,setosa`;

const ds = DatasetLoader.fromCSV(csv, {
  featureCols: ["sepal", "petal"],
  targetCols: ["species"], // a string column is one-hot encoded
});
console.log(ds.numRows, ds.inputs[0], ds.targets[1], ds.targetNames);
// 4 [ 5.1, 1.4 ] [ 0, 1, 0 ] [ 'species_setosa', 'species_versicolor', 'species_virginica' ]
if (ds.numRows !== 4 || ds.inputs[0][1] !== 1.4 || ds.targets[1].join(",") !== "0,1,0")
  throw new Error(`unexpected dataset: ${ds.numRows}, ${ds.targets[1]}`);

// A row with a missing field throws (the header has 3 fields).
let message: string | null = null;
try {
  DatasetLoader.fromCSV("sepal,petal,species\n5.1,1.4", { featureCols: ["sepal", "petal"], targetCols: ["species"] });
} catch (e) {
  message = (e as Error).message;
}
console.log(message); // DatasetLoader.fromCSV: row 1 has 2 fields, expected 3.
if (message === null) throw new Error("a short CSV row was not rejected");

// The result feeds a DataLoader directly.
const loader = new DataLoader(ds, 2);
console.log(loader.length); // 4
if (loader.length !== 4) throw new Error("DataLoader did not take the dataset");
```

### DataAugmentation — split, normalize and augment

`DataAugmentation` splits a dataset, scales it using the training statistics, and adds noisy copies of the samples. `normalize` gives the min and max per feature, which `normalizePoint` reuses for new samples.

```ts
import { DataAugmentation } from "@dniskav/neuron";

// Seeded generator (Park-Miller) for the shuffling and the noise.
let seed = 17;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Two features on different scales.
const X = Array.from({ length: 10 }, (_, i) => [i, 100 - i * 10]);
const y = Array.from({ length: 10 }, (_, i) => i % 2);

const { trainX, trainY, valX, testX } = DataAugmentation.split(X, y, 0.6, 0.2);
console.log(trainX.length, valX.length, testX.length); // 6 2 2
if (trainX.length !== 6 || valX.length !== 2 || testX.length !== 2) throw new Error("unexpected split sizes");

// Fit the scaling on the training part only, then apply it to the validation part.
const { normalized, min, max } = DataAugmentation.normalize(trainX);
const valNormalized = valX.map((x) => DataAugmentation.normalizePoint(x, min, max));
const inRange = normalized.every((row) => row.every((v) => v >= 0 && v <= 1));
console.log(inRange, valNormalized.length); // true 2
if (!inRange || valNormalized.length !== 2) throw new Error("normalization left the [0, 1] range");

// Three copies of each training sample (the original and two noisy ones).
const augmented = DataAugmentation.augmentBatch(normalized, trainY, 3, 0.02);
console.log(augmented.X.length, augmented.y.length); // 18 18
if (augmented.X.length !== 18 || augmented.y.length !== 18) throw new Error("augmentBatch returned the wrong size");

// Invalid input throws: mismatched lengths, empty data, and ratios that leave no test set.
const errorOf = (fn: () => unknown): string | null => {
  try {
    fn();
    return null;
  } catch (e) {
    return (e as Error).message;
  }
};
const badShuffle = errorOf(() => DataAugmentation.shuffle([[1], [2]], [0]));
const emptyNormalize = errorOf(() => DataAugmentation.normalize([]));
const badSplit = errorOf(() => DataAugmentation.split(X, y, 0.8, 0.2));
console.log(badShuffle);     // DataAugmentation.shuffle: X has 2 rows but y has 1 labels
console.log(emptyNormalize); // DataAugmentation.normalize: X is empty
console.log(badSplit);       // trainRatio (0.8) + valRatio (0.2) must be < 1
if (!badShuffle || !emptyNormalize || !badSplit) throw new Error("DataAugmentation accepted invalid input");
```

### WeightInspector — diagnose your network

`WeightInspector` reports, for each layer, the mean, standard deviation, range, number of near-zero weights and number of parameters (weights and biases). `print` shows them as a table.

```ts
import { NetworkN, WeightInspector, relu, sigmoid } from "@dniskav/neuron";

// Seeded generator (Park-Miller) so the initial weights are reproducible.
let seed = 29;
Math.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

const net = new NetworkN([8, 6, 4, 3], { activations: [relu, relu, sigmoid] });
WeightInspector.print(net);

const stats = WeightInspector.inspect(net);
const total = WeightInspector.inspectAll(net).totalParams;
console.log(stats.map((s) => s.totalParams), total); // [ 54, 28, 15 ] 97 (weights and biases)
if (JSON.stringify(stats.map((s) => s.totalParams)) !== "[54,28,15]" || total !== 97)
  throw new Error(`unexpected parameter counts: ${stats.map((s) => s.totalParams)}, ${total}`);

// A negative or NaN threshold throws.
let rejected = false;
try {
  WeightInspector.inspect(net, NaN);
} catch {
  rejected = true;
}
if (!rejected) throw new Error("NaN threshold was not rejected");
```

### LossPlotter — ASCII loss curve

`LossPlotter` draws a loss curve as text, with a size you choose in the constructor, so the output does not depend on the terminal. `render()` returns the chart as a string; `print()` writes it to the console.

```ts
import { LossPlotter } from "@dniskav/neuron";

const plotter = new LossPlotter({ width: 40, height: 6, title: "Training loss" });
for (let e = 0; e < 100; e++) plotter.add(2 * Math.exp(-0.05 * e), e);

const chart = plotter.render();
console.log(chart);
const footer = chart.split("\n").pop()!;
console.log(footer); // min=0.01417  max=2.000  last=0.01417  n=100
if (!footer.includes("max=2.000") || !footer.includes("n=100") || chart.split("\n").length !== 10)
  throw new Error(`unexpected chart footer: ${footer}`);

// Only finite values are accepted.
let message: string | null = null;
try {
  plotter.add(NaN);
} catch (e) {
  message = (e as Error).message;
}
console.log(message); // LossPlotter.add: loss must be a finite number, got NaN
if (message === null) throw new Error("NaN was not rejected");
```
