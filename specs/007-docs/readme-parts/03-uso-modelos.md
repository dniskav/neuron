### Autoencoder — learn compressed representations

An `Autoencoder` is trained to reproduce its own input through a narrow bottleneck. `train` takes one gradient step on the mean squared error, so its step is the gradient of the loss it returns. Because the loss is averaged over the `inputSize` outputs, a useful `lr` is larger than in 0.3.2 (this example uses 2). A latent size that is not smaller than the input size is accepted without error or warning (an over-complete autoencoder is valid, for example for denoising).

```ts
import { Autoencoder } from "@dniskav/neuron";

// Fixed seed so the run is reproducible (the library has no global seed).
let s = 1;
Math.random = () => {
  s = (s + 0x6d2b79f5) | 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// 40 samples that really live on a 2-D surface inside 8-D space.
const data = Array.from({ length: 40 }, () => {
  const a = Math.random(), b = Math.random();
  return [a, b, (a + b) / 2, 1 - a, 1 - b, a * b, (a + 1) / 2, (b + 1) / 2];
});

// 8 → [6] → 2 (latent) → [6] → 8
const ae = new Autoencoder(8, [6], 2, [6]);

const before = ae.trainBatch(data, 2); // mean reconstruction MSE of the first epoch
let after = before;
for (let epoch = 0; epoch < 500; epoch++) after = ae.trainBatch(data, 2);

const latent = ae.encode(data[0]);        // compressed: 2 values
const reconstructed = ae.reconstruct(data[0]); // decoded back: 8 values

if (!(after < 0.5 * before)) throw new Error("the reconstruction error did not drop");
if (latent.length !== 2 || reconstructed.length !== 8) throw new Error("unexpected sizes");
new Autoencoder(4, [4], 6, [4]); // over-complete: no error, no warning

console.log(before.toFixed(4), "->", after.toFixed(4)); // 0.0717 -> 0.0019
```

### GAN — generative adversarial training

The discriminator must end in a sigmoid (the exported `sigmoid`, which is the default, or an activation equal to it by value); any other output activation makes the constructor throw. The generator receives the real gradient of `-log D(G(z))` through the discriminator. Adversarial training does not converge to a fixed point: the two losses oscillate, so look at the samples rather than at a single loss value. The example below uses a tiny target and only claims that the generated samples end up closer to it than they started.

```ts
import { GAN } from "@dniskav/neuron";

// Fixed seed so the run is reproducible (the library has no global seed).
let s = 1;
Math.random = () => {
  s = (s + 0x6d2b79f5) | 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// Real data: 2-D points clustered around (0.8, 0.2).
const realSample = () => [0.8 + 0.05 * (Math.random() - 0.5), 0.2 + 0.05 * (Math.random() - 0.5)];

const gan = new GAN(
  4,       // latentDim
  [8],     // generator hidden layers
  2,       // outputDim (size of generated samples)
  [8],     // discriminator hidden layers (its output is a sigmoid)
);

// Distance between the mean of 100 generated samples and the real mean.
const distance = () => {
  const m = [0, 0];
  for (let i = 0; i < 100; i++) {
    const g = gan.generate();
    m[0] += g[0] / 100;
    m[1] += g[1] / 100;
  }
  return Math.hypot(m[0] - 0.8, m[1] - 0.2);
};

const start = distance();
let last = { dLoss: NaN, gLoss: NaN };
for (let step = 0; step < 300; step++) {
  const realBatch = Array.from({ length: 8 }, realSample);
  last = gan.trainStep(realBatch, 0.1); // { dLoss, gLoss } of this step
}
const end = distance();

const fake = gan.generate(); // new synthetic sample, 2 values in (0, 1)
if (!Number.isFinite(last.dLoss) || !Number.isFinite(last.gLoss)) throw new Error("non-finite loss");
if (fake.length !== 2 || fake.some((v) => !(v > 0 && v < 1))) throw new Error("unexpected sample");
if (!(end < start)) throw new Error("the generated samples did not get closer to the real ones");

console.log(start.toFixed(3), "->", end.toFixed(3)); // 0.318 -> 0.107
console.log(`D: ${last.dLoss.toFixed(3)}  G: ${last.gLoss.toFixed(3)}`); // D: 1.429  G: 0.708
```

### VAE — variational autoencoder

The encoder output is linear (`mu` and `logVar` are unbounded) and the reconstruction gradient reaches the encoder. The loss is `MSE + klWeight · KL`, where the MSE is averaged over the inputs and the KL is summed over the latent dimensions. With the default `klWeight = 1` the latent code collapses on small data in [0, 1] (every `mu` is the same and the model can only predict the mean); `klWeight` of about 0.003 let it learn in the measured case. This is one measured case, not a law: tune `klWeight` for your data. The example trains both and checks the difference.

```ts
import { VAE } from "@dniskav/neuron";

// Fixed seed so the run is reproducible (the library has no global seed).
let s = 5;
Math.random = () => {
  s = (s + 0x6d2b79f5) | 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const data = [
  [0.1, 0.9, 0.1, 0.9], [0.9, 0.1, 0.9, 0.1], [0.2, 0.8, 0.2, 0.8],
  [0.8, 0.2, 0.8, 0.2], [0.1, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1],
];

// MSE of the best constant predictor (the mean of the data): the bar to beat.
const mean = [0, 1, 2, 3].map((j) => data.reduce((a, x) => a + x[j], 0) / data.length);
const sq = (x: number[], r: number[]) => x.reduce((a, v, j) => a + (v - r[j]) ** 2, 0) / x.length;
const baseline = data.reduce((a, x) => a + sq(x, mean), 0) / data.length;

function trainAndScore(klWeight: number) {
  // VAE(inputSize, encoderHidden, latentDim, decoderHidden, options, klWeight)
  const vae = new VAE(4, [6], 2, [6], undefined, klWeight);
  for (let epoch = 0; epoch < 3000; epoch++) for (const x of data) vae.train(x, 0.1);
  const score = data.reduce((a, x) => a + sq(x, vae.decode(vae.encode(x).mu)), 0) / data.length;
  return { vae, score };
}

const collapsed = trainAndScore(1);    // default klWeight
const learned = trainAndScore(0.003);  // small klWeight for small [0, 1] data

const { vae } = learned;
const { mu, logVar } = vae.encode(data[0]);  // linear outputs: distribution parameters
const z = vae.reparametrize(mu, logVar);     // sample z ~ N(mu, sigma^2)
const generated = vae.generate();            // decode a random z ~ N(0, I): 4 values

if (!(collapsed.score > 0.9 * baseline)) throw new Error("klWeight = 1 was expected to collapse here");
if (!(learned.score < 0.7 * baseline)) throw new Error("klWeight = 0.003 was expected to beat the baseline");
if (z.length !== 2 || generated.length !== 4) throw new Error("unexpected sizes");

console.log(baseline.toFixed(4), collapsed.score.toFixed(4), learned.score.toFixed(4)); // 0.1367 0.1367 0.0540
```

### Word2Vec — learn word embeddings

`train(corpus, lr, epochs)` needs a finite `lr` >= 0 and an integer `epochs` >= 1, and returns the mean loss of each epoch. On a corpus this small the embeddings only show coarse structure (words that share contexts end up similar), so the example checks that, not the exact ranking of the neighbours. Vector arithmetic with `analogy` needs a much larger corpus to mean anything.

```ts
import { Word2Vec } from "@dniskav/neuron";

// Fixed seed so the run is reproducible (the library has no global seed).
let s = 1;
Math.random = () => {
  s = (s + 0x6d2b79f5) | 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const corpus = [
  ["the", "king", "rules", "the", "kingdom"],
  ["the", "queen", "rules", "the", "land"],
  ["man", "and", "woman", "are", "human"],
];

const w2v = new Word2Vec(16, { model: "skipgram", windowSize: 2 }); // 16-dim embeddings
w2v.buildVocab(corpus);
const losses = w2v.train(corpus, 0.05, 200); // lr 0.05, 200 epochs

const kingQueen = w2v.similarity("king", "queen"); // share the context "the _ rules the"
const kingHuman = w2v.similarity("king", "human"); // never share a context
const neighbours = w2v.mostSimilar("king", 3).map((r) => r.word);

if (!(losses[losses.length - 1] < 0.8 * losses[0])) throw new Error("the loss did not drop");
if (!(kingQueen > 0.5 && kingQueen > kingHuman + 0.5)) throw new Error("king should be close to queen");
if (neighbours.length !== 3) throw new Error("expected 3 neighbours");

console.log(losses[0].toFixed(3), "->", losses[losses.length - 1].toFixed(3)); // 2.373 -> 1.272
console.log(kingQueen.toFixed(3), kingHuman.toFixed(3)); // 0.976 -0.158
console.log(neighbours); // [ 'queen', 'land', 'kingdom' ]
```

### t-SNE — visualize embeddings in 2D

The default `lr` of 200 is not safe for a small number of points: in a measured case with 30 points it diverges, while 5 to 10 converges (the working range depends on n; from about 100 points on the default works, see `TSNEOptions.lr` in `src/TSNE.ts`). `perplexity` must be smaller than the number of points. `seed` makes the embedding reproducible.

```ts
import { TSNE } from "@dniskav/neuron";

// Fixed seed for the data (TSNE has its own `seed` option).
let s = 1;
Math.random = () => {
  s = (s + 0x6d2b79f5) | 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// 30 points in 5-D: two clusters of 15, centred 6 apart on every axis.
const X = Array.from({ length: 30 }, (_, i) =>
  Array.from({ length: 5 }, () => (i < 15 ? 0 : 6) + 2 * Math.random() - 1),
);

// Distance between the cluster centres divided by the mean spread inside a cluster.
function separation(Y: number[][]): number {
  const stats = [Y.slice(0, 15), Y.slice(15)].map((pts) => {
    const c = [0, 1].map((j) => pts.reduce((a, p) => a + p[j], 0) / pts.length);
    const spread = pts.reduce((a, p) => a + Math.hypot(p[0] - c[0], p[1] - c[1]), 0) / pts.length;
    return { c, spread };
  });
  const gap = Math.hypot(stats[0].c[0] - stats[1].c[0], stats[0].c[1] - stats[1].c[1]);
  return gap / ((stats[0].spread + stats[1].spread) / 2);
}

// A small lr suited to n = 30, and the default (200) for comparison.
const good = new TSNE({ perplexity: 5, lr: 10, nIter: 300, seed: 42 });
const points2D = good.fitTransform(X); // [30][2]
const bad = new TSNE({ perplexity: 5, nIter: 300, seed: 42 }); // lr = 200
bad.fitTransform(X);

const sepGood = separation(points2D);
const sepBad = separation(bad.embedding);
if (points2D.length !== 30 || points2D[0].length !== 2) throw new Error("unexpected shape");
if (!(sepGood > 3 && good.kl() < 0.5)) throw new Error("lr = 10 should separate the clusters");
if (!(sepBad < 1.5 && bad.kl() > 0.8)) throw new Error("lr = 200 was expected to diverge at n = 30");

console.log(`lr 10:  separation ${sepGood.toFixed(2)}, KL ${good.kl().toFixed(3)}`); // lr 10:  separation 5.41, KL 0.215
console.log(`lr 200: separation ${sepBad.toFixed(2)}, KL ${bad.kl().toFixed(3)}`); // lr 200: separation 0.52, KL 1.585
```

### PositionalEncoding — order without parameters

`PositionalEncoding` needs `dModel` to be an integer >= 1 (odd values are fine: the last dimension then uses only the sine) and `pos`/`seqLen` to be integers >= 0; anything else throws. `LearnedPositionalEncoding` adds `weights[pos][d]` to each embedding, and its `update(dWeights, lr)` does `weights += lr * dWeights`, so pass the negative loss gradient (the same delta convention as the rest of the library).

```ts
import { PositionalEncoding, LearnedPositionalEncoding } from "@dniskav/neuron";

// Token embeddings: 3 tokens of 4 values each.
const tokenEmbeddings = [
  [0, 0, 0, 0],
  [1, 1, 1, 1],
  [2, 2, 2, 2],
];

// Sinusoidal: deterministic, no training needed.
const pe = PositionalEncoding.encodeSequence(8, 4);         // [8][4]
const withPos = PositionalEncoding.apply(tokenEmbeddings);  // adds the encoding of positions 0, 1, 2
const odd = PositionalEncoding.encode(1, 5).map((v) => +v.toFixed(4)); // dModel = 5 is valid

// Learned: trainable, with a fixed maxSeqLen (a longer sequence throws).
const lpe = new LearnedPositionalEncoding(8, 4);
const withLearnedPos = lpe.apply(tokenEmbeddings);
const before = lpe.weights[1][2];
const delta = lpe.weights.map((row) => row.map(() => 0));
delta[1][2] = 1;
lpe.update(delta, 0.5); // weights[1][2] += 0.5 * 1

let rejected = 0;
try { PositionalEncoding.encode(-1, 4); } catch { rejected++; } // pos must be an integer >= 0
try { PositionalEncoding.encode(0, 0); } catch { rejected++; }  // dModel must be an integer >= 1
try { lpe.apply(new Array(9).fill([0, 0, 0, 0])); } catch { rejected++; } // 9 > maxSeqLen = 8

if (pe.length !== 8 || pe[0].length !== 4) throw new Error("unexpected shape");
if (withPos[0].join() !== "0,1,0,1") throw new Error("position 0 should add [0, 1, 0, 1]");
if (withLearnedPos.length !== 3 || withLearnedPos[0].length !== 4) throw new Error("unexpected shape");
if (Math.abs(lpe.weights[1][2] - before - 0.5) > 1e-12) throw new Error("update should add lr * dWeights");
if (rejected !== 3) throw new Error("invalid arguments should throw");

console.log(odd);        // [ 0.8415, 0.5403, 0.0251, 0.9997, 0.0006 ]
console.log(withPos[0]); // [ 0, 1, 0, 1 ]
```

### ContrastiveLearning — representations without labels

The constructor is `(inputSize, encoderHidden, projectionDim, options)`. The encoder is `inputSize → ...encoderHidden` (ReLU), so `encode` returns `encoderHidden[last]` values. The projection head is `encoderHidden[last] → max(projectionDim, floor(encoderHidden[last] / 2)) → projectionDim` with a linear output, so `project` can return negative values. `trainStep` returns the NT-Xent loss before the update and estimates gradients by finite differences (slow: about `O(parameters)` forward passes per step).

Known limitation: with the raw random initialization training is not reliable. In the maintainers' measurements over 40 initializations, 19 started dead (the encoder or the projection mapped everything to zero), and only 12 of the 21 healthy ones learned. The example therefore shows the API and checks only what always holds (the returned loss is the loss of the pairs before the step, and it is finite); it does not claim that the representation improves. `Augmenter.dropoutFeatures` sets each value to 0 with probability `rate` and does not rescale the kept ones (it is not inverted dropout).

```ts
import { ContrastiveLearning, Augmenter } from "@dniskav/neuron";

// Fixed seed so the run is reproducible (the library has no global seed).
let s = 1;
Math.random = () => {
  s = (s + 0x6d2b79f5) | 0;
  let t = Math.imul(s ^ (s >>> 15), 1 | s);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// Unlabeled data: 8 samples of 8 features.
const unlabeledData = Array.from({ length: 8 }, () =>
  Array.from({ length: 8 }, () => Math.random()),
);

// Encoder: 8 → [16, 8]; projection head: 8 → 8 → 4 (linear output).
const cl = new ContrastiveLearning(8, [16, 8], 4, { temperature: 0.5 });

// Two augmented views per sample: [original, noisy + feature dropout].
const pairs = unlabeledData.map((x) => Augmenter.makePair(x));

const expected = cl.computeLoss(pairs); // forward only, no update
const loss = cl.trainStep(pairs, 0.05); // loss of these pairs BEFORE the update
for (let step = 0; step < 4; step++) cl.trainStep(unlabeledData.map((x) => Augmenter.makePair(x)), 0.05);

const representation = cl.encode(unlabeledData[0]);  // 8 values: use this downstream
const projection = cl.project(unlabeledData[0]);     // 4 values: only for the loss

// dropoutFeatures zeroes values and does NOT rescale the kept ones.
const dropped = Augmenter.dropoutFeatures([1, 1, 1, 1, 1, 1, 1, 1], 0.5);

if (!Number.isFinite(loss) || loss <= 0) throw new Error("the loss should be finite and positive");
if (Math.abs(loss - expected) > 1e-12) throw new Error("trainStep should return the loss before the update");
if (representation.length !== 8 || projection.length !== 4) throw new Error("unexpected sizes");
if (!dropped.every((v) => v === 0 || v === 1)) throw new Error("kept values must not be rescaled");

console.log(loss.toFixed(4), representation.length, projection.length); // 2.7051 8 4
console.log(dropped.join(",")); // 1,0,1,1,0,0,0,1
```
