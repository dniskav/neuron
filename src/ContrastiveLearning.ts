// ─── CONTRASTIVE LEARNING ─────────────────────────────────────────────────────
//
// Supervised learning requires labels. Contrastive learning sidesteps this:
// it trains an encoder to produce useful representations using only the
// structure of the data itself — no annotation needed.
//
// Core intuition:
//   Two "views" (augmented versions) of the same example should map to nearby
//   points in embedding space. Views from different examples should map far apart.
//
// This file implements a simplified SimCLR pipeline:
//   Augmenter      — creates two views of each input
//   ContrastiveLearning — encoder + projection head + NT-Xent loss
//
// References:
//   Chen et al., "A Simple Framework for Contrastive Learning of Visual
//   Representations" (SimCLR), ICML 2020.
//
// ─────────────────────────────────────────────────────────────────────────────

import { NetworkN }          from './NetworkN';
import { NetworkNOptions }   from './NetworkN';
import { relu, linear }      from './activations';

// ─── AUGMENTER ────────────────────────────────────────────────────────────────
//
// Data augmentation is what makes contrastive learning work.
// The encoder must learn invariances to the transformations we apply — those
// invariances are precisely the useful structure we want it to capture.
//
// For image data: crops, color jitter, blur.
// For tabular/feature data: Gaussian noise and random feature dropout.
//
// The key constraint: augmentations must be LABEL-PRESERVING.
// "A dog with noise added" is still a dog. The encoder should agree.
//
export class Augmenter {
  // Add zero-mean Gaussian noise with standard deviation `sigma`.
  //
  // Uses the Box-Muller transform to produce normally distributed noise from
  // two uniform random variables:
  //   z = √(-2·ln(u₁)) · cos(2π·u₂)   where u₁, u₂ ~ Uniform(0, 1)
  //
  // This keeps us dependency-free while yielding proper Gaussian samples.
  static addNoise(x: number[], sigma = 0.05): number[] {
    return x.map(v => {
      // Box-Muller: generates one N(0,1) sample per input dimension.
      const u1 = Math.max(1e-10, Math.random()); // guard against log(0)
      const u2 = Math.random();
      const z  = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      return v + sigma * z;
    });
  }

  // Randomly zero out features with probability `rate`.
  //
  // Analogous to masking in BERT or random crops in vision contrastive learning.
  // The encoder must learn representations that are robust to missing features —
  // it cannot simply memorize individual dimensions.
  static dropoutFeatures(x: number[], rate = 0.1): number[] {
    return x.map(v => (Math.random() < rate ? 0 : v));
  }

  // Apply both noise and feature dropout in sequence.
  //
  // Combining augmentations is standard in SimCLR — stronger augmentations
  // force the encoder to learn more robust, abstract representations.
  static augment(x: number[], noiseStd = 0.05, dropRate = 0.1): number[] {
    return Augmenter.dropoutFeatures(Augmenter.addNoise(x, noiseStd), dropRate);
  }

  // Generate a positive pair: [original, augmented_copy].
  //
  // These two views are used as the (i, j) positive pair in NT-Xent.
  // Everything else in the batch acts as a negative.
  static makePair(x: number[]): [number[], number[]] {
    return [x, Augmenter.augment(x)];
  }
}

// ─── CONTRASTIVE LEARNING (SimCLR) ───────────────────────────────────────────
//
// Architecture:
//
//   x ──→ [ Encoder f(·) ] ──→ h ──→ [ Projection Head g(·) ] ──→ z
//                                ↑ used for downstream tasks
//                                              ↑ used for NT-Xent loss only
//
// Why separate encoder h from projection z?
// ───────────────────────────────────────────
// The NT-Xent loss optimizes the projection space z to be contrastive.
// This optimization causes information that is NOT useful for contrastiveness
// (e.g. fine-grained texture in images) to be discarded.
//
// If we computed the loss directly on h, the encoder would lose that information
// too — hurting downstream tasks like classification.
//
// The projection head "absorbs" the destructive effect of the loss, shielding
// the encoder's richer representation h. At inference time, we discard g(·)
// and use h directly. This separation is one of SimCLR's key findings.
//
// Representation collapse — the trivial failure mode:
// ────────────────────────────────────────────────────
// A naive encoder could minimize any pairwise loss by mapping all inputs to
// the same constant vector. Then all similarities equal 1 and all losses → 0.
//
// NT-Xent prevents this via the denominator: it sums over ALL 2(N-1) negatives
// in the batch. If z_i = z_j for all i,j (collapse), the numerator exp(sim/τ)
// equals the denominator — giving loss = log(2N-1) ≠ 0, not 0.
// The loss is only minimized when positives are close AND negatives are far.
//
// NT-Xent Loss:
// ─────────────
//   For a batch of N pairs → 2N samples { z_1, z_1', z_2, z_2', ..., z_N, z_N' }
//
//   loss(i, j) = -log [ exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) ]
//
//   sim(u, v) = uᵀv / (||u|| · ||v||)   (cosine similarity, range [-1, 1])
//   τ          = temperature (default 0.5)
//
//   Total loss = mean over all 2N positive pairs (each sample is an anchor once).
//
// Temperature τ:
//   Low  τ → sharp distribution, hard negatives dominate, risk of instability.
//   High τ → soft distribution, treats all negatives equally, weaker signal.
//   τ = 0.5 is the SimCLR default and works well in practice.
//
export class ContrastiveLearning {
  encoder:        NetworkN;
  projectionHead: NetworkN;
  temperature:    number;

  // encoderHidden: hidden layer sizes for the encoder (not counting input/output).
  //   e.g. inputSize=64, encoderHidden=[256, 128] → NetworkN([64, 256, 128])
  //   The encoder output dimension is encoderHidden[last].
  //
  // projectionDim: dimension of the projection head output (the z space).
  //   e.g. 64. Typically smaller than the encoder's output.
  //
  // The encoder uses ReLU activations throughout — empirically stronger than
  // sigmoid for representation learning because it doesn't saturate.
  constructor(
    inputSize:      number,
    encoderHidden:  number[],
    projectionDim:  number,
    options: {
      temperature?:      number;
      encoderOptions?:   NetworkNOptions;
    } = {},
  ) {
    if (encoderHidden.length === 0) {
      throw new Error('encoderHidden must have at least one element.');
    }

    this.temperature = options.temperature ?? 0.5;

    // Build encoder: inputSize → hidden[0] → ... → hidden[last]
    const encoderStructure = [inputSize, ...encoderHidden];
    const encoderActivations = encoderHidden.map(() => relu);
    this.encoder = new NetworkN(encoderStructure, {
      activations: encoderActivations,
      ...options.encoderOptions,
    });

    // Build projection head: encoderOut → projectionDim
    // The output layer is LINEAR (no final ReLU): a final ReLU can map every projection
    // to the zero vector (dead head), where cosine similarity is undefined (guarded to 0)
    // and the gradient is exactly 0, so training never recovers. Breaking change (v0.4):
    // project() may now return negative values (before it was always >= 0).
    // One hidden layer at half the encoder output size gives a good balance
    // between capacity and regularization.
    const encoderOut    = encoderHidden[encoderHidden.length - 1];
    const projHidden    = Math.max(projectionDim, Math.floor(encoderOut / 2));
    this.projectionHead = new NetworkN(
      [encoderOut, projHidden, projectionDim],
      { activations: [relu, linear] },
    );
  }

  // ── Inference (downstream tasks use this, not project()) ─────────────────
  //
  // Returns h — the encoder representation before the projection head.
  // This is the vector to use for classification, clustering, retrieval, etc.
  //
  // The projection head is only active during training.
  encode(x: number[]): number[] {
    return this.encoder.predict(x);
  }

  // ── Training path: encode then project ───────────────────────────────────
  //
  // Returns z — the projected representation used to compute NT-Xent.
  // Do NOT use this for downstream tasks (see encode() above).
  project(x: number[]): number[] {
    const h = this.encoder.predict(x);
    return this.projectionHead.predict(h);
  }

  // ── Cosine similarity ─────────────────────────────────────────────────────
  //
  // sim(u, v) = uᵀv / (||u|| · ||v||)
  //
  // Range: [-1, 1]. We use cosine rather than Euclidean distance because it is
  // scale-invariant — only the direction of the projection matters, not its
  // magnitude. This prevents the trivial solution of making ||z|| → ∞.
  static cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0, normA = 0, normB = 0;
    for (let d = 0; d < a.length; d++) {
      dot   += a[d] * b[d];
      normA += a[d] * a[d];
      normB += b[d] * b[d];
    }
    // Soft epsilon in the denominator instead of a jump to 0 below a norm threshold:
    // the loss stays CONTINUOUS as ||z|| -> 0 (a zero-norm jump made finite differences
    // return garbage near dead projections). For ||a||·||b|| >> 1e-8 the effect is < 1e-8
    // relative; at exactly zero norm the similarity is 0. Observable change (v0.4): the
    // loss differs from the old one by O(1e-8 / (||a||·||b||)).
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
  }

  // ── NT-Xent loss (no weight update) ──────────────────────────────────────
  //
  // Forward-only pass. Used for validation / monitoring during training.
  computeLoss(pairs: [number[], number[]][]): number {
    const { projections, N } = this._forwardProjections(pairs);
    return this._ntXentLoss(projections, N);
  }

  // ── Training step ─────────────────────────────────────────────────────────
  //
  // Given a batch of positive pairs, compute NT-Xent loss and update weights
  // via finite-difference gradient approximation.
  //
  // Full analytical backprop through NT-Xent is complex to implement from
  // scratch without an autograd engine. Finite differences are slower but
  // correct and keep the implementation readable for educational purposes.
  // For production use, couple this with the Tape (autograd) module.
  //
  // Step-by-step:
  //   1. Forward all 2N inputs through encoder + projection head → { z_i }.
  //   2. Build the 2N×2N cosine similarity matrix (scaled by 1/τ).
  //   3. For each anchor i, identify its positive pair and all 2N-2 negatives.
  //   4. Apply softmax over the row; loss = -log(softmax at positive index).
  //   5. Average over all 2N anchors.
  //   6. Approximate ∂L/∂w per weight with finite differences and apply update.
  //
  // Returns: NT-Xent loss before the weight update.
  trainStep(pairs: [number[], number[]][], lr: number): number {
    const loss = this.computeLoss(pairs);

    // Finite-difference gradient approximation, SIMULTANEOUS step:
    // every gradient is evaluated at the ORIGINAL weights and only after all of them
    // are known are the weights updated (w += lr * (-grad)). Each probed value is saved
    // and restored exactly (no += eps / -= 2eps / += eps drift, no in-place update that
    // would make later gradients depend on already-updated earlier weights).
    //
    // gradient ≈ (L(w+ε) - L(w-ε)) / (2ε). O(P) forward passes, P = number of parameters.
    //
    // Known limitation (documented, not a defect of the step): cosine similarity has
    // curvature ~ 1/||z||. With very small projection norms the loss changes quickly
    // along the step, so a finite lr departs from the first-order gradient there.
    // eps = 1e-6 (was 1e-4). Measured on raw Xavier inits (20 seeds): with 1e-4 the
    // step departed from the oracle gradient (cosine 0.90..0.95) on seeds whose live
    // projections have norm 5e-3..6e-2, because the 1/||z|| curvature makes a central
    // difference with a big step inexact; with 1e-6 all of them give cosine >= 0.99999.
    // Rounding error is ~1e-16/1e-6 = 1e-10, far below the gradients.
    const eps = 1e-6;

    type Slot = { owner: { weights: number[] } | { bias: number }; index: number };
    const slots: Slot[] = [];
    const grads: number[] = [];
    for (const net of [this.encoder, this.projectionHead]) {
      for (const layer of net.layers) {
        for (const neuron of layer.neurons) {
          for (let j = 0; j < neuron.weights.length; j++) slots.push({ owner: neuron, index: j });
          slots.push({ owner: neuron, index: -1 }); // bias
        }
      }
    }

    const read = (s: Slot): number =>
      s.index >= 0 ? (s.owner as { weights: number[] }).weights[s.index] : (s.owner as { bias: number }).bias;
    const write = (s: Slot, v: number): void => {
      if (s.index >= 0) (s.owner as { weights: number[] }).weights[s.index] = v;
      else (s.owner as { bias: number }).bias = v;
    };

    for (const s of slots) {
      const orig = read(s);
      write(s, orig + eps);
      const lossPlus = this.computeLoss(pairs);
      write(s, orig - eps);
      const lossMinus = this.computeLoss(pairs);
      write(s, orig); // exact restore
      grads.push((lossPlus - lossMinus) / (2 * eps));
    }

    for (let k = 0; k < slots.length; k++) {
      write(slots[k], read(slots[k]) + lr * (-grads[k])); // gradient descent
    }

    return loss;
  }

  // ── Private: forward all pairs through the projection head ───────────────
  //
  // Returns a flat array of 2N projections.
  // Layout: [ z_0, z_0', z_1, z_1', ..., z_{N-1}, z_{N-1}' ]
  // Even indices 2i   → original view of pair i
  // Odd  indices 2i+1 → augmented view of pair i (the positive)
  private _forwardProjections(
    pairs: [number[], number[]][],
  ): { projections: number[][]; N: number } {
    const N = pairs.length;
    const projections: number[][] = [];
    for (const [x, xAug] of pairs) {
      projections.push(this.project(x));
      projections.push(this.project(xAug));
    }
    return { projections, N };
  }

  // ── Private: NT-Xent loss over a set of 2N projections ───────────────────
  //
  // pairs[2i]   and pairs[2i+1] are positives.
  // All other 2N-2 samples are negatives for each anchor.
  private _ntXentLoss(projections: number[][], N: number): number {
    const total = 2 * N; // total number of views
    const tau   = this.temperature;

    // Pre-compute the full similarity matrix (2N × 2N) scaled by 1/τ.
    // sim[i][j] = cosine(z_i, z_j) / τ
    // We mask the diagonal (self-similarity) when computing the softmax denominator.
    const sim: number[][] = Array.from({ length: total }, (_, i) =>
      Array.from({ length: total }, (_, j) =>
        ContrastiveLearning.cosineSimilarity(projections[i], projections[j]) / tau,
      ),
    );

    let totalLoss = 0;

    // For each anchor i, find its positive j and compute the NT-Xent term.
    for (let i = 0; i < total; i++) {
      // The positive partner of sample 2i is 2i+1 (and vice versa).
      const posIdx = i % 2 === 0 ? i + 1 : i - 1;

      // Numerator: exp(sim(z_i, z_j+) / τ)
      const numerator = Math.exp(sim[i][posIdx]);

      // Denominator: sum of exp(sim(z_i, z_k) / τ) for all k ≠ i
      // We exclude k=i (self) but include k=posIdx (positive pair is NOT excluded
      // from the denominator — this is the correct NT-Xent formulation, which
      // makes it a harder problem than excluding positives).
      let denominator = 0;
      for (let k = 0; k < total; k++) {
        if (k !== i) {
          denominator += Math.exp(sim[i][k]);
        }
      }

      // -log(softmax at positive position) — cross-entropy form of the loss.
      // When positives have high similarity and negatives low: numerator ≈ denominator
      // → log(1) = 0, perfect. When collapsed: all similarities equal →
      // numerator/denominator = 1/(2N-1) → loss = log(2N-1) > 0.
      totalLoss += -Math.log(numerator / (denominator + 1e-10));
    }

    // Average over all 2N anchor positions.
    return totalLoss / total;
  }
}
