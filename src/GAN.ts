// ─── GENERATIVE ADVERSARIAL NETWORK ──────────────────────────────────────────
//
// A GAN consists of two networks trained in opposition (Goodfellow et al., 2014):
//
//   Generator     G(z)  : latent noise z ∈ ℝᵏ  →  synthetic data x̂ ∈ ℝⁿ
//   Discriminator D(x)  : data x ∈ ℝⁿ           →  P(real) ∈ [0, 1]
//
// Objective — min-max game:
//
//   min_G  max_D  E[log D(x_real)] + E[log(1 - D(G(z)))]
//
//   • D tries to maximize: assign 1 to real, 0 to fake.
//   • G tries to minimize (i.e., maximize log D(G(z))): fool D into outputting 1.
//
// Nash Equilibrium:
//   Training converges (in theory) when G produces data indistinguishable from
//   real and D outputs 0.5 everywhere — it can no longer tell real from fake.
//   In practice, this is rarely a clean fixed point.
//
// Mode Collapse:
//   G may find a single output (or small set) that reliably fools D, ignoring
//   the true diversity of the training distribution. Symptoms: G generates
//   nearly identical samples regardless of z. Mitigations: mini-batch
//   discrimination, Wasserstein loss (WGAN), spectral normalisation.
//
// Training loop (one step):
//   1. Sample real batch x ~ P_data
//   2. Sample latent batch z ~ N(0, I)
//   3. Generate fake batch x̂ = G(z)
//   4. Update D: maximise log D(x) + log(1 - D(x̂))
//      → delta_real = (1 - D(x))·D(x)·(1-D(x))      (label = 1)
//      → delta_fake = (0 - D(x̂))·D(x̂)·(1-D(x̂))   (label = 0)
//   5. Update G: maximise log D(G(z))
//      → provide gradient that pushes D(G(z)) → 1
//

import { NetworkN, NetworkNOptions } from "./NetworkN";

// ─── GAN ─────────────────────────────────────────────────────────────────────

export class GAN {
  readonly generator: NetworkN;
  readonly discriminator: NetworkN;
  readonly latentDim: number;

  constructor(
    latentDim: number,
    generatorHidden: number[],
    outputDim: number,
    discriminatorHidden: number[],
    options?: {
      generatorOptions?: NetworkNOptions;
      discriminatorOptions?: NetworkNOptions;
    },
  ) {
    this.latentDim = latentDim;

    // Generator: latentDim → ...hidden... → outputDim
    const gStructure = [latentDim, ...generatorHidden, outputDim];
    this.generator = new NetworkN(gStructure, options?.generatorOptions ?? {});

    // Discriminator: outputDim → ...hidden... → 1 (probability of being real)
    const dStructure = [outputDim, ...discriminatorHidden, 1];
    this.discriminator = new NetworkN(dStructure, options?.discriminatorOptions ?? {});
  }

  // ── Public API ───────────────────────────────────────────────────────────

  // Generate a synthetic sample. If z is not provided, samples from N(0, 1).
  generate(z?: number[]): number[] {
    const latent = z ?? this.sampleLatent();
    return this.generator.predict(latent);
  }

  // Returns the discriminator's estimate that x is real, in [0, 1].
  discriminate(x: number[]): number {
    return this.discriminator.predict(x)[0];
  }

  // ── Training Step ────────────────────────────────────────────────────────
  //
  // Runs one discriminator update and one generator update over the provided
  // real batch. Returns per-step losses for monitoring.
  //
  // Discriminator loss (binary cross-entropy, minimised via SGD):
  //   L_D = -[ log D(x_real) + log(1 - D(G(z))) ]
  //
  // Generator loss:
  //   L_G = -log D(G(z))   (non-saturating variant — avoids vanishing gradients
  //                          in early training when D is confident)
  //
  trainStep(
    realBatch: number[][],
    lr: number,
  ): { dLoss: number; gLoss: number } {
    const eps = 1e-15;
    let dLossSum = 0;
    let gLossSum = 0;

    for (const xReal of realBatch) {
      // ── Discriminator update (real sample, label = 1) ──────────────────
      const dReal = Math.max(eps, Math.min(1 - eps, this.discriminate(xReal)));
      // delta = (target - prediction) for sigmoid output
      const dRealDelta = [1 - dReal];
      this.discriminator.trainWithDeltas(xReal, dRealDelta, lr);
      dLossSum += -Math.log(dReal);

      // ── Discriminator update (fake sample, label = 0) ──────────────────
      const z = this.sampleLatent();
      const xFake = this.generate(z);
      const dFake = Math.max(eps, Math.min(1 - eps, this.discriminate(xFake)));
      const dFakeDelta = [0 - dFake];
      this.discriminator.trainWithDeltas(xFake, dFakeDelta, lr);
      dLossSum += -Math.log(1 - dFake);

      // ── Generator update (wants D to output 1 for fake samples) ────────
      //
      // We cannot backprop through the discriminator directly, so we use a
      // proxy: train the generator to produce outputs that the (now-updated)
      // discriminator rates as real.
      //
      // Practical approach: run D on the fake sample, compute the error
      // signal as if the real label were 1, then use that signal to compute
      // generator output deltas via the chain rule approximation.
      //
      const z2 = this.sampleLatent();
      const xFake2 = this.generate(z2);
      const dScore = Math.max(eps, Math.min(1 - eps, this.discriminate(xFake2)));

      // Error from generator's perspective: it wants D(G(z)) = 1
      const gError = 1 - dScore;  // how far we are from fooling D

      // Approximate gradient w.r.t. generator output:
      // ∂L_G/∂x̂ ≈ -D'(x̂) where D' is the discriminator's input sensitivity.
      // We estimate this by the sign of the discriminator's "wish": push x̂
      // toward what D considers real, i.e. use dScore's complementary error.
      const gDelta = xFake2.map(() => gError / xFake2.length);
      this.generator.trainWithDeltas(z2, gDelta, lr);
      gLossSum += -Math.log(dScore);
    }

    const n = realBatch.length;
    return {
      dLoss: dLossSum / n,
      gLoss: gLossSum / n,
    };
  }

  // Samples a latent vector z ~ N(0, 1)^latentDim using Box-Muller transform.
  sampleLatent(): number[] {
    const z: number[] = [];
    for (let i = 0; i < this.latentDim; i += 2) {
      // Box-Muller: converts two uniform samples into two standard normals
      const u1 = Math.random();
      const u2 = Math.random();
      const r  = Math.sqrt(-2 * Math.log(u1 + 1e-15));
      const theta = 2 * Math.PI * u2;
      z.push(r * Math.cos(theta));
      if (i + 1 < this.latentDim) z.push(r * Math.sin(theta));
    }
    return z;
  }
}
