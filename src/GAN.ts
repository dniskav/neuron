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
//   5. Update G: minimise -log D(G(z2)) with a fresh z2 and the D just updated.
//      The gradient flows through D (frozen: D.inputGradient does not touch its
//      weights) and through G's output derivative, then trainWithDeltas.
//
// Contract: D must have a SIGMOID output (BCE deltas `1 - D` / `-D` assume it).
// The constructor checks it by value: the output activation of D is accepted if it is the
// `sigmoid` object of ./activations, or any Activation whose fn(x) = 1/(1+e^(-x)) and
// dfn(out) = out·(1-out) agree with those formulas on a fixed probe set (tolerance 1e-9).
// Otherwise it throws. The check only calls fn/dfn: it consumes no Math.random and does not
// touch the network. The generator step is exact w.r.t. D at
// its post-update weights; an oracle that freezes D at the pre-step weights
// differs by O(lr).
//

import { NetworkN, NetworkNOptions } from "./NetworkN";
import { Activation, sigmoid } from "./activations";
import { validateArray } from "./Validation";

// Probe set for the by-value sigmoid check (see the GAN constructor).
const SIGMOID_PROBE_X = [-6, -2, -0.5, 0, 0.5, 2, 6];
const SIGMOID_PROBE_OUT = [0.05, 0.25, 0.5, 0.75, 0.95];
const SIGMOID_PROBE_TOL = 1e-9;

// Returns null when `act` behaves as a sigmoid on the probe set; otherwise the reason it does not.
function sigmoidMismatch(act: Activation): string | null {
  const EVAL_FAILED = "fn or dfn threw or returned NaN on the probe set";
  const isNum = (v: unknown): v is number => typeof v === "number" && !Number.isNaN(v);
  try {
    for (const x of SIGMOID_PROBE_X) {
      const got = act.fn(x);
      if (!isNum(got)) return EVAL_FAILED;
      if (!(Math.abs(got - 1 / (1 + Math.exp(-x))) <= SIGMOID_PROBE_TOL)) return "fn differs from 1/(1+e^(-x))";
    }
    for (const out of SIGMOID_PROBE_OUT) {
      const got = act.dfn(out);
      if (!isNum(got)) return EVAL_FAILED;
      if (!(Math.abs(got - out * (1 - out)) <= SIGMOID_PROBE_TOL)) return "dfn differs from out·(1-out)";
    }
  } catch {
    return EVAL_FAILED;
  }
  return null;
}

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

    // The BCE deltas of trainStep assume D's output is sigmoid (see the header contract).
    // Identity is a shortcut; any other activation is accepted if it matches sigmoid by value.
    const dOutput = this.discriminator.layers[this.discriminator.layers.length - 1].neurons[0].activation;
    if (dOutput !== sigmoid) {
      const reason = sigmoidMismatch(dOutput);
      if (reason !== null) {
        throw new Error(
          'GAN: the discriminator output must use the sigmoid activation (the BCE deltas assume it; ' +
          'checked by value: fn(x) = 1/(1+e^(-x)) and dfn(out) = out·(1-out) on a fixed probe set); ' +
          `${reason}. End discriminatorOptions.activations with sigmoid or an equivalent activation`,
        );
      }
    }
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
    // Validated before any weight changes: a non-empty batch whose rows have G's output width
    // (D's input width) and finite values, and lr finite and >= 0 (lr = 0 is a valid no-op).
    if (!Array.isArray(realBatch) || realBatch.length === 0) {
      throw new Error('GAN.trainStep: realBatch must be a non-empty array');
    }
    const outputDim = this.generator.structure[this.generator.structure.length - 1];
    realBatch.forEach((row, i) => validateArray(row, outputDim, `GAN.trainStep: realBatch[${i}]`));
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`GAN.trainStep: lr must be a finite number >= 0, got ${lr}`);
    }

    const eps = 1e-15;
    let dLossSum = 0;
    let gLossSum = 0;

    for (const xReal of realBatch) {
      // ── Discriminator update (real sample, label = 1) ──────────────────
      // Every D forward of the step runs in training mode (dropout active) and its update
      // reuses that mask (reuseMasks = true; Q9): the score, the delta and the weight update
      // belong to the same network. With dropoutRate = 0 this is exactly discriminate().
      const dReal = Math.max(eps, Math.min(1 - eps, this.discriminator.predict(xReal, true)[0]));
      // delta = (target - prediction) for sigmoid output
      const dRealDelta = [1 - dReal];
      this.discriminator.trainWithDeltas(xReal, dRealDelta, lr, true);
      dLossSum += -Math.log(dReal);

      // ── Discriminator update (fake sample, label = 0) ──────────────────
      const z = this.sampleLatent();
      const xFake = this.generate(z);
      const dFake = Math.max(eps, Math.min(1 - eps, this.discriminator.predict(xFake, true)[0]));
      const dFakeDelta = [0 - dFake];
      this.discriminator.trainWithDeltas(xFake, dFakeDelta, lr, true);
      dLossSum += -Math.log(1 - dFake);

      // ── Generator update (wants D to output 1 for fake samples) ────────
      //
      // L_G = -log D(G(z2)). Chain rule, with D FROZEN at its current weights
      // (the ones it has just after its own update above; the update order is
      // D first, then G):
      //   -dL/dD            = 1 / D
      //   deltaD            = dfn_D(D) / D          (sigmoid output: 1 - D)
      //   dxHat             = D.inputGradient(xFake2, [deltaD])   = -dL/dx̂
      //   gDelta[i]         = dxHat[i] · dfn_G(x̂_i)   (delta of G's output layer)
      // inputGradient does not modify any weight of D, so only G learns here.
      // Math.random order is unchanged: z, then z2.
      const z2 = this.sampleLatent();
      // G also runs in training mode and D scores in training mode; D.inputGradient and
      // G.trainWithDeltas reuse those masks (Q9). Math.random order: z, z2 (plus the dropout
      // masks, which exist only with dropoutRate > 0).
      const xFake2 = this.generator.predict(z2, true);
      const dScore = Math.max(eps, Math.min(1 - eps, this.discriminator.predict(xFake2, true)[0]));
      const dOutAct = this.discriminator.layers[this.discriminator.layers.length - 1].neurons[0].activation;
      const dxHat = this.discriminator.inputGradient(xFake2, [dOutAct.dfn(dScore) / dScore], true);
      const gOutLayer = this.generator.layers[this.generator.layers.length - 1];
      const gDelta = xFake2.map((xh, i) => dxHat[i] * gOutLayer.neurons[i].activation.dfn(xh));
      this.generator.trainWithDeltas(z2, gDelta, lr, true);
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
