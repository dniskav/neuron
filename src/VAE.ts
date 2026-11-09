// ─── VARIATIONAL AUTOENCODER ──────────────────────────────────────────────────
//
// A VAE (Kingma & Welling, 2013) learns a probabilistic latent space P(z|x).
// Unlike a plain autoencoder (which learns a deterministic embedding), the VAE
// encodes each input as a *distribution* N(μ, σ²) over the latent space,
// forcing the latent space to be smooth and generative.
//
// Architecture:
//   Encoder:   x ∈ ℝⁿ  →  [μ₁…μₖ, logVar₁…logVarₖ] ∈ ℝ²ᵏ
//   Sampling:  z = μ + σ·ε,   ε ~ N(0, I)   (reparametrisation trick)
//   Decoder:   z ∈ ℝᵏ  →  x̂ ∈ ℝⁿ
//
// Reparametrisation Trick:
//   Sampling z ~ N(μ, σ²) is not differentiable with respect to μ and σ.
//   The trick rewrites it as z = μ + σ·ε where ε ~ N(0,1) is a fixed noise
//   sample. Now μ and σ are parameters through which gradients can flow.
//
// ELBO Loss (Evidence Lower BOund) — maximised during training:
//   L = E[log P(x|z)]  −  KL[ Q(z|x) || P(z) ]
//
//   In practice (minimised):
//   L = ReconstructionLoss(x, x̂)  +  KL divergence
//
// Gradient flow (feature 004): the reconstruction gradient reaches the encoder through
// z = μ + σ·ε (dz/dμ = 1, dz/dlogVar = 0.5·σ·ε). The encoder output is linear, and an
// optional klWeight (β, default 1) scales the KL term: with MSE averaged over n and KL
// summed over the latent, β = 1 collapses the posterior on small [0,1] data (μ constant).
//
// Reconstruction Loss (MSE):
//   L_recon = (1/n) Σ (xᵢ - x̂ᵢ)²
//
// KL Divergence (closed form for N(μ,σ²) vs N(0,1)):
//   KL = -½ Σ [ 1 + logVar − μ² − exp(logVar) ]
//   where logVar = log(σ²) to keep the encoder output unconstrained.
//
// Why KL regularises the latent space:
//   KL = 0 only when μ = 0 and σ = 1 everywhere. The encoder is pushed to
//   produce distributions close to the standard normal, so the latent space
//   is dense and interpolation between samples is meaningful.
//

import { NetworkN, NetworkNOptions } from "./NetworkN";
import { linear, sigmoid } from "./activations";
import { validateArray } from "./Validation";

// ─── VAE ─────────────────────────────────────────────────────────────────────

export class VAE {
  readonly encoder: NetworkN;  // x → [μ₁…μₖ, logVar₁…logVarₖ]
  readonly decoder: NetworkN;  // z → x̂
  readonly latentDim: number;
  // β of the β-VAE: L = MSE + klWeight·KL. Default 1 is the plain ELBO-style loss.
  // With MSE averaged over n and KL summed over the latent dimensions, a small
  // klWeight is needed for the latent code to carry information (see train()).
  klWeight: number;

  constructor(
    inputSize: number,
    encoderHidden: number[],
    latentDim: number,
    decoderHidden: number[],
    options?: NetworkNOptions,
    klWeight = 1,
  ) {
    if (!(klWeight >= 0) || !Number.isFinite(klWeight)) {
      throw new Error(`VAE: klWeight must be a finite number >= 0 (got ${klWeight})`);
    }
    this.latentDim = latentDim;
    this.klWeight = klWeight;

    // Encoder outputs 2·latentDim values: first half = μ, second half = logVar.
    // The output layer is ALWAYS linear: μ and logVar must be unbounded (with a
    // sigmoid, μ ∈ (0,1) and σ ≥ 1, and the VAE cannot beat predicting the mean).
    // If `options.activations` is given, it must have one entry per decoder layer
    // (decoderHidden.length + 1): the decoder receives it as-is. Its first
    // `encoderHidden.length` entries apply to the encoder hidden layers.
    const encoderStructure = [inputSize, ...encoderHidden, latentDim * 2];
    const encoderOptions: NetworkNOptions = { ...(options ?? {}) };
    if (options?.activations) {
      const nDecoderLayers = decoderHidden.length + 1;
      const nActivations = options.activations.length;
      if (nActivations !== nDecoderLayers || encoderHidden.length > nActivations) {
        throw new Error(
          `VAE: options.activations needs ${nDecoderLayers} entries (one per decoder layer, ` +
          `passed to the decoder as-is) and must cover the ${encoderHidden.length} encoder ` +
          `hidden layers; got ${nActivations}`,
        );
      }
      encoderOptions.activations = [...options.activations.slice(0, encoderHidden.length), linear];
    } else {
      encoderOptions.activations = [...encoderHidden.map(() => sigmoid), linear];
    }
    this.encoder = new NetworkN(encoderStructure, encoderOptions);

    // Decoder reconstructs the original input from z
    const decoderStructure = [latentDim, ...decoderHidden, inputSize];
    this.decoder = new NetworkN(decoderStructure, options ?? {});
  }

  // ── Encode ───────────────────────────────────────────────────────────────
  // Splits the encoder output into μ and logVar vectors.
  encode(x: number[]): { mu: number[]; logVar: number[] } {
    const out = this.encoder.predict(x);
    const mu     = out.slice(0, this.latentDim);
    const logVar = out.slice(this.latentDim);
    return { mu, logVar };
  }

  // ── Reparametrisation Trick ──────────────────────────────────────────────
  // z = μ + σ·ε,  ε ~ N(0,1)
  // σ = exp(0.5 · logVar)  (ensures σ > 0 without constraining the network)
  reparametrize(mu: number[], logVar: number[]): number[] {
    return this._reparametrizeWithEps(mu, logVar).z;
  }

  // ── Decode ───────────────────────────────────────────────────────────────
  decode(z: number[]): number[] {
    return this.decoder.predict(z);
  }

  // ── Forward Pass ─────────────────────────────────────────────────────────
  // Encodes, samples z, and decodes. `eps` is the noise used (needed to
  // backpropagate through z = μ + σ·ε); it is drawn in the same order as before.
  forward(x: number[]): {
    reconstruction: number[];
    mu: number[];
    logVar: number[];
    z: number[];
    eps: number[];
  } {
    const { mu, logVar } = this.encode(x);
    const { z, eps } = this._reparametrizeWithEps(mu, logVar);
    const reconstruction = this.decode(z);
    return { reconstruction, mu, logVar, z, eps };
  }

  // ── Training Step ────────────────────────────────────────────────────────
  //
  // Loss (minimised):  L = MSE(x, x̂) + klWeight·KL      (ε fixed during the step; klWeight = 1 by default)
  //   MSE = (1/n) Σ (xᵢ − x̂ᵢ)²
  //   KL  = −½ Σ (1 + logVarᵢ − μᵢ² − exp(logVarᵢ))
  //
  // Convention: the optimizers do `w + lr·g`, so every delta below is −dL/d(pre-activation).
  //
  //   decoder output : −dMSE/dx̂ · dfn = 2(x − x̂)/n · dfn(x̂)
  //   gz = −dMSE/dz  : decoder.inputGradient(z, those deltas), computed BEFORE
  //                    the decoder is updated (it needs the weights of this step)
  //   z = μ + σε     : dz/dμ = 1,  dz/dlogVar = 0.5·σ·ε   (σ = exp(0.5 logVar))
  //   encoder μ      : (gz − β·μ) · dfn(μ_out)
  //   encoder logVar : (gz·0.5σε − β·0.5(exp(logVar) − 1)) · dfn(lv_out)
  //
  train(x: number[], lr: number): {
    totalLoss: number;
    reconLoss: number;
    klLoss: number;
  } {
    // Validated before any Math.random draw or weight change: input width and finiteness, and
    // lr finite and >= 0 (lr = 0 is a valid no-op).
    validateArray(x, this.encoder.structure[0], 'VAE.train: x');
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`VAE.train: lr must be a finite number >= 0, got ${lr}`);
    }

    // Training forward (dropout ACTIVE, one mask per sub-network, reused below; Q9). Same
    // Math.random order as forward(): encoder masks, eps, decoder masks. With dropoutRate = 0
    // it is exactly forward().
    const encOut = this.encoder.predict(x, true);
    const mu     = encOut.slice(0, this.latentDim);
    const logVar = encOut.slice(this.latentDim);
    const { z, eps } = this._reparametrizeWithEps(mu, logVar);
    const reconstruction = this.decoder.predict(z, true);

    const reconLoss = x.reduce((s, xi, i) => s + (xi - reconstruction[i]) ** 2, 0) / x.length;
    const klLoss = mu.reduce((s, m, i) => {
      return s - 0.5 * (1 + logVar[i] - m * m - Math.exp(logVar[i]));
    }, 0);
    const totalLoss = reconLoss + this.klWeight * klLoss;

    // ── Decoder deltas and gradient at its input (before updating it) ─────
    const decLayer = this.decoder.layers[this.decoder.layers.length - 1];
    const decoderDeltas = reconstruction.map(
      (r, i) => (2 * (x[i] - r) / x.length) * decLayer.neurons[i].activation.dfn(r),
    );
    const gz = this.decoder.inputGradient(z, decoderDeltas, true);
    this.decoder.trainWithDeltas(z, decoderDeltas, lr, true);

    // ── Encoder deltas ────────────────────────────────────────────────────
    const encLayer = this.encoder.layers[this.encoder.layers.length - 1];
    const k = this.latentDim;
    const beta = this.klWeight;
    const encoderDeltas: number[] = [];
    for (let i = 0; i < k; i++) {
      encoderDeltas.push((gz[i] - beta * mu[i]) * encLayer.neurons[i].activation.dfn(mu[i]));
    }
    for (let i = 0; i < k; i++) {
      const sigma = Math.exp(0.5 * logVar[i]);
      const g = gz[i] * 0.5 * sigma * eps[i] - beta * 0.5 * (Math.exp(logVar[i]) - 1);
      encoderDeltas.push(g * encLayer.neurons[k + i].activation.dfn(logVar[i]));
    }
    this.encoder.trainWithDeltas(x, encoderDeltas, lr, true);

    return { totalLoss, reconLoss, klLoss };
  }

  // ── Generate ─────────────────────────────────────────────────────────────
  // Samples z ~ N(0, I) and decodes it (pure generation, no input required).
  generate(z?: number[]): number[] {
    const latent = z ?? Array.from({ length: this.latentDim }, () => this._sampleNormal());
    return this.decode(latent);
  }

  // ── Private ──────────────────────────────────────────────────────────────

  private _reparametrizeWithEps(mu: number[], logVar: number[]): { z: number[]; eps: number[] } {
    const eps = mu.map(() => 0);
    const z = mu.map((m, i) => {
      const sigma = Math.exp(0.5 * logVar[i]);
      eps[i] = this._sampleNormal();
      return m + sigma * eps[i];
    });
    return { z, eps };
  }

  // Box-Muller transform: samples one value from N(0, 1).
  private _sampleNormal(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
  }
}
