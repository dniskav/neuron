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

// ─── VAE ─────────────────────────────────────────────────────────────────────

export class VAE {
  readonly encoder: NetworkN;  // x → [μ₁…μₖ, logVar₁…logVarₖ]
  readonly decoder: NetworkN;  // z → x̂
  readonly latentDim: number;

  constructor(
    inputSize: number,
    encoderHidden: number[],
    latentDim: number,
    decoderHidden: number[],
    options?: NetworkNOptions,
  ) {
    this.latentDim = latentDim;

    // Encoder outputs 2·latentDim values: first half = μ, second half = logVar
    const encoderStructure = [inputSize, ...encoderHidden, latentDim * 2];
    this.encoder = new NetworkN(encoderStructure, options ?? {});

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
    return mu.map((m, i) => {
      const sigma = Math.exp(0.5 * logVar[i]);
      const eps   = this._sampleNormal();
      return m + sigma * eps;
    });
  }

  // ── Decode ───────────────────────────────────────────────────────────────
  decode(z: number[]): number[] {
    return this.decoder.predict(z);
  }

  // ── Forward Pass ─────────────────────────────────────────────────────────
  // Encodes, samples z, and decodes.
  forward(x: number[]): {
    reconstruction: number[];
    mu: number[];
    logVar: number[];
    z: number[];
  } {
    const { mu, logVar } = this.encode(x);
    const z = this.reparametrize(mu, logVar);
    const reconstruction = this.decode(z);
    return { reconstruction, mu, logVar, z };
  }

  // ── Training Step ────────────────────────────────────────────────────────
  //
  // Performs one forward pass, computes the ELBO loss, and updates both
  // encoder and decoder weights via their built-in SGD.
  //
  // Reconstruction loss:   L_recon = MSE(x, x̂)
  // KL divergence:         L_kl    = -½ Σ(1 + logVarᵢ - μᵢ² - exp(logVarᵢ))
  // Total:                 L       = L_recon + L_kl
  //
  train(x: number[], lr: number): {
    totalLoss: number;
    reconLoss: number;
    klLoss: number;
  } {
    const { reconstruction, mu, logVar, z } = this.forward(x);

    // ── Reconstruction loss (MSE) ────────────────────────────────────────
    const reconLoss = x.reduce((s, xi, i) => s + (xi - reconstruction[i]) ** 2, 0) / x.length;

    // ── KL divergence ───────────────────────────────────────────────────
    // KL = -½ Σ [ 1 + logVar - μ² - exp(logVar) ]
    const klLoss = mu.reduce((s, m, i) => {
      return s - 0.5 * (1 + logVar[i] - m * m - Math.exp(logVar[i]));
    }, 0);

    const totalLoss = reconLoss + klLoss;

    // ── Update decoder (reconstruction gradient) ─────────────────────────
    // delta for decoder output = (x - x̂) / n  (MSE gradient)
    const decoderDeltas = reconstruction.map((r, i) => (x[i] - r) / x.length);
    this.decoder.trainWithDeltas(z, decoderDeltas, lr);

    // ── Update encoder ───────────────────────────────────────────────────
    // Gradient of ELBO w.r.t. encoder output [μ, logVar]:
    //   ∂L_kl/∂μᵢ     = μᵢ
    //   ∂L_kl/∂logVarᵢ = 0.5·(exp(logVarᵢ) - 1)
    // We negate because the encoder minimises the loss.
    const encoderDeltas: number[] = [
      ...mu.map((m) => -m),
      ...logVar.map((lv) => -0.5 * (Math.exp(lv) - 1)),
    ];
    this.encoder.trainWithDeltas(x, encoderDeltas, lr);

    return { totalLoss, reconLoss, klLoss };
  }

  // ── Generate ─────────────────────────────────────────────────────────────
  // Samples z ~ N(0, I) and decodes it (pure generation, no input required).
  generate(z?: number[]): number[] {
    const latent = z ?? Array.from({ length: this.latentDim }, () => this._sampleNormal());
    return this.decode(latent);
  }

  // ── Private ──────────────────────────────────────────────────────────────

  // Box-Muller transform: samples one value from N(0, 1).
  private _sampleNormal(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-15)) * Math.cos(2 * Math.PI * u2);
  }
}
