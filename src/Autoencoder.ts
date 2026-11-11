// ─── AUTOENCODER ─────────────────────────────────────────────────────────────
//
// An unsupervised model that learns a compressed representation of its input.
// The network is trained to reproduce its own input at the output.
//
// ─── ARCHITECTURE ────────────────────────────────────────────────────────────
//
//   Input ──► [ Encoder ] ──► Latent (bottleneck) ──► [ Decoder ] ──► Output
//    xᵢ ∈ ℝ^d         z ∈ ℝ^k (k << d)                  x̂ ∈ ℝ^d
//
//   The encoder compresses the input to a lower-dimensional latent space.
//   The decoder reconstructs the original input from the latent code.
//   By forcing information through a narrow bottleneck, the network learns
//   to discard noise and retain only the most salient structure in the data.
//
// ─── ENCODER ─────────────────────────────────────────────────────────────────
//   Structure: [inputSize, ...encoderHidden, latentSize]
//   z = encoderNetwork.predict(x)
//
// ─── DECODER ─────────────────────────────────────────────────────────────────
//   Structure: [latentSize, ...decoderHidden, inputSize]
//   x̂ = decoderNetwork.predict(z)
//
// ─── LOSS ────────────────────────────────────────────────────────────────────
//   L = MSE(x, x̂) = (1/d) · Σᵢ (xᵢ − x̂ᵢ)²
//
//   Training minimizes L jointly over encoder and decoder weights.
//   The gradient flows: ∂L/∂x̂ → decoder (backprop) → ∂L/∂z → encoder (backprop).
//
// ─── APPLICATIONS ────────────────────────────────────────────────────────────
//   - Dimensionality reduction (like PCA but non-linear)
//   - Denoising: train on noisy inputs, targets = clean inputs
//   - Anomaly detection: high reconstruction error signals an out-of-distribution input
//   - Feature extraction: use the latent code z as a learned feature vector
//
// ─────────────────────────────────────────────────────────────────────────────

import { NetworkN }        from "./NetworkN";
import { NetworkNOptions } from "./NetworkN";
import { mse }             from "./losses";
import { validateArray }   from "./Validation";

export class Autoencoder {
  /** Encoder sub-network: maps input → latent representation. */
  readonly encoder: NetworkN;
  /** Decoder sub-network: maps latent representation → reconstructed input. */
  readonly decoder: NetworkN;

  private readonly _inputSize: number;
  private readonly _latentSize: number;

  constructor(
    inputSize: number,
    encoderHidden: number[],
    latentSize: number,
    decoderHidden: number[],
    options: NetworkNOptions = {}
  ) {
    if (inputSize < 1) {
      throw new Error(`Autoencoder: inputSize must be ≥ 1, got ${inputSize}`);
    }
    if (latentSize < 1) {
      throw new Error(`Autoencoder: latentSize must be ≥ 1, got ${latentSize}`);
    }
    if (latentSize >= inputSize) {
      // No error and no warning: an over-complete autoencoder is valid (e.g. for
      // denoising). Nothing is thrown or logged. With latentSize >= inputSize the
      // latent code need not compress, and the network can learn the identity map.
      // throw new Error(...)  // intentionally not thrown
    }

    this._inputSize  = inputSize;
    this._latentSize = latentSize;

    // ── Build encoder ──────────────────────────────────────────────────────
    // Structure: [inputSize, ...encoderHidden, latentSize]
    const encoderStructure = [inputSize, ...encoderHidden, latentSize];

    // If the caller supplied activations, they apply to the encoder only.
    // We build separate options objects to keep encoder/decoder independent.
    const encoderOptions: NetworkNOptions = { ...options };
    if (options.activations) {
      const nEncoderLayers = encoderStructure.length - 1;
      if (options.activations.length >= nEncoderLayers) {
        encoderOptions.activations = options.activations.slice(0, nEncoderLayers);
      } else {
        // Not enough specified — drop the override and let NetworkN use defaults
        encoderOptions.activations = undefined;
      }
    }
    this.encoder = new NetworkN(encoderStructure, encoderOptions);

    // ── Build decoder ──────────────────────────────────────────────────────
    // Structure: [latentSize, ...decoderHidden, inputSize]
    const decoderStructure = [latentSize, ...decoderHidden, inputSize];

    const decoderOptions: NetworkNOptions = { ...options };
    if (options.activations) {
      const nEncoderLayers = encoderStructure.length - 1;
      const nDecoderLayers = decoderStructure.length - 1;
      const remaining = options.activations.slice(nEncoderLayers);
      if (remaining.length >= nDecoderLayers) {
        decoderOptions.activations = remaining.slice(0, nDecoderLayers);
      } else {
        decoderOptions.activations = undefined;
      }
    }
    this.decoder = new NetworkN(decoderStructure, decoderOptions);
  }

  // ── encode ─────────────────────────────────────────────────────────────────
  // Maps an input vector to its latent representation.
  //   z = encoder(x) ∈ ℝ^latentSize
  encode(x: number[]): number[] {
    if (x.length !== this._inputSize) {
      throw new Error(
        `Autoencoder.encode: expected input of length ${this._inputSize}, got ${x.length}`
      );
    }
    return this.encoder.predict(x);
  }

  // ── decode ─────────────────────────────────────────────────────────────────
  // Reconstructs an input from its latent code.
  //   x̂ = decoder(z) ∈ ℝ^inputSize
  decode(z: number[]): number[] {
    if (z.length !== this._latentSize) {
      throw new Error(
        `Autoencoder.decode: expected latent vector of length ${this._latentSize}, got ${z.length}`
      );
    }
    return this.decoder.predict(z);
  }

  // ── reconstruct ───────────────────────────────────────────────────────────
  // Convenience: encode then decode in a single call.
  //   x̂ = decode(encode(x))
  reconstruct(x: number[]): number[] {
    return this.decode(this.encode(x));
  }

  // ── train ──────────────────────────────────────────────────────────────────
  // Trains on a single example using backpropagation through both sub-networks.
  //
  // Gradient flow:
  //   1. Forward:  z = encoder(x),  x̂ = decoder(z)
  //   2. Compute MSE output deltas at x̂: δᵢ = (2/d)(xᵢ − x̂ᵢ) · act'(x̂ᵢ)
  //   3. ∂L/∂z = decoder.inputGradient(z, outputDeltas) (BEFORE updating weights)
  //   4. Update encoder weights via trainWithDeltas(x, dLdz, lr)
  //   5. Update decoder weights via trainWithDeltas(z, outputDeltas, lr)
  //
  // A residual skip on the LAST layer of the decoder (or encoder) throws (Q2).
  //
  // Returns the MSE reconstruction loss: (1/d) · Σᵢ (xᵢ − x̂ᵢ)².
  train(x: number[], lr: number): number {
    if (x.length !== this._inputSize) {
      throw new Error(
        `Autoencoder.train: expected input of length ${this._inputSize}, got ${x.length}`
      );
    }
    // lr must be finite and >= 0 (lr = 0 is a valid no-op). Checked before any weight changes.
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`Autoencoder.train: lr must be a finite number >= 0, got ${lr}`);
    }

    // ── Forward pass ──────────────────────────────────────────────────────
    const z    = this.encoder.predict(x, true);       // latent code
    const xHat = this.decoder.predict(z, true);       // reconstruction

    const loss = mse(xHat, x);

    // ── Output-layer deltas: δᵢ = (2/d) · (xᵢ − x̂ᵢ) · act'(x̂ᵢ) ───────────
    // L = (1/d)·Σ(xᵢ − x̂ᵢ)² (the MEAN MSE this method returns), so
    // −∂L/∂x̂ᵢ = (2/d)·(xᵢ − x̂ᵢ). The step is therefore exactly the gradient of the
    // returned loss (AUDIT N10). Before, the 2/d factor was missing and the step was
    // d/2 times larger (1.5x with d = 3), so the effective lr is now d/2 times smaller:
    // to keep the old dynamics, multiply lr by d/2.
    const d             = x.length;
    const decoderOutAct = this.decoder.layers[this.decoder.layers.length - 1].neurons[0].activation;
    const outputDeltas  = xHat.map((xh, i) => (2 / d) * (x[i] - xh) * decoderOutAct.dfn(xh));

    // ── ∂L/∂z through the decoder, BEFORE any weight is updated ───────────
    // decoder.inputGradient(z, outputDeltas) = −dL/dz with the weights of this step. It
    // reuses NetworkN's own backward pass, so residual connections and dropout masks of
    // the decoder are honoured (the old hand-written layer walk ignored both: cosine
    // 0.9949 with residual). Same pattern as VAE. If the decoder's LAST layer has a
    // residual skip it throws (Q2, same as VAE/GAN) before touching any weight.
    // reuseMasks = true (Q9): with dropout, loss, gradient and update share the masks drawn by
    // the two predict(…, true) above (one mask per step); it draws no Math.random.
    const gz = this.decoder.inputGradient(z, outputDeltas, true);

    // gz is the gradient at the encoder's OUTPUT (post-activation z). trainWithDeltas
    // expects deltas at the pre-activation of the output layer, so apply the chain rule
    // through the encoder's output activation: δz[k] = gz[k]·act'(z[k]) (AUDIT N6).
    const encoderOutAct = this.encoder.layers[this.encoder.layers.length - 1].neurons[0].activation;
    const dLdz = gz.map((g, k) => g * encoderOutAct.dfn(z[k]));

    // ── Updates ───────────────────────────────────────────────────────────
    // Encoder first: it only needs dLdz (already computed), and if its last layer has a
    // residual skip trainWithDeltas throws before the decoder is modified, so a failing
    // call never leaves a half-updated model. The two sub-networks share no weights, so
    // the order does not change the result.
    this.encoder.trainWithDeltas(x, dLdz, lr, true);
    this.decoder.trainWithDeltas(z, outputDeltas, lr, true);

    return loss;
  }

  // ── trainBatch ────────────────────────────────────────────────────────────
  // Trains on a batch of examples and returns the mean reconstruction MSE.
  trainBatch(X: number[][], lr: number): number {
    if (X.length === 0) {
      throw new Error("Autoencoder.trainBatch: batch X must be non-empty");
    }
    // T060: validate the WHOLE batch (and lr) first. Without this, a bad row in the middle
    // left the rows before it already trained.
    if (!(Number.isFinite(lr) && lr >= 0)) {
      throw new Error(`Autoencoder.trainBatch: lr must be a finite number >= 0, got ${lr}`);
    }
    X.forEach((x, i) => validateArray(x, this._inputSize, `Autoencoder.trainBatch: X[${i}]`));
    let totalLoss = 0;
    for (const x of X) totalLoss += this.train(x, lr);
    return totalLoss / X.length;
  }
}
