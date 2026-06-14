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
      // Not a hard error — an over-complete autoencoder is valid for denoising —
      // but we warn so the user is aware.
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
  //   2. Compute MSE output deltas at x̂: δᵢ = (xᵢ − x̂ᵢ) · act'(x̂ᵢ)
  //   3. Walk backward through decoder layers to get ∂L/∂z (BEFORE updating weights)
  //   4. Update decoder weights via trainWithDeltas(z, outputDeltas, lr)
  //   5. Update encoder weights via trainWithDeltas(x, dLdz, lr)
  //
  // Returns the MSE reconstruction loss: (1/d) · Σᵢ (xᵢ − x̂ᵢ)².
  train(x: number[], lr: number): number {
    if (x.length !== this._inputSize) {
      throw new Error(
        `Autoencoder.train: expected input of length ${this._inputSize}, got ${x.length}`
      );
    }

    // ── Forward pass ──────────────────────────────────────────────────────
    const z    = this.encoder.predict(x, true);       // latent code
    const xHat = this.decoder.predict(z, true);       // reconstruction

    const loss = mse(xHat, x);

    // ── Output-layer deltas: δᵢ = (xᵢ − x̂ᵢ) · act'(x̂ᵢ) ─────────────────
    const decoderOutAct = this.decoder.layers[this.decoder.layers.length - 1].neurons[0].activation;
    const outputDeltas  = xHat.map((xh, i) => (x[i] - xh) * decoderOutAct.dfn(xh));

    // ── Propagate deltas backward through all decoder layers to reach ∂L/∂z ─
    // We walk through the decoder layer stack manually (using CURRENT weights,
    // before any update) to compute the gradient at the decoder's input (= z).
    //
    // For each layer l (from output to input):
    //   ∂L/∂aₗ[j] = Σk δ[k] · W[k][j]    error backpropagated to layer l input
    //   δₗ₋₁[j]   = ∂L/∂aₗ[j] · act'(aₗ₋₁[j])   (chain rule through activation)
    //
    // We need the post-activation values at each decoder layer to apply act'.
    // We recompute them with a fresh forward pass before mutating anything.
    const decoderLayers  = this.decoder.layers;
    const decoderActVals: number[][] = [z];
    let cur = [...z];
    for (const layer of decoderLayers) {
      cur = layer.predict(cur);
      decoderActVals.push(cur);
    }

    let deltas = outputDeltas;
    for (let l = decoderLayers.length - 1; l >= 0; l--) {
      const layer   = decoderLayers[l];
      const prevAct = decoderActVals[l];               // input activations to layer l
      const prevLayerActivation = l > 0
        ? decoderLayers[l - 1].neurons[0].activation
        : null;                                         // null = latent input (no act')

      // Compute gradient at layer l's INPUT
      const prevDeltas = prevAct.map((out, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0);
        return prevLayerActivation ? errProp * prevLayerActivation.dfn(out) : errProp;
      });

      deltas = prevDeltas;
    }

    // `deltas` now holds ∂L/∂z — the gradient at the encoder's output.
    const dLdz = deltas;

    // ── Update decoder weights (after we have extracted ∂L/∂z) ───────────
    this.decoder.trainWithDeltas(z, outputDeltas, lr);

    // ── Update encoder weights ────────────────────────────────────────────
    // dLdz acts as the output-layer delta for the encoder:
    //   encoder output neuron k receives gradient dLdz[k].
    this.encoder.trainWithDeltas(x, dLdz, lr);

    return loss;
  }

  // ── trainBatch ────────────────────────────────────────────────────────────
  // Trains on a batch of examples and returns the mean reconstruction MSE.
  trainBatch(X: number[][], lr: number): number {
    if (X.length === 0) {
      throw new Error("Autoencoder.trainBatch: batch X must be non-empty");
    }
    let totalLoss = 0;
    for (const x of X) totalLoss += this.train(x, lr);
    return totalLoss / X.length;
  }
}
