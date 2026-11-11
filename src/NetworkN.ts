import { Layer }                            from "./Layer";
import { Dropout }                          from "./Dropout";
import { Activation, sigmoid }              from "./activations";
import { OptimizerFactory, defaultOptimizer } from "./optimizers";
import { validateArray }                    from "./Validation";

export interface NetworkNOptions {
  // One activation per layer (hidden layers + output layer).
  // Defaults to sigmoid for all layers.
  activations?: Activation[];
  // Optimizer factory shared across all weights in the network.
  // Defaults to SGD.
  optimizer?: OptimizerFactory;
  // Residual (skip) connections. When true, adds input to output for layers
  // where the number of inputs equals the number of outputs.
  // Can also be a function that returns true/false per layer index.
  residual?: boolean | ((layerIndex: number) => boolean);
  // Dropout rate for hidden layers (0 = disabled). Applied during training only.
  dropoutRate?: number;
}

// ─── N-LAYER NETWORK ──────────────────────────────────────────────────────────
// Generalization of Network for arbitrary depth.
//
//   new NetworkN([3, 16, 8, 1])
//   → 3 inputs → layer 16 → layer 8 → 1 output
//
//   new NetworkN([3, 16, 8, 1], {
//     activations: [relu, relu, sigmoid],
//     optimizer: () => new Adam(),
//   })
//
export class NetworkN {
  layers: Layer[];
  private _dropouts: Dropout[];
  private _residual: boolean | ((layerIndex: number) => boolean);

  constructor(readonly structure: number[], options: NetworkNOptions = {}) {
    const nLayers    = structure.length - 1;
    const activations = options.activations ?? Array.from({ length: nLayers }, () => sigmoid);
    const optimizer   = options.optimizer   ?? defaultOptimizer;
    const dropoutRate = options.dropoutRate ?? 0;

    if (activations.length !== nLayers) {
      throw new Error(`Expected ${nLayers} activations, got ${activations.length}`);
    }

    if (dropoutRate < 0 || dropoutRate >= 1) {
      throw new Error(`Dropout rate must be in [0, 1), got ${dropoutRate}`);
    }

    this._residual = options.residual ?? false;

    this.layers = [];
    for (let i = 1; i < structure.length; i++) {
      this.layers.push(new Layer(structure[i], structure[i - 1], activations[i - 1], optimizer));
    }

    // Create dropout instances for hidden layers only (not output layer)
    this._dropouts = [];
    if (dropoutRate > 0) {
      for (let i = 0; i < nLayers - 1; i++) {
        this._dropouts.push(new Dropout(dropoutRate));
      }
    }

    // Validate all output neurons share the same activation
    const outputLayer = this.layers[this.layers.length - 1];
    const outputActivation = outputLayer.neurons[0].activation;
    for (let i = 1; i < outputLayer.neurons.length; i++) {
      if (outputLayer.neurons[i].activation !== outputActivation) {
        throw new Error('All output neurons must share the same activation function');
      }
    }
  }

  predict(inputs: number[], training = false): number[] {
    validateArray(inputs, this.structure[0], 'NetworkN.predict');

    let current = [...inputs];
    for (let i = 0; i < this.layers.length; i++) {
      const layerInput = [...current];
      const layerOutput = this.layers[i].predict(current);

      // Residual connection: add input to output when sizes match
      if (this._shouldResidual(i)) {
        if (this.structure[i] === this.structure[i + 1]) {
          current = layerOutput.map((v, j) => v + layerInput[j]);
        } else {
          current = [...layerOutput];
        }
      } else {
        current = [...layerOutput];
      }

      // Apply dropout to hidden layers only (not output layer)
      if (i < this._dropouts.length) {
        current = this._dropouts[i].forward(current, training);
      }
    }
    return current;
  }

  // Generalized backpropagation across L layers.
  // Returns the mean squared error for the example, L = (1/n) sum (t - p)^2, and the step is
  // the gradient of THAT loss: g = -dL/d(output) = (2/n)(t - p) (N10, Q8). With n = 1 the step
  // is twice the one of versions before 0.4.0 (which trained 0.5 sum (t - p)^2).
  // trainWithDeltas / inputGradient take deltas computed by the caller and do not rescale.
  train(inputs: number[], targets: number[], lr: number): number {
    validateArray(inputs, this.structure[0], 'NetworkN.train');
    validateArray(targets, this.structure[this.structure.length - 1], 'NetworkN.train');

    const { act, pre } = this._forwardAll(inputs, true);
    const pred = act[act.length - 1];
    // g_L = -dL/d(output) for L = (1/n) * sum (t - p)^2, i.e. (2/n)(t - p). The output
    // derivative is applied inside _backward, on the output of f (before any residual).
    const g = pred.map((p, i) => (2 / pred.length) * (targets[i] - p));
    this._backward(act, pre, g, null, lr);

    return pred.reduce((s, p, i) => s + (targets[i] - p) ** 2, 0) / pred.length;
  }

  // Backprop with externally provided output-layer deltas.
  // Useful for custom loss functions (e.g. physics-based gradients).
  // Deltas are already multiplied by the output activation derivative, so a
  // residual skip on the LAST layer (which needs dL/d(output)) cannot be
  // propagated: that case throws instead of training with a wrong gradient.
  // Lengths and values are validated before any Math.random draw (dropout) or weight write.
  //
  // reuseMasks (default false): with dropoutRate > 0 the internal forward normally draws a
  // FRESH dropout mask. With reuseMasks = true it re-applies the masks saved by the last
  // predict(x, true) and draws nothing, so the update belongs to the very network that
  // produced the loss (Q9). Throws if dropoutRate > 0 and no masks are saved. With
  // dropoutRate = 0 the flag changes nothing.
  trainWithDeltas(inputs: number[], outputDeltas: number[], lr: number, reuseMasks = false): void {
    validateArray(inputs, this.structure[0], 'NetworkN.trainWithDeltas');
    validateArray(outputDeltas, this.structure[this.structure.length - 1], 'NetworkN.trainWithDeltas');
    this._assertDeltasSupported('trainWithDeltas');
    if (reuseMasks) this._assertMasksSaved('trainWithDeltas');
    const { act, pre } = this._forwardAll(inputs, true, reuseMasks);
    this._backward(act, pre, null, outputDeltas, lr);
  }

  // Gradient with respect to the INPUT, without touching any weight.
  //
  //   inputGradient(inputs, outputDeltas) -> -dL/d(inputs)
  //
  // Same contract and sign convention as trainWithDeltas: `outputDeltas` are
  // -dL/d(pre-activation of the output layer), i.e. -dL/d(output) already
  // multiplied by the output activation derivative. The result is -dL/dx (the
  // library convention: optimizers do `w + lr * g`, so "ascent direction of
  // -L"). To MINIMIZE L w.r.t. x, step x += lr * result. Typical use (GAN):
  // pass [dfn(D)/D] for the discriminator and chain the result through the
  // generator's own dfn, then call trainWithDeltas on the generator.
  //
  // Throws if the last layer has a residual skip (it would need dL/d(output),
  // not deltas; Q2). With dropoutRate > 0 and reuseMasks = false (default) a
  // fresh mask is drawn (training mode, consuming Math.random exactly like
  // trainWithDeltas) and applied in the backward pass; reseed Math.random for
  // a fixed mask. With reuseMasks = true it re-applies the masks of the last
  // predict(x, true), draws nothing, and the result is the exact gradient of
  // THAT forward pass (Q9); it throws if dropoutRate > 0 and no masks are saved.
  inputGradient(inputs: number[], outputDeltas: number[], reuseMasks = false): number[] {
    this._assertDeltasSupported('inputGradient');
    validateArray(inputs, this.structure[0], 'NetworkN.inputGradient');
    validateArray(outputDeltas, this.structure[this.structure.length - 1], 'NetworkN.inputGradient');
    if (reuseMasks) this._assertMasksSaved('inputGradient');
    const { act, pre } = this._forwardAll(inputs, true, reuseMasks);
    return this._backward(act, pre, null, outputDeltas, null);
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: layer 0 (all neurons), layer 1, ..., layer N.
  // Also clears the saved dropout masks (see setWeights). After getWeights(), a
  // trainWithDeltas or inputGradient call with reuseMasks = true throws "needs the
  // dropout masks"; call predict(x, true) again before using reuseMasks.
  getWeights(): number[] {
    // Reset dropout masks to avoid stale state
    for (const d of this._dropouts) d.resetMask();
    const w: number[] = [];
    for (const layer of this.layers) {
      for (const n of layer.neurons) {
        // Loops, not a push spread: a spread of >~125 000 arguments overflows the stack.
        for (const v of n.weights) w.push(v);
        w.push(n.bias);
      }
    }
    return w;
  }

  // Validates the total length BEFORE resetting dropout masks or writing, so a
  // rejected call leaves the network exactly as it was (T010, I2).
  // A successful call also clears the saved dropout masks, as getWeights() does: a
  // following trainWithDeltas or inputGradient with reuseMasks = true throws "needs
  // the dropout masks" until predict(x, true) is called again.
  setWeights(weights: number[]): void {
    validateArray(weights, this._weightCount(), 'NetworkN.setWeights');
    // Reset dropout masks to avoid stale state
    for (const d of this._dropouts) d.resetMask();
    let idx = 0;
    for (const layer of this.layers) {
      for (const n of layer.neurons) {
        for (let j = 0; j < n.weights.length; j++) n.weights[j] = weights[idx++];
        n.bias = weights[idx++];
      }
    }
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  // Flat weight count: each neuron of layer i has structure[i] weights and one bias,
  // and layer i has structure[i + 1] neurons.
  private _weightCount(): number {
    let total = 0;
    for (let i = 0; i < this.structure.length - 1; i++) {
      total += (this.structure[i] + 1) * this.structure[i + 1];
    }
    return total;
  }

  private _shouldResidual(layerIndex: number): boolean {
    if (typeof this._residual === 'function') return this._residual(layerIndex);
    return this._residual;
  }

  private _lastHasResidual(): boolean {
    const last = this.layers.length - 1;
    return this._shouldResidual(last) && this.structure[last] === this.structure[last + 1];
  }

  // reuseMasks needs the masks of a previous predict(x, true) in EVERY dropout layer. Checked
  // before the forward pass so a failing call draws nothing and writes nothing.
  private _assertMasksSaved(method: string): void {
    if (this._dropouts.some(d => !d.hasMask())) {
      throw new Error(
        `NetworkN.${method}: reuseMasks = true needs the dropout masks of a previous ` +
        `predict(x, true), but none are saved (an inference predict or setWeights/getWeights ` +
        `clears them). Call predict(x, true) first, or use reuseMasks = false.`
      );
    }
  }

  private _assertDeltasSupported(method: string): void {
    if (this._lastHasResidual()) {
      throw new Error(
        `NetworkN.${method}: residual on the last layer needs dL/d(output), but only ` +
        `deltas (already multiplied by the activation derivative) are given. ` +
        `Use train(), or disable residual on the last layer.`
      );
    }
  }

  // Forward pass storing activations at every layer boundary.
  //   act[l]  = input of layer l (after residual and dropout of layer l-1)
  //   pre[l]  = output of f in layer l, BEFORE residual and dropout
  // The activation derivative must be evaluated on pre[l], not on act[l+1].
  // With reuseMasks the dropout layers re-apply their saved masks instead of drawing new ones.
  private _forwardAll(
    inputs: number[], training: boolean, reuseMasks = false,
  ): { act: number[][]; pre: number[][] } {
    const act: number[][] = [inputs];
    const pre: number[][] = [];
    for (let i = 0; i < this.layers.length; i++) {
      const layerInput  = act[act.length - 1];
      const layerOutput = this.layers[i].predict(layerInput);
      pre.push([...layerOutput]);
      let current: number[];
      if (this._shouldResidual(i) && this.structure[i] === this.structure[i + 1]) {
        current = layerOutput.map((v, j) => v + layerInput[j]);
      } else {
        current = layerOutput;
      }
      if (i < this._dropouts.length) {
        current = reuseMasks
          ? this._dropouts[i].reuseMask(current)
          : this._dropouts[i].forward(current, training);
      }
      act.push(current);
    }
    return { act, pre };
  }

  // Backward pass carrying g_l = -dL/d(output of layer l) (after dropout/residual).
  //   dropout:   g' = mask * g                      (gradient of the sum f + x)
  //   delta_l  = g' * dfn(pre[l])                   (unless given externally)
  //   g_{l-1}  = W^T delta_l  (+ g' if residual)    (the skip carries g, not delta)
  // Exactly one of `g` (dL/d(output) of the last layer) or `deltas` (last-layer
  // deltas already multiplied by dfn) is given. Weights are updated only when
  // lr != null. Weights are read before their layer is updated. Returns
  // -dL/d(inputs).
  private _backward(
    act: number[][], pre: number[][],
    g: number[] | null, deltas: number[] | null, lr: number | null,
  ): number[] {
    const last = this.layers.length - 1;
    let grad = g ?? [];
    for (let l = last; l >= 0; l--) {
      const layer = this.layers[l];
      const gm = l < this._dropouts.length ? this._dropouts[l].backward(grad) : grad;
      const delta = (l === last && deltas)
        ? deltas
        : gm.map((v, k) => v * layer.neurons[k].activation.dfn(pre[l][k]));
      const layerIn = act[l];
      const residual = this._shouldResidual(l) && this.structure[l] === this.structure[l + 1];
      const gPrev = layerIn.map((_, j) => {
        const errProp = layer.neurons.reduce((s, n, k) => s + delta[k] * n.weights[j], 0);
        return residual ? errProp + gm[j] : errProp;
      });
      if (lr !== null) {
        layer.neurons.forEach((n, k) => {
          n._update(layerIn.map(inp => delta[k] * inp), delta[k], lr);
        });
      }
      grad = gPrev;
    }
    return grad;
  }
}
