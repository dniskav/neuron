## How it works

The neural classes, from `Neuron` to `NetworkN`, apply an **activation function** to the weighted sum of their inputs and use **gradient descent** to adjust the weights. For one sigmoid output neuron with plain SGD, the update is:

```text
delta   = (target − output) × activation′(sum)
weight += lr × delta × input
bias   += lr × delta
```

This is the rule in `Neuron.train` (`src/Neuron.ts`) and `NeuronN.train` (`src/NeuronN.ts`). The update adds `lr × delta × input`, because the step follows the negative gradient of the squared error.

`NetworkN` implements full **backpropagation** across all layers: deltas flow from the output back to the first layer with the chain rule (`_backward`, `src/NetworkN.ts`). Its `train` starts from the gradient of the **mean** squared error that it returns: `(2 / n) × (target − output)` for each of the `n` outputs (`src/NetworkN.ts`; the returned loss is computed in the same method). So `lr` is the step size of that loss, whatever the number of outputs. `Network.train` uses the same factor (`src/Network.ts`). This changed in 0.4.0, see [Upgrading to 0.4](#upgrading-to-04).

`NeuronN` uses a **simplified Xavier initialization**: each weight is drawn uniformly from `[-√(1/n), +√(1/n)]`, where `n` is the number of inputs, and the bias starts at 0 (`src/NeuronN.ts`). `NetworkN` builds its layers from `NeuronN` (`src/Layer.ts`), so it starts the same way.

When an **optimizer** is used (e.g., Adam), the raw gradient is passed to the optimizer instead of being applied directly. `SGD` applies `w ← w + lr·g` (`src/optimizers.ts`); other optimizers take different steps. Each weight keeps its own optimizer state (`src/NeuronN.ts`).

The `Value` class implements **reverse-mode automatic differentiation**: every operation records its inputs and a `_backward` closure. Calling `.backward()` on the output node builds a topological order and propagates `∂L/∂w` through the graph (`src/Tape.ts`).

## Known limitations

- **`ContrastiveLearning`:** the raw initialization can leave a projection head dead, with a zero-norm output (`src/ContrastiveLearning.ts`). On small problems the loss may not go down.
- **`VAE`:** with the default `klWeight` of 1, the model collapses on small datasets and the latent code stops carrying information. In the measured case a `klWeight` around 0.003 learned (`src/VAE.ts`).
- **`PCA`:** with rank-deficient data, the axes with zero variance can have a norm close to 0, so they are not an orthonormal basis. Do not use them as one.
- **`TSNE`:** with a small number of points, the default `lr` of 200 diverges. The stable range depends on the number of points (`src/TSNE.ts`).
- **`TCN`:** sensitive to `lr`. The header of `src/TCN.ts` lists the measured range.
- **Randomness:** there is no global seed. Components that draw random numbers use `Math.random` unless they have their own option (for example, `KMeans` takes a `seed`). The examples in this README fix `Math.random` in their own block.
- **Training:** most training is online (one sample per update) and runs on the CPU in plain JavaScript. There is no GPU backend.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## Upgrading to 0.4

0.4.0 changes how several models train, so 0.3.2 results are not directly comparable:

- Weights and learning rates tuned with 0.3.2 are not a good starting point. Retrain, and re-tune `lr`.
- `NetworkN.train` with `n` outputs, and `Network.train` with one output: `lr_new = lr_old × n / 2` reproduces the old step with the default optimizer or `Momentum` (with `Adam`, keep `lr`). `Network.train` with several outputs and `NetworkN` with `residual` have no recipe, so retrain.
- Many methods now throw on invalid input (empty data, wrong lengths, `NaN`, invalid options) instead of failing silently.
- `Trainer` rejects `lr = 0`, and the `clipValue` option is gone: use `ClippedOptimizerFactory`.

The full guide, with the breaking changes and their migration steps, is in [CHANGELOG.md](CHANGELOG.md).

## Build

```bash
npm run build      # tsup: CJS, ESM and type declarations into dist/ (cleans dist/ first)
npm run dev        # the same build in watch mode
npm test           # vitest run (one pass)
npm run test:watch # vitest in watch mode
npm run coverage   # vitest run with v8 coverage
npm run typecheck  # tsc --noEmit over src/ and tests/
```

The build writes `dist/index.js` (CommonJS), `dist/index.mjs` (ES modules), and `dist/index.d.ts` and `dist/index.d.mts` (types).

## For AI agents

If you are an AI agent or LLM working with this codebase, read [AGENTS.md](AGENTS.md) first. It is the entry point for agents working on this repository.

## Roadmap (nice to have)

These features are intentionally out of scope for the current didactic focus but are documented here for reference.

### ONNX export

Export trained models to the [ONNX](https://onnx.ai/) interchange format so they can be run in Python (onnxruntime), browsers (onnxruntime-web), mobile, or production inference servers.

**What it would require:**
- Serialize each layer's weights + op type to the protobuf ONNX schema (`onnx.proto`).
- Map neuron layer types to standard ONNX operators (`Gemm`, `MatMul`, `LSTM`, `Conv`, `Relu`, `Softmax`, …).
- Handle dynamic batch dimensions in the graph IR.
- Ship a build step that compiles the `.proto` definitions (adds a dev dependency on `protobufjs` or `onnx-proto`).

**Why it's skipped:** It adds a non-trivial build pipeline and a dependency. The library has zero runtime dependencies by design. ONNX export makes sense once you outgrow the library for training — at that point PyTorch/TF are the right tools.

### WebGL / WASM backend

Replace the current pure-JS number arrays with a GPU-accelerated or WASM-compiled backend so larger models (e.g. ViT, GPT-2 scale) become feasible in the browser.

**What it would require:**
- Abstract `Tensor` type that backends implement (JS arrays, WebGL textures, WASM memory).
- WebGL backend: encode matrix ops as fragment-shader programs (similar to `gpu.js` or `tfjs-backend-webgl`).
- WASM backend: compile a BLAS-like C/Rust core (e.g. `wasm-bindgen` + `ndarray`) and bind it to TypeScript.
- Every layer's `forward` / `backward` rewritten against the `Tensor` API.

**Why it's skipped:** The goal is to make the math readable. GPU shader code and WASM bindings are implementation details that obscure the algorithms. The library intentionally trades performance for pedagogical clarity.

## Changelog

Release notes for every version, including the 0.4.0 breaking changes, the migration guide and the known limitations, are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
