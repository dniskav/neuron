## Guides

The long examples are in `docs/`, one guide per topic. Every example there is run by the same gate as the examples in this README, and each one ends with a check that throws if the result is wrong. The links are absolute because the `docs/` folder is not in the npm package.

- [Basic usage](https://github.com/dniskav/neuron/blob/master/docs/basics.md): a single neuron, `NetworkN` with Adam, the Perceptron, linear and logistic regression, Gaussian Naive Bayes, decision trees, K-Means, PCA, SOM and the Hopfield network (11 examples).
- [Models](https://github.com/dniskav/neuron/blob/master/docs/models.md): Autoencoder, GAN, VAE, Word2Vec, t-SNE, positional encodings and contrastive learning (7 examples).
- [Sequences](https://github.com/dniskav/neuron/blob/master/docs/sequences.md): the `Value` autograd, Conv2D with MaxPool2D and Flatten, Conv1D, RNN, TCN and NetworkLSTM (6 examples).
- [Training and evaluation tools](https://github.com/dniskav/neuron/blob/master/docs/tools.md): metrics, early stopping, `Trainer`, `ClippedOptimizerFactory`, `ModelSaver`, `DataLoader`, `Tokenizer`, `DatasetLoader`, `DataAugmentation`, `WeightInspector` and `LossPlotter` (12 examples).
- [Fast core and extending the library](https://github.com/dniskav/neuron/blob/master/docs/fast-core.md): seeds with `setRandom`, the vectorized optimizers, `setBackend` and `checkBackend`, `NetworkFast` and batches, format 2 with `ModelSaver.save` and `load`, `infer`, your own embeddings, and custom activations, optimizers and models (12 examples).
