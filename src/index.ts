export { Neuron }      from "./Neuron";
export { NeuronN }     from "./NeuronN";
export { Layer }       from "./Layer";
export { Network }     from "./Network";
export { NetworkN }    from "./NetworkN";
export { NetworkLSTM } from "./NetworkLSTM";
export { LSTMLayer }   from "./LSTMLayer";
export { NetworkTransformer }    from "./NetworkTransformer";
export { NetworkTransformerRL }  from "./NetworkTransformerRL";
export { TransformerBlock }      from "./TransformerBlock";
export { MultiHeadAttention }    from "./MultiHeadAttention";
export { AttentionHead }      from "./AttentionHead";
export { LayerNorm }          from "./LayerNorm";
export { WeightMatrix, EmbeddingMatrix, BiasVector, matMul, transpose, softmax, softmaxBackward } from "./MatMul";

export { sigmoid, relu, tanh, linear, leakyRelu, elu, makeLeakyRelu, makeElu } from "./activations";
export type { Activation }             from "./activations";

export { SGD, Momentum, Adam, ClipOptimizer, ClippedOptimizerFactory, defaultOptimizer } from "./optimizers";
export type { Optimizer, OptimizerFactory } from "./optimizers";

export { mse, crossEntropy, mseDelta, crossEntropyDelta, crossEntropyDeltaRaw } from "./losses";

export type { NetworkNOptions }           from "./NetworkN";
export type { NetworkLSTMOptions }        from "./NetworkLSTM";
export type { NetworkTransformerOptions }    from "./NetworkTransformer";
export type { NetworkTransformerRLOptions }  from "./NetworkTransformerRL";
export type { TransformerBlockOptions }      from "./TransformerBlock";

// New layers
export { Dropout }     from "./Dropout";
export { GRULayer }    from "./GRU";
export { BatchNorm }   from "./BatchNorm";
export { Conv1D }      from "./Conv1D";

// New utilities
export { Trainer }     from "./Trainer";
export type { TrainerOptions, TrainDataset, TrainableNetwork, TrainableNetworkWithWeights, TrainMetrics } from "./Trainer";
export { DataLoader }  from "./DataLoader";
export type { DataPair } from "./DataLoader";
export { LRScheduler } from "./LRScheduler";
export { ModelSaver }  from "./ModelSaver";
export type { Serializable } from "./ModelSaver";

// Validation
export { validateArray, validateArrayMinLength, validate2DArray, validateNumber } from "./Validation";

// Classical ML
export { Perceptron }                       from "./Perceptron";
export { LinearRegression }                 from "./LinearRegression";
export { LogisticRegression, SoftmaxRegression } from "./LogisticRegression";
export { GaussianNaiveBayes }               from "./NaiveBayes";
export { DecisionTree }                     from "./DecisionTree";

// Unsupervised learning
export { KMeans }                           from "./KMeans";
export type { KMeansOptions }               from "./KMeans";
export { PCA }                              from "./PCA";
export { SOM }                              from "./SOM";
export type { SOMOptions }                  from "./SOM";
export { HopfieldNetwork }                  from "./HopfieldNetwork";
export { Autoencoder }                      from "./Autoencoder";

// CNN extras
export { Conv2D }                           from "./Conv2D";
export { MaxPool2D }                        from "./MaxPool2D";
export { Flatten }                          from "./Flatten";

// Recurrent extras
export { RNN }                              from "./RNN";
export { Seq2Seq }                          from "./Seq2Seq";
export { CausalConv1D, TCN }               from "./TCN";

// Generative models
export { GAN }  from "./GAN";
export { VAE }  from "./VAE";

// Automatic differentiation
export { Value } from "./Tape";

// Evaluation & diagnostics
export { WeightInspector } from "./WeightInspector";
export type { WeightStats } from "./WeightInspector";
export {
  confusionMatrix,
  precision,
  recall,
  f1Score,
  accuracy,
  rocCurve,
  auc,
  mae,
  rmse,
  r2Score,
  perplexity,
  printConfusionMatrix,
  classificationReport,
} from "./Metrics";

// Training utilities
export { EarlyStopping }    from "./EarlyStopping";
export { LossPlotter }      from "./LossPlotter";
export { DataAugmentation } from "./DataAugmentation";
