// ─── Trainer ─────────────────────────────────────────────────────────────────
//
// High-level training loop for feedforward networks.
// Handles epoch iteration, shuffling, learning rate decay, and loss history.
//
// Usage:
//   const trainer = new Trainer(network, { epochs: 1000, lr: 0.1 })
//   trainer.train({ inputs, targets })
//   console.log(trainer.getHistory())
//
// ─────────────────────────────────────────────────────────────────────────────

export interface TrainerOptions {
  epochs?: number
  lr?: number
  lrDecay?: number       // multiply lr by this each epoch (e.g. 0.999)
  verbose?: boolean      // print loss every 100 epochs
}

export interface TrainDataset {
  inputs: number[][]
  targets: number[][]
}

export interface TrainableNetwork {
  train(inputs: number[], targets: number[], lr: number): number
}

export class Trainer {
  readonly network: TrainableNetwork
  readonly epochs: number
  readonly lrInitial: number
  readonly lrDecay: number
  readonly verbose: boolean

  private _history: number[] = []

  constructor(network: TrainableNetwork, options: TrainerOptions = {}) {
    this.network = network
    this.epochs = options.epochs ?? 1000
    this.lrInitial = options.lr ?? 0.1
    this.lrDecay = options.lrDecay ?? 1.0
    this.verbose = options.verbose ?? false
  }

  // ── Train on dataset ──────────────────────────────────────────────────────
  train(dataset: TrainDataset): number[] {
    const { inputs, targets } = dataset
    if (inputs.length !== targets.length) {
      throw new Error('Trainer.train: inputs and targets must have the same length')
    }

    const n = inputs.length
    let lr = this.lrInitial
    this._history = []

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Shuffle indices
      const indices = Array.from({ length: n }, (_, i) => i)
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]]
      }

      // Train on each sample
      let epochLoss = 0
      for (const i of indices) {
        epochLoss += this.network.train(inputs[i], targets[i], lr)
      }
      epochLoss /= n
      this._history.push(epochLoss)

      // Apply learning rate decay
      lr *= this.lrDecay

      if (this.verbose && (epoch + 1) % 100 === 0) {
        console.log(`Epoch ${epoch + 1}/${this.epochs}, loss: ${epochLoss.toFixed(6)}, lr: ${lr.toFixed(6)}`)
      }
    }

    return this._history
  }

  // ── Get loss history ──────────────────────────────────────────────────────
  getHistory(): number[] {
    return [...this._history]
  }
}
