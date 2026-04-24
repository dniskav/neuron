// ─── NetworkTransformerRL ─────────────────────────────────────────────────────
//
// Transformer para Reinforcement Learning con memoria de secuencia.
// Inspirado en Decision Transformer, pero simplificado para fines educativos.
//
// Arquitectura:
//   Input:  secuencia de últimos N estados  (seqLen × inputDim)
//   Output: Q-values para las acciones      (nActions)
//
// Diferencias con NetworkTransformer (Sudoku):
//   - Entrada continua (no tokens discretos) → proyección lineal en vez de embedding
//   - Output único (no por-token) → pooling sobre el último estado
//   - Causal mask: cada paso solo ve el pasado (no el futuro)
//
// Uso en RL:
//   - El agente recuerda sus últimos N pasos (sensores + acciones previas)
//   - La atención aprende qué momentos pasados son relevantes para decidir ahora
//   - Ej: "hace 3 pasos giré a la izquierda y choqué → ahora evito esa acción"
//
// ─────────────────────────────────────────────────────────────────────────────

import { TransformerBlock } from './TransformerBlock'
import { WeightMatrix } from './MatMul'
import { Adam } from './optimizers'

export interface NetworkTransformerRLOptions {
  d_model?:  number    // dimensión del modelo (default: 32)
  nHeads?:   number    // cabezas de atención (default: 2)
  d_ff?:     number    // dimensión feed-forward (default: 64)
  nBlocks?:  number    // bloques transformer (default: 2)
  nActions?: number    // acciones posibles (default: 2)
}

export class NetworkTransformerRL {
  readonly seqLen:   number
  readonly inputDim: number
  readonly d_model:  number
  readonly nActions: number

  // Proyección de entrada: inputDim → d_model
  inputProj: WeightMatrix

  // Transformer stack
  blocks: TransformerBlock[]

  // Pooling + output: d_model → nActions
  outputProj: WeightMatrix
  outputBias: number[]
  private outBiasOpts: Adam[]

  // Forward caches para backprop
  private _projected: number[][] | null = null

  constructor(seqLen: number, inputDim: number, options: NetworkTransformerRLOptions = {}) {
    const {
      d_model  = 32,
      nHeads   = 2,
      d_ff     = 64,
      nBlocks  = 2,
      nActions = 2,
    } = options

    this.seqLen   = seqLen
    this.inputDim = inputDim
    this.d_model  = d_model
    this.nActions = nActions

    // Proyección de entrada
    this.inputProj = new WeightMatrix(d_model, inputDim)

    // Bloques transformer
    this.blocks = Array.from({ length: nBlocks }, () =>
      new TransformerBlock({ d_model, nHeads, d_ff })
    )

    // Proyección de salida
    this.outputProj  = new WeightMatrix(nActions, d_model)
    this.outputBias  = new Array(nActions).fill(0)
    this.outBiasOpts = Array.from({ length: nActions }, () => new Adam())
  }

  // ── Forward ────────────────────────────────────────────────────────────────
  // sequence: seqLen × inputDim → nActions Q-values

  predict(sequence: number[][]): number[] {
    const h = this._forward(sequence)
    // Pooling: promediar sobre la secuencia, pero dar más peso al último paso
    const pooled = this._pool(h)
    // Output: Q-values
    return this.outputProj.W.map((row, c) =>
      row.reduce((s, w, m) => s + w * pooled[m], this.outputBias[c])
    )
  }

  // ── Training ────────────────────────────────────────────────────────────────
  // sequence: seqLen × inputDim
  // target:   nActions Q-values (one-hot style para Q-learning)
  // lr:       learning rate
  // Returns: MSE loss

  train(sequence: number[][], target: number[], lr: number): number {
    // Forward
    const h = this._forward(sequence)
    const pooled = this._pool(h)

    // Output projection
    const pred: number[] = this.outputProj.W.map((row, c) =>
      row.reduce((s, w, m) => s + w * pooled[m], this.outputBias[c])
    )

    // MSE loss: L = (1/n) * Σ (pred[c] - target[c])²
    const n = this.nActions
    let loss = 0
    for (let c = 0; c < n; c++) {
      const diff = pred[c] - target[c]
      loss += diff * diff
    }
    loss /= n

    // Backward: dL/dpred[c] = 2 * (pred[c] - target[c]) / n
    const dPred = pred.map((p, c) => 2 * (p - target[c]) / n)

    // Output projection backward
    //   dPooled[m] = Σ_c dPred[c] * outputProj.W[c][m]
    const dPooled: number[] = Array.from({ length: this.d_model }, (_, m) =>
      dPred.reduce((s, dp, c) => s + dp * this.outputProj.W[c][m], 0)
    )

    const dWout: number[][] = Array.from({ length: this.nActions }, (_, c) =>
      Array.from({ length: this.d_model }, (_, m) =>
        dPred[c] * pooled[m]
      )
    )
    const dBout = dPred.slice()

    this.outputProj.update(dWout, lr)
    for (let c = 0; c < this.nActions; c++)
      this.outputBias[c] = this.outBiasOpts[c].step(this.outputBias[c], dBout[c], lr)

    // Backprop through transformer blocks
    // dH: gradient w.r.t. pooled output, broadcast to seqLen positions
    let dH: number[][] = Array.from({ length: this.seqLen }, (_, i) =>
      dPooled.map(v => v / this.seqLen)  // Gradiente dividido entre posiciones
    )

    for (let b = this.blocks.length - 1; b >= 0; b--)
      dH = this.blocks[b].backward(dH, lr)

    // Backprop into input projection
    // dH[i] es el gradiente w.r.t. projected[i]
    // inputProj: projected[i] = inputProj @ sequence[i]
    for (let i = 0; i < this.seqLen; i++) {
      const dInputProj: number[][] = Array.from({ length: this.d_model }, (_, k) =>
        Array.from({ length: this.inputDim }, (_, m) =>
          dH[i][k] * sequence[i][m]
        )
      )
      this.inputProj.update(dInputProj, lr)
    }

    return loss
  }

  // Attention weights from every block for visualization.
  getAttentionWeights(): (number[][] | null)[][] {
    return this.blocks.map(b => b.getAttentionWeights())
  }

  // ── Internal ────────────────────────────────────────────────────────────────

  private _forward(sequence: number[][]): number[][] {
    // Proyectar entrada: inputDim → d_model
    let h: number[][] = sequence.map(step =>
      this.inputProj.W.map((row, k) =>
        row.reduce((s, w, m) => s + w * step[m], 0)
      )
    )

    // Pasar por bloques transformer
    for (const block of this.blocks)
      h = block.predict(h)

    this._projected = h
    return h
  }

  private _pool(h: number[][]): number[] {
    // Pooling con peso: último paso tiene 2x peso
    const weights = Array.from({ length: this.seqLen }, (_, i) =>
      i === this.seqLen - 1 ? 2 : 1
    )
    const totalWeight = weights.reduce((a, b) => a + b, 0)

    return Array.from({ length: this.d_model }, (_, m) => {
      let sum = 0
      for (let i = 0; i < this.seqLen; i++)
        sum += weights[i] * h[i][m]
      return sum / totalWeight
    })
  }
}
