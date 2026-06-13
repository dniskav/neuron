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
  pooling?:  'avg' | 'max' | 'last' | 'weighted'  // pooling strategy (default: 'weighted')
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

  // Pooling config
  private _pooling: 'avg' | 'max' | 'last' | 'weighted'

  // For max pooling backward: argmax per dimension across all positions
  private _argmax: number[] | null = null

  constructor(seqLen: number, inputDim: number, options: NetworkTransformerRLOptions = {}) {
    const {
      d_model  = 32,
      nHeads   = 2,
      d_ff     = 64,
      nBlocks  = 2,
      nActions = 2,
      pooling  = 'weighted',
    } = options

    this.seqLen   = seqLen
    this.inputDim = inputDim
    this.d_model  = d_model
    this.nActions = nActions
    this._pooling = pooling

    // Proyección de entrada
    this.inputProj = new WeightMatrix(d_model, inputDim)

    // Bloques transformer (causal: each step only sees the past)
    this.blocks = Array.from({ length: nBlocks }, () =>
      new TransformerBlock({ d_model, nHeads, d_ff, causal: true })
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
    // dH: gradient w.r.t. pooled output, distributed using the same pooling as _pool()
    let dH: number[][] = this._distributePoolGradient(dPooled)

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

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: inputProj, block0, block1, ..., blockN, outputProj, outputBias.

  getWeightsFlat(): number[] {
    const w: number[] = [];
    for (const row of this.inputProj.W) w.push(...row);
    for (const block of this.blocks) w.push(...block.getWeights());
    for (const row of this.outputProj.W) w.push(...row);
    w.push(...this.outputBias);
    return w;
  }

  setWeightsFlat(weights: number[]): void {
    let idx = 0;
    for (let i = 0; i < this.inputProj.W.length; i++)
      for (let j = 0; j < this.inputProj.W[i].length; j++) this.inputProj.W[i][j] = weights[idx++];
    for (const block of this.blocks) {
      const blockLen = block.getWeights().length;
      block.setWeights(weights.slice(idx, idx + blockLen));
      idx += blockLen;
    }
    for (let i = 0; i < this.outputProj.W.length; i++)
      for (let j = 0; j < this.outputProj.W[i].length; j++) this.outputProj.W[i][j] = weights[idx++];
    for (let i = 0; i < this.outputBias.length; i++) this.outputBias[i] = weights[idx++];
  }

  getWeightsStructured() {
    return {
      inputProj: this.inputProj.W.map(r => [...r]),
      blocks: this.blocks.map(b => ({
        attn: {
          heads: b.attn.heads.map(h => ({
            Wq: h.Wq.W.map(r => [...r]),
            Wk: h.Wk.W.map(r => [...r]),
            Wv: h.Wv.W.map(r => [...r]),
          })),
          Wo: b.attn.Wo.W.map(r => [...r]),
        },
        norm1: { gamma: [...b.norm1.gamma], beta: [...b.norm1.beta] },
        norm2: { gamma: [...b.norm2.gamma], beta: [...b.norm2.beta] },
        ff1: b.ff1.W.map(r => [...r]),
        ff2: b.ff2.W.map(r => [...r]),
        b1: [...b.b1],
        b2: [...b.b2],
      })),
      outputProj: this.outputProj.W.map(r => [...r]),
      outputBias: [...this.outputBias],
    }
  }

  setWeightsStructured(data: ReturnType<NetworkTransformerRL['getWeightsStructured']>): void {
    data.inputProj.forEach((row, i) => { this.inputProj.W[i] = [...row] })
    data.blocks.forEach((bd, b) => {
      const blk = this.blocks[b]
      bd.attn.heads.forEach((hd, h) => {
        blk.attn.heads[h].Wq.W = hd.Wq.map(r => [...r])
        blk.attn.heads[h].Wk.W = hd.Wk.map(r => [...r])
        blk.attn.heads[h].Wv.W = hd.Wv.map(r => [...r])
      })
      blk.attn.Wo.W = bd.attn.Wo.map(r => [...r])
      blk.norm1.gamma = [...bd.norm1.gamma]
      blk.norm1.beta  = [...bd.norm1.beta]
      blk.norm2.gamma = [...bd.norm2.gamma]
      blk.norm2.beta  = [...bd.norm2.beta]
      blk.ff1.W = bd.ff1.map(r => [...r])
      blk.ff2.W = bd.ff2.map(r => [...r])
      blk.b1 = [...bd.b1]
      blk.b2 = [...bd.b2]
    })
    this.outputProj.W = data.outputProj.map(r => [...r])
    this.outputBias   = [...data.outputBias]
  }

  // ── Serializable interface (flat array) ────────────────────────────────────
  // These satisfy the Serializable interface from ModelSaver, which requires
  // getWeights(): number[] and setWeights(weights: number[]): void.

  getWeights(): number[] {
    return this.getWeightsFlat()
  }

  setWeights(weights: number[]): void {
    this.setWeightsFlat(weights)
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
    switch (this._pooling) {
      case 'avg':
        return this._poolAvg(h)
      case 'max':
        return this._poolMax(h)
      case 'last':
        return this._poolLast(h)
      case 'weighted':
      default:
        return this._poolWeighted(h)
    }
  }

  private _poolAvg(h: number[][]): number[] {
    const n = h.length
    return Array.from({ length: this.d_model }, (_, m) => {
      let sum = 0
      for (let i = 0; i < n; i++)
        sum += h[i][m]
      return sum / n
    })
  }

  private _poolMax(h: number[][]): number[] {
    // Element-wise max over all positions, tracking argmax for backward
    this._argmax = new Array(this.d_model).fill(0)
    return Array.from({ length: this.d_model }, (_, m) => {
      let maxVal = -Infinity
      for (let i = 0; i < h.length; i++) {
        if (h[i][m] > maxVal) {
          maxVal = h[i][m]
          this._argmax![m] = i
        }
      }
      return maxVal
    })
  }

  private _poolLast(h: number[][]): number[] {
    // Return last position only
    return [...h[h.length - 1]]
  }

  private _poolWeighted(h: number[][]): number[] {
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

  /** Returns the current pooling type for inspection. */
  getPoolingType(): string {
    return this._pooling
  }

  // ── Helper: distribute pooled gradient back to each position ────────────────
  // Must match the same distribution as _pool() used during forward.

  private _distributePoolGradient(dPooled: number[]): number[][] {
    switch (this._pooling) {
      case 'avg': {
        const n = this.seqLen
        return Array.from({ length: n }, () =>
          dPooled.map(v => v / n)
        )
      }
      case 'max': {
        // Route gradient only to the argmax position per dimension
        if (!this._argmax) {
          // Fallback: if no argmax (shouldn't happen), distribute uniformly
          const n = this.seqLen
          return Array.from({ length: n }, () =>
            dPooled.map(v => v / n)
          )
        }
        const argmax = this._argmax
        return Array.from({ length: this.seqLen }, (_, i) =>
          dPooled.map((v, m) => i === argmax[m] ? v : 0)
        )
      }
      case 'last': {
        // Send all gradient to the last position
        return Array.from({ length: this.seqLen }, (_, i) =>
          i === this.seqLen - 1 ? [...dPooled] : new Array(this.d_model).fill(0)
        )
      }
      case 'weighted':
      default: {
        const weights = Array.from({ length: this.seqLen }, (_, i) =>
          i === this.seqLen - 1 ? 2 : 1
        )
        const totalWeight = weights.reduce((a, b) => a + b, 0)
        return Array.from({ length: this.seqLen }, (_, i) =>
          dPooled.map(v => v * weights[i] / totalWeight)
        )
      }
    }
  }
}
