// ─── NetworkTransformer ───────────────────────────────────────────────────────
//
// A full token-classification Transformer:
//
//   Input:  sequence of integer token ids   (seqLen)
//   Output: per-token logits                (seqLen × nClasses), flattened
//
// Architecture:
//   1. Embedding: tokenEmb[id] + posEmb[pos]  →  seqLen × d_model
//   2. N transformer blocks (MHA + FFN + LayerNorm × 2 each)
//   3. Output projection: d_model → nClasses  per token
//
// Designed for Sudoku solving:
//   seqLen   = 81  (one token per cell)
//   vocabSize = 10  (digits 0–9; 0 = empty)
//   d_model   = 64
//   nHeads    = 4
//   d_ff      = 128
//   nBlocks   = 4
//   nClasses  = 9  (digits 1–9)
//
// The key advantage over a plain MLP is that each token can attend to all
// others — the network can learn "cell i looks at all cells in its row,
// column, and box to decide which digit fits" without being told the rules.
//
// ─────────────────────────────────────────────────────────────────────────────

import { TransformerBlock }  from './TransformerBlock'
import { EmbeddingMatrix, WeightMatrix } from './MatMul'
import { Adam }              from './optimizers'

export interface NetworkTransformerOptions {
  vocabSize?: number    // default: 10 (0–9)
  d_model?:   number    // default: 64
  nHeads?:    number    // default: 4
  d_ff?:      number    // default: 128
  nBlocks?:   number    // default: 4
  nClasses?:  number    // default: 9
}

export class NetworkTransformer {
  readonly seqLen:    number
  readonly vocabSize: number
  readonly d_model:   number
  readonly nClasses:  number

  // Embeddings
  tokenEmb: EmbeddingMatrix   // vocabSize × d_model
  posEmb:   EmbeddingMatrix   // seqLen    × d_model

  // Transformer stack
  blocks: TransformerBlock[]

  // Output projection: nClasses × d_model  (one row per class)
  outputProj: WeightMatrix
  outputBias: number[]
  private outBiasOpts: Adam[]

  constructor(seqLen: number, options: NetworkTransformerOptions = {}) {
    const {
      vocabSize = 10,
      d_model   = 64,
      nHeads    = 4,
      d_ff      = 128,
      nBlocks   = 4,
      nClasses  = 9,
    } = options

    this.seqLen    = seqLen
    this.vocabSize = vocabSize
    this.d_model   = d_model
    this.nClasses  = nClasses

    this.tokenEmb = new EmbeddingMatrix(vocabSize, d_model)
    this.posEmb   = new EmbeddingMatrix(seqLen,    d_model)

    this.blocks = Array.from({ length: nBlocks }, () =>
      new TransformerBlock({ d_model, nHeads, d_ff })
    )

    this.outputProj  = new WeightMatrix(nClasses, d_model)
    this.outputBias  = new Array(nClasses).fill(0)
    this.outBiasOpts = Array.from({ length: nClasses }, () => new Adam())
  }

  // ── Forward pass ──────────────────────────────────────────────────────────
  // tokens: seqLen integer ids  →  seqLen * nClasses logits (flattened)

  predict(tokens: number[]): number[] {
    const h = this._forward(tokens)
    // Output projection per token: logits[i][c] = outputProj[c] · h[i] + bias[c]
    return h.flatMap(hi =>
      this.outputProj.W.map((row, c) =>
        row.reduce((s, w, m) => s + w * hi[m], this.outputBias[c])
      )
    )
  }

  // ── Training step (online, one sample at a time) ───────────────────────────
  // tokens:  seqLen integer ids
  // targets: seqLen * nClasses values (e.g. one-hot per cell)
  // mask:    optional boolean[seqLen] — only compute loss/gradients for
  //          positions where mask[i] = true (e.g. empty cells in Sudoku)
  // Returns: MSE loss over the masked positions.

  train(tokens: number[], targets: number[], lr: number, mask?: boolean[]): number {
    // ── Forward ──────────────────────────────────────────────────────────────
    const h = this._forward(tokens)

    // Output projection
    const logits: number[][] = h.map(hi =>
      this.outputProj.W.map((row, c) =>
        row.reduce((s, w, m) => s + w * hi[m], this.outputBias[c])
      )
    )

    // ── Loss (MSE over masked positions) ─────────────────────────────────────
    let loss  = 0
    let count = 0
    const dLogits: number[][] = Array.from({ length: this.seqLen }, (_, i) => {
      if (mask && !mask[i]) return new Array(this.nClasses).fill(0)
      count++
      return Array.from({ length: this.nClasses }, (_, c) => {
        const t = targets[i * this.nClasses + c]
        const p = logits[i][c]
        loss += (p - t) ** 2
        return 2 * (p - t)   // MSE gradient (sign: loss = (p-t)², dL/dp = 2(p-t))
      })
    })
    if (count > 0) loss /= count * this.nClasses

    // ── Backward ─────────────────────────────────────────────────────────────

    // Output projection backward
    //   dH[i][m] = Σ_c dLogits[i][c] * outputProj.W[c][m]
    const dH: number[][] = Array.from({ length: this.seqLen }, (_, i) =>
      Array.from({ length: this.d_model }, (_, m) =>
        dLogits[i].reduce((s, dl, c) => s + dl * this.outputProj.W[c][m], 0)
      )
    )

    const dWout: number[][] = Array.from({ length: this.nClasses }, (_, c) =>
      Array.from({ length: this.d_model }, (_, m) =>
        dLogits.reduce((s, dl, i) => s + dl[c] * h[i][m], 0)
      )
    )
    const dBout = Array.from({ length: this.nClasses }, (_, c) =>
      dLogits.reduce((s, dl) => s + dl[c], 0)
    )
    this.outputProj.update(dWout, lr)
    for (let c = 0; c < this.nClasses; c++)
      this.outputBias[c] = this.outBiasOpts[c].step(this.outputBias[c], dBout[c], lr)

    // Backprop through transformer blocks (reverse order)
    let dX = dH
    for (let b = this.blocks.length - 1; b >= 0; b--)
      dX = this.blocks[b].backward(dX, lr)

    // Backprop into embeddings (SGD)
    // dX[i] is the gradient w.r.t. (tokenEmb[tokens[i]] + posEmb[i])
    // Both embeddings receive the same gradient.
    for (let i = 0; i < this.seqLen; i++) {
      this.tokenEmb.update(tokens[i], dX[i], lr)
      this.posEmb.update(i, dX[i], lr)
    }

    return loss
  }

  // Attention weights from every block for visualization.
  // Returns: nBlocks × nHeads × seqLen × seqLen  (nulls if not yet run)
  getAttentionWeights(): (number[][] | null)[][] {
    return this.blocks.map(b => b.getAttentionWeights())
  }

  // ── Internal ──────────────────────────────────────────────────────────────
  // Shared embedding + block forward pass.

  private _forward(tokens: number[]): number[][] {
    // Embed: h[i] = tokenEmb[tokens[i]] + posEmb[i]
    let h: number[][] = tokens.map((id, i) => {
      const te = this.tokenEmb.get(id)
      const pe = this.posEmb.get(i)
      return te.map((v, m) => v + pe[m])
    })

    // Pass through transformer blocks
    for (const block of this.blocks)
      h = block.predict(h)

    return h
  }
}
