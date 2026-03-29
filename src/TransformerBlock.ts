// ─── Transformer Block ────────────────────────────────────────────────────────
//
// One standard transformer block (post-norm, "Attention Is All You Need"):
//
//   Attention sub-layer:
//     attnOut = MHA(X)
//     h1      = LayerNorm(X + attnOut)       ← residual + normalize
//
//   Feed-Forward sub-layer:
//     ff1     = ReLU(h1 @ ff1.W^T + b1)     ← expand to d_ff
//     ff2     = h1 @ ff2.W^T + b2            ← project back to d_model
//     out     = LayerNorm(h1 + ff2)          ← residual + normalize
//
// The residual connections allow gradients to flow directly to earlier layers,
// preventing the vanishing gradient problem in deep stacks.
//
// LayerNorm stabilizes activations after each sub-layer, making training
// significantly more stable than without normalization.
//
// ─────────────────────────────────────────────────────────────────────────────

import { MultiHeadAttention } from './MultiHeadAttention'
import { LayerNorm }          from './LayerNorm'
import { WeightMatrix }       from './MatMul'
import { Adam }               from './optimizers'

export interface TransformerBlockOptions {
  d_model: number
  nHeads:  number
  d_ff:    number
}

export class TransformerBlock {
  readonly d_model: number
  readonly d_ff:    number

  attn:  MultiHeadAttention
  norm1: LayerNorm
  norm2: LayerNorm

  ff1: WeightMatrix   // d_ff    × d_model  (expand)
  ff2: WeightMatrix   // d_model × d_ff     (project)
  b1:  number[]       // d_ff
  b2:  number[]       // d_model

  private b1Opts: Adam[]
  private b2Opts: Adam[]

  // Forward caches (needed for backprop)
  private _X:       number[][] | null = null
  private _attnOut: number[][] | null = null
  private _h1:      number[][] | null = null
  private _ff1Pre:  number[][] | null = null   // pre-ReLU
  private _ff1Out:  number[][] | null = null   // post-ReLU
  private _ff2Out:  number[][] | null = null

  constructor({ d_model, nHeads, d_ff }: TransformerBlockOptions) {
    this.d_model = d_model
    this.d_ff    = d_ff

    this.attn  = new MultiHeadAttention(d_model, nHeads)
    this.norm1 = new LayerNorm(d_model)
    this.norm2 = new LayerNorm(d_model)

    this.ff1 = new WeightMatrix(d_ff, d_model)
    this.ff2 = new WeightMatrix(d_model, d_ff)
    this.b1  = new Array(d_ff).fill(0)
    this.b2  = new Array(d_model).fill(0)

    this.b1Opts = Array.from({ length: d_ff },    () => new Adam())
    this.b2Opts = Array.from({ length: d_model }, () => new Adam())
  }

  // ── Forward ───────────────────────────────────────────────────────────────
  // X: seqLen × d_model  →  out: seqLen × d_model

  predict(X: number[][]): number[][] {
    const seqLen = X.length

    // ── Attention sub-layer ──────────────────────────────────────────────────
    const attnOut = this.attn.predict(X)

    this.norm1.resetCache(seqLen)
    const h1 = X.map((x, i) => {
      const added = x.map((v, k) => v + attnOut[i][k])    // residual
      return this.norm1.predictOne(added, i)
    })

    // ── FFN sub-layer ────────────────────────────────────────────────────────
    // ff1: expand  d_model → d_ff  with ReLU
    const ff1Pre: number[][] = h1.map(h =>
      this.ff1.W.map((row, k) => row.reduce((s, w, m) => s + w * h[m], this.b1[k]))
    )
    const ff1Out: number[][] = ff1Pre.map(pre => pre.map(v => Math.max(0, v)))

    // ff2: project  d_ff → d_model  (linear)
    const ff2Out: number[][] = ff1Out.map(h =>
      this.ff2.W.map((row, k) => row.reduce((s, w, m) => s + w * h[m], this.b2[k]))
    )

    this.norm2.resetCache(seqLen)
    const out = h1.map((h, i) => {
      const added = h.map((v, k) => v + ff2Out[i][k])     // residual
      return this.norm2.predictOne(added, i)
    })

    // Cache for backward
    this._X       = X
    this._attnOut = attnOut
    this._h1      = h1
    this._ff1Pre  = ff1Pre
    this._ff1Out  = ff1Out
    this._ff2Out  = ff2Out

    return out
  }

  // ── Backward ──────────────────────────────────────────────────────────────
  // dOut: seqLen × d_model  →  dX: seqLen × d_model

  backward(dOut: number[][], lr: number): number[][] {
    const seqLen  = dOut.length
    const d_model = this.d_model
    const h1      = this._h1!
    const ff1Out  = this._ff1Out!
    const ff1Pre  = this._ff1Pre!

    // ── FFN backward ─────────────────────────────────────────────────────────

    // norm2 backward (one token at a time, using position-matched cache)
    const dAdded2 = dOut.map((do_, i) => this.norm2.backwardOne(do_, i, lr))

    // Residual: dAdded2 splits into dFf2Out (ff2 path) and dH1_skip (h1 path)
    // Since both paths receive the same gradient, we use dAdded2 for both.

    // ff2 backward: ff2Out = ff2 @ ff1Out + b2
    //   dFf1Out[i][k] = Σ_m dAdded2[i][m] * ff2.W[m][k]
    const dFf1Out: number[][] = dAdded2.map(da =>
      Array.from({ length: this.d_ff }, (_, k) =>
        this.ff2.W.reduce((s, row, m) => s + row[k] * da[m], 0)
      )
    )

    //   dff2.W[m][k] = Σ_i dAdded2[i][m] * ff1Out[i][k]
    const dW2: number[][] = Array.from({ length: d_model }, (_, m) =>
      Array.from({ length: this.d_ff }, (_, k) =>
        dAdded2.reduce((s, da, i) => s + da[m] * ff1Out[i][k], 0)
      )
    )
    const db2 = Array.from({ length: d_model }, (_, m) =>
      dAdded2.reduce((s, da) => s + da[m], 0)
    )
    this.ff2.update(dW2, lr)
    for (let m = 0; m < d_model; m++)
      this.b2[m] = this.b2Opts[m].step(this.b2[m], db2[m], lr)

    // ReLU backward
    const dFf1Pre = dFf1Out.map((d, i) =>
      d.map((v, k) => ff1Pre[i][k] > 0 ? v : 0)
    )

    // ff1 backward: ff1Pre = ff1 @ h1 + b1
    //   dH1_fromFf[i][m] = Σ_k dFf1Pre[i][k] * ff1.W[k][m]
    const dH1_fromFf: number[][] = dFf1Pre.map(dp =>
      Array.from({ length: d_model }, (_, m) =>
        this.ff1.W.reduce((s, row, k) => s + dp[k] * row[m], 0)
      )
    )

    const dW1: number[][] = Array.from({ length: this.d_ff }, (_, k) =>
      Array.from({ length: d_model }, (_, m) =>
        dFf1Pre.reduce((s, dp, i) => s + dp[k] * h1[i][m], 0)
      )
    )
    const db1 = Array.from({ length: this.d_ff }, (_, k) =>
      dFf1Pre.reduce((s, dp) => s + dp[k], 0)
    )
    this.ff1.update(dW1, lr)
    for (let k = 0; k < this.d_ff; k++)
      this.b1[k] = this.b1Opts[k].step(this.b1[k], db1[k], lr)

    // Total dH1 = gradient from ff1 path + gradient through skip to norm2 residual
    const dH1: number[][] = Array.from({ length: seqLen }, (_, i) =>
      dH1_fromFf[i].map((v, m) => v + dAdded2[i][m])
    )

    // ── Attention backward ────────────────────────────────────────────────────

    // norm1 backward
    const dAdded1 = dH1.map((d, i) => this.norm1.backwardOne(d, i, lr))

    // Residual: dAdded1 flows to both MHA output and X skip connection
    const dAttnOut   = dAdded1   // into MHA backward
    const dX_skip    = dAdded1   // through skip

    const dX_fromAttn = this.attn.backward(dAttnOut, lr)

    // Combine
    const dX: number[][] = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: d_model }, (_, m) =>
        dX_fromAttn[i][m] + dX_skip[i][m]
      )
    )

    return dX
  }

  // Attention weights from the last predict() — for visualization.
  getAttentionWeights(): (number[][] | null)[] {
    return this.attn.getAttentionWeights()
  }
}
