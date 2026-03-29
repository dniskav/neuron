// ─── Multi-Head Attention ─────────────────────────────────────────────────────
//
// Runs nHeads independent attention heads in parallel, concatenates their
// outputs, then projects back to d_model via an output weight matrix Wo.
//
//   head_h  = AttentionHead_h(X)          each: seqLen × d_v
//   concat  = [head_0 | head_1 | … ]      seqLen × (nHeads * d_v)
//   out     = concat @ Wo^T               seqLen × d_model
//
// d_k = d_v = d_model / nHeads  (equal split of capacity across heads).
//
// Each head can specialize in a different type of relationship:
//   Head 0 → might learn "look at same row"
//   Head 1 → might learn "look at same column"
//   Head 2 → might learn "look at same 3×3 box"
//   …
// The network figures this out on its own through training.
//
// ─────────────────────────────────────────────────────────────────────────────

import { AttentionHead } from './AttentionHead'
import { WeightMatrix }  from './MatMul'

export class MultiHeadAttention {
  readonly nHeads:  number
  readonly d_model: number
  readonly d_k:     number   // d_k = d_v = d_model / nHeads

  heads: AttentionHead[]
  Wo:   WeightMatrix   // d_model × (nHeads * d_k)

  // Cached for backward
  private _concat: number[][] | null = null   // seqLen × (nHeads * d_k)

  constructor(d_model: number, nHeads: number) {
    this.nHeads  = nHeads
    this.d_model = d_model
    this.d_k     = Math.floor(d_model / nHeads)

    this.heads = Array.from({ length: nHeads }, () =>
      new AttentionHead(d_model, this.d_k, this.d_k)
    )
    this.Wo = new WeightMatrix(d_model, nHeads * this.d_k)
  }

  // ── Forward ───────────────────────────────────────────────────────────────
  // X: seqLen × d_model  →  out: seqLen × d_model

  predict(X: number[][]): number[][] {
    const seqLen   = X.length
    const headOuts = this.heads.map(h => h.predict(X))   // nHeads × seqLen × d_k

    // Concatenate head outputs along feature axis
    const concat: number[][] = Array.from({ length: seqLen }, (_, i) =>
      headOuts.flatMap(ho => ho[i])
    )

    // Project: out[i] = Wo @ concat[i]
    const out: number[][] = concat.map(c =>
      this.Wo.W.map(row => row.reduce((s, w, j) => s + w * c[j], 0))
    )

    this._concat = concat
    return out
  }

  // ── Backward ──────────────────────────────────────────────────────────────
  // dOut: seqLen × d_model  →  dX: seqLen × d_model

  backward(dOut: number[][], lr: number): number[][] {
    const seqLen   = dOut.length
    const concatD  = this.nHeads * this.d_k
    const d_model  = this.d_model
    const concat   = this._concat!

    // dConcat[i] = Wo^T @ dOut[i]
    const dConcat: number[][] = dOut.map(do_ =>
      Array.from({ length: concatD }, (_, j) =>
        this.Wo.W.reduce((s, row, k) => s + do_[k] * row[j], 0)
      )
    )

    // dWo[k][j] = Σ_i dOut[i][k] * concat[i][j]
    const dWo: number[][] = Array.from({ length: d_model }, (_, k) =>
      Array.from({ length: concatD }, (_, j) =>
        dOut.reduce((s, row, i) => s + row[k] * concat[i][j], 0)
      )
    )
    this.Wo.update(dWo, lr)

    // Split dConcat into per-head gradients and backprop each head
    const dX: number[][] = Array.from({ length: seqLen }, () =>
      new Array(d_model).fill(0)
    )

    for (let h = 0; h < this.nHeads; h++) {
      const start      = h * this.d_k
      const dHeadOut   = dConcat.map(dc => dc.slice(start, start + this.d_k))
      const dXh        = this.heads[h].backward(dHeadOut, lr)
      for (let i = 0; i < seqLen; i++)
        for (let m = 0; m < d_model; m++)
          dX[i][m] += dXh[i][m]
    }

    return dX
  }

  // Attention weights per head from the last predict() — for visualization.
  // Returns: nHeads × seqLen × seqLen
  getAttentionWeights(): (number[][] | null)[] {
    return this.heads.map(h => h.getAttentionWeights())
  }
}
