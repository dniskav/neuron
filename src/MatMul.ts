// ─── Matrix utilities for Transformer ────────────────────────────────────────
//
// Low-level helpers used by AttentionHead, MultiHeadAttention and
// TransformerBlock. Kept separate to avoid circular imports.
//
// ─────────────────────────────────────────────────────────────────────────────

import { Adam } from './optimizers'

// ── Matrix multiply ───────────────────────────────────────────────────────────
// C = A @ B   |  A: rows × inner,  B: inner × cols  →  C: rows × cols

export function matMul(A: number[][], B: number[][]): number[][] {
  const rows  = A.length
  const inner = B.length
  const cols  = B[0].length
  const C = Array.from({ length: rows }, () => new Array(cols).fill(0))
  for (let i = 0; i < rows; i++)
    for (let k = 0; k < inner; k++) {
      const aik = A[i][k]
      for (let j = 0; j < cols; j++)
        C[i][j] += aik * B[k][j]
    }
  return C
}

// ── Transpose ─────────────────────────────────────────────────────────────────

export function transpose(A: number[][]): number[][] {
  const rows = A.length, cols = A[0].length
  const T = Array.from({ length: cols }, () => new Array(rows).fill(0))
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      T[j][i] = A[i][j]
  return T
}

// ── Softmax (row-wise) ────────────────────────────────────────────────────────

export function softmax(row: number[]): number[] {
  const max  = Math.max(...row)
  const exps = row.map(v => Math.exp(v - max))
  const sum  = exps.reduce((a, b) => a + b, 0)
  return exps.map(e => e / sum)
}

// ── Softmax backward (Jacobian-vector product) ────────────────────────────────
//
// Given s = softmax(z) and upstream gradient dS = dL/ds, compute dL/dz:
//   dL/dz_i = s_i * (dS_i − dot(dS, s))
//
// This is the compact form of the Jacobian·dS product.

export function softmaxBackward(dS: number[], s: number[]): number[] {
  const dot = s.reduce((acc, si, i) => acc + dS[i] * si, 0)
  return s.map((si, i) => si * (dS[i] - dot))
}

// ── WeightMatrix ──────────────────────────────────────────────────────────────
//
// A 2D weight matrix with one Adam optimizer per scalar weight.
// Xavier initialization: limit = √(2 / (rows + cols)).
//
// Usage:
//   const W = new WeightMatrix(d_out, d_in)
//   const out = W.W.map(row => row.reduce((s,w,j) => s + w * x[j], 0))
//   W.update(dW, lr)  // dW has same shape as W.W

export class WeightMatrix {
  W:    number[][]
  private opts: Adam[][]

  constructor(rows: number, cols: number) {
    const limit = Math.sqrt(2 / (rows + cols))
    this.W    = Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() * 2 - 1) * limit)
    )
    this.opts = Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => new Adam())
    )
  }

  // Apply pre-computed gradient (same shape as W).
  // clipValue: optional per-element gradient clipping before the Adam step.
  // Prevents gradient explosion in deep networks (e.g. Transformers without
  // global norm clipping). Pass e.g. 1.0 to clip to [-1, 1].
  update(dW: number[][], lr: number, clipValue = Infinity): void {
    for (let i = 0; i < this.W.length; i++)
      for (let j = 0; j < this.W[0].length; j++) {
        const g = isFinite(clipValue)
          ? Math.max(-clipValue, Math.min(clipValue, dW[i][j]))
          : dW[i][j]
        this.W[i][j] = this.opts[i][j].step(this.W[i][j], g, lr)
      }
  }
}

// ── EmbeddingMatrix ───────────────────────────────────────────────────────────
//
// Lookup table: vocabSize × d_model.
// get(idx) returns a copy of row idx.
// update(idx, grad, lr) applies SGD to row idx.
//
// SGD is used here (not Adam) because embeddings are updated sparsely and
// a per-row Adam state would require tracking per-row step counts, adding
// complexity for modest gain in this context.

export class EmbeddingMatrix {
  W: number[][]

  constructor(vocabSize: number, d_model: number) {
    const limit = Math.sqrt(1 / d_model)
    this.W = Array.from({ length: vocabSize }, () =>
      Array.from({ length: d_model }, () => (Math.random() * 2 - 1) * limit)
    )
  }

  get(idx: number): number[] {
    return [...this.W[idx]]
  }

  update(idx: number, grad: number[], lr: number): void {
    for (let m = 0; m < this.W[idx].length; m++)
      this.W[idx][m] += lr * grad[m]
  }
}
