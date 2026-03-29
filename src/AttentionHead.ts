// ─── Single Attention Head ────────────────────────────────────────────────────
//
// Implements scaled dot-product self-attention for one head:
//
//   Q = X @ Wq^T   (seqLen × d_k)
//   K = X @ Wk^T   (seqLen × d_k)
//   V = X @ Wv^T   (seqLen × d_v)
//
//   scores[i][j] = Q[i] · K[j] / √d_k   (seqLen × seqLen)
//   attn[i]      = softmax(scores[i])
//   out[i]       = Σ_j  attn[i][j] · V[j]    (seqLen × d_v)
//
// The scaling factor √d_k prevents the dot products from growing large
// (which would push softmax into a saturated region with tiny gradients).
//
// Weights are stored transposed for efficient row-major access:
//   Wq: d_k × d_model   so  Q[i][k] = Σ_m Wq[k][m] * X[i][m]
//
// ─────────────────────────────────────────────────────────────────────────────

import { WeightMatrix, softmax, softmaxBackward } from './MatMul'

interface Cache {
  X:      number[][]   // seqLen × d_model  (input)
  Q:      number[][]   // seqLen × d_k
  K:      number[][]   // seqLen × d_k
  V:      number[][]   // seqLen × d_v
  scores: number[][]   // seqLen × seqLen   (pre-softmax)
  attn:   number[][]   // seqLen × seqLen   (post-softmax)
}

export class AttentionHead {
  readonly d_k: number
  readonly d_v: number

  Wq: WeightMatrix   // d_k × d_model
  Wk: WeightMatrix   // d_k × d_model
  Wv: WeightMatrix   // d_v × d_model

  private cache: Cache | null = null

  constructor(d_model: number, d_k: number, d_v: number) {
    this.d_k = d_k
    this.d_v = d_v
    this.Wq  = new WeightMatrix(d_k, d_model)
    this.Wk  = new WeightMatrix(d_k, d_model)
    this.Wv  = new WeightMatrix(d_v, d_model)
  }

  // ── Forward ───────────────────────────────────────────────────────────────
  // X: seqLen × d_model  →  out: seqLen × d_v

  predict(X: number[][]): number[][] {
    const seqLen = X.length
    const scale  = 1 / Math.sqrt(this.d_k)

    // Project to Q, K, V
    const Q: number[][] = X.map(x =>
      this.Wq.W.map(wq => wq.reduce((s, w, m) => s + w * x[m], 0))
    )
    const K: number[][] = X.map(x =>
      this.Wk.W.map(wk => wk.reduce((s, w, m) => s + w * x[m], 0))
    )
    const V: number[][] = X.map(x =>
      this.Wv.W.map(wv => wv.reduce((s, w, m) => s + w * x[m], 0))
    )

    // Attention scores: scores[i][j] = Q[i] · K[j] * scale
    const scores: number[][] = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: seqLen }, (_, j) =>
        Q[i].reduce((s, q, k) => s + q * K[j][k], 0) * scale
      )
    )

    // Softmax over keys dimension
    const attn = scores.map(row => softmax(row))

    // Weighted sum of values: out[i] = Σ_j attn[i][j] * V[j]
    const out: number[][] = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: this.d_v }, (_, d) =>
        attn[i].reduce((s, a, j) => s + a * V[j][d], 0)
      )
    )

    this.cache = { X, Q, K, V, scores, attn }
    return out
  }

  // ── Backward ──────────────────────────────────────────────────────────────
  // dOut: seqLen × d_v  →  dX: seqLen × d_model
  //
  // Steps:
  //   1. dV    = attn^T @ dOut
  //   2. dAttn = dOut @ V^T            (attention weight gradients)
  //   3. dScores = softmaxBwd(dAttn) / √d_k
  //   4. dQ   = dScores @ K,   dK = dScores^T @ Q
  //   5. dWq  = dQ^T @ X,      dWk = dK^T @ X,  dWv = dV^T @ X
  //   6. dX   = dQ @ Wq  +  dK @ Wk  +  dV @ Wv

  backward(dOut: number[][], lr: number): number[][] {
    const { X, Q, K, V, attn } = this.cache!
    const seqLen  = X.length
    const d_model = X[0].length
    const scale   = 1 / Math.sqrt(this.d_k)

    // 1. dV[j][d] = Σ_i attn[i][j] * dOut[i][d]
    const dV: number[][] = Array.from({ length: seqLen }, (_, j) =>
      Array.from({ length: this.d_v }, (_, d) =>
        attn.reduce((s, a, i) => s + a[j] * dOut[i][d], 0)
      )
    )

    // 2. dAttn[i][j] = dOut[i] · V[j]
    const dAttn: number[][] = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: seqLen }, (_, j) =>
        dOut[i].reduce((s, d, k) => s + d * V[j][k], 0)
      )
    )

    // 3. dScores via softmax backward, scaled
    const dScores = dAttn.map((da, i) =>
      softmaxBackward(da, attn[i]).map(v => v * scale)
    )

    // 4a. dQ[i][k] = Σ_j dScores[i][j] * K[j][k]
    const dQ: number[][] = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: this.d_k }, (_, k) =>
        dScores[i].reduce((s, ds, j) => s + ds * K[j][k], 0)
      )
    )

    // 4b. dK[j][k] = Σ_i dScores[i][j] * Q[i][k]
    const dK: number[][] = Array.from({ length: seqLen }, (_, j) =>
      Array.from({ length: this.d_k }, (_, k) =>
        dScores.reduce((s, ds, i) => s + ds[j] * Q[i][k], 0)
      )
    )

    // 5. Weight gradients
    const dWq: number[][] = Array.from({ length: this.d_k }, (_, k) =>
      Array.from({ length: d_model }, (_, m) =>
        dQ.reduce((s, dq, i) => s + dq[k] * X[i][m], 0)
      )
    )
    const dWk: number[][] = Array.from({ length: this.d_k }, (_, k) =>
      Array.from({ length: d_model }, (_, m) =>
        dK.reduce((s, dk, i) => s + dk[k] * X[i][m], 0)
      )
    )
    const dWv: number[][] = Array.from({ length: this.d_v }, (_, k) =>
      Array.from({ length: d_model }, (_, m) =>
        dV.reduce((s, dv, i) => s + dv[k] * X[i][m], 0)
      )
    )

    this.Wq.update(dWq, lr)
    this.Wk.update(dWk, lr)
    this.Wv.update(dWv, lr)

    // 6. dX[i][m] = dQ[i] · Wq[:,m]  +  dK[i] · Wk[:,m]  +  dV[i] · Wv[:,m]
    const dX: number[][] = Array.from({ length: seqLen }, (_, i) =>
      Array.from({ length: d_model }, (_, m) =>
        dQ[i].reduce((s, dq, k) => s + dq * this.Wq.W[k][m], 0) +
        dK[i].reduce((s, dk, k) => s + dk * this.Wk.W[k][m], 0) +
        dV[i].reduce((s, dv, k) => s + dv * this.Wv.W[k][m], 0)
      )
    )

    return dX
  }

  // Attention weights from the last predict() call — useful for visualization.
  getAttentionWeights(): number[][] | null {
    return this.cache ? this.cache.attn : null
  }
}
