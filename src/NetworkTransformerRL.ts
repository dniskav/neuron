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
import { WeightMatrix, BiasVector } from './MatMul'
import { validateArray, validate2DArray } from './Validation'

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
  outputBias: BiasVector

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
    this.outputProj = new WeightMatrix(nActions, d_model)
    this.outputBias = new BiasVector(nActions)
  }

  // ── Forward ────────────────────────────────────────────────────────────────
  // sequence: seqLen × inputDim → nActions Q-values

  predict(sequence: number[][]): number[] {
    const h = this._forward(sequence)
    // Pooling: promediar sobre la secuencia, pero dar más peso al último paso
    const pooled = this._pool(h)
    // Output: Q-values
    return this.outputProj.W.map((row, c) =>
      row.reduce((s, w, m) => s + w * pooled[m], this.outputBias.values[c])
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
      row.reduce((s, w, m) => s + w * pooled[m], this.outputBias.values[c])
    )

    // MSE loss: L = (1/n) * Σ (pred[c] - target[c])²
    const n = this.nActions
    let loss = 0
    for (let c = 0; c < n; c++) {
      const diff = pred[c] - target[c]
      loss += diff * diff
    }
    loss /= n

    // Backward: optimizers do `w + lr*g`, so the chain carries -dL/dpred[c]
    // = -2 * (pred[c] - target[c]) / n.
    const dPred = pred.map((p, c) => -2 * (p - target[c]) / n)

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
    this.outputBias.update(dBout, lr)

    // Backprop through transformer blocks
    // dH: gradient w.r.t. pooled output, distributed using the same pooling as _pool()
    let dH: number[][] = this._distributePoolGradient(dPooled)

    for (let b = this.blocks.length - 1; b >= 0; b--)
      dH = this.blocks[b].backward(dH, lr)

    // Backprop into input projection
    // dH[i] es el gradiente w.r.t. projected[i]
    // inputProj: projected[i] = inputProj @ sequence[i]
    // Accumulate over positions, then update once (a single Adam step per call).
    const dInputProj: number[][] = Array.from({ length: this.d_model }, () =>
      new Array(this.inputDim).fill(0)
    )
    for (let i = 0; i < this.seqLen; i++)
      for (let k = 0; k < this.d_model; k++)
        for (let m = 0; m < this.inputDim; m++)
          dInputProj[k][m] += dH[i][k] * sequence[i][m]
    this.inputProj.update(dInputProj, lr)

    return loss
  }

  // Attention weights from every block for visualization.
  getAttentionWeights(): (number[][] | null)[][] {
    return this.blocks.map(b => b.getAttentionWeights())
  }

  // ── Flat weight serialization ─────────────────────────────────────────────
  // Order: inputProj, block0, block1, ..., blockN, outputProj, outputBias.

  // Elemento a elemento: pasar el array como argumentos supera el límite de la VM.
  getWeightsFlat(): number[] {
    const w: number[] = [];
    for (const v of this.inputProj.getWeights()) w.push(v);
    for (const block of this.blocks) for (const v of block.getWeights()) w.push(v);
    for (const v of this.outputProj.getWeights()) w.push(v);
    for (const v of this.outputBias.getWeights()) w.push(v);
    return w;
  }

  // Longitud total de la estructura, en el orden de getWeightsFlat.
  private _expectedFlatLength(): number {
    return this.inputProj.getWeights().length
      + this.blocks.reduce((n, block) => n + block.getWeights().length, 0)
      + this.outputProj.getWeights().length
      + this.outputBias.values.length;
  }

  setWeightsFlat(weights: number[]): void {
    // Validar antes de escribir: un error no deja el modelo a medias.
    validateArray(weights, this._expectedFlatLength(), 'NetworkTransformerRL.setWeightsFlat');
    let idx = 0;
    const inputProjLen = this.inputProj.getWeights().length;
    this.inputProj.setWeights(weights.slice(idx, idx + inputProjLen));
    idx += inputProjLen;
    for (const block of this.blocks) {
      const blockLen = block.getWeights().length;
      block.setWeights(weights.slice(idx, idx + blockLen));
      idx += blockLen;
    }
    const outProjLen = this.outputProj.getWeights().length;
    this.outputProj.setWeights(weights.slice(idx, idx + outProjLen));
    idx += outProjLen;
    this.outputBias.setWeights(weights.slice(idx, idx + this.outputBias.values.length));
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
        b1: [...b.b1.values],
        b2: [...b.b2.values],
      })),
      outputProj: this.outputProj.W.map(r => [...r]),
      outputBias: [...this.outputBias.values],
    }
  }

  // Valida la FORMA de cada matriz/vector contra el modelo actual, sin escribir nada.
  private _validateStructured(data: ReturnType<NetworkTransformerRL['getWeightsStructured']>): void {
    const m = 'NetworkTransformerRL.setWeightsStructured';
    const mat = (arr: unknown, w: WeightMatrix, name: string) =>
      validate2DArray(arr, w.W.length, w.W[0]?.length ?? 0, `${m}(${name})`);
    const vec = (arr: unknown, len: number, name: string) =>
      validateArray(arr, len, `${m}(${name})`);

    if (data === null || typeof data !== 'object') {
      throw new Error(`${m}: expected object, got ${data === null ? 'null' : typeof data}`);
    }
    mat(data.inputProj, this.inputProj, 'inputProj');
    if (!Array.isArray(data.blocks) || data.blocks.length !== this.blocks.length) {
      throw new Error(
        `${m}: expected ${this.blocks.length} blocks, got ${Array.isArray(data.blocks) ? data.blocks.length : typeof data.blocks}`
      );
    }
    data.blocks.forEach((bd, b) => {
      const blk = this.blocks[b];
      const p = `blocks[${b}]`;
      if (bd === null || typeof bd !== 'object' || bd.attn === null || typeof bd.attn !== 'object') {
        throw new Error(`${m}: ${p} is malformed`);
      }
      if (!Array.isArray(bd.attn.heads) || bd.attn.heads.length !== blk.attn.heads.length) {
        throw new Error(
          `${m}: expected ${blk.attn.heads.length} heads in ${p}, got ${Array.isArray(bd.attn.heads) ? bd.attn.heads.length : typeof bd.attn.heads}`
        );
      }
      bd.attn.heads.forEach((hd, h) => {
        const hp = `${p}.heads[${h}]`;
        mat(hd?.Wq, blk.attn.heads[h].Wq, `${hp}.Wq`);
        mat(hd?.Wk, blk.attn.heads[h].Wk, `${hp}.Wk`);
        mat(hd?.Wv, blk.attn.heads[h].Wv, `${hp}.Wv`);
      });
      mat(bd.attn.Wo, blk.attn.Wo, `${p}.attn.Wo`);
      vec(bd.norm1?.gamma, blk.norm1.gamma.length, `${p}.norm1.gamma`);
      vec(bd.norm1?.beta, blk.norm1.beta.length, `${p}.norm1.beta`);
      vec(bd.norm2?.gamma, blk.norm2.gamma.length, `${p}.norm2.gamma`);
      vec(bd.norm2?.beta, blk.norm2.beta.length, `${p}.norm2.beta`);
      mat(bd.ff1, blk.ff1, `${p}.ff1`);
      mat(bd.ff2, blk.ff2, `${p}.ff2`);
      vec(bd.b1, blk.b1.values.length, `${p}.b1`);
      vec(bd.b2, blk.b2.values.length, `${p}.b2`);
    });
    mat(data.outputProj, this.outputProj, 'outputProj');
    vec(data.outputBias, this.outputBias.values.length, 'outputBias');
  }

  setWeightsStructured(data: ReturnType<NetworkTransformerRL['getWeightsStructured']>): void {
    // Validar la forma de TODO antes de escribir: un error deja el modelo intacto.
    this._validateStructured(data);
    this.inputProj.setWeights(data.inputProj.flat())
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
      blk.b1.setWeights(bd.b1)
      blk.b2.setWeights(bd.b2)
    })
    this.outputProj.W = data.outputProj.map(r => [...r])
    this.outputBias.setWeights(data.outputBias)
  }

  // ── Serializable interface (flat array) ────────────────────────────────────
  // These satisfy the Serializable interface from ModelSaver, which requires
  // getWeights(): number[] and setWeights(weights: number[]): void.

  getWeights(): number[] {
    return this.getWeightsFlat()
  }

  setWeights(weights: number[]): void {
    // Valida con el nombre del método que llama el usuario; setWeightsFlat valida de nuevo.
    validateArray(weights, this._expectedFlatLength(), 'NetworkTransformerRL.setWeights')
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
