// Default numeric backend: plain loops over Float64Array, row-major.
//
// Layout: a layer weight W has rows = nOut and cols = nIn, and
// W[k * cols + j] is the weight of neuron k for input j.
//
// Every operation writes into a caller-provided buffer (`out` or `G`) and
// returns it. None of them allocates, and none writes outside its buffer.
//
// Summation order is the reference order used by NetworkN, so that a
// vectorised model can reproduce it bit for bit: j increasing in matVec
// (starting from the bias), k increasing in matTVec.

export type Vec = Float64Array

export interface Backend {
  // out[k] = b[k] + sum_j W[k,j] * x[j]             (b may be null)
  matVec(W: Vec, rows: number, cols: number, x: Vec, b: Vec | null, out: Vec): Vec
  // out[j] = sum_k W[k,j] * d[k]                    (W^T d, for backprop)
  matTVec(W: Vec, rows: number, cols: number, d: Vec, out: Vec): Vec
  // G[k,j] += s * d[k] * x[j]                       (accumulates one sample)
  addOuter(G: Vec, rows: number, cols: number, d: Vec, x: Vec, s: number): Vec
  // Batch of n rows. X is n x cols, OUT is n x rows.
  // OUT[i,k] = b[k] + sum_j W[k,j] * X[i,j]
  matMulBatch(W: Vec, rows: number, cols: number, X: Vec, n: number, b: Vec | null, OUT: Vec): Vec
  // D is n x rows, OUT is n x cols. OUT[i,j] = sum_k W[k,j] * D[i,k]
  matMulBatchT(W: Vec, rows: number, cols: number, D: Vec, n: number, OUT: Vec): Vec
  // G[k,j] += s * sum_i D[i,k] * X[i,j]
  addOuterBatch(G: Vec, rows: number, cols: number, D: Vec, X: Vec, n: number, s: number): Vec
}

export const defaultBackend: Backend = {
  matVec(W, rows, cols, x, b, out) {
    for (let k = 0; k < rows; k++) {
      let s = b === null ? 0 : b[k]
      const base = k * cols
      for (let j = 0; j < cols; j++) s += W[base + j] * x[j]
      out[k] = s
    }
    return out
  },

  matTVec(W, rows, cols, d, out) {
    for (let j = 0; j < cols; j++) out[j] = 0
    for (let k = 0; k < rows; k++) {
      const dk = d[k]
      const base = k * cols
      for (let j = 0; j < cols; j++) out[j] += W[base + j] * dk
    }
    return out
  },

  addOuter(G, rows, cols, d, x, s) {
    for (let k = 0; k < rows; k++) {
      const sd = s * d[k]
      const base = k * cols
      for (let j = 0; j < cols; j++) G[base + j] += sd * x[j]
    }
    return G
  },

  matMulBatch(W, rows, cols, X, n, b, OUT) {
    for (let i = 0; i < n; i++) {
      const xBase = i * cols
      const oBase = i * rows
      for (let k = 0; k < rows; k++) {
        let s = b === null ? 0 : b[k]
        const wBase = k * cols
        for (let j = 0; j < cols; j++) s += W[wBase + j] * X[xBase + j]
        OUT[oBase + k] = s
      }
    }
    return OUT
  },

  matMulBatchT(W, rows, cols, D, n, OUT) {
    for (let i = 0; i < n; i++) {
      const dBase = i * rows
      const oBase = i * cols
      for (let j = 0; j < cols; j++) OUT[oBase + j] = 0
      for (let k = 0; k < rows; k++) {
        const dk = D[dBase + k]
        const wBase = k * cols
        for (let j = 0; j < cols; j++) OUT[oBase + j] += W[wBase + j] * dk
      }
    }
    return OUT
  },

  addOuterBatch(G, rows, cols, D, X, n, s) {
    for (let k = 0; k < rows; k++) {
      const gBase = k * cols
      for (let j = 0; j < cols; j++) {
        let acc = 0
        for (let i = 0; i < n; i++) acc += D[i * rows + k] * X[i * cols + j]
        G[gBase + j] += s * acc
      }
    }
    return G
  },
}
