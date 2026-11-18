import { describe, it, expect } from 'vitest'
import { matMul, transpose, softmax, softmaxBackward, WeightMatrix, BiasVector, EmbeddingMatrix } from '../src/MatMul'
import { Adam } from '../src/optimizers'
import { VecAdam } from '../src/core/optim'

describe('matMul', () => {
  it('multiplies two matrices correctly', () => {
    const A = [[1, 2], [3, 4]]
    const B = [[5, 6], [7, 8]]
    const C = matMul(A, B)
    expect(C).toEqual([[19, 22], [43, 50]])
  })

  it('handles non-square matrices', () => {
    const A = [[1, 2, 3], [4, 5, 6]]
    const B = [[7, 8], [9, 10], [11, 12]]
    const C = matMul(A, B)
    expect(C).toEqual([[58, 64], [139, 154]])
  })

  it('throws on incompatible dimensions', () => {
    const A = [[1, 2], [3, 4]]
    const B = [[5, 6, 7], [8, 9, 10], [11, 12, 13]]
    expect(() => matMul(A, B)).toThrow('Incompatible dimensions')
  })
})

describe('transpose', () => {
  it('transposes a matrix', () => {
    const A = [[1, 2, 3], [4, 5, 6]]
    const T = transpose(A)
    expect(T).toEqual([[1, 4], [2, 5], [3, 6]])
  })

  it('transposes a square matrix', () => {
    const A = [[1, 2], [3, 4]]
    const T = transpose(A)
    expect(T).toEqual([[1, 3], [2, 4]])
  })
})

describe('softmax', () => {
  it('returns probabilities that sum to 1', () => {
    const s = softmax([1, 2, 3])
    const sum = s.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(1, 10)
  })

  it('returns uniform distribution for equal inputs', () => {
    const s = softmax([5, 5, 5])
    s.forEach(v => expect(v).toBeCloseTo(1 / 3, 10))
  })

  it('assigns higher probability to larger input', () => {
    const s = softmax([1, 10, 100])
    expect(s[2]).toBeGreaterThan(s[1])
    expect(s[1]).toBeGreaterThan(s[0])
  })
})

describe('softmaxBackward', () => {
  it('computes correct gradient', () => {
    const s = softmax([1, 2, 3])
    const dS = [1, 0, 0]
    const dz = softmaxBackward(dS, s)
    // Jacobian-vector product should satisfy sum(dz) = 0
    const sum = dz.reduce((a, b) => a + b, 0)
    expect(sum).toBeCloseTo(0, 10)
  })
})

describe('WeightMatrix', () => {
  it('initializes with correct dimensions', () => {
    const W = new WeightMatrix(3, 4)
    expect(W.W.length).toBe(3)
    expect(W.W[0].length).toBe(4)
  })

  it('update applies Adam optimizer', () => {
    const W = new WeightMatrix(2, 2)
    const orig = W.W.map(r => [...r])
    const dW = [[0.1, 0.2], [0.3, 0.4]]
    W.update(dW, 0.01)
    // Weights should change
    expect(W.W[0][0]).not.toBe(orig[0][0])
  })

  it('getWeights and setWeights work correctly', () => {
    const W = new WeightMatrix(2, 3)
    const flat = W.getWeights()
    expect(flat.length).toBe(6)
    const newFlat = flat.map((v, i) => v + i)
    W.setWeights(newFlat)
    expect(W.getWeights()).toEqual(newFlat)
  })
})

describe('EmbeddingMatrix', () => {
  it('initializes with correct dimensions', () => {
    const E = new EmbeddingMatrix(10, 4)
    expect(E.W.length).toBe(10)
    expect(E.W[0].length).toBe(4)
  })

  it('get returns a copy of the row', () => {
    const E = new EmbeddingMatrix(5, 3)
    const row = E.get(2)
    expect(row.length).toBe(3)
    expect(row).not.toBe(E.W[2]) // should be a copy
  })

  it('update modifies the row', () => {
    const E = new EmbeddingMatrix(5, 3)
    const orig = E.get(2)
    E.update(2, [0.1, 0.2, 0.3], 0.01)
    const updated = E.get(2)
    expect(updated[0]).not.toBe(orig[0])
  })

  it('getWeights returns flat array', () => {
    const E = new EmbeddingMatrix(5, 3)
    const flat = E.getWeights()
    expect(flat.length).toBe(15) // 5 * 3
    expect(Array.isArray(flat)).toBe(true)
    expect(flat.every(v => isFinite(v))).toBe(true)
  })

  it('setWeights and getWeights roundtrip', () => {
    const E = new EmbeddingMatrix(5, 3)
    const orig = E.getWeights()
    const modified = orig.map(v => v + 0.1)
    E.setWeights(modified)
    expect(E.getWeights()).toEqual(modified)
  })
})

// ── WeightMatrix and BiasVector: exactness against the 0.4.0 scalar Adam ──────
// The reference classes below are the 0.4.0 code, copied as it was: one scalar
// Adam per element. The vectorized state must reproduce them bit for bit.

class RefWeightMatrix {
  W: number[][]
  private opts: Adam[][]

  constructor(init: number[][]) {
    this.W = init.map(r => [...r])
    this.opts = init.map(r => r.map(() => new Adam()))
  }

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

class RefBiasVector {
  values: number[]
  private opts: Adam[]

  constructor(init: number[]) {
    this.values = [...init]
    this.opts = init.map(() => new Adam())
  }

  update(grad: number[], lr: number): void {
    for (let i = 0; i < this.values.length; i++)
      this.values[i] = this.opts[i].step(this.values[i], grad[i], lr)
  }
}

// Deterministic gradients; magnitudes go above and below 0.5 so clipping is exercised.
const gradAt = (step: number, i: number, j: number): number =>
  3 * Math.sin(0.37 * step + 1.3 * i - 0.71 * j + 0.2)

const grid = (rows: number, cols: number, f: (i: number, j: number) => number): number[][] =>
  Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => f(i, j)))

function expectSameBits(actual: number[], expected: number[], label: string): void {
  expect(actual.length, `${label}: length`).toBe(expected.length)
  actual.forEach((v, k) => {
    expect(Object.is(v, expected[k]), `${label}[${k}]: ${v} vs ${expected[k]}`).toBe(true)
  })
}

describe('WeightMatrix and BiasVector against the 0.4.0 scalar Adam', () => {
  const cases = [
    { rows: 3, cols: 3, clip: Infinity },
    { rows: 3, cols: 3, clip: 0.5 },
    { rows: 2, cols: 4, clip: Infinity },
    { rows: 2, cols: 4, clip: 0.5 },
  ]

  it.each(cases)(
    'is bit-identical after 200 interleaved steps ($rows x $cols, clipValue $clip)',
    ({ rows, cols, clip }) => {
      const W = new WeightMatrix(rows, cols)
      const b = new BiasVector(rows)
      const refW = new RefWeightMatrix(W.W)
      const refB = new RefBiasVector(b.values)
      const lr = 0.01
      for (let step = 0; step < 200; step++) {
        const dW = grid(rows, cols, (i, j) => gradAt(step, i, j))
        const db = Array.from({ length: rows }, (_, i) => gradAt(step, i, cols + 1))
        W.update(dW, lr, clip)
        refW.update(dW, lr, clip)
        b.update(db, lr)
        refB.update(db, lr)
        expectSameBits(W.W.flat(), refW.W.flat(), `W after step ${step}`)
        expectSameBits(b.values, refB.values, `bias after step ${step}`)
      }
    },
  )

  it('keeps one VecAdam per row of WeightMatrix and one for BiasVector', () => {
    const W = new WeightMatrix(3, 2)
    const opts = (W as unknown as { opts: unknown[] }).opts
    expect(opts).toHaveLength(3)
    for (const o of opts) expect(o).toBeInstanceOf(VecAdam)
    const b = new BiasVector(4)
    expect((b as unknown as { opt: unknown }).opt).toBeInstanceOf(VecAdam)
  })
})
