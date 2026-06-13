import { describe, it, expect } from 'vitest'
import { MultiHeadAttention } from '../src/MultiHeadAttention'

describe('MultiHeadAttention', () => {
  it('creates with correct dimensions', () => {
    const mha = new MultiHeadAttention(8, 4)
    expect(mha.nHeads).toBe(4)
    expect(mha.d_model).toBe(8)
    expect(mha.d_k).toBe(2)
  })

  it('predict returns correct shape', () => {
    const mha = new MultiHeadAttention(8, 2)
    const X = [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]
    const out = mha.predict(X)
    expect(out.length).toBe(2)
    expect(out[0].length).toBe(8)
  })

  it('predict returns finite values', () => {
    const mha = new MultiHeadAttention(4, 2)
    const X = [[1, 0, 0, 0], [0, 1, 0, 0]]
    const out = mha.predict(X)
    out.forEach(row => row.forEach(v => expect(isFinite(v)).toBe(true)))
  })

  it('getAttentionWeights returns per-head weights', () => {
    const mha = new MultiHeadAttention(4, 2)
    mha.predict([[1, 0, 0, 0], [0, 1, 0, 0]])
    const weights = mha.getAttentionWeights()
    expect(weights.length).toBe(2) // 2 heads
    weights.forEach(w => {
      expect(w).not.toBeNull()
      expect(w!.length).toBe(2)
    })
  })

  it('causal mask propagates to heads', () => {
    const mha = new MultiHeadAttention(4, 2, true)
    mha.predict([[1, 0, 0, 0], [0, 1, 0, 0]])
    const weights = mha.getAttentionWeights()
    // Check causal mask in each head
    weights.forEach(w => {
      expect(w![0][1]).toBeCloseTo(0, 5)
    })
  })

  it('backward returns gradient w.r.t. input', () => {
    const mha = new MultiHeadAttention(4, 2)
    mha.predict([[1, 2, 3, 4], [4, 3, 2, 1]])
    const dOut = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    const dX = mha.backward(dOut, 0.01)
    expect(dX.length).toBe(2)
    expect(dX[0].length).toBe(4)
  })

  it('getWeights and setWeights work correctly', () => {
    const mha = new MultiHeadAttention(4, 2)
    const w = mha.getWeights()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    mha.setWeights(newW)
    expect(mha.getWeights()).toEqual(newW)
  })
})
