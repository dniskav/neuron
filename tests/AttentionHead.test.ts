import { describe, it, expect } from 'vitest'
import { AttentionHead } from '../src/AttentionHead'

describe('AttentionHead', () => {
  it('creates with correct dimensions', () => {
    const head = new AttentionHead(8, 4, 4)
    expect(head.d_k).toBe(4)
    expect(head.d_v).toBe(4)
  })

  it('predict returns correct shape', () => {
    const head = new AttentionHead(8, 4, 4)
    const X = [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]
    const out = head.predict(X)
    expect(out.length).toBe(2)
    expect(out[0].length).toBe(4)
  })

  it('predict returns finite values', () => {
    const head = new AttentionHead(4, 2, 2)
    const X = [[1, 0, 0, 0], [0, 1, 0, 0]]
    const out = head.predict(X)
    out.forEach(row => row.forEach(v => expect(isFinite(v)).toBe(true)))
  })

  it('getAttentionWeights returns null before predict', () => {
    const head = new AttentionHead(4, 2, 2)
    expect(head.getAttentionWeights()).toBeNull()
  })

  it('getAttentionWeights returns matrix after predict', () => {
    const head = new AttentionHead(4, 2, 2)
    head.predict([[1, 0, 0, 0], [0, 1, 0, 0]])
    const attn = head.getAttentionWeights()
    expect(attn).not.toBeNull()
    expect(attn!.length).toBe(2)
    expect(attn![0].length).toBe(2)
  })

  it('attention weights sum to 1 for each row', () => {
    const head = new AttentionHead(4, 2, 2)
    head.predict([[1, 2, 3, 4], [4, 3, 2, 1]])
    const attn = head.getAttentionWeights()!
    attn.forEach(row => {
      const sum = row.reduce((s, v) => s + v, 0)
      expect(sum).toBeCloseTo(1, 5)
    })
  })

  it('causal mask prevents attending to future positions', () => {
    const head = new AttentionHead(4, 2, 2, true)
    head.predict([[1, 2, 3, 4], [4, 3, 2, 1]])
    const attn = head.getAttentionWeights()!
    // Position 0 should only attend to position 0
    expect(attn[0][0]).toBeGreaterThan(0)
    expect(attn[0][1]).toBeCloseTo(0, 5)
    // Position 1 can attend to both
    expect(attn[1][0]).toBeGreaterThanOrEqual(0)
    expect(attn[1][1]).toBeGreaterThanOrEqual(0)
  })

  it('backward returns gradient w.r.t. input', () => {
    const head = new AttentionHead(4, 2, 2)
    head.predict([[1, 2, 3, 4], [4, 3, 2, 1]])
    const dOut = [[0.1, 0.2], [0.3, 0.4]]
    const dX = head.backward(dOut, 0.01)
    expect(dX.length).toBe(2)
    expect(dX[0].length).toBe(4)
    expect(dX.flat().every(v => isFinite(v))).toBe(true)
  })

  it('backward updates weights', () => {
    const head = new AttentionHead(4, 2, 2)
    head.predict([[1, 2, 3, 4], [4, 3, 2, 1]])

    const wBefore = head.getWeights()
    head.backward([[0.1, 0.2], [0.3, 0.4]], 0.01)
    const wAfter = head.getWeights()

    // Weights should have changed
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
  })
})
