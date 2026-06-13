import { describe, it, expect } from 'vitest'
import { TransformerBlock } from '../src/TransformerBlock'

describe('TransformerBlock', () => {
  it('creates with correct dimensions', () => {
    const block = new TransformerBlock({ d_model: 8, nHeads: 2, d_ff: 16 })
    expect(block.d_model).toBe(8)
    expect(block.d_ff).toBe(16)
  })

  it('predict returns correct shape', () => {
    const block = new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 8 })
    const X = [[1, 2, 3, 4], [4, 3, 2, 1]]
    const out = block.predict(X)
    expect(out.length).toBe(2)
    expect(out[0].length).toBe(4)
  })

  it('predict returns finite values', () => {
    const block = new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 8 })
    const X = [[1, 0, 0, 0], [0, 1, 0, 0]]
    const out = block.predict(X)
    out.forEach(row => row.forEach(v => expect(isFinite(v)).toBe(true)))
  })

  it('backward returns gradient w.r.t. input', () => {
    const block = new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 8 })
    block.predict([[1, 2, 3, 4], [4, 3, 2, 1]])
    const dOut = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    const dX = block.backward(dOut, 0.01)
    expect(dX.length).toBe(2)
    expect(dX[0].length).toBe(4)
  })

  it('backward updates weights', () => {
    const block = new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 8 })
    block.predict([[1, 2, 3, 4], [4, 3, 2, 1]])

    const wBefore = block.getWeights()
    block.backward([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], 0.01)
    const wAfter = block.getWeights()

    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
  })

  it('getWeights and setWeights work correctly', () => {
    const block = new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 8 })
    const w = block.getWeights()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    block.setWeights(newW)
    expect(block.getWeights()).toEqual(newW)
  })

  it('getAttentionWeights returns per-head weights', () => {
    const block = new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 8 })
    block.predict([[1, 0, 0, 0], [0, 1, 0, 0]])
    const weights = block.getAttentionWeights()
    expect(weights.length).toBe(2) // 2 heads
  })
})
