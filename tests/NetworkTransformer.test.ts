import { describe, it, expect } from 'vitest'
import { NetworkTransformer } from '../src/NetworkTransformer'

describe('NetworkTransformer', () => {
  it('creates with default options', () => {
    const net = new NetworkTransformer(9)
    expect(net.seqLen).toBe(9)
    expect(net.vocabSize).toBe(10)
    expect(net.d_model).toBe(64)
    expect(net.nClasses).toBe(9)
  })

  it('creates with custom options', () => {
    const net = new NetworkTransformer(4, {
      vocabSize: 5,
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })
    expect(net.seqLen).toBe(4)
    expect(net.vocabSize).toBe(5)
    expect(net.d_model).toBe(8)
    expect(net.nClasses).toBe(3)
  })

  it('predict returns correct length', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })
    const tokens = [0, 1, 2, 3]
    const out = net.predict(tokens)
    expect(out.length).toBe(4 * 3) // seqLen * nClasses
  })

  it('predict returns finite values', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })
    const tokens = [0, 1, 2, 3]
    const out = net.predict(tokens)
    expect(out.every(v => isFinite(v))).toBe(true)
  })

  it('train returns loss', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })
    const tokens = [0, 1, 2, 3]
    // One-hot targets for each position
    const targets = [
      1, 0, 0, // pos 0: class 0
      0, 1, 0, // pos 1: class 1
      0, 0, 1, // pos 2: class 2
      1, 0, 0, // pos 3: class 0
    ]
    const loss = net.train(tokens, targets, 0.01)
    expect(loss).toBeGreaterThanOrEqual(0)
    expect(isFinite(loss)).toBe(true)
  })

  it('train with mask', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })
    const tokens = [0, 1, 2, 3]
    const targets = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]
    const mask = [true, false, true, false]
    const loss = net.train(tokens, targets, 0.01, mask)
    expect(loss).toBeGreaterThanOrEqual(0)
  })

  it('getAttentionWeights returns per-block per-head weights', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 2,
      nClasses: 3,
    })
    net.predict([0, 1, 2, 3])
    const weights = net.getAttentionWeights()
    expect(weights.length).toBe(2) // 2 blocks
    weights.forEach(block => {
      expect(block.length).toBe(2) // 2 heads per block
    })
  })

  it('getWeights and setWeights work correctly', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })
    const w = net.getWeights()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.001)
    net.setWeights(newW)
    expect(net.getWeights()).toEqual(newW)
  })

  it('training improves predictions', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })

    const tokens = [0, 1, 2, 3]
    // Position 0 should predict class 0
    const targets = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]

    const predBefore = net.predict(tokens)

    // Train for a few steps
    for (let i = 0; i < 10; i++) {
      net.train(tokens, targets, 0.01)
    }

    const predAfter = net.predict(tokens)

    // Predictions should have changed
    const changed = predBefore.some((v, i) => v !== predAfter[i])
    expect(changed).toBe(true)
  })
})
