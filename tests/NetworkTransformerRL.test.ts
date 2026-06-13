import { describe, it, expect } from 'vitest'
import { NetworkTransformerRL } from '../src/NetworkTransformerRL'

describe('NetworkTransformerRL', () => {
  it('creates with default options', () => {
    const net = new NetworkTransformerRL(4, 3)
    expect(net.seqLen).toBe(4)
    expect(net.inputDim).toBe(3)
    expect(net.d_model).toBe(32)
    expect(net.nActions).toBe(2)
  })

  it('creates with custom options', () => {
    const net = new NetworkTransformerRL(4, 3, {
      d_model: 16,
      nHeads: 2,
      d_ff: 32,
      nBlocks: 1,
      nActions: 4,
    })
    expect(net.d_model).toBe(16)
    expect(net.nActions).toBe(4)
  })

  it('predict returns correct length', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 4 })
    const sequence = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    const out = net.predict(sequence)
    expect(out.length).toBe(4)
  })

  it('predict returns finite values', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const out = net.predict(sequence)
    expect(out.every(v => isFinite(v))).toBe(true)
  })

  it('train returns loss', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const target = [1, 0]
    const loss = net.train(sequence, target, 0.01)
    expect(loss).toBeGreaterThanOrEqual(0)
    expect(isFinite(loss)).toBe(true)
  })

  it('causal mask is applied', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    net.predict(sequence)
    const weights = net.getAttentionWeights()
    // Check causal mask in each block's heads
    weights.forEach(block => {
      block.forEach(head => {
        if (head) {
          // Position 0 should not attend to position 1
          expect(head[0][1]).toBeCloseTo(0, 5)
        }
      })
    })
  })

  it('training improves predictions', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const target = [1, 0]

    const predBefore = net.predict(sequence)

    // Train for a few steps
    for (let i = 0; i < 10; i++) {
      net.train(sequence, target, 0.01)
    }

    const predAfter = net.predict(sequence)

    // Predictions should have changed
    const changed = predBefore.some((v, i) => v !== predAfter[i])
    expect(changed).toBe(true)
  })

  it('structured getWeights/setWeights work correctly', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const w = net.getWeights()
    expect(w.inputProj).toBeDefined()
    expect(w.blocks).toBeDefined()
    expect(w.outputProj).toBeDefined()

    net.setWeights(w)
    const w2 = net.getWeights()
    // Should be equivalent
    expect(w2.inputProj).toEqual(w.inputProj)
  })

  it('flat getWeightsFlat/setWeightsFlat work correctly', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const w = net.getWeightsFlat()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.001)
    net.setWeightsFlat(newW)
    expect(net.getWeightsFlat()).toEqual(newW)
  })
})
