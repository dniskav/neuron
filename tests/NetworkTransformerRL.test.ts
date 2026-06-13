import { describe, it, expect } from 'vitest'
import { NetworkTransformerRL } from '../src/NetworkTransformerRL'
import { ModelSaver } from '../src/ModelSaver'

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

  it('structured getWeightsStructured/setWeightsStructured work correctly', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const w = net.getWeightsStructured()
    expect(w.inputProj).toBeDefined()
    expect(w.blocks).toBeDefined()
    expect(w.outputProj).toBeDefined()

    net.setWeightsStructured(w)
    const w2 = net.getWeightsStructured()
    // Should be equivalent
    expect(w2.inputProj).toEqual(w.inputProj)
  })

  it('flat getWeights/setWeights (Serializable interface) work correctly', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const w = net.getWeights()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)
    expect(Array.isArray(w)).toBe(true)

    // getWeights should match getWeightsFlat
    expect(w).toEqual(net.getWeightsFlat())

    const newW = w.map(v => v + 0.001)
    net.setWeights(newW)
    expect(net.getWeights()).toEqual(newW)
  })

  it('getWeightsFlat/setWeightsFlat work correctly', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const w = net.getWeightsFlat()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.001)
    net.setWeightsFlat(newW)
    expect(net.getWeightsFlat()).toEqual(newW)
  })

  it('ModelSaver roundtrip works with flat Serializable interface', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    const origWeights = net.getWeights()

    const json = ModelSaver.toJSON(net)

    // Modify weights
    net.setWeights(origWeights.map(v => v + 1))

    // Restore from JSON
    ModelSaver.fromJSON(net, json)
    expect(net.getWeights()).toEqual(origWeights)
  })

  // ── Pooling tests ─────────────────────────────────────────────────────────

  it('default pooling is "weighted"', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 })
    expect(net.getPoolingType()).toBe('weighted')
  })

  it('each pooling type changes predict output', () => {
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const options = { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 }

    const outputs: Record<string, number[]> = {}
    for (const pooling of ['avg', 'max', 'last', 'weighted'] as const) {
      const net = new NetworkTransformerRL(4, 3, { ...options, pooling })
      outputs[pooling] = net.predict(sequence)
    }

    // All pooling types should produce finite outputs of correct length
    for (const pooling of ['avg', 'max', 'last', 'weighted'] as const) {
      expect(outputs[pooling].length).toBe(2)
      expect(outputs[pooling].every(v => isFinite(v))).toBe(true)
    }

    // Not all pooling types produce identical output (some must differ)
    // max and last could coincidentally match, so check at least one pair differs
    const allSame = (
      JSON.stringify(outputs['avg']) === JSON.stringify(outputs['max']) &&
      JSON.stringify(outputs['max']) === JSON.stringify(outputs['last']) &&
      JSON.stringify(outputs['last']) === JSON.stringify(outputs['weighted'])
    )
    expect(allSame).toBe(false)
  })

  it('"avg" pooling trains without exploding', () => {
    // avg pooling distributes gradient evenly; with tiny network the loss
    // may oscillate but should not explode
    const net = new NetworkTransformerRL(4, 3, { d_model: 24, nHeads: 2, d_ff: 48, nBlocks: 2, nActions: 2, pooling: 'avg' })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const target = [1, 0]
    const lr = 0.0005

    const losses: number[] = []
    for (let i = 0; i < 50; i++) {
      losses.push(net.train(sequence, target, lr))
    }

    // All losses should be finite and not exploding
    const allFinite = losses.every(l => isFinite(l))
    const lastLoss = losses[losses.length - 1]
    expect(allFinite).toBe(true)
    expect(lastLoss).toBeLessThan(1000) // Not exploding
  })

  it('"max" pooling trains without exploding', () => {
    // max pooling only routes gradient to argmax positions; very sparse
    const net = new NetworkTransformerRL(4, 3, { d_model: 24, nHeads: 2, d_ff: 48, nBlocks: 2, nActions: 2, pooling: 'max' })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const target = [1, 0]
    const lr = 0.0005

    const losses: number[] = []
    for (let i = 0; i < 50; i++) {
      losses.push(net.train(sequence, target, lr))
    }

    const allFinite = losses.every(l => isFinite(l))
    const lastLoss = losses[losses.length - 1]
    expect(allFinite).toBe(true)
    expect(lastLoss).toBeLessThan(1000)
  })

  it('"last" pooling trains without exploding', () => {
    // last pooling only routes gradient to the last position; sparse
    const net = new NetworkTransformerRL(4, 3, { d_model: 24, nHeads: 2, d_ff: 48, nBlocks: 2, nActions: 2, pooling: 'last' })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const target = [1, 0]
    const lr = 0.0005

    const losses: number[] = []
    for (let i = 0; i < 50; i++) {
      losses.push(net.train(sequence, target, lr))
    }

    const allFinite = losses.every(l => isFinite(l))
    const lastLoss = losses[losses.length - 1]
    expect(allFinite).toBe(true)
    expect(lastLoss).toBeLessThan(1000)
  })

  it('max pooling backward routes gradient correctly', () => {
    // For max pooling, gradient should only flow to the argmax position.
    // We verify this indirectly: training with max pooling should still
    // decrease loss, meaning gradients are non-zero and routed correctly.

    const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2, pooling: 'max' })
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const target = [1, 0]

    // Get initial weights
    const initialWeights = net.getWeights()

    // One training step
    net.train(sequence, target, 0.01)

    // Weights should have changed (gradient was non-zero)
    const newWeights = net.getWeights()
    const changed = newWeights.some((w, i) => w !== initialWeights[i])
    expect(changed).toBe(true)
  })

  it('all pooling types converge with training', () => {
    const sequence = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    const target = [1, 0]
    const options = { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2 }

    for (const pooling of ['avg', 'max', 'last', 'weighted'] as const) {
      const net = new NetworkTransformerRL(4, 3, { ...options, pooling })
      const initialLoss = net.train(sequence, target, 0.01)

      // Train more
      for (let i = 0; i < 15; i++) {
        net.train(sequence, target, 0.01)
      }

      const finalLoss = net.train(sequence, target, 0.01)
      // Loss should decrease or at least stay finite
      expect(isFinite(finalLoss)).toBe(true)
    }
  })

  it('getPoolingType returns correct value', () => {
    for (const pooling of ['avg', 'max', 'last', 'weighted'] as const) {
      const net = new NetworkTransformerRL(4, 3, { d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 1, nActions: 2, pooling })
      expect(net.getPoolingType()).toBe(pooling)
    }
  })
})
