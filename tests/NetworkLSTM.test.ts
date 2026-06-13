import { describe, it, expect } from 'vitest'
import { NetworkLSTM } from '../src/NetworkLSTM'

describe('NetworkLSTM', () => {
  it('creates with correct dimensions', () => {
    const net = new NetworkLSTM(3, 8, [4, 2])
    expect(net.inputSize).toBe(3)
    expect(net.hiddenSize).toBe(8)
  })

  it('predict returns array of correct length', () => {
    const net = new NetworkLSTM(3, 8, [4, 2])
    net.resetState()
    const out = net.predict([1, 2, 3])
    expect(out.length).toBe(2)
  })

  it('validates input size', () => {
    const net = new NetworkLSTM(3, 8, [2])
    net.resetState()
    expect(() => net.predict([1, 2])).toThrow()
  })

  it('resetState clears trajectory', () => {
    const net = new NetworkLSTM(2, 4, [2])
    net.resetState()
    net.predict([1, 0])
    net.predict([0, 1])
    net.resetState()
    // Should be able to predict again without error
    const out = net.predict([1, 0])
    expect(out.length).toBe(2)
  })

  it('train on episode works', () => {
    const net = new NetworkLSTM(2, 4, [2])
    net.resetState()

    // Run a few steps
    net.predict([1, 0])
    net.predict([0, 1])
    net.predict([1, 0])

    // Train with targets for each step
    const targets = [[1, 0], [0, 1], [1, 0]]
    net.train(targets, 0.01)
  })

  it('learns simple sequence pattern', () => {
    const net = new NetworkLSTM(1, 8, [1])
    const lr = 0.1

    // Train to predict the next value in [0, 1, 0, 1]
    for (let epoch = 0; epoch < 2000; epoch++) {
      net.resetState()
      net.predict([0])
      net.predict([1])
      net.predict([0])
      net.predict([1])
      net.train([[1], [0], [1], [0]], lr)
    }

    // Test
    net.resetState()
    net.predict([0])
    net.predict([1])
    net.predict([0])
    const out = net.predict([1])
    // Should predict something close to 0 (next in pattern)
    expect(out[0]).toBeLessThan(0.6)
  }, 30000)

  it('getWeightsFlat and setWeightsFlat work correctly', () => {
    const net = new NetworkLSTM(2, 4, [2])
    const w = net.getWeightsFlat()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    net.setWeightsFlat(newW)
    expect(net.getWeightsFlat()).toEqual(newW)
  })

  it('structured getWeights/setWeights work correctly', () => {
    const net = new NetworkLSTM(2, 4, [2])
    const w = net.getWeights()
    expect(w.lstm).toBeDefined()
    expect(w.dense).toBeDefined()

    net.setWeights(w)
    const w2 = net.getWeights()
    // Should be equivalent
    expect(w2.lstm.forgetGate.W).toEqual(w.lstm.forgetGate.W)
  })
})
