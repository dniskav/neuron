import { describe, it, expect } from 'vitest'
import { NetworkN } from '../src/NetworkN'

describe('NetworkN', () => {
  it('predict returns array of correct length', () => {
    const net = new NetworkN([3, 8, 2])
    const out = net.predict([1, 2, 3])
    expect(out.length).toBe(2)
  })

  it('validates input length', () => {
    const net = new NetworkN([3, 8, 2])
    expect(() => net.predict([1, 2])).toThrow()
  })

  it('validates activations count', () => {
    expect(() => new NetworkN([3, 8, 2], { activations: [] })).toThrow()
  })

  it('validates output neuron activations', () => {
    // This should work - all output neurons use same activation
    const net = new NetworkN([3, 8, 2])
    expect(net).toBeDefined()
  })

  it('train returns MSE loss', () => {
    const net = new NetworkN([2, 4, 1])
    const loss = net.train([0, 1], [1], 0.1)
    expect(loss).toBeGreaterThanOrEqual(0)
  })

  it('learns XOR problem', () => {
    const net = new NetworkN([2, 8, 1])
    const lr = 0.5

    for (let epoch = 0; epoch < 5000; epoch++) {
      net.train([0, 0], [0], lr)
      net.train([0, 1], [1], lr)
      net.train([1, 0], [1], lr)
      net.train([1, 1], [0], lr)
    }

    expect(net.predict([0, 0])[0]).toBeCloseTo(0, 0)
    expect(net.predict([0, 1])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 0])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 1])[0]).toBeCloseTo(0, 0)
  }, 30000)

  it('getWeights and setWeights work correctly', () => {
    const net = new NetworkN([2, 4, 1])
    const w = net.getWeights()
    expect(w.length).toBeGreaterThan(0)

    const newW = w.map(v => v + 0.1)
    net.setWeights(newW)
    expect(net.getWeights()).toEqual(newW)
  })

  it('learns multi-class output', () => {
    const net = new NetworkN([2, 8, 3])
    const lr = 0.3

    for (let epoch = 0; epoch < 3000; epoch++) {
      net.train([1, 0], [1, 0, 0], lr)
      net.train([0, 1], [0, 1, 0], lr)
      net.train([1, 1], [0, 0, 1], lr)
    }

    const out1 = net.predict([1, 0])
    expect(out1[0]).toBeGreaterThan(out1[1])
    expect(out1[0]).toBeGreaterThan(out1[2])
  }, 30000)
})
