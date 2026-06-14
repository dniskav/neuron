import { describe, it, expect } from 'vitest'
import { Network } from '../src/Network'

describe('Network', () => {
  it('predict returns a number[]', () => {
    const net = new Network(2, 4, 1)
    const out = net.predict([0, 1])
    expect(Array.isArray(out)).toBe(true)
    expect(typeof out[0]).toBe('number')
    expect(isFinite(out[0])).toBe(true)
  })

  it('validates input length', () => {
    const net = new Network(2, 4, 1)
    expect(() => net.predict([1])).toThrow()
    expect(() => net.predict([1, 2, 3])).toThrow()
  })

  it('train returns squared error', () => {
    const net = new Network(2, 4, 1)
    const err = net.train([0, 1], 1.0, 0.1)
    expect(err).toBeGreaterThanOrEqual(0)
  })

  it('learns XOR problem', () => {
    const net = new Network(2, 8, 1)
    const lr = 0.5

    // Train for many epochs
    for (let epoch = 0; epoch < 5000; epoch++) {
      net.train([0, 0], 0, lr)
      net.train([0, 1], 1, lr)
      net.train([1, 0], 1, lr)
      net.train([1, 1], 0, lr)
    }

    // Check predictions
    expect(net.predict([0, 0])[0]).toBeCloseTo(0, 0)
    expect(net.predict([0, 1])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 0])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 1])[0]).toBeCloseTo(0, 0)
  }, 30000)

  it('getWeights and setWeights work correctly', () => {
    const net = new Network(2, 4, 1)
    const w = net.getWeights()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    // Modify weights
    const newW = w.map(v => v + 0.1)
    net.setWeights(newW)
    const w2 = net.getWeights()
    expect(w2).toEqual(newW)
  })

  it('setWeights produces same predictions', () => {
    const net = new Network(2, 4, 1)
    const w = net.getWeights()
    const pred1 = net.predict([0.5, 0.5])

    // Set same weights
    net.setWeights(w)
    const pred2 = net.predict([0.5, 0.5])
    expect(pred1[0]).toBeCloseTo(pred2[0], 10)
  })
})
