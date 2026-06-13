import { describe, it, expect } from 'vitest'
import { Neuron } from '../src/Neuron'

describe('Neuron', () => {
  it('predict returns a number between 0 and 1', () => {
    const n = new Neuron()
    const out = n.predict(1.0)
    expect(out).toBeGreaterThanOrEqual(0)
    expect(out).toBeLessThanOrEqual(1)
  })

  it('predict with zero input', () => {
    const n = new Neuron()
    const out = n.predict(0)
    // sigmoid(bias) should be between 0 and 1
    expect(out).toBeGreaterThanOrEqual(0)
    expect(out).toBeLessThanOrEqual(1)
  })

  it('train updates weights', () => {
    const n = new Neuron()
    const origW = n.weight
    const origB = n.bias
    n.train(1.0, 1.0, 0.5)
    // Weights should have changed
    expect(n.weight).not.toBe(origW)
    expect(n.bias).not.toBe(origB)
  })

  it('learns to approximate a simple function', () => {
    const n = new Neuron()
    // Train to output 1 for input 1
    for (let i = 0; i < 1000; i++) {
      n.train(1.0, 1.0, 0.5)
    }
    const out = n.predict(1.0)
    expect(out).toBeGreaterThan(0.5)
  })

  it('validates inputs', () => {
    const n = new Neuron()
    expect(() => n.predict(NaN)).toThrow()
    expect(() => n.predict(Infinity)).toThrow()
    expect(() => n.train(NaN, 1, 0.1)).toThrow()
  })
})
