import { describe, it, expect } from 'vitest'
import { NeuronN } from '../src/NeuronN'
import { sigmoid } from '../src/activations'

describe('NeuronN', () => {
  it('predict returns a number', () => {
    const n = new NeuronN(3)
    const out = n.predict([1, 2, 3])
    expect(typeof out).toBe('number')
    expect(isFinite(out)).toBe(true)
  })

  it('predict with sigmoid returns between 0 and 1', () => {
    const n = new NeuronN(2, sigmoid)
    const out = n.predict([0.5, -0.5])
    expect(out).toBeGreaterThanOrEqual(0)
    expect(out).toBeLessThanOrEqual(1)
  })

  it('train updates weights', () => {
    const n = new NeuronN(2)
    const origW = [...n.weights]
    n.train([1, 2], 1.0, 0.1)
    // At least one weight should have changed
    const changed = n.weights.some((w, i) => w !== origW[i])
    expect(changed).toBe(true)
  })

  it('validates input length', () => {
    const n = new NeuronN(3)
    expect(() => n.predict([1, 2])).toThrow()
    expect(() => n.predict([1, 2, 3, 4])).toThrow()
  })

  it('validates input values', () => {
    const n = new NeuronN(2)
    expect(() => n.predict([NaN, 1])).toThrow()
    expect(() => n.predict([1, Infinity])).toThrow()
  })

  it('learns XOR-like pattern with single neuron (limited)', () => {
    const n = new NeuronN(2)
    // Train on (1,1)->1 pattern
    for (let i = 0; i < 500; i++) {
      n.train([1, 1], 1.0, 0.5)
    }
    const out = n.predict([1, 1])
    expect(out).toBeGreaterThan(0.5)
  })
})
