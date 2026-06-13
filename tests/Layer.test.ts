import { describe, it, expect } from 'vitest'
import { Layer } from '../src/Layer'

describe('Layer', () => {
  it('creates correct number of neurons', () => {
    const layer = new Layer(4, 3)
    expect(layer.neurons.length).toBe(4)
  })

  it('each neuron has correct number of weights', () => {
    const layer = new Layer(4, 3)
    layer.neurons.forEach(n => {
      expect(n.weights.length).toBe(3)
    })
  })

  it('predict returns array of correct length', () => {
    const layer = new Layer(4, 3)
    const out = layer.predict([1, 2, 3])
    expect(out.length).toBe(4)
  })

  it('predict returns finite values', () => {
    const layer = new Layer(4, 3)
    const out = layer.predict([0.5, -0.5, 1.0])
    out.forEach(v => {
      expect(isFinite(v)).toBe(true)
    })
  })

  it('predict with different inputs gives different outputs', () => {
    const layer = new Layer(2, 3)
    const out1 = layer.predict([1, 0, 0])
    const out2 = layer.predict([0, 1, 0])
    // At least one output should differ
    const differs = out1.some((v, i) => v !== out2[i])
    expect(differs).toBe(true)
  })
})
