import { describe, it, expect } from 'vitest'
import { LSTMLayer } from '../src/LSTMLayer'

describe('LSTMLayer', () => {
  it('creates with correct dimensions', () => {
    const lstm = new LSTMLayer(3, 4)
    expect(lstm.inputSize).toBe(3)
    expect(lstm.hSize).toBe(4)
  })

  it('predict returns array of hidden size', () => {
    const lstm = new LSTMLayer(3, 4)
    const h = lstm.predict([1, 2, 3])
    expect(h.length).toBe(4)
  })

  it('predict returns finite values', () => {
    const lstm = new LSTMLayer(3, 4)
    const h = lstm.predict([0.5, -0.5, 1.0])
    h.forEach(v => expect(isFinite(v)).toBe(true))
  })

  it('maintains state across calls', () => {
    const lstm = new LSTMLayer(2, 3)
    const h1 = lstm.predict([1, 0])
    const h2 = lstm.predict([0, 1])
    // h2 should be different from h1 because state changed
    const differs = h1.some((v, i) => v !== h2[i])
    expect(differs).toBe(true)
  })

  it('reset clears state', () => {
    const lstm = new LSTMLayer(2, 3)
    lstm.predict([1, 0])
    lstm.predict([0, 1])
    lstm.reset()
    // After reset, h should be all zeros
    lstm.h.forEach(v => expect(v).toBe(0))
    lstm.c.forEach(v => expect(v).toBe(0))
  })

  it('validates input size', () => {
    const lstm = new LSTMLayer(3, 4)
    expect(() => lstm.predict([1, 2])).toThrow()
    expect(() => lstm.predict([1, 2, 3, 4])).toThrow()
  })

  it('validates constructor parameters', () => {
    expect(() => new LSTMLayer(0, 4)).toThrow()
    expect(() => new LSTMLayer(3, 0)).toThrow()
    expect(() => new LSTMLayer(-1, 4)).toThrow()
  })

  it('getWeightsFlat and setWeightsFlat work correctly', () => {
    const lstm = new LSTMLayer(3, 4)
    const w = lstm.getWeightsFlat()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    lstm.setWeightsFlat(newW)
    expect(lstm.getWeightsFlat()).toEqual(newW)
  })

  it('backprop updates weights', () => {
    const lstm = new LSTMLayer(2, 3)
    // Run a few steps
    lstm.predict([1, 0])
    lstm.predict([0, 1])

    const wBefore = lstm.getWeightsFlat()

    // Backprop with dummy gradients
    const dh_seq = [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]
    lstm.backprop(dh_seq, 0.01)

    const wAfter = lstm.getWeightsFlat()
    // Weights should have changed
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
  })
})
