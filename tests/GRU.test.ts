import { describe, it, expect } from 'vitest'
import { GRULayer } from '../src/GRU'

describe('GRULayer', () => {
  it('creates with correct dimensions', () => {
    const gru = new GRULayer(3, 4)
    expect(gru.inputSize).toBe(3)
    expect(gru.hSize).toBe(4)
  })

  it('predict returns array of hidden size', () => {
    const gru = new GRULayer(3, 4)
    const h = gru.predict([1, 2, 3])
    expect(h.length).toBe(4)
  })

  it('predict returns finite values', () => {
    const gru = new GRULayer(3, 4)
    const h = gru.predict([0.5, -0.5, 1.0])
    h.forEach(v => expect(isFinite(v)).toBe(true))
  })

  it('maintains state across calls', () => {
    const gru = new GRULayer(2, 3)
    const h1 = gru.predict([1, 0])
    const h2 = gru.predict([0, 1])
    const differs = h1.some((v, i) => v !== h2[i])
    expect(differs).toBe(true)
  })

  it('reset clears state', () => {
    const gru = new GRULayer(2, 3)
    gru.predict([1, 0])
    gru.predict([0, 1])
    gru.reset()
    gru.h.forEach(v => expect(v).toBe(0))
  })

  it('validates input size', () => {
    const gru = new GRULayer(3, 4)
    expect(() => gru.predict([1, 2])).toThrow()
    expect(() => gru.predict([1, 2, 3, 4])).toThrow()
  })

  it('validates constructor parameters', () => {
    expect(() => new GRULayer(0, 4)).toThrow()
    expect(() => new GRULayer(3, 0)).toThrow()
  })

  it('backprop updates weights', () => {
    const gru = new GRULayer(2, 3)
    gru.predict([1, 0])
    gru.predict([0, 1])

    const wBefore = gru.getWeightsFlat()

    const dh_seq = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    gru.backprop(dh_seq, 0.01)

    const wAfter = gru.getWeightsFlat()
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
  })

  it('getWeightsFlat and setWeightsFlat work correctly', () => {
    const gru = new GRULayer(3, 4)
    const w = gru.getWeightsFlat()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    gru.setWeightsFlat(newW)
    expect(gru.getWeightsFlat()).toEqual(newW)
  })

  it('structured getWeights/setWeights work correctly', () => {
    const gru = new GRULayer(3, 4)
    const w = gru.getWeights()
    expect(w.resetGate).toBeDefined()
    expect(w.updateGate).toBeDefined()
    expect(w.newGate).toBeDefined()

    gru.setWeights(w)
    const w2 = gru.getWeights()
    expect(w2.resetGate.W).toEqual(w.resetGate.W)
  })
})
