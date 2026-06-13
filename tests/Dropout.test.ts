import { describe, it, expect } from 'vitest'
import { Dropout } from '../src/Dropout'

describe('Dropout', () => {
  it('creates with valid rate', () => {
    const d = new Dropout(0.5)
    expect(d.rate).toBe(0.5)
  })

  it('throws for invalid rate', () => {
    expect(() => new Dropout(-0.1)).toThrow()
    expect(() => new Dropout(1)).toThrow()
  })

  it('forward in inference mode returns unchanged input', () => {
    const d = new Dropout(0.5)
    const x = [1, 2, 3, 4, 5]
    const out = d.forward(x, false)
    expect(out).toEqual(x)
  })

  it('forward in training mode applies mask', () => {
    const d = new Dropout(0.5)
    const x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    const out = d.forward(x, true)
    // Some elements should be zeroed (probabilistic, but very likely with 10 elements)
    const zeros = out.filter(v => v === 0).length
    // With rate=0.5, expect roughly half zeros (but allow some variance)
    expect(zeros).toBeGreaterThan(0)
    // Non-zero elements should be scaled by 1/(1-0.5) = 2
    out.filter(v => v !== 0).forEach(v => {
      expect(v).toBeCloseTo(2, 5)
    })
  })

  it('backward applies same mask', () => {
    const d = new Dropout(0.5)
    const x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    d.forward(x, true)
    const dOut = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    const grad = d.backward(dOut)
    // Gradient should be zero where mask was zero
    expect(grad.length).toBe(10)
  })

  it('rate 0 does nothing', () => {
    const d = new Dropout(0)
    const x = [1, 2, 3, 4, 5]
    const out = d.forward(x, true)
    expect(out).toEqual(x)
  })

  it('getWeights returns empty array', () => {
    const d = new Dropout(0.5)
    expect(d.getWeights()).toEqual([])
  })

  it('resetMask clears mask', () => {
    const d = new Dropout(0.5)
    d.forward([1, 2, 3], true)
    d.resetMask()
    // After reset, backward should return unchanged gradient
    const grad = d.backward([1, 1, 1])
    expect(grad).toEqual([1, 1, 1])
  })
})
