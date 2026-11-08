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

// ── reuseMask (T015, Q9) ─────────────────────────────────────────────────────
// reuseMask(x) re-applies the mask saved by the last forward(x, true) without
// drawing from Math.random, so a second pass over the same activations sees the
// SAME units dropped (forward, loss, gradient and update share one mask).
describe('Dropout.reuseMask', () => {
  const spyRandom = () => {
    const original = Math.random
    let calls = 0
    Math.random = () => {
      calls++
      return original()
    }
    return { count: () => calls, restore: () => (Math.random = original) }
  }

  it('applies exactly the mask of the last training forward and consumes no Math.random', () => {
    const d = new Dropout(0.5)
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const first = d.forward(x, true)
    expect(first.some((v) => v === 0)).toBe(true) // the case discriminates
    expect(first.some((v) => v !== 0)).toBe(true)
    const spy = spyRandom()
    try {
      expect(d.reuseMask(x)).toEqual(first)
      expect(d.reuseMask(x)).toEqual(first) // repeatable: the mask is not consumed
      expect(spy.count()).toBe(0)
    } finally {
      spy.restore()
    }
  })

  it('applies the saved mask to DIFFERENT values (same positions, same scale)', () => {
    const d = new Dropout(0.5)
    d.forward([1, 1, 1, 1, 1, 1, 1, 1], true)
    const y = [3, -1, 4, 1, -5, 9, 2, 6]
    const mask = d.backward([1, 1, 1, 1, 1, 1, 1, 1])
    expect(d.reuseMask(y)).toEqual(y.map((v, i) => v * mask[i]))
  })

  it('the backward of a reused mask is the same mask', () => {
    const d = new Dropout(0.5)
    const x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    d.forward(x, true)
    const g1 = d.backward(x)
    d.reuseMask(x)
    expect(d.backward(x)).toEqual(g1)
  })

  it('hasMask reflects whether a training mask is saved', () => {
    const d = new Dropout(0.5)
    expect(d.hasMask()).toBe(false)
    d.forward([1, 2, 3], true)
    expect(d.hasMask()).toBe(true)
    d.forward([1, 2, 3], false)
    expect(d.hasMask()).toBe(false)
    d.forward([1, 2, 3], true)
    d.resetMask()
    expect(d.hasMask()).toBe(false)
  })

  it('throws a clear error without a saved mask (new, after inference forward, after resetMask)', () => {
    const d = new Dropout(0.5)
    expect(() => d.reuseMask([1, 2, 3])).toThrow(/Dropout\.reuseMask/)
    d.forward([1, 2, 3], false)
    expect(() => d.reuseMask([1, 2, 3])).toThrow(/Dropout\.reuseMask/)
    d.forward([1, 2, 3], true)
    d.resetMask()
    expect(() => d.reuseMask([1, 2, 3])).toThrow(/Dropout\.reuseMask/)
  })

  it('throws when the saved mask has another length than the input, and keeps the mask', () => {
    const d = new Dropout(0.5)
    const first = d.forward([1, 2, 3, 4], true)
    expect(() => d.reuseMask([1, 2, 3])).toThrow(/Dropout\.reuseMask.*4.*3/)
    expect(d.reuseMask([1, 2, 3, 4])).toEqual(first)
  })

  it('rate 0 has no mask to reuse: it throws (NetworkN never builds a Dropout with rate 0)', () => {
    const d = new Dropout(0)
    d.forward([1, 2], true)
    expect(d.hasMask()).toBe(false)
    expect(() => d.reuseMask([1, 2])).toThrow(/Dropout\.reuseMask/)
  })
})
