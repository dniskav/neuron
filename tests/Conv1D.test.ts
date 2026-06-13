import { describe, it, expect } from 'vitest'
import { Conv1D } from '../src/Conv1D'

describe('Conv1D', () => {
  it('creates with correct dimensions', () => {
    const conv = new Conv1D(10, 3, 4)
    expect(conv.inputLength).toBe(10)
    expect(conv.kernelSize).toBe(3)
    expect(conv.filters).toBe(4)
  })

  it('forward returns correct shape with valid padding', () => {
    const conv = new Conv1D(10, 3, 4)
    const input = Array.from({ length: 10 }, (_, i) => i)
    const out = conv.forward(input)
    expect(out.length).toBe(4) // filters
    expect(out[0].length).toBe(8) // 10 - 3 + 1
  })

  it('forward returns correct shape with same padding', () => {
    const conv = new Conv1D(10, 3, 4, 1, 'same')
    const input = Array.from({ length: 10 }, (_, i) => i)
    const out = conv.forward(input)
    expect(out.length).toBe(4) // filters
    expect(out[0].length).toBe(10) // same as input
  })

  it('forward returns finite values', () => {
    const conv = new Conv1D(10, 3, 4)
    const input = Array.from({ length: 10 }, (_, i) => i * 0.1)
    const out = conv.forward(input)
    out.forEach(row => row.forEach(v => expect(isFinite(v)).toBe(true)))
  })

  it('forward validates input length', () => {
    const conv = new Conv1D(10, 3, 4)
    expect(() => conv.forward([1, 2, 3])).toThrow()
  })

  it('getOutputLength returns correct length', () => {
    const conv1 = new Conv1D(10, 3, 4)
    expect(conv1.getOutputLength()).toBe(8)

    const conv2 = new Conv1D(10, 3, 4, 1, 'same')
    expect(conv2.getOutputLength()).toBe(10)
  })

  it('backward returns gradient w.r.t. input', () => {
    const conv = new Conv1D(10, 3, 4)
    const input = Array.from({ length: 10 }, (_, i) => i)
    conv.forward(input)
    const dOut = Array.from({ length: 4 }, () => new Array(8).fill(0.1))
    const dX = conv.backward(dOut)
    expect(dX.length).toBe(10)
    expect(dX.every(v => isFinite(v))).toBe(true)
  })

  it('getWeights and setWeights work correctly', () => {
    const conv = new Conv1D(10, 3, 4)
    const w = conv.getWeights()
    // 4 filters * 3 kernelSize * 1 + 4 biases = 16
    expect(w.length).toBe(16)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    conv.setWeights(newW)
    expect(conv.getWeights()).toEqual(newW)
  })

  it('constructor validates parameters', () => {
    expect(() => new Conv1D(0, 3, 4)).toThrow()
    expect(() => new Conv1D(10, 0, 4)).toThrow()
    expect(() => new Conv1D(10, 3, 0)).toThrow()
    expect(() => new Conv1D(2, 3, 4)).toThrow() // kernelSize > inputLength with valid padding
  })
})
