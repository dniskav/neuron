import { describe, it, expect } from 'vitest'
import { Conv1D } from '../src/Conv1D'
import { Adam, SGD } from '../src/optimizers'

describe('Conv1D', () => {
  it('creates with correct dimensions', () => {
    const conv = new Conv1D(10, 3, 4)
    expect(conv.inputLength).toBe(10)
    expect(conv.kernelSize).toBe(3)
    expect(conv.filters).toBe(4)
    expect(conv.inputChannels).toBe(1)
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
    dX.forEach(row => row.forEach(v => expect(isFinite(v)).toBe(true)))
  })

  it('backward with lr updates weights', () => {
    const conv = new Conv1D(10, 3, 4)
    const input = Array.from({ length: 10 }, (_, i) => i)
    conv.forward(input)
    const wBefore = conv.getWeights()
    const dOut = Array.from({ length: 4 }, () => new Array(8).fill(0.1))
    conv.backward(dOut, 0.01)
    const wAfter = conv.getWeights()
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
  })

  it('getWeights and setWeights work correctly', () => {
    const conv = new Conv1D(10, 3, 4)
    const w = conv.getWeights()
    // 4 filters * 3 kernelSize * 1 inputChannels + 4 biases = 16
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

  it('works with Adam optimizer', () => {
    const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new Adam())
    const input = Array.from({ length: 10 }, (_, i) => i)
    conv.forward(input)
    const wBefore = conv.getWeights()
    const dOut = Array.from({ length: 4 }, () => new Array(8).fill(0.1))
    conv.backward(dOut, 0.01)
    const wAfter = conv.getWeights()
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
    expect(wAfter.every(v => isFinite(v))).toBe(true)
  })

  it('Adam produces different updates than SGD', () => {
    const convSGD = new Conv1D(10, 3, 4)
    const convAdam = new Conv1D(10, 3, 4, 1, 'valid', () => new Adam())

    // Copy weights so they start the same
    convSGD.setWeights(convAdam.getWeights())

    const input = Array.from({ length: 10 }, (_, i) => i)
    convSGD.forward(input); convAdam.forward(input)
    const dOut = Array.from({ length: 4 }, () => new Array(8).fill(0.1))
    convSGD.backward(dOut, 0.01)
    convAdam.backward(dOut, 0.01)

    const wSGD = convSGD.getWeights()
    const wAdam = convAdam.getWeights()
    const differs = wSGD.some((v, i) => v !== wAdam[i])
    expect(differs).toBe(true)
  })

  // ── Multi-Channel Tests ─────────────────────────────────────────────────

  describe('multi-channel input', () => {
    it('creates with inputChannels parameter', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      expect(conv.inputChannels).toBe(3)
      expect(conv.inputLength).toBe(10)
      expect(conv.kernelSize).toBe(3)
      expect(conv.filters).toBe(4)
    })

    it('validates inputChannels >= 1', () => {
      expect(() => new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 0)).toThrow()
      expect(() => new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), -1)).toThrow()
    })

    it('kernel shape matches inputChannels', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      // kernels: [filters][kernelSize][inputChannels]
      expect(conv.kernels.length).toBe(4)
      expect(conv.kernels[0].length).toBe(3)
      expect(conv.kernels[0][0].length).toBe(3)
    })

    it('forward processes 3-channel input correctly', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      // Input: [inputLength][inputChannels] = [10][3]
      const input: number[][] = Array.from({ length: 10 }, (_, i) => [i, i * 0.1, i * 0.01])
      const out = conv.forward(input)
      expect(out.length).toBe(4) // filters
      expect(out[0].length).toBe(8) // 10 - 3 + 1
      out.forEach(row => row.forEach(v => expect(isFinite(v)).toBe(true)))
    })

    it('forward validates 2D input dimensions', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      // Wrong number of channels
      const badInput: number[][] = Array.from({ length: 10 }, () => [1, 2])
      expect(() => conv.forward(badInput)).toThrow()

      // Wrong length
      const badLength: number[][] = Array.from({ length: 5 }, () => [1, 2, 3])
      expect(() => conv.forward(badLength)).toThrow()
    })

    it('forward rejects 1D input when inputChannels > 1', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      const input1D = Array.from({ length: 10 }, (_, i) => i)
      expect(() => conv.forward(input1D)).toThrow()
    })

    it('backward computes correct gradients for multi-channel', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      const input: number[][] = Array.from({ length: 10 }, (_, i) => [i, i * 0.1, i * 0.01])
      conv.forward(input)
      const dOut = Array.from({ length: 4 }, () => new Array(8).fill(0.1))
      const dX = conv.backward(dOut)
      expect(dX.length).toBe(10)
      dX.forEach(row => {
        expect(row.length).toBe(3)
        row.forEach(v => expect(isFinite(v)).toBe(true))
      })
    })

    it('backward updates multi-channel kernels', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      const input: number[][] = Array.from({ length: 10 }, (_, i) => [i, i * 0.1, i * 0.01])
      conv.forward(input)
      const wBefore = conv.getWeights()
      const dOut = Array.from({ length: 4 }, () => new Array(8).fill(0.1))
      conv.backward(dOut, 0.01)
      const wAfter = conv.getWeights()
      const changed = wBefore.some((v, i) => v !== wAfter[i])
      expect(changed).toBe(true)
    })

    it('getWeights count is correct for multi-channel', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      const w = conv.getWeights()
      // 4 filters * 3 kernelSize * 3 inputChannels + 4 biases = 40
      expect(w.length).toBe(40)
    })

    it('serialization round-trip works with multi-channel', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      const input: number[][] = Array.from({ length: 10 }, (_, i) => [i, i * 0.1, i * 0.01])
      const out1 = conv.forward(input)

      const w = conv.getWeights()
      const conv2 = new Conv1D(10, 3, 4, 1, 'valid', () => new SGD(), 3)
      conv2.setWeights(w)

      const out2 = conv2.forward(input)
      for (let f = 0; f < 4; f++) {
        for (let pos = 0; pos < 8; pos++) {
          expect(out1[f][pos]).toBeCloseTo(out2[f][pos], 10)
        }
      }
    })

    it('multi-channel with same padding works', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'same', () => new SGD(), 2)
      const input: number[][] = Array.from({ length: 10 }, (_, i) => [i, i * 0.5])
      const out = conv.forward(input)
      expect(out.length).toBe(4)
      expect(out[0].length).toBe(10)
    })

    it('multi-channel with Adam optimizer', () => {
      const conv = new Conv1D(10, 3, 4, 1, 'valid', () => new Adam(), 3)
      const input: number[][] = Array.from({ length: 10 }, (_, i) => [i, i * 0.1, i * 0.01])
      conv.forward(input)
      const wBefore = conv.getWeights()
      const dOut = Array.from({ length: 4 }, () => new Array(8).fill(0.1))
      conv.backward(dOut, 0.01)
      const wAfter = conv.getWeights()
      const changed = wBefore.some((v, i) => v !== wAfter[i])
      expect(changed).toBe(true)
      expect(wAfter.every(v => isFinite(v))).toBe(true)
    })
  })
})
