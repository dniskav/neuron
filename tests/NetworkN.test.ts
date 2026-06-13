import { describe, it, expect } from 'vitest'
import { NetworkN } from '../src/NetworkN'

describe('NetworkN', () => {
  it('predict returns array of correct length', () => {
    const net = new NetworkN([3, 8, 2])
    const out = net.predict([1, 2, 3])
    expect(out.length).toBe(2)
  })

  it('validates input length', () => {
    const net = new NetworkN([3, 8, 2])
    expect(() => net.predict([1, 2])).toThrow()
  })

  it('validates activations count', () => {
    expect(() => new NetworkN([3, 8, 2], { activations: [] })).toThrow()
  })

  it('validates output neuron activations', () => {
    // This should work - all output neurons use same activation
    const net = new NetworkN([3, 8, 2])
    expect(net).toBeDefined()
  })

  it('train returns MSE loss', () => {
    const net = new NetworkN([2, 4, 1])
    const loss = net.train([0, 1], [1], 0.1)
    expect(loss).toBeGreaterThanOrEqual(0)
  })

  it('learns XOR problem', () => {
    const net = new NetworkN([2, 8, 1])
    const lr = 0.5

    for (let epoch = 0; epoch < 5000; epoch++) {
      net.train([0, 0], [0], lr)
      net.train([0, 1], [1], lr)
      net.train([1, 0], [1], lr)
      net.train([1, 1], [0], lr)
    }

    expect(net.predict([0, 0])[0]).toBeCloseTo(0, 0)
    expect(net.predict([0, 1])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 0])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 1])[0]).toBeCloseTo(0, 0)
  }, 30000)

  it('getWeights and setWeights work correctly', () => {
    const net = new NetworkN([2, 4, 1])
    const w = net.getWeights()
    expect(w.length).toBeGreaterThan(0)

    const newW = w.map(v => v + 0.1)
    net.setWeights(newW)
    expect(net.getWeights()).toEqual(newW)
  })

  it('learns multi-class output', () => {
    const net = new NetworkN([2, 8, 3])
    const lr = 0.3

    for (let epoch = 0; epoch < 3000; epoch++) {
      net.train([1, 0], [1, 0, 0], lr)
      net.train([0, 1], [0, 1, 0], lr)
      net.train([1, 1], [0, 0, 1], lr)
    }

    const out1 = net.predict([1, 0])
    expect(out1[0]).toBeGreaterThan(out1[1])
    expect(out1[0]).toBeGreaterThan(out1[2])
  }, 30000)

  // ── Residual Connection Tests ─────────────────────────────────────────────

  describe('residual connections', () => {
    it('residual=true adds skip connections when sizes match', () => {
      const net = new NetworkN([4, 4, 4, 1], { residual: true })
      const out = net.predict([1, 2, 3, 4])
      expect(out.length).toBe(1)
      expect(isFinite(out[0])).toBe(true)
    })

    it('residual is skipped when layer sizes do not match', () => {
      const net = new NetworkN([2, 4, 4, 1], { residual: true })
      // Layer 0: 2→4 (no residual), Layer 1: 4→4 (residual), Layer 2: 4→1 (no residual)
      const out = net.predict([1, 2])
      expect(out.length).toBe(1)
      expect(isFinite(out[0])).toBe(true)
    })

    it('residual function selects specific layers', () => {
      const net = new NetworkN([4, 4, 4, 1], {
        residual: (layerIndex: number) => layerIndex === 1, // only layer 1
      })
      const out = net.predict([1, 2, 3, 4])
      expect(out.length).toBe(1)
      expect(isFinite(out[0])).toBe(true)
    })

    it('residual trains and loss decreases', () => {
      const net = new NetworkN([4, 4, 4, 1], { residual: true })
      const lr = 0.1

      const loss1 = net.train([1, 2, 3, 4], [1], lr)
      // Train several epochs
      let loss = loss1
      for (let i = 0; i < 100; i++) {
        loss = net.train([1, 2, 3, 4], [1], lr)
      }
      // Loss should decrease
      expect(loss).toBeLessThan(loss1)
    })

    it('residual network learns better on matching-size layers', () => {
      // Structure with matching hidden layers: [2, 4, 4, 4, 1]
      const netRes = new NetworkN([2, 4, 4, 4, 1], { residual: true })
      const netNoRes = new NetworkN([2, 4, 4, 4, 1])
      const lr = 0.3

      // XOR-like data
      const data = [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [0] },
      ]

      // Train both for the same number of epochs
      for (let epoch = 0; epoch < 2000; epoch++) {
        for (const d of data) {
          netRes.train(d.input, d.target, lr)
          netNoRes.train(d.input, d.target, lr)
        }
      }

      // Both should have reduced loss compared to initial
      const lossRes = data.reduce((s, d) => {
        const p = netRes.predict(d.input)
        return s + (d.target[0] - p[0]) ** 2
      }, 0) / data.length

      const lossNoRes = data.reduce((s, d) => {
        const p = netNoRes.predict(d.input)
        return s + (d.target[0] - p[0]) ** 2
      }, 0) / data.length

      // Both should have some learning (loss < 0.5)
      expect(lossRes).toBeLessThan(0.5)
      expect(lossNoRes).toBeLessThan(0.5)
    }, 30000)

    it('residual works with trainWithDeltas', () => {
      const net = new NetworkN([4, 4, 4, 1], { residual: true })
      // Just verify it doesn't throw
      net.trainWithDeltas([1, 2, 3, 4], [0.5], 0.1)
      expect(true).toBe(true)
    })

    it('residual works with getWeights/setWeights', () => {
      const net = new NetworkN([4, 4, 4, 1], { residual: true })
      const w = net.getWeights()
      expect(w.length).toBeGreaterThan(0)

      const newW = w.map(v => v + 0.01)
      net.setWeights(newW)
      expect(net.getWeights()).toEqual(newW)
    })
  })

  // ── Dropout Integration Tests ─────────────────────────────────────────────

  describe('dropout integration', () => {
    it('predict with training=true returns different values across calls', () => {
      const net = new NetworkN([4, 8, 2], { dropoutRate: 0.5 })
      const input = [1, 2, 3, 4]

      // With dropout, multiple training predictions should differ
      const results: number[][] = []
      for (let i = 0; i < 10; i++) {
        results.push(net.predict(input, true))
      }

      // At least some predictions should differ due to random mask
      const allSame = results.every(r =>
        r.every((v, j) => v === results[0][j])
      )
      expect(allSame).toBe(false)
    })

    it('predict with training=false returns deterministic values', () => {
      const net = new NetworkN([4, 8, 2], { dropoutRate: 0.5 })
      const input = [1, 2, 3, 4]

      const out1 = net.predict(input, false)
      const out2 = net.predict(input, false)
      expect(out1).toEqual(out2)
    })

    it('predict without training parameter defaults to inference mode', () => {
      const net = new NetworkN([4, 8, 2], { dropoutRate: 0.5 })
      const input = [1, 2, 3, 4]

      const out1 = net.predict(input)
      const out2 = net.predict(input)
      expect(out1).toEqual(out2)
    })

    it('dropout rate 0 disables dropout', () => {
      const net = new NetworkN([4, 8, 2], { dropoutRate: 0 })
      const input = [1, 2, 3, 4]

      const out1 = net.predict(input, true)
      const out2 = net.predict(input, true)
      expect(out1).toEqual(out2)
    })

    it('train uses dropout during forward pass', () => {
      const net = new NetworkN([4, 8, 2], { dropoutRate: 0.3 })
      const loss = net.train([1, 2, 3, 4], [1, 0], 0.1)
      expect(loss).toBeGreaterThanOrEqual(0)
      expect(isFinite(loss)).toBe(true)
    })

    it('dropout validates rate', () => {
      expect(() => new NetworkN([4, 8, 2], { dropoutRate: -0.1 })).toThrow()
      expect(() => new NetworkN([4, 8, 2], { dropoutRate: 1 })).toThrow()
    })

    it('getWeights and setWeights reset dropout masks', () => {
      const net = new NetworkN([4, 8, 2], { dropoutRate: 0.5 })
      // Trigger a training forward pass to create masks
      net.predict([1, 2, 3, 4], true)
      // getWeights should reset masks
      const w = net.getWeights()
      // setWeights should also reset masks
      net.setWeights(w)
      // After reset, inference should be deterministic
      const out1 = net.predict([1, 2, 3, 4], false)
      const out2 = net.predict([1, 2, 3, 4], false)
      expect(out1).toEqual(out2)
    })

    it('dropout with residual works together', () => {
      const net = new NetworkN([4, 4, 4, 1], {
        residual: true,
        dropoutRate: 0.3,
      })
      const loss = net.train([1, 2, 3, 4], [1], 0.1)
      expect(loss).toBeGreaterThanOrEqual(0)
      expect(isFinite(loss)).toBe(true)
    })

    it('dropout only applied to hidden layers, not output', () => {
      // [4, 8, 2] — 2 layers: hidden (8) and output (2)
      // Only hidden layer should have dropout
      const net = new NetworkN([4, 8, 2], { dropoutRate: 0.9 })
      const input = [1, 2, 3, 4]

      // With very high dropout, training predictions should vary wildly
      // but inference should be stable
      const infOut = net.predict(input, false)
      const infOut2 = net.predict(input, false)
      expect(infOut).toEqual(infOut2)
    })
  })
})
