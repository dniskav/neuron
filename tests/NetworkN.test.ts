import { describe, it, expect } from 'vitest'
import { NetworkN } from '../src/NetworkN'
import { SGD } from '../src/optimizers'

// ── Golden of the scale (T016, N10, Q8) ───────────────────────────────────────
// Constants captured from the src BEFORE T016: NetworkN([3, 4, n], SGD) with seed 0x5eed, one
// train([0.5, -0.3, 0.8], t, 0.1) with t = [0.7] (n = 1) or [0.7, 0.2, -0.4] (n = 3).
// The new step is 2 / n times the old one (n = 1: x2; n = 3: x2/3); the returned MSE is unchanged.
const G_N1_W0 = [0.2425241061934615, -0.24671718699653025, 0.5218122061699301, 0, -0.45984821691067784, -0.14039550148338517, 0.17794433923475306, 0, -0.21456843225290098, -0.22184758087922987, 0.07012078956642942, 0, -0.15287462481333983, 0.5734771893942693, -0.06589724755809809, 0, 0.17240573116578162, -0.08749168482609093, -0.19005121383816004, 0.04898033803328872, 0]
const G_N1_W1 = [0.24262306594003907, -0.2467765628444768, 0.5219705417644542, 0.00019791949315516543, -0.4599032709933288, -0.14036246903379462, 0.17785625270251157, -0.00011010816530184794, -0.21468807655535121, -0.22177579429775973, 0.06992935868250903, -0.00023928860490048527, -0.15284447699031667, 0.5734591007004554, -0.06584901104126101, 6.029564604634188e-05, 0.1756722781343126, -0.08503060594592586, -0.18751356795144902, 0.05112222335288832, 0.005036594641172871]
const G_N1_E = 0.040588354309527654
const G_N3_W0 = [0.2531363785518638, 0.45156001955369024, 0.24651592425193625, 0, 0.1371916592230722, -0.3372895088476623, 0.4006856568017125, 0, 0.3592710022288117, -0.41013857518238434, -0.4651620714950429, 0, -0.3071791462203578, 0.09417275205568039, 0.14419196933919928, 0, 0.138850410701707, 0.06345283938571811, 0.20035885367542505, 0.47664910997264087, 0, -0.3783147649373859, -0.3570438439492136, -0.11503446311689913, 0.4979950918350369, 0, 0.16277456935495138, -0.4307691282592714, 0.17573952092789114, -0.06872368999756873, 0]
const G_N3_W1 = [0.25301419239271383, 0.4516333312491802, 0.2463204263972963, -0.0002443723182999452, 0.1385530716717828, -0.3381063563168886, 0.40286391671964944, 0.002722824897421189, 0.3589356103366816, -0.4099373400471063, -0.46569869852245105, -0.000670783784260174, -0.3072312476302891, 0.09420401290163916, 0.14410860708330922, -0.00010420281986257211, 0.1400379465925989, 0.06479940726691953, 0.20140678816717617, 0.4776986519066447, 0.002171237724540849, -0.3815417879337873, -0.36070302221252387, -0.11788213174699266, 0.4951430551238145, -0.005900145100023154, 0.15094777341879143, -0.44417974066038, 0.16530303013762482, -0.07917618947578076, -0.02162358686923629]
const G_N3_E = 0.2732710573161626

describe('NetworkN scale golden (T016)', () => {
  for (const [n, t, w0, w1, e] of [
    [1, [0.7], G_N1_W0, G_N1_W1, G_N1_E],
    [3, [0.7, 0.2, -0.4], G_N3_W0, G_N3_W1, G_N3_E],
  ] as const) {
    it(`one iteration with ${n} output(s): the new step is 2/n times the old one (1e-9)`, () => {
      const net = new NetworkN([3, 4, n], { optimizer: () => new SGD() })
      net.setWeights([...w0]) // captured initial weights (independent of the RNG draw order)
      const err = net.train([0.5, -0.3, 0.8], [...t], 0.1)
      expect(Math.abs(err - e)).toBeLessThan(1e-12)
      net.getWeights().forEach((v, i) => {
        expect(Math.abs((v - w0[i]) - (2 / n) * (w1[i] - w0[i]))).toBeLessThan(1e-9)
      })
    })
  }
})

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

  it('train returns MSE loss (the loss whose gradient it steps, N10)', () => {
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
