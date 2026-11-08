// Network with several outputs (T007, I5 / N4, decision Q5).
//
// Network.train(inputs, target: number | number[], lr):
//   - nOutputs = 1: a number or a one-element array (same trajectory as before).
//   - nOutputs > 1: a number[] of length nOutputs; every output neuron is trained.
//   - a number with nOutputs > 1, or a wrong length, throws before touching weights.
//   - returns the mean squared error over the outputs.
// Also: setWeights validates the total length before writing.
import { describe, it, expect } from 'vitest'
import { Network } from '../../src/Network'
import { Trainer } from '../../src/Trainer'
import { seedRandom } from '../helpers/seed'

const INPUT = [0.5, -0.8, 0.3]

function build(nOut: number): Network {
  seedRandom(0x5eed)
  return new Network(3, 4, nOut)
}

describe('Network multi-output: vector targets', () => {
  it('outputs 1..n-1 change after train (they did not before)', () => {
    const net = build(3)
    const before = net.outputLayer.neurons.map((n) => [...n.weights, n.bias])
    net.train(INPUT, [0.2, 0.8, 0.5], 0.5)
    const after = net.outputLayer.neurons.map((n) => [...n.weights, n.bias])
    for (let k = 0; k < 3; k++) {
      const changed = after[k].some((v, i) => v !== before[k][i])
      expect(changed, `output neuron ${k} must be trained`).toBe(true)
    }
  })

  it('hidden deltas sum over all outputs with the ORIGINAL output weights (one step, explicit formula)', () => {
    const net = build(2)
    const x = INPUT
    const t = [0.2, 0.8]
    const lr = 0.5
    // Pre-step values, written out by hand:
    //   d_k = (t_k - p_k) * f'(p_k)                      output deltas
    //   delta_i = (sum_k d_k * W_k[i]) * f'(h_i)         hidden delta, original W
    //   w_ij <- w_ij + lr * delta_i * x_j                 SGD step
    const hOut = net.hiddenLayer.predict(x)
    const p = net.outputLayer.predict(hOut)
    const d = net.outputLayer.neurons.map((o, k) => (t[k] - p[k]) * o.activation.dfn(p[k]))
    const W0 = net.outputLayer.neurons.map((o) => [...o.weights])
    const expected = net.hiddenLayer.neurons.map((n, i) => {
      const sum = d.reduce((s, dk, k) => s + dk * W0[k][i], 0)
      const delta = sum * n.activation.dfn(hOut[i])
      return n.weights.map((w, j) => w + lr * delta * x[j])
    })
    net.train(x, t, lr)
    net.hiddenLayer.neurons.forEach((n, i) =>
      n.weights.forEach((w, j) => expect(Math.abs(w - expected[i][j])).toBeLessThan(1e-12)),
    )
  })

  it('returns the mean squared error over the outputs (computed before the update)', () => {
    const net = build(2)
    const target = [0.2, 0.8]
    const p = net.predict(INPUT)
    const expected = ((target[0] - p[0]) ** 2 + (target[1] - p[1]) ** 2) / 2
    const err = net.train(INPUT, target, 0.5)
    expect(Math.abs(err - expected)).toBeLessThan(1e-12)
  })

  it('a one-element array with nOutputs = 1 behaves like the scalar', () => {
    const a = build(1)
    const b = build(1)
    expect(a.train(INPUT, [0.9], 0.5)).toBe(b.train(INPUT, 0.9, 0.5))
    expect(a.getWeights()).toEqual(b.getWeights())
  })

  it('a number with nOutputs = 2 throws and leaves the weights untouched', () => {
    const net = build(2)
    const w = net.getWeights()
    expect(() => net.train(INPUT, 0.9, 0.5)).toThrow(/Network\.train.*expected 2 targets/)
    expect(net.getWeights()).toEqual(w)
  })

  it('a number[] shorter than nOutputs throws with both lengths', () => {
    const net = build(2)
    const w = net.getWeights()
    expect(() => net.train(INPUT, [0.9], 0.5)).toThrow(/Network\.train: expected 2 targets, got 1/)
    expect(net.getWeights()).toEqual(w)
  })

  it('a number[] longer than nOutputs throws with both lengths', () => {
    const net = build(2)
    const w = net.getWeights()
    expect(() => net.train(INPUT, [0.1, 0.2, 0.3], 0.5)).toThrow(
      /Network\.train: expected 2 targets, got 3/,
    )
    expect(net.getWeights()).toEqual(w)
  })

  it('a NaN inside the target throws and names the index', () => {
    const net = build(2)
    const w = net.getWeights()
    expect(() => net.train(INPUT, [0.1, NaN], 0.5)).toThrow(/Network\.train.*index 1/)
    expect(net.getWeights()).toEqual(w)
  })

  it('a target that is neither a number nor an array throws', () => {
    const net = build(2)
    const w = net.getWeights()
    expect(() => net.train(INPUT, 'x' as unknown as number[], 0.5)).toThrow(/Network\.train/)
    expect(net.getWeights()).toEqual(w)
  })

  it('nOutputs = 1 rejects an array of two elements', () => {
    const net = build(1)
    expect(() => net.train(INPUT, [0.1, 0.2], 0.5)).toThrow(/Network\.train: expected 1 targets, got 2/)
  })
})

describe('Network multi-output: setWeights length', () => {
  it('setWeights with the wrong length throws with both lengths and keeps the weights', () => {
    const net = build(2)
    // (3 + 1) * 4 + (4 + 1) * 2 = 26
    const w = net.getWeights()
    expect(w).toHaveLength(26)
    expect(() => net.setWeights(w.slice(0, 25))).toThrow(/Network\.setWeights.*26.*25/)
    expect(() => net.setWeights([...w, 0])).toThrow(/Network\.setWeights.*26.*27/)
    expect(net.getWeights()).toEqual(w)
  })

  it('round trip of getWeights/setWeights with two outputs is exact', () => {
    const net = build(2)
    const w = net.getWeights()
    net.setWeights(w.map((v) => v + 0.1))
    net.setWeights(w)
    expect(net.getWeights()).toEqual(w)
  })
})

describe('Network multi-output: Trainer compatibility', () => {
  it('new Trainer(new Network(2, 3, 2)) trains for the given epochs', () => {
    seedRandom(0x5eed)
    const net = new Network(2, 3, 2)
    const trainer = new Trainer(net, { epochs: 5 })
    const history = trainer.train({
      inputs: [[0, 1], [1, 0]],
      targets: [[0.2, 0.8], [0.9, 0.1]],
    })
    expect(history).toHaveLength(5)
    expect(history.every((v) => Number.isFinite(v))).toBe(true)
  })
})
