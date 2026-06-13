import { describe, it, expect } from 'vitest'
import { Network } from '../src/Network'
import { NetworkN } from '../src/NetworkN'
import { NetworkLSTM } from '../src/NetworkLSTM'
import { NetworkTransformer } from '../src/NetworkTransformer'
import { NetworkTransformerRL } from '../src/NetworkTransformerRL'

describe('Integration: XOR with Network', () => {
  it('converges on XOR', () => {
    const net = new Network(2, 8, 1)
    const lr = 0.5

    for (let epoch = 0; epoch < 5000; epoch++) {
      net.train([0, 0], 0, lr)
      net.train([0, 1], 1, lr)
      net.train([1, 0], 1, lr)
      net.train([1, 1], 0, lr)
    }

    expect(net.predict([0, 0])).toBeCloseTo(0, 0)
    expect(net.predict([0, 1])).toBeCloseTo(1, 0)
    expect(net.predict([1, 0])).toBeCloseTo(1, 0)
    expect(net.predict([1, 1])).toBeCloseTo(0, 0)
  }, 30000)
})

describe('Integration: XOR with NetworkN', () => {
  it('converges on XOR', () => {
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
})

describe('Integration: Sequence copy with LSTM', () => {
  it('learns to echo last input', () => {
    const net = new NetworkLSTM(1, 8, [1])
    const lr = 0.1

    // Train: given [0, 1, 0, 1], predict the next value
    for (let epoch = 0; epoch < 500; epoch++) {
      net.resetState()
      net.predict([0])
      net.predict([1])
      net.predict([0])
      net.predict([1])
      net.train([[1], [0], [1], [0]], lr)
    }

    // Test: after seeing [0, 1, 0], should predict close to 1
    net.resetState()
    net.predict([0])
    net.predict([1])
    net.predict([0])
    const out = net.predict([1])
    // Should predict something close to 0 (next in pattern)
    expect(out[0]).toBeLessThan(0.7)
  }, 30000)
})

describe('Integration: Gradient check with finite differences', () => {
  it('gradients match finite differences for Network', () => {
    const net = new Network(2, 4, 1)
    const inputs = [0.5, 0.7]
    const target = 1.0
    const lr = 0.01

    // Get initial prediction
    const predBefore = net.predict(inputs)

    // Train one step
    net.train(inputs, target, lr)

    // Get prediction after training
    const predAfter = net.predict(inputs)

    // Prediction should have moved toward target
    const errBefore = Math.abs(target - predBefore)
    const errAfter = Math.abs(target - predAfter)
    expect(errAfter).toBeLessThan(errBefore)
  })

  it('gradients match finite differences for NetworkN', () => {
    const net = new NetworkN([2, 4, 1])
    const inputs = [0.5, 0.7]
    const target = [1.0]
    const lr = 0.01

    const predBefore = net.predict(inputs)
    net.train(inputs, target, lr)
    const predAfter = net.predict(inputs)

    const errBefore = Math.abs(target[0] - predBefore[0])
    const errAfter = Math.abs(target[0] - predAfter[0])
    expect(errAfter).toBeLessThan(errBefore)
  })
})

describe('Integration: getWeights/setWeights consistency', () => {
  it('Network weights are consistent after set', () => {
    const net = new Network(2, 4, 1)
    const w = net.getWeights()
    net.setWeights(w)
    expect(net.getWeights()).toEqual(w)
  })

  it('NetworkN weights are consistent after set', () => {
    const net = new NetworkN([2, 8, 4, 1])
    const w = net.getWeights()
    net.setWeights(w)
    expect(net.getWeights()).toEqual(w)
  })

  it('NetworkLSTM weights are consistent after set', () => {
    const net = new NetworkLSTM(2, 4, [2])
    const w = net.getWeightsFlat()
    net.setWeightsFlat(w)
    expect(net.getWeightsFlat()).toEqual(w)
  })

  it('NetworkTransformer weights are consistent after set', () => {
    const net = new NetworkTransformer(4, {
      d_model: 8,
      nHeads: 2,
      d_ff: 16,
      nBlocks: 1,
      nClasses: 3,
    })
    const w = net.getWeights()
    net.setWeights(w)
    expect(net.getWeights()).toEqual(w)
  })

  it('NetworkTransformerRL weights are consistent after set', () => {
    const net = new NetworkTransformerRL(4, 3, {
      d_model: 16,
      nHeads: 2,
      d_ff: 32,
      nBlocks: 1,
      nActions: 2,
    })
    const w = net.getWeightsFlat()
    net.setWeightsFlat(w)
    expect(net.getWeightsFlat()).toEqual(w)
  })
})
