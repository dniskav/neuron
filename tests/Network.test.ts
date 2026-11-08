import { describe, it, expect } from 'vitest'
import { Network } from '../src/Network'
import { seedRandom } from './helpers/seed'

// ── Golden (T007, nOutputs = 1) ───────────────────────────────────────────────
// Constants captured from the src BEFORE T007 with seed 0x5eed (tests/setup.ts):
//   A = Network(3, 4, 1), 3 steps of train(x_s, t_s, 0.5)
//   B = Network(2, 3, 1), 4 steps of train(x_s, t_s, 0.3)
// Tolerance 1e-12. The scalar path must keep producing the same trajectory.
const GOLD_A_ERRS = [0.16352477613008828, 0.17459284738219838, 0.092577941235036751]
const GOLD_A_W = [0.24613405591855053, -0.24978105331973158, 0.52071712992794128, 0.0011106539634801523, -0.46144867865183276, -0.13910113045129183, 0.17833741443220913, -0.0010872072937272020, -0.21821116996436263, -0.21882724666641509, 0.071066087977429526, -0.0020312529264291126, -0.15193899247468234, 0.57254509064907699, -0.066135795980469009, 0.000072274536663841126, 0.19670666809574783, -0.076984114502970852, -0.17276235059686373, 0.054973900563283003, 0.036410503337469591]
const GOLD_B_ERRS = [0.28425558789512040, 0.29338437593952388, 0.23925965257610726, 0.22406019651853365]
const GOLD_B_W = [0.29667931915122864, -0.30265333401934480, -0.00073262729957948298, 0.63885119084324971, -0.56358102220312367, -0.00050759780425866144, -0.17203929176754851, 0.21761250467594345, -0.00021382686589313711, -0.21041517735102061, -0.21765437885497607, 0.074143941729493790, 0.0080840073309179361]

// Algorithm of Network.train BEFORE T007 (scalar target, one output), copied verbatim.
function oldTrain(net: Network, inputs: number[], target: number, lr: number): number {
  const hiddenOut = net.hiddenLayer.predict(inputs)
  const prediction = net.outputLayer.predict(hiddenOut)[0]
  const outputNeuron = net.outputLayer.neurons[0]
  const outputError = target - prediction
  const outputDelta = outputError * outputNeuron.activation.dfn(prediction)
  const hiddenDeltas = net.hiddenLayer.neurons.map((neuron, i) => {
    const hiddenError = outputDelta * outputNeuron.weights[i]
    return hiddenError * neuron.activation.dfn(hiddenOut[i])
  })
  net.hiddenLayer.neurons.forEach((neuron, i) => {
    neuron._update(inputs.map(inp => hiddenDeltas[i] * inp), hiddenDeltas[i], lr)
  })
  outputNeuron._update(hiddenOut.map(h => outputDelta * h), outputDelta, lr)
  return outputError * outputError
}

describe('Network golden (scalar target, nOutputs = 1)', () => {
  it('scalar trajectory matches the captured values of the previous src', () => {
    seedRandom(0x5eed)
    const net = new Network(3, 4, 1)
    const X = [[0.5, -0.8, 0.3], [-0.4, 0.6, 0.2], [0.9, 0.1, -0.7]]
    const T = [0.9, 0.1, 0.8]
    const errs = X.map((x, s) => net.train(x, T[s], 0.5))
    errs.forEach((e, s) => expect(Math.abs(e - GOLD_A_ERRS[s])).toBeLessThan(1e-12))
    const w = net.getWeights()
    expect(w).toHaveLength(GOLD_A_W.length)
    w.forEach((v, i) => expect(Math.abs(v - GOLD_A_W[i])).toBeLessThan(1e-12))
  })

  it('second configuration Network(2, 3, 1) matches the captured values', () => {
    seedRandom(0x5eed)
    const net = new Network(2, 3, 1)
    const X = [[0, 1], [1, 0], [1, 1], [0, 0]]
    const T = [1, 1, 0, 0]
    const errs = X.map((x, s) => net.train(x, T[s], 0.3))
    errs.forEach((e, s) => expect(Math.abs(e - GOLD_B_ERRS[s])).toBeLessThan(1e-12))
    const w = net.getWeights()
    expect(w).toHaveLength(GOLD_B_W.length)
    w.forEach((v, i) => expect(Math.abs(v - GOLD_B_W[i])).toBeLessThan(1e-12))
  })

  it('train(x, t) equals the old algorithm step by step (toEqual) and train(x, [t]) equals both', () => {
    seedRandom(0x5eed)
    const netNew = new Network(3, 4, 1)
    seedRandom(0x5eed)
    const netOld = new Network(3, 4, 1)
    seedRandom(0x5eed)
    const netArr = new Network(3, 4, 1)
    const X = [[0.5, -0.8, 0.3], [-0.4, 0.6, 0.2], [0.9, 0.1, -0.7]]
    const T = [0.9, 0.1, 0.8]
    for (let s = 0; s < X.length; s++) {
      const eNew = netNew.train(X[s], T[s], 0.5)
      const eOld = oldTrain(netOld, X[s], T[s], 0.5)
      const eArr = netArr.train(X[s], [T[s]], 0.5)
      expect(eNew).toBe(eOld)
      expect(eArr).toBe(eOld)
      expect(netNew.getWeights()).toEqual(netOld.getWeights())
      expect(netArr.getWeights()).toEqual(netOld.getWeights())
    }
  })
})

describe('Network', () => {
  it('predict returns a number[]', () => {
    const net = new Network(2, 4, 1)
    const out = net.predict([0, 1])
    expect(Array.isArray(out)).toBe(true)
    expect(typeof out[0]).toBe('number')
    expect(isFinite(out[0])).toBe(true)
  })

  it('validates input length', () => {
    const net = new Network(2, 4, 1)
    expect(() => net.predict([1])).toThrow()
    expect(() => net.predict([1, 2, 3])).toThrow()
  })

  it('train returns squared error', () => {
    const net = new Network(2, 4, 1)
    const err = net.train([0, 1], 1.0, 0.1)
    expect(err).toBeGreaterThanOrEqual(0)
  })

  it('learns XOR problem', () => {
    const net = new Network(2, 8, 1)
    const lr = 0.5

    // Train for many epochs
    for (let epoch = 0; epoch < 5000; epoch++) {
      net.train([0, 0], 0, lr)
      net.train([0, 1], 1, lr)
      net.train([1, 0], 1, lr)
      net.train([1, 1], 0, lr)
    }

    // Check predictions
    expect(net.predict([0, 0])[0]).toBeCloseTo(0, 0)
    expect(net.predict([0, 1])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 0])[0]).toBeCloseTo(1, 0)
    expect(net.predict([1, 1])[0]).toBeCloseTo(0, 0)
  }, 30000)

  it('getWeights and setWeights work correctly', () => {
    const net = new Network(2, 4, 1)
    const w = net.getWeights()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    // Modify weights
    const newW = w.map(v => v + 0.1)
    net.setWeights(newW)
    const w2 = net.getWeights()
    expect(w2).toEqual(newW)
  })

  it('setWeights produces same predictions', () => {
    const net = new Network(2, 4, 1)
    const w = net.getWeights()
    const pred1 = net.predict([0.5, 0.5])

    // Set same weights
    net.setWeights(w)
    const pred2 = net.predict([0.5, 0.5])
    expect(pred1[0]).toBeCloseTo(pred2[0], 10)
  })
})
