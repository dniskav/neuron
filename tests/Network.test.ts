import { describe, it, expect } from 'vitest'
import { Network } from '../src/Network'
import { seedRandom } from './helpers/seed'

// ── Golden (T007, nOutputs = 1) ───────────────────────────────────────────────
// Constants captured from the src BEFORE T007 with seed 0x5eed (tests/setup.ts):
//   A = Network(3, 4, 1), 3 steps of train(x_s, t_s, 0.5)
//   B = Network(2, 3, 1), 4 steps of train(x_s, t_s, 0.3)
// Tolerance 1e-12. The scalar path must keep producing the same trajectory.
// T016 (N10, Q8): since 0.4.0 train() scales the step by 2 / nOutputs (x2 here), so these old
// trajectories are reproduced by calling train with HALF the lr (a power of 2: exact in
// floating point): lr 0.25 instead of 0.5, lr 0.15 instead of 0.3. Thresholds unchanged.
const GOLD_A_ERRS = [0.16352477613008828, 0.17459284738219838, 0.092577941235036751]
const GOLD_A_W = [0.24613405591855053, -0.24978105331973158, 0.52071712992794128, 0.0011106539634801523, -0.46144867865183276, -0.13910113045129183, 0.17833741443220913, -0.0010872072937272020, -0.21821116996436263, -0.21882724666641509, 0.071066087977429526, -0.0020312529264291126, -0.15193899247468234, 0.57254509064907699, -0.066135795980469009, 0.000072274536663841126, 0.19670666809574783, -0.076984114502970852, -0.17276235059686373, 0.054973900563283003, 0.036410503337469591]
const GOLD_B_ERRS = [0.28425558789512040, 0.29338437593952388, 0.23925965257610726, 0.22406019651853365]
const GOLD_B_W = [0.29667931915122864, -0.30265333401934480, -0.00073262729957948298, 0.63885119084324971, -0.56358102220312367, -0.00050759780425866144, -0.17203929176754851, 0.21761250467594345, -0.00021382686589313711, -0.21041517735102061, -0.21765437885497607, 0.074143941729493790, 0.0080840073309179361]

// Algorithm of Network.train BEFORE T007 (and before the 2/n scale of T016, so it is called with the doubled lr) (scalar target, one output), copied verbatim.
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
    const errs = X.map((x, s) => net.train(x, T[s], 0.25))
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
    const errs = X.map((x, s) => net.train(x, T[s], 0.15))
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
      const eNew = netNew.train(X[s], T[s], 0.25)
      const eOld = oldTrain(netOld, X[s], T[s], 0.5)
      const eArr = netArr.train(X[s], [T[s]], 0.25)
      expect(eNew).toBe(eOld)
      expect(eArr).toBe(eOld)
      expect(netNew.getWeights()).toEqual(netOld.getWeights())
      expect(netArr.getWeights()).toEqual(netOld.getWeights())
    }
  })
})

// ── Golden of the scale (T016, N10, Q8) ───────────────────────────────────────
// Constants captured from the src BEFORE T016: Network(3, 4, n) with seed 0x5eed, one
// train([0.5, -0.3, 0.8], t, 0.1) with t = [0.7] (n = 1) or [0.7, 0.2, -0.4] (n = 3).
// The new step is 2 / n times the old one (n = 1: x2; n = 3: x2/3); the returned error is unchanged.
const G_W1_W0 = [0.39172362242953424, 0.3174042960248416, -0.36168475526371036, 0, -0.30425084502904925, -0.17981409216963884, -0.1989414820289893, 0, -0.5170233007291418, -0.2406247867850135, -0.510084143158857, 0, -0.15453784770090523, 0.022384982696402888, -0.2813588427735337, 0, 0.4420683328062296, -0.48672677110880613, 0.22279948624782264, -0.4052538531832397, 0]
const G_W1_W1 = [0.39203230305796527, 0.31721908764778295, -0.36119086625822067, 0.0006173612568621013, -0.3045881290249195, -0.1796117217721167, -0.1994811364223817, -0.000674567991740471, -0.5168794250740816, -0.2407111121780497, -0.5098539421107605, 0.00028775131012057206, -0.15481663774354662, 0.02255225672198772, -0.2818049068417599, -0.0005575800852827688, 0.44462122815039334, -0.48426936976586504, 0.22480380308965953, -0.4028679032695096, 0.005635994855889261]
const G_W1_E = 0.05110021389235828
const G_W3_W0 = [-0.48805820274378386, -0.22966399949179309, 0.3228561339382224, 0, -0.017407297470028848, -0.1598941990836909, 0.4339710482636922, 0, 0.3322115209177123, 0.12072205186670319, 0.2045425151091562, 0, -0.06034496095577078, -0.13781889652503485, -0.14593256259122492, 0, 0.21906630625016987, 0.4818814138416201, -0.22758114570751786, 0.003309764666482806, 0, 0.43019590806216, -0.2028336450457573, -0.09872845583595335, -0.25043528992682695, 0, 0.062239858554676175, -0.4516724643763155, 0.10766511643305421, -0.17707310989499092, 0]
const G_W3_W1 = [-0.488506980651543, -0.2293947327471376, 0.3221380892858077, -0.0008975558155183147, -0.015928942857947895, -0.16078121185093946, 0.4363364156430217, 0.0029567092241619087, 0.33193548152464875, 0.12088767550254133, 0.2041008520802545, -0.0005520787861271048, -0.0596702896589389, -0.13822369930313397, -0.14485308851629392, 0.0013493425936637591, 0.22075809993734347, 0.48381571194429, -0.22572015971628343, 0.004848401612010831, 0.0032485966161072467, 0.4265287731822597, -0.20702643291002482, -0.10276233224251627, -0.2537704419924879, -0.007041663562154345, 0.051539842479099277, -0.4639062378857222, 0.09589501693829912, -0.18680446199569012, -0.020546261804230026]
const G_W3_E = 0.2651408583353752

describe('Network scale golden (T016)', () => {
  for (const [n, t, w0, w1, e] of [
    [1, 0.7, G_W1_W0, G_W1_W1, G_W1_E],
    [3, [0.7, 0.2, -0.4], G_W3_W0, G_W3_W1, G_W3_E],
  ] as const) {
    it(`one iteration with ${n} output(s): the new step is 2/n times the old one (1e-9)`, () => {
      const net = new Network(3, 4, n)
      net.setWeights([...w0]) // captured initial weights (independent of the RNG draw order)
      const err = net.train([0.5, -0.3, 0.8], t as number | number[], 0.1)
      expect(Math.abs(err - e)).toBeLessThan(1e-12)
      net.getWeights().forEach((v, i) => {
        expect(Math.abs((v - w0[i]) - (2 / n) * (w1[i] - w0[i]))).toBeLessThan(1e-9)
      })
    })
  }
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

  it('train returns squared error (the one-output case of the mean MSE)', () => {
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
