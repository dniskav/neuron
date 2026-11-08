// Gradient-oracle test for Network (feature 002, T006).
//
// Template (same structure as NetworkN.oracle.test.ts):
//   1. smoke   - builds, predicts finite outputs in (0, 1), train() returns the squared error.
//   2. oracle  - expectGradientMatches: the update train() really performs follows -dL/dw
//                from central finite differences (cosine ~ 1). Loss = 0.5 (y - t)^2; its
//                scale does not matter for the cosine. Optimizer is the default SGD.
//   3. learns  - expectLearns: MSE on fixed data drops below half after a few epochs.
// Weights are read/written with getWeights/setWeights; a fixed seed (tests/setup.ts)
// makes the initial weights deterministic.
import { describe, it, expect } from 'vitest'
import { Network } from '../../src/Network'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT = [0.5, -0.8, 0.3]
const TARGET = 0.9
const LR = 1e-3

function build(): Network {
  return new Network(3, 4, 1)
}

describe('Network oracle', () => {
  it('smoke: predicts finite outputs and train returns the squared error', () => {
    const net = build()
    const out = net.predict(INPUT)
    expect(out).toHaveLength(1)
    expect(out[0]).toBeGreaterThan(0)
    expect(out[0]).toBeLessThan(1)
    const err = net.train(INPUT, TARGET, LR)
    expect(Number.isFinite(err)).toBe(true)
    expect(err).toBeGreaterThanOrEqual(0)
  })

  it('oracle: train() follows -dL/dw from finite differences', () => {
    const net = build()
    const r = expectGradientMatches({
      getParams: () => net.getWeights(),
      setParams: (p) => net.setWeights(p),
      loss: () => 0.5 * (net.predict(INPUT)[0] - TARGET) ** 2,
      step: (lr) => {
        net.train(INPUT, TARGET, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: MSE on a fixed mini-set drops below half', () => {
    const net = build()
    const data: [number[], number][] = [
      [[0.5, -0.8, 0.3], 0.9],
      [[-0.4, 0.6, 0.2], 0.1],
      [[0.9, 0.1, -0.7], 0.8],
    ]
    const loss = () => data.reduce((s, [x, t]) => s + (net.predict(x)[0] - t) ** 2, 0) / data.length
    expectLearns({
      loss,
      step: () => data.forEach(([x, t]) => net.train(x, t, 0.5)),
      steps: 300,
    })
  })

  // AUDIT I5: Network(nIn, nHid, nOut > 1) builds nOut output neurons, but
  // train(inputs, target: number, lr) takes a single scalar target and only
  // updates outputLayer.neurons[0]. The API cannot express a vector target, so
  // no executable test is written here (it would only assert the bug by hand).
  it.todo('AUDIT I5: with several outputs only the first one is trained (train accepts one scalar target)')
})
