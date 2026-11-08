// Gradient-oracle test for Network (nOutputs = 1, 2 and 3).
//
// Template (same structure as NetworkN.oracle.test.ts):
//   1. smoke   - builds, predicts finite outputs in (0, 1), train() returns the squared error.
//   2. oracle  - expectGradientMatches: the update train() really performs follows -dL/dw
//                from central finite differences (cosine ~ 1). Loss = (y - t)^2, which for one
//                output is the MSE that train() returns; the ratio checks the scale. Optimizer is
//                the default SGD.
//   3. learns  - expectLearns: MSE on fixed data drops below half after a few epochs.
//   4. vector  - nOutputs = 1, 2 and 3 with a vector target: the oracle covers ALL weights
//                (hidden and every output neuron), loss = mean_k (y_k - t_k)^2, the MSE that
//                train() returns (the output delta carries the factor 2 / nOutputs).
//                Robustness: 5 seeds x lr * 0.97 / 1.03 -> cosine > 0.999 and
//                |ratio - 1| < 0.01 in every run (ratio = |delta| / (lr |g|)).
//                "learns" for nOutputs = 2: both outputs get error < 0.5 x initial with
//                5 seeds x lr * 0.7 / 1.0 / 1.3 (see the header of that test for figures).
// Weights are read/written with getWeights/setWeights; a fixed seed (tests/setup.ts)
// makes the initial weights deterministic; the robustness loops reseed explicitly.
import { describe, it, expect } from 'vitest'
import { Network } from '../../src/Network'
import { expectGradientMatches, gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

const INPUT = [0.5, -0.8, 0.3]
const TARGET = 0.9
const LR = 1e-3

function build(): Network {
  return new Network(3, 4, 1)
}

const SEEDS = [1, 2, 3, 4, 5]
const LR_SCALES_ORACLE = [0.97, 1, 1.03]
const TARGETS_BY_OUTPUTS: Record<number, number[]> = {
  1: [0.8],
  2: [0.2, 0.8],
  3: [0.2, 0.8, 0.5],
}

/** Oracle of Network(3, 4, nOut) trained with a vector target, for one seed and one lr. */
function vectorOracle(nOut: number, seed: number, lr: number) {
  seedRandom(seed)
  const net = new Network(3, 4, nOut)
  const target = TARGETS_BY_OUTPUTS[nOut]
  return gradientOracle({
    getParams: () => net.getWeights(),
    setParams: (p) => net.setWeights(p),
    loss: () =>
      target.reduce((s, t, k) => s + (net.predict(INPUT)[k] - t) ** 2, 0) / target.length,
    step: (l) => {
      net.train(INPUT, target, l)
    },
    lr,
  })
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
      loss: () => (net.predict(INPUT)[0] - TARGET) ** 2,
      step: (lr) => {
        net.train(INPUT, TARGET, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
    expect(Math.abs(r.ratio - 1)).toBeLessThan(0.01)
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

  for (const nOut of [1, 2, 3]) {
    it(`oracle (nOutputs = ${nOut}): vector-target train() follows -dL/dw on all weights, 5 seeds x lr x0.97/x1.03`, () => {
      for (const seed of SEEDS) {
        for (const s of LR_SCALES_ORACLE) {
          const r = vectorOracle(nOut, seed, LR * s)
          expect(r.cosine, `seed ${seed}, lr x${s}`).toBeGreaterThan(0.999)
          expect(Math.abs(r.ratio - 1), `seed ${seed}, lr x${s}, ratio ${r.ratio}`).toBeLessThan(0.01)
        }
      }
    })
  }

  // The step of the HIDDEN weights alone is also the gradient of the mean MSE. The hidden
  // delta derives from the output delta, so scaling only the output layer (not the delta that
  // flows back) would leave the output ratio right and the hidden one wrong. Slice = the first
  // (3 + 1) * 4 weights; ratio = (delta . -g) / (lr |g|^2) on that slice, n = 1 and 3.
  for (const nOut of [1, 3]) {
    it(`scale (N10): the hidden-weights step has ratio in [0.99, 1.01] (nOutputs = ${nOut})`, () => {
      for (const seed of SEEDS) {
        seedRandom(seed)
        const net = new Network(3, 4, nOut)
        const target = TARGETS_BY_OUTPUTS[nOut]
        const mse = () => target.reduce((s, t, k) => s + (net.predict(INPUT)[k] - t) ** 2, 0) / nOut
        const w0 = net.getWeights()
        const nHidden = (3 + 1) * 4
        const eps = 1e-6
        const g: number[] = []
        for (let i = 0; i < nHidden; i++) {
          const wp = [...w0]; wp[i] += eps
          const wm = [...w0]; wm[i] -= eps
          net.setWeights(wp); const lp = mse()
          net.setWeights(wm); const lm = mse()
          g.push(-(lp - lm) / (2 * eps)) // -dL/dw
        }
        net.setWeights(w0)
        net.train(INPUT, target, LR)
        const delta = net.getWeights().slice(0, nHidden).map((v, i) => v - w0[i])
        const dot = delta.reduce((s, v, i) => s + v * g[i], 0)
        const ratio = dot / (LR * g.reduce((s, v) => s + v * v, 0))
        expect(ratio, `seed ${seed}, ratio ${ratio}`).toBeGreaterThan(0.99)
        expect(ratio, `seed ${seed}, ratio ${ratio}`).toBeLessThan(1.01)
      }
    })
  }

  // Figures (seeds 1..5, lr x0.7 / x1.0 / x1.3, 300 epochs, 3 samples):
  // worst final/initial MSE ratio = 0.0158 (output 0) and 0.0694 (output 1).
  // Threshold is 0.5; the robustness rule requires the worst case below 0.7 x 0.5 = 0.35.
  // Oracle (nOutputs 2 and 3, seeds 1..5, lr x0.97/x1.03): min cosine 0.9999999999999997,
  // max |ratio - 1| 1.7e-11.
  it('learns (nOutputs = 2): both outputs drop below half their initial MSE, 5 seeds x lr x0.7/x1.3', () => {
    const data: [number[], number[]][] = [
      [[0.5, -0.8, 0.3], [0.9, 0.2]],
      [[-0.4, 0.6, 0.2], [0.1, 0.7]],
      [[0.9, 0.1, -0.7], [0.8, 0.4]],
    ]
    const perOutputMse = (net: Network): number[] => {
      const sums = [0, 0]
      for (const [x, t] of data) {
        const p = net.predict(x)
        sums[0] += (p[0] - t[0]) ** 2
        sums[1] += (p[1] - t[1]) ** 2
      }
      return sums.map((s) => s / data.length)
    }
    for (const seed of SEEDS) {
      for (const s of [0.7, 1, 1.3]) {
        seedRandom(seed)
        const net = new Network(3, 4, 2)
        const before = perOutputMse(net)
        for (let epoch = 0; epoch < 300; epoch++) {
          for (const [x, t] of data) net.train(x, t, 0.5 * s)
        }
        const after = perOutputMse(net)
        for (let k = 0; k < 2; k++) {
          expect(Number.isFinite(after[k]), `seed ${seed}, lr x${s}, output ${k}`).toBe(true)
          expect(
            after[k] < 0.5 * before[k],
            `seed ${seed}, lr x${s}, output ${k}: initial ${before[k]}, final ${after[k]}`,
          ).toBe(true)
        }
      }
    }
  })
})
