// Gradient-oracle test for GRULayer (feature 002, T009).
//
// Template (same structure as NetworkN.oracle.test.ts and Network.oracle.test.ts):
//   1. smoke   - predicts a 3-step sequence with finite outputs in (-1, 1) and
//                backprop() moves the weights.
//   2. oracle  - expectGradientMatches: the update backprop() really performs follows
//                -dL/dw from central finite differences (cosine ~ 1). Loss is
//                L = 0.5 * sum_t sum_i (target_t,i - h_t,i)^2 over the episode;
//                backprop receives dh_t = -dL/dh_t = target_t - h_t (negated
//                convention, as in the LSTM template). Optimizer is the default SGD.
//                The step is averaged over the episode length T (lr / T), so the
//                ratio |delta| / (lr |g|) is 1/T.
//   3. learns  - expectLearns: the same loss, measured outside backprop on fixed
//                data, drops below half after a few episodes.
// Weights are read/written with getWeightsFlat/setWeightsFlat; reset() clears h and
// the trajectory before every loss() and step(). A fixed seed (tests/setup.ts) makes
// the initial weights deterministic.
import { describe, it, expect } from 'vitest'
import { GRULayer } from '../../src/GRU'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const IN = 3
const HID = 4
const T = 3
const INPUTS: number[][] = [
  [0.5, -0.8, 0.3],
  [-0.4, 0.6, 0.2],
  [0.9, 0.1, -0.7],
]
const TARGETS: number[][] = [
  [0.4, -0.2, 0.1, 0.6],
  [-0.3, 0.5, 0.2, -0.1],
  [0.1, 0.1, -0.4, 0.3],
]
const LR = 1e-3

function build(): GRULayer {
  return new GRULayer(IN, HID)
}

/** Runs the whole episode from a clean state and returns the hidden states. */
function run(gru: GRULayer): number[][] {
  gru.reset()
  return INPUTS.map((x) => [...gru.predict(x)])
}

/** L = 0.5 * sum_t sum_i (target - h)^2, measured with predict() only. */
function loss(gru: GRULayer): number {
  const hs = run(gru)
  let s = 0
  for (let t = 0; t < T; t++)
    for (let i = 0; i < HID; i++) s += 0.5 * (TARGETS[t][i] - hs[t][i]) ** 2
  return s
}

describe('GRULayer oracle', () => {
  it('smoke: predicts finite bounded states and backprop moves the weights', () => {
    const gru = build()
    const hs = run(gru)
    expect(hs).toHaveLength(T)
    for (const h of hs) {
      expect(h).toHaveLength(HID)
      for (const v of h) {
        expect(Number.isFinite(v)).toBe(true)
        expect(v).toBeGreaterThan(-1)
        expect(v).toBeLessThan(1)
      }
    }
    const before = gru.getWeightsFlat()
    gru.reset()
    const hs2 = INPUTS.map((x) => [...gru.predict(x)])
    gru.backprop(
      hs2.map((h, t) => h.map((v, i) => TARGETS[t][i] - v)),
      LR,
    )
    const after = gru.getWeightsFlat()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: backprop() follows -dL/dw from finite differences', () => {
    const gru = build()
    const r = expectGradientMatches({
      getParams: () => gru.getWeightsFlat(),
      setParams: (p) => gru.setWeightsFlat(p),
      loss: () => loss(gru),
      step: (lr) => {
        gru.reset()
        const hs = INPUTS.map((x) => [...gru.predict(x)])
        gru.backprop(
          hs.map((h, t) => h.map((v, i) => TARGETS[t][i] - v)),
          lr,
        )
      },
      lr: LR,
      resetRng: () => gru.reset(),
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: the episode loss drops below half after a few hundred steps', () => {
    const gru = build()
    expectLearns({
      loss: () => loss(gru),
      step: () => {
        gru.reset()
        const hs = INPUTS.map((x) => [...gru.predict(x)])
        gru.backprop(
          hs.map((h, t) => h.map((v, i) => TARGETS[t][i] - v)),
          1.0,
        )
      },
      steps: 300,
    })
  })
})
