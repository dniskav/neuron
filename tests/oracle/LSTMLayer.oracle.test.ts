// Gradient-oracle test for LSTMLayer (feature 002, T008).
//
// Template (same structure as NetworkN.oracle.test.ts / Network.oracle.test.ts):
//   1. smoke   - reset + predict over a 3-step sequence gives finite, bounded
//                outputs; backprop() changes the weights and keeps them finite.
//   2. oracle  - expectGradientMatches: the update backprop() really performs
//                follows -dL/dw from central finite differences (cosine ~ 1).
//                Loss = sum over the episode of 0.5 (h_t - target_t)^2.
//   3. learns  - expectLearns: the episode loss on fixed data drops below half.
//
// Convention: backprop(dh_seq, lr) expects dh_seq[t] = -dL/dh_t (the layer
// applies w + (lr/T) * g with g = dL/dw negated internally), so a correct
// layer gives cosine ~ +1 against -dL/dw. The loss is recomputed with
// reset() before each forward pass; the weights are read/written with
// getWeightsFlat/setWeightsFlat. The initial weights come from the seeded
// Math.random in tests/setup.ts, so the run is deterministic.
import { describe, it, expect } from 'vitest'
import { LSTMLayer } from '../../src/LSTMLayer'
import { SGD } from '../../src/optimizers'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT_SIZE = 2
const HIDDEN = 3
// 3-step episode: inputs and per-step hidden targets (fixed, independent of the model).
const SEQ: number[][] = [
  [0.5, -0.3],
  [0.8, 0.1],
  [-0.4, 0.6],
]
const TARGET: number[][] = [
  [0.6, -0.2, 0.1],
  [-0.3, 0.4, 0.5],
  [0.2, 0.0, -0.4],
]
const LR = 0.5

function setup() {
  const layer = new LSTMLayer(INPUT_SIZE, HIDDEN, () => new SGD())

  // Full forward pass over the episode from a clean state.
  const forward = (): number[][] => {
    layer.reset()
    return SEQ.map((x) => layer.predict(x))
  }

  // L = sum_t sum_k 0.5 (h_t[k] - target_t[k])^2
  const loss = (): number =>
    forward().reduce(
      (s, h, t) => s + h.reduce((a, v, k) => a + 0.5 * (v - TARGET[t][k]) ** 2, 0),
      0,
    )

  // One real update: forward, then backprop with dh = -dL/dh.
  const step = (lr: number): void => {
    const hs = forward()
    const dh = hs.map((h, t) => h.map((v, k) => -(v - TARGET[t][k])))
    layer.backprop(dh, lr)
  }

  return {
    layer,
    forward,
    loss,
    step,
    getParams: (): number[] => layer.getWeightsFlat(),
    setParams: (p: number[]): void => layer.setWeightsFlat(p),
  }
}

describe('LSTMLayer oracle', () => {
  it('smoke: bounded finite outputs over the episode and backprop changes the weights', () => {
    const m = setup()
    const hs = m.forward()
    expect(hs).toHaveLength(SEQ.length)
    for (const h of hs) {
      expect(h).toHaveLength(HIDDEN)
      for (const v of h) {
        expect(Number.isFinite(v)).toBe(true)
        // h = o * tanh(c) with o in (0, 1), so |h| < 1.
        expect(Math.abs(v)).toBeLessThan(1)
      }
    }

    const before = m.getParams()
    m.step(LR)
    const after = m.getParams()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: backprop follows -dL/dw from finite differences over 3 steps (BPTT)', () => {
    const m = setup()
    const r = expectGradientMatches({
      getParams: m.getParams,
      setParams: m.setParams,
      loss: m.loss,
      step: (lr) => m.step(lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('expectLearns: the episode loss drops below half', () => {
    const m = setup()
    expectLearns({ loss: m.loss, step: () => m.step(LR), steps: 600 })
  })
})
