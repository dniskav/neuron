// Gradient-oracle test for NetworkLSTM (feature 002, T017).
//
// Template (same structure as LSTMLayer.oracle.test.ts and GRU.oracle.test.ts):
//   1. smoke   - a 3-step episode gives finite outputs in (0, 1) (sigmoid dense
//                output) and train() moves the weights.
//   2. oracle  - expectGradientMatches: the update train() really performs follows
//                -dL/dw from central finite differences (cosine > 0.999). Loss is
//                L = sum_t sum_k 0.5 (y_t,k - target_t,k)^2 over the episode.
//                train() averages the dense gradients and the LSTM BPTT step by T
//                (a uniform scale), so only the direction is compared (Q9).
//   3. learns  - expectLearns: the episode loss on fixed data drops below half.
//
// Convention: train(targets, lr) receives the targets, not the gradient. It computes
// (target - y) * sigma'(y), which is -dL/dy * dy/dz, and the layers apply w + lr * g,
// so a correct network gives cosine ~ +1 against -dL/dw. The loss is recomputed with
// resetState() before every forward pass. Weights are read/written with
// getWeightsFlat/setWeightsFlat. The initial weights come from the seeded Math.random
// in tests/setup.ts, so the run is deterministic.
//
// Architecture NetworkLSTM(2, 3, [4, 2]): LSTMLayer(2 -> 3) -> Layer(3 -> 4) -> Layer(4 -> 2).
// Two dense layers exercise both the output-layer path and the hidden-dense derivative path.
import { describe, it, expect } from 'vitest'
import { NetworkLSTM } from '../../src/NetworkLSTM'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT_SIZE = 2
const HIDDEN = 3
const DENSE = [4, 2]
const OUT = 2
// 3-step episode: inputs and per-step targets (fixed, independent of the model).
const SEQ: number[][] = [
  [0.5, -0.3],
  [0.8, 0.1],
  [-0.4, 0.6],
]
// Targets away from the initial sigmoid outputs (~0.5), so that fitting them is a real
// reduction of the loss and not only the constant-output floor (the variance of the targets).
const TARGET: number[][] = [
  [0.9, 0.2],
  [0.8, 0.1],
  [0.9, 0.3],
]
const LR = 0.5

function setup() {
  const net = new NetworkLSTM(INPUT_SIZE, HIDDEN, DENSE)

  // Full forward pass over the episode from a clean state; returns the outputs per step.
  const forward = (): number[][] => {
    net.resetState()
    return SEQ.map((x) => net.predict(x))
  }

  // L = sum_t sum_k 0.5 (y_t[k] - target_t[k])^2
  const loss = (): number =>
    forward().reduce(
      (s, y, t) => s + y.reduce((a, v, k) => a + 0.5 * (v - TARGET[t][k]) ** 2, 0),
      0,
    )

  // One real update: forward (fills the trajectory), then train() with the targets.
  const step = (lr: number): void => {
    forward()
    net.train(TARGET, lr)
  }

  return {
    net,
    forward,
    loss,
    step,
    getParams: (): number[] => net.getWeightsFlat(),
    setParams: (p: number[]): void => net.setWeightsFlat(p),
  }
}

describe('NetworkLSTM oracle', () => {
  it('smoke: finite sigmoid outputs over the episode and train() moves the weights', () => {
    const m = setup()
    const ys = m.forward()
    expect(ys).toHaveLength(SEQ.length)
    for (const y of ys) {
      expect(y).toHaveLength(OUT)
      for (const v of y) {
        expect(Number.isFinite(v)).toBe(true)
        expect(v).toBeGreaterThan(0)
        expect(v).toBeLessThan(1)
      }
    }

    const before = m.getParams()
    m.step(LR)
    const after = m.getParams()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: train() follows -dL/dw from finite differences over the episode (BPTT)', () => {
    const m = setup()
    const r = expectGradientMatches({
      getParams: m.getParams,
      setParams: m.setParams,
      loss: m.loss,
      step: (lr) => m.step(lr),
      lr: LR,
      resetRng: () => m.net.resetState(),
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('expectLearns: the episode loss drops below half on fixed data', () => {
    const m = setup()
    expectLearns({ loss: m.loss, step: () => m.step(LR), steps: 300 })
  })
})
