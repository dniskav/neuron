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
import { expectGradientMatches, gradientOracle } from '../helpers/gradientOracle'
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

// ── Puertas de reset y update (T029) ──────────────────────────────────────────
//
// The cases above start from h = 0 and only 3 steps, and their loss sums over time.
// Here the reset gate changes the loss on purpose: the episode starts from a non-zero
// state H0, lasts 4 steps (h_prev != 0 from t = 1 on) and newGate.W reads h with
// non-negligible weights, so r ⊙ h_prev reaches the candidate n.
//
// The loss is averaged over time, L = (1/T) Σ_t 0.5 Σ_i (target - h)². backprop divides
// its step by T, so the real step is lr·(-dL/dw) and the oracle ratio is 1. Each gate
// is checked on its own (its W block, the rest of the parameters held fixed), because
// a small block barely moves the cosine of the whole vector.
const IN2 = 3
const HID2 = 4
const T2 = 4
const H0 = [0.6, -0.5, 0.4, 0.7]
const INPUTS2: number[][] = [
  [0.5, -0.8, 0.3],
  [-0.4, 0.6, 0.2],
  [0.9, 0.1, -0.7],
  [0.2, 0.7, -0.3],
]
const TARGETS2: number[][] = [
  [0.4, -0.2, 0.1, 0.6],
  [-0.3, 0.5, 0.2, -0.1],
  [0.1, 0.1, -0.4, 0.3],
  [-0.2, 0.3, 0.5, -0.6],
]
const LR2 = 1e-3

function build2(): GRULayer {
  return new GRULayer(IN2, HID2)
}

/** Starts an episode from H0 (not zero), with the trajectory cleared. */
function start2(gru: GRULayer): void {
  gru.reset()
  gru.h = [...H0]
}

/** Runs the 4-step episode from H0 and returns the hidden states. */
function run2(gru: GRULayer): number[][] {
  start2(gru)
  return INPUTS2.map((x) => [...gru.predict(x)])
}

/** L = (1/T) Σ_t 0.5 Σ_i (target - h)², measured with predict() only. */
function meanLoss2(gru: GRULayer): number {
  const hs = run2(gru)
  let s = 0
  for (let t = 0; t < T2; t++)
    for (let i = 0; i < HID2; i++) s += 0.5 * (TARGETS2[t][i] - hs[t][i]) ** 2
  return s / T2
}

/** One real update: the same episode, then backprop with dh = target - h. */
function step2(gru: GRULayer, lr: number): void {
  start2(gru)
  const hs = INPUTS2.map((x) => [...gru.predict(x)])
  gru.backprop(
    hs.map((h, t) => h.map((v, i) => TARGETS2[t][i] - v)),
    lr,
  )
}

/** Reads and writes only the W block of one gate, in place. */
function gateW(gru: GRULayer, name: 'resetGate' | 'updateGate') {
  return {
    getParams(): number[] {
      const out: number[] = []
      for (const row of gru[name].W) for (const v of row) out.push(v)
      return out
    },
    setParams(p: number[]): void {
      let k = 0
      for (const row of gru[name].W) for (let j = 0; j < row.length; j++) row[j] = p[k++]
    },
  }
}

describe('GRULayer oracle: reset gate', () => {
  it('precondition: the episode carries a non-zero state and newGate reads it', () => {
    const gru = build2()
    const hs = run2(gru)
    // h_prev at t >= 1 is the previous hidden state, so it must be non-zero there.
    for (let t = 0; t < T2 - 1; t++) {
      expect(hs[t].some((v) => Math.abs(v) > 0.01)).toBe(true)
    }
    // newGate.W over the h part of [x, r ⊙ h_prev] must be far from zero.
    const hPart = gru.newGate.W.flatMap((row) => row.slice(IN2))
    expect(Math.max(...hPart.map((v) => Math.abs(v)))).toBeGreaterThan(0.1)
  })

  it('oracle (whole vector): backprop() follows -dL/dw on the 4-step episode from H0', () => {
    const gru = build2()
    const r = gradientOracle({
      getParams: () => gru.getWeightsFlat(),
      setParams: (p) => gru.setWeightsFlat(p),
      loss: () => meanLoss2(gru),
      step: (lr) => step2(gru, lr),
      lr: LR2,
      resetRng: () => start2(gru),
    })
    expect(r.cosine).toBeGreaterThan(0.999)
    expect(r.ratio).toBeGreaterThan(0.99)
    expect(r.ratio).toBeLessThan(1.01)
  })

  it('oracle (resetGate.W block): its update follows -dL/dw on its own', () => {
    const gru = build2()
    const block = gateW(gru, 'resetGate')
    const r = expectGradientMatches({
      getParams: block.getParams,
      setParams: block.setParams,
      loss: () => meanLoss2(gru),
      step: (lr) => step2(gru, lr),
      lr: LR2,
      resetRng: () => start2(gru),
    })
    expect(r.cosine).toBeGreaterThan(0.999)
    expect(r.ratio).toBeGreaterThanOrEqual(0.9)
    expect(r.ratio).toBeLessThanOrEqual(1.1)
  })

  it('oracle (updateGate.W block): its update follows -dL/dw on its own', () => {
    const gru = build2()
    const block = gateW(gru, 'updateGate')
    const r = expectGradientMatches({
      getParams: block.getParams,
      setParams: block.setParams,
      loss: () => meanLoss2(gru),
      step: (lr) => step2(gru, lr),
      lr: LR2,
      resetRng: () => start2(gru),
    })
    expect(r.cosine).toBeGreaterThan(0.999)
    expect(r.ratio).toBeGreaterThanOrEqual(0.9)
    expect(r.ratio).toBeLessThanOrEqual(1.1)
  })
})
