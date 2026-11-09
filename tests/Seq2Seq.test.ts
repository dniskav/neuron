// Behavior tests for Seq2Seq (T029, feature 006). The gradient oracle lives in
// tests/oracle/Seq2Seq.oracle.test.ts and is not repeated here.
//
// Hand-computed values. Most tests use a net whose LSTM weights and biases are all 0,
// except (where stated) the encoder's cell-gate bias a. For one unit:
//   zero weights: f = i = o = σ(0) = 0.5 and g = tanh(0) = 0, so
//                 c_t = 0.5·c_{t-1}                 and  h_t = 0.5·tanh(c_t).
//   cell bias a (weights 0): g = tanh(a), so c_t = 0.5·c_{t-1} + 0.5·tanh(a), hence
//                 c_T = (1 − 0.5^T)·tanh(a).   With T = 3 and a = 1: c_3 = 0.875·tanh(1).
// Inputs have no effect when the weights are 0, so they are arbitrary in those tests.
//
// Learning test (T029 "baja ≥ 30 %"): inversion of [0.1, 0.5, 0.9] with hidden 8, 300 steps,
// 5 seeds and lr 0.5 × {0.7, 1, 1.3}. Figures measured with the code before writing this
// test (threshold NOT adjusted to them): worst last/first ratio 0.091 (lr 0.65, seed 1).
// Required: ratio ≤ 0.7 (−30 %), so the worst case has a margin of 0.091 ≤ 0.7·0.7 = 0.49.
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type { LSTMLayer } from '../src/LSTMLayer'
import { Seq2Seq } from '../src/Seq2Seq'
import { restoreRandom, seedRandom, withSeed } from './helpers/seed'

/** tanh(1) = 0.7615941559…, the reference value used in the derivations. */
const TANH1 = Math.tanh(1)

/**
 * Sets the LSTM flat weights to 0 and the cell-gate bias to `cellBias`.
 * Flat layout (see LSTMLayer.getWeightsFlat): forget (W, b), input (W, b), cell (W, b), output (W, b).
 * The cell-gate bias block therefore starts after forget and input (W and b) and the cell W.
 */
function setLSTM(layer: LSTMLayer, hidden: number, input: number, cellBias = 0): void {
  const comb = input + hidden
  const flat = new Array<number>(4 * (hidden * comb + hidden)).fill(0)
  const cellBiasStart = 2 * (hidden * comb + hidden) + hidden * comb
  for (let k = 0; k < hidden; k++) flat[cellBiasStart + k] = cellBias
  layer.setWeightsFlat(flat)
}

/** Seq2Seq with zero LSTM weights (encoder cell bias optional). W_out and b_out are left as built. */
function zeroNet(input: number, hidden: number, output: number, encCellBias = 0): Seq2Seq {
  const m = new Seq2Seq(input, hidden, output)
  setLSTM(m.encoder, hidden, input, encCellBias)
  setLSTM(m.decoder, hidden, output) // the decoder is fed the outputs, so its input size is outputSize
  return m
}

const SEQ3: number[][] = [[0.3], [-0.7], [0.9]]
const LEARN_IN: number[][] = [[0.1], [0.5], [0.9]]
const LEARN_OUT: number[][] = [[0.9], [0.5], [0.1]]

beforeEach(() => seedRandom())
afterEach(() => restoreRandom())

describe('Seq2Seq constructor', () => {
  const BAD: [number, number, number][] = [
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [-1, 2, 1],
    [1, -2, 1],
    [1, 1, -1],
  ]
  for (const [i, h, o] of BAD) {
    it(`throws when a size is not positive (${i}, ${h}, ${o})`, () => {
      expect(() => new Seq2Seq(i, h, o)).toThrow('Seq2Seq: all sizes must be positive')
    })
  }

  it('builds the encoder with inputSize and the decoder with outputSize as its input', () => {
    const m = new Seq2Seq(2, 3, 1)
    // A flat LSTM has 4·(hidden·(input + hidden) + hidden) weights.
    expect(m.encoder.getWeightsFlat()).toHaveLength(4 * (3 * (2 + 3) + 3)) // 4·18 = 72
    expect(m.decoder.getWeightsFlat()).toHaveLength(4 * (3 * (1 + 3) + 3)) // 4·15 = 60
    expect(m.W_out).toHaveLength(1)
    expect(m.W_out[0]).toHaveLength(3)
    expect(m.b_out).toHaveLength(1)
  })
})

describe('Seq2Seq.encode', () => {
  it('returns h and c of length hiddenSize, both 0 for an empty sequence', () => {
    const ctx = zeroNet(1, 2, 1).encode([])
    expect(ctx.h).toEqual([0, 0]) // the state starts at 0 and nothing is applied
    expect(ctx.c).toEqual([0, 0])
  })

  it('final state with cell bias 1: c_3 = 0.875·tanh(1), h_3 = 0.5·tanh(c_3)', () => {
    const ctx = zeroNet(1, 2, 1, 1).encode(SEQ3)
    const cT = 0.875 * TANH1 // (1 − 0.5³)·tanh(1)
    expect(ctx.c[0]).toBeCloseTo(cT, 12)
    expect(ctx.c[1]).toBeCloseTo(cT, 12)
    expect(ctx.h[0]).toBeCloseTo(0.5 * Math.tanh(cT), 12) // o = 0.5
    expect(ctx.h[1]).toBeCloseTo(0.5 * Math.tanh(cT), 12)
  })

  it('resets the encoder state on each call (encoding twice gives the same context)', () => {
    const m = zeroNet(1, 2, 1, 1)
    const first = m.encode(SEQ3)
    const second = m.encode(SEQ3)
    // Without the reset the cell would start from the previous c_3 and keep growing,
    // so the second pass would not equal 0.875·tanh(1).
    expect(second.c[0]).toBeCloseTo(0.875 * TANH1, 12)
    expect(second.c).toEqual(first.c)
    expect(second.h).toEqual(first.h)
  })

  it('returns copies, so changing the result does not change the encoder state', () => {
    const m = zeroNet(1, 2, 1, 1)
    const ctx = m.encode(SEQ3)
    ctx.h[0] = 999
    ctx.c[0] = 999
    expect(m.encoder.h[0]).not.toBe(999)
    expect(m.encoder.c[0]).not.toBe(999)
  })
})

describe('Seq2Seq.decode', () => {
  // Zero decoder weights, W_out = [[1,0],[0,1]], b_out = [0.25, −0.5].
  // Context c = [2, 0], h = [0, 0]. Unit 0: c_t = 2·0.5^(t+1), h_t = 0.5·tanh(c_t).
  // Unit 1 stays at c = 0, h = 0. So output_t = [0.5·tanh(c_t) + 0.25, −0.5].
  function decodeNet(): Seq2Seq {
    const m = zeroNet(1, 2, 2)
    m.W_out = [[1, 0], [0, 1]]
    m.b_out = [0.25, -0.5]
    return m
  }

  it('produces `steps` outputs of size outputSize, with the values from the context', () => {
    const out = decodeNet().decode({ h: [0, 0], c: [2, 0] }, 3)
    expect(out).toHaveLength(3)
    expect(out[0]).toHaveLength(2)
    // t = 0: c = 1, t = 1: c = 0.5, t = 2: c = 0.25
    expect(out[0][0]).toBeCloseTo(0.5 * Math.tanh(1) + 0.25, 12)
    expect(out[1][0]).toBeCloseTo(0.5 * Math.tanh(0.5) + 0.25, 12)
    expect(out[2][0]).toBeCloseTo(0.5 * Math.tanh(0.25) + 0.25, 12)
    for (const o of out) expect(o[1]).toBeCloseTo(-0.5, 12)
  })

  it('uses the context: a zero context gives the bias alone at every step', () => {
    const out = decodeNet().decode({ h: [0, 0], c: [0, 0] }, 4)
    // h stays 0, so output = W_out·0 + b_out = [0.25, −0.5].
    for (const o of out) {
      expect(o[0]).toBeCloseTo(0.25, 12)
      expect(o[1]).toBeCloseTo(-0.5, 12)
    }
  })

  it('follows `steps`: 0 gives [], and a longer run starts with the shorter one', () => {
    const m = decodeNet()
    const ctx = { h: [0, 0], c: [2, 0] }
    expect(m.decode(ctx, 0)).toEqual([])
    const five = m.decode(ctx, 5)
    expect(five).toHaveLength(5)
    // Free-running decoding is deterministic, so the first 3 steps match the 3-step run.
    expect(five.slice(0, 3)).toEqual(m.decode(ctx, 3))
  })
})

describe('Seq2Seq.trainStep', () => {
  it('returns the teacher-forced MSE computed before the update (encoder context reaches the decoder)', () => {
    // Encoder cell bias 1 gives context c = 0.875·tanh(1) per unit (see the header).
    // Decoder weights 0: c_t = 0.5^(t+1)·0.875·tanh(1) at step t, output = h_t[0] = 0.5·tanh(c_t).
    //   u_0 = 0.5·tanh(0.4375·tanh(1)), u_1 = 0.5·tanh(0.21875·tanh(1)).
    // Targets 0 and T·O = 2, so loss = (u_0² + u_1²)/2.
    // If the decoder did not receive the encoder c (c = 0), every output would be 0 and the loss 0.
    const m = zeroNet(1, 2, 1, 1)
    m.W_out = [[1, 0]]
    m.b_out = [0]
    const u0 = 0.5 * Math.tanh(0.4375 * TANH1)
    const u1 = 0.5 * Math.tanh(0.21875 * TANH1)
    const loss = m.trainStep(SEQ3, [[0], [0]], 0.5)
    expect(loss).toBeCloseTo((u0 * u0 + u1 * u1) / 2, 12)
    expect(loss).toBeGreaterThan(0)
  })

  it('normalizes the loss by T·outputSize', () => {
    // Zero weights and W_out = 0, b_out = 0: every output is 0, so loss = Σ y² / (T·O).
    // Targets [[1,0],[0,2]]: Σ y² = 1 + 4 = 5, T·O = 2·2 = 4, loss = 5/4 = 1.25.
    const m = zeroNet(1, 2, 2)
    m.W_out = [[0, 0], [0, 0]]
    m.b_out = [0, 0]
    const loss = m.trainStep([[0.5]], [[1, 0], [0, 2]], 0.5)
    expect(loss).toBeCloseTo(1.25, 12)
  })

  it('one step moves b_out by lr·(2/(T·O))·Σ(y − o) and W_out by the decoder hidden states', () => {
    // Encoder cell bias 1 (context c = 0.875·tanh(1)), decoder zero, W_out = 0, b_out = 0.
    // Targets y_t = 1 for t = 0, 1 (T = 2, O = 1). Outputs are 0 before the update, so
    //   loss = ((0−1)² + (0−1)²)/2 = 1.
    // Gradient step: Δb = lr·(2/(T·O))·Σ_t (y_t − o_t) = 0.5·(2/2)·2 = 1.
    // ΔW[0][j] = lr·(2/(T·O))·Σ_t (y_t − o_t)·h_t[j], with h_t[j] = u_t (both units equal):
    //   ΔW[0][j] = 0.5·Σ_t u_t = 0.5·(u_0 + u_1).
    // The decoder and encoder get zero gradient (W_out = 0 before the step).
    const m = zeroNet(1, 2, 1, 1)
    m.W_out = [[0, 0]]
    m.b_out = [0]
    const u0 = 0.5 * Math.tanh(0.4375 * TANH1)
    const u1 = 0.5 * Math.tanh(0.21875 * TANH1)
    const loss = m.trainStep(SEQ3, [[1], [1]], 0.5)
    expect(loss).toBeCloseTo(1, 12)
    expect(m.b_out[0]).toBeCloseTo(1, 12)
    expect(m.W_out[0][0]).toBeCloseTo(0.5 * (u0 + u1), 12)
    expect(m.W_out[0][1]).toBeCloseTo(0.5 * (u0 + u1), 12)
  })

  it('lr = 0 leaves every weight unchanged, and the loss is still finite', () => {
    const m = new Seq2Seq(1, 3, 1)
    const enc = m.encoder.getWeightsFlat()
    const dec = m.decoder.getWeightsFlat()
    const wOut = m.W_out.map((row) => [...row])
    const bOut = [...m.b_out]
    const loss = m.trainStep(LEARN_IN, LEARN_OUT, 0)
    expect(Number.isFinite(loss)).toBe(true)
    expect(m.encoder.getWeightsFlat()).toEqual(enc)
    expect(m.decoder.getWeightsFlat()).toEqual(dec)
    expect(m.W_out).toEqual(wOut)
    expect(m.b_out).toEqual(bOut)
  })

  it('learns: inverting [0.1, 0.5, 0.9] drops the loss by ≥ 30 % (5 seeds, lr ×0.7 / ×1.3)', () => {
    for (const f of [0.7, 1, 1.3]) {
      for (const seed of [1, 2, 3, 4, 5]) {
        withSeed(seed, () => {
          const m = new Seq2Seq(1, 8, 1)
          let first = NaN
          let last = NaN
          for (let s = 0; s < 300; s++) {
            const l = m.trainStep(LEARN_IN, LEARN_OUT, 0.5 * f)
            expect(Number.isFinite(l), `finite loss, seed ${seed}, lr ${0.5 * f}, step ${s}`).toBe(true)
            if (s === 0) first = l
            last = l
          }
          // Threshold: last ≤ 0.7·first (−30 %). Measured worst case 0.091 (see the header).
          expect(last, `seed ${seed}, lr ${0.5 * f}`).toBeLessThanOrEqual(0.7 * first)
        })
      }
    }
  })
})
