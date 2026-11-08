// Gradient-oracle test for Seq2Seq (feature 002, T023).
//
//   1. smoke   - encode/decode have the right shapes, trainStep returns the teacher-forced
//                MSE measured outside it.
//   2. oracle  - the update trainStep() really performs follows -dL/dw from central finite
//                differences. Loss = teacher-forced MSE, the value trainStep returns. Only
//                the cosine is asserted (> 0.999), never the ratio (Q9).
//   3. blocks  - the same oracle on encoder only, decoder only and W_out/b_out only, so
//                feature 003 can see which part fails.
//   4. learns  - expectLearns: the loss on a fixed pair drops below half.
//   5. control - with the delta negated by hand, decoder and W_out/b_out follow -dL/dw
//                (cosine ~ +1): the oracle is sound and, for those blocks, the only defect
//                is the sign.
// AUDIT C1: trainStep hands +dL/dw (dProjOut = 2*diff/outputSize) to optimizers that do
// `w + lr*g`, so every block moves UP the loss. Measured cosines: global -0.996, encoder
// -0.940, decoder -1, W_out/b_out -1.
// AUDIT Q3: the encoder gradient is also approximate (only dhSeq[0] goes to the last
// encoder step; dc and dh of the decoder's initial state are ignored). That is why the
// encoder cosine is -0.94 and not -1, and why the negated encoder still misses 0.999.
// Remove `.fails` when the defects are fixed (feature 003).
import { describe, it, expect } from 'vitest'
import { Seq2Seq } from '../../src/Seq2Seq'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT: number[][] = [
  [0.5, -0.3],
  [0.2, 0.8],
  [-0.6, 0.4],
]
const TARGET: number[][] = [[0.3], [-0.2], [0.5]]
const HID = 3
const LR = 1e-3

function build(): Seq2Seq {
  return new Seq2Seq(2, HID, 1)
}

/** Teacher-forced MSE, computed with encode/predict only (same loss trainStep returns). */
function loss(m: Seq2Seq): number {
  const ctx = m.encode(INPUT)
  m.decoder.reset()
  m.decoder.h = [...ctx.h]
  m.decoder.c = [...ctx.c]
  let prev = [0]
  let s = 0
  for (let t = 0; t < TARGET.length; t++) {
    const h = m.decoder.predict(prev)
    const o = (m as any)._project(h) as number[]
    s += (o[0] - TARGET[t][0]) ** 2
    prev = TARGET[t]
  }
  return s / TARGET.length
}

type Block = 'all' | 'enc' | 'dec' | 'out'

function parts(m: Seq2Seq) {
  return {
    enc: m.encoder.getWeightsFlat(),
    dec: m.decoder.getWeightsFlat(),
    out: [...m.W_out.flat(), ...m.b_out],
  }
}

function setParts(m: Seq2Seq, enc: number[], dec: number[], out: number[]): void {
  m.encoder.setWeightsFlat(enc)
  m.decoder.setWeightsFlat(dec)
  m.W_out = [out.slice(0, HID)]
  m.b_out = [out[HID]]
}

/** Runs the oracle on one block. `negate` flips the delta by hand (sign control). */
function oracle(block: Block, negate = false) {
  const m = build()
  return gradientOracle({
    getParams: () => {
      const p = parts(m)
      return block === 'all' ? [...p.enc, ...p.dec, ...p.out] : p[block]
    },
    setParams: (v) => {
      const p = parts(m)
      const ne = p.enc.length
      const nd = p.dec.length
      if (block === 'all') setParts(m, v.slice(0, ne), v.slice(ne, ne + nd), v.slice(ne + nd))
      else if (block === 'enc') setParts(m, v, p.dec, p.out)
      else if (block === 'dec') setParts(m, p.enc, v, p.out)
      else setParts(m, p.enc, p.dec, v)
    },
    loss: () => loss(m),
    step: (lr) => {
      const before = parts(m)
      m.trainStep(INPUT, TARGET, lr)
      if (negate) {
        const after = parts(m)
        const flip = (a: number[], b: number[]) => a.map((x, i) => 2 * x - b[i])
        setParts(
          m,
          flip(before.enc, after.enc),
          flip(before.dec, after.dec),
          flip(before.out, after.out),
        )
      }
    },
    lr: LR,
  })
}

describe('Seq2Seq oracle', () => {
  it('smoke: shapes are right, outputs finite, trainStep returns the teacher-forced MSE', () => {
    const m = build()
    const ctx = m.encode(INPUT)
    expect(ctx.h).toHaveLength(HID)
    expect(ctx.c).toHaveLength(HID)
    const out = m.decode(ctx, 4)
    expect(out).toHaveLength(4)
    expect(out.every((o) => o.length === 1 && Number.isFinite(o[0]))).toBe(true)

    const before = loss(m)
    const reported = m.trainStep(INPUT, TARGET, LR)
    expect(Number.isFinite(reported)).toBe(true)
    expect(reported).toBeCloseTo(before, 12)
  })

  // AUDIT C1 (measured global cosine -0.996).
  it.fails('oracle: trainStep() follows -dL/dw from finite differences (cosine > 0.999)', () => {
    expect(oracle('all').cosine).toBeGreaterThan(0.999)
  })

  // AUDIT C1 + Q3 (measured cosine -0.940).
  it.fails('oracle (encoder alone): follows -dL/dw (cosine > 0.999)', () => {
    expect(oracle('enc').cosine).toBeGreaterThan(0.999)
  })

  // AUDIT C1 (measured cosine -1).
  it.fails('oracle (decoder alone): follows -dL/dw (cosine > 0.999)', () => {
    expect(oracle('dec').cosine).toBeGreaterThan(0.999)
  })

  // AUDIT C1 (measured cosine -1).
  it.fails('oracle (W_out/b_out alone): follows -dL/dw (cosine > 0.999)', () => {
    expect(oracle('out').cosine).toBeGreaterThan(0.999)
  })

  // AUDIT Q3: even with the sign fixed by hand the encoder gradient is approximate.
  it.fails('oracle (encoder alone, sign negated): follows -dL/dw (cosine > 0.999)', () => {
    expect(oracle('enc', true).cosine).toBeGreaterThan(0.999)
  })

  it('control: with the delta negated by hand, decoder and W_out/b_out follow -dL/dw', () => {
    expect(oracle('dec', true).cosine).toBeGreaterThan(0.999)
    expect(oracle('out', true).cosine).toBeGreaterThan(0.999)
  })

  // AUDIT C1: measured loss 0.126 -> 7e11 after 300 steps at lr 0.01 (it diverges).
  it.fails('learns: the loss on a fixed pair drops below half', () => {
    const m = build()
    expectLearns({
      loss: () => loss(m),
      step: () => {
        m.trainStep(INPUT, TARGET, 0.01)
      },
      steps: 300,
    })
  })
})
