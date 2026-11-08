// Gradient-oracle test for Seq2Seq (feature 002, T023).
//
//   1. smoke   - encode/decode have the right shapes, trainStep returns the teacher-forced
//                MSE measured outside it.
//   2. oracle  - the update trainStep() really performs follows -dL/dw from central finite
//                differences. Loss = teacher-forced MSE, the value trainStep returns. Only
//                the cosine is asserted (> 0.999), never the ratio (Q9).
//   3. blocks  - the same oracle on encoder only, decoder only and W_out/b_out only,
//                to localise a failure to one part.
//   4. learns  - expectLearns: the loss on a fixed pair drops below half.
//   5. T_in != T_out - the same oracle with sequences of different lengths (the encoder
//                update is scaled by T_in/T_out so the total gradient is that of the loss
//                averaged over T_out*outputSize).
// The decoder's gradient w.r.t. its initial state (dh0, dc0) feeds the last encoder step
// (full BPTT through the context vector).
import { describe, it, expect } from 'vitest'
import { Seq2Seq } from '../../src/Seq2Seq'
import { withSeed } from '../helpers/seed'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

interface Data {
  input: number[][]
  target: number[][]
}

const INPUT: number[][] = [
  [0.5, -0.3],
  [0.2, 0.8],
  [-0.6, 0.4],
]
const TARGET: number[][] = [[0.3], [-0.2], [0.5]]
const SAME: Data = { input: INPUT, target: TARGET }
// T_in = 4, T_out = 2 and T_in = 2, T_out = 5.
const LONG_IN: Data = {
  input: [...INPUT, [0.1, 0.9]],
  target: [[0.3], [-0.4]],
}
const LONG_OUT: Data = {
  input: [INPUT[0], INPUT[1]],
  target: [[0.3], [-0.2], [0.5], [0.1], [-0.6]],
}
const HID = 3
const LR = 1e-3

function build(outSize = 1): Seq2Seq {
  return new Seq2Seq(2, HID, outSize)
}

/** Teacher-forced MSE, computed with encode/predict only (same loss trainStep returns). */
function loss(m: Seq2Seq, d: Data = SAME): number {
  const ctx = m.encode(d.input)
  m.decoder.reset()
  m.decoder.h = [...ctx.h]
  m.decoder.c = [...ctx.c]
  let prev = new Array(m.outputSize).fill(0)
  let s = 0
  for (let t = 0; t < d.target.length; t++) {
    const h = m.decoder.predict(prev)
    const o = (m as any)._project(h) as number[]
    o.forEach((v, k) => (s += (v - d.target[t][k]) ** 2))
    prev = d.target[t]
  }
  return s / (d.target.length * m.outputSize)
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
  const n = m.outputSize
  m.W_out = Array.from({ length: n }, (_, i) => out.slice(i * HID, (i + 1) * HID))
  m.b_out = out.slice(n * HID, n * HID + n)
}

/** Runs the oracle on one block. */
function oracle(block: Block, d: Data = SAME, lr = LR, outSize = 1) {
  const m = build(outSize)
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
    loss: () => loss(m, d),
    step: (l) => {
      m.trainStep(d.input, d.target, l)
    },
    lr,
  })
}

// outputSize 2 and 3 (T_in = 3, T_out = 3); closes mutant M14 (no /outputSize in dProjOut).
const MULTI: Record<number, Data> = {
  2: { input: INPUT, target: [[0.3, -0.4], [-0.2, 0.5], [0.5, 0.1]] },
  3: { input: INPUT, target: [[0.3, -0.4, 0.2], [-0.2, 0.5, -0.1], [0.5, 0.1, -0.6]] },
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

  it('oracle: trainStep() follows -dL/dw from finite differences (cosine > 0.999)', () => {
    expect(oracle('all').cosine).toBeGreaterThan(0.999)
  })

  it('oracle (encoder alone): follows -dL/dw (cosine > 0.999)', () => {
    expect(oracle('enc').cosine).toBeGreaterThan(0.999)
  })

  it('oracle (decoder alone): follows -dL/dw (cosine > 0.999)', () => {
    expect(oracle('dec').cosine).toBeGreaterThan(0.999)
  })

  it('oracle (W_out/b_out alone): follows -dL/dw (cosine > 0.999)', () => {
    expect(oracle('out').cosine).toBeGreaterThan(0.999)
  })

  it('oracle holds with lr x0.97 / x1.03 and across seeds', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const f of [0.97, 1.03]) {
        withSeed(seed, () => {
          expect(oracle('all', SAME, LR * f).cosine).toBeGreaterThan(0.999)
        })
      }
    }
  })

  for (const [name, d] of [
    ['T_in=4 > T_out=2', LONG_IN],
    ['T_in=2 < T_out=5', LONG_OUT],
  ] as const) {
    it(`oracle with ${name}: global, encoder, decoder and W_out/b_out follow -dL/dw`, () => {
      for (const block of ['all', 'enc', 'dec', 'out'] as const) {
        expect(oracle(block, d).cosine).toBeGreaterThan(0.999)
      }
      for (const seed of [1, 2, 3, 4, 5]) {
        withSeed(seed, () => {
          expect(oracle('all', d).cosine).toBeGreaterThan(0.999)
        })
      }
    })
  }

  for (const n of [2, 3]) {
    it(`oracle with outputSize ${n}: follows -dL/dw with ratio ~ 1 (lr x0.97/x1.03, 5 seeds)`, () => {
      for (const block of ['all', 'enc', 'dec', 'out'] as const) {
        const r = oracle(block, MULTI[n], LR, n)
        expect(r.cosine).toBeGreaterThan(0.999)
        expect(r.ratio).toBeGreaterThan(0.9)
        expect(r.ratio).toBeLessThan(1.1)
      }
      for (const seed of [1, 2, 3, 4, 5]) {
        for (const f of [0.97, 1.03]) {
          withSeed(seed, () => {
            const r = oracle('all', MULTI[n], LR * f, n)
            expect(r.cosine).toBeGreaterThan(0.999)
            expect(r.ratio).toBeGreaterThan(0.9)
            expect(r.ratio).toBeLessThan(1.1)
          })
        }
      }
    })
  }

  it('learns: the loss on a fixed pair drops below half', () => {
    const m = build()
    expectLearns({
      loss: () => loss(m),
      step: () => {
        m.trainStep(INPUT, TARGET, 0.5)
      },
      steps: 300,
    })
  })
})
