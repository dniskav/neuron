// Gradient-oracle test for VAE (feature 002, T025).
//
//   1. smoke   - encode/forward/generate have the right sizes, train returns finite losses
//                that match an independent computation (no it.fails).
//   2. oracle  - reference loss (Q4) = MSE(x, x_hat) + KL(N(mu, sigma^2) || N(0, I)) with the
//                reparametrisation noise fixed: resetRng reseeds Math.random before every
//                loss() and step(); forward() draws the same eps values in the same order.
//                Global and per block (decoder, encoder mu, encoder logVar, KL term only).
//   3. learns  - reconstruction better than predicting the mean, and mu varies with the input.
// AUDIT C2: train() sends the encoder only the KL gradient; the reconstruction gradient
// never goes back through z = mu + sigma * eps. Which blocks fail is MEASURED (cosines in the
// report), not assumed. Optimizer is the default SGD, so no Adam patch (Q1). Only the
// cosine is asserted (Q9). Remove `.fails` when the defect is fixed (feature 003).
import { describe, it, expect } from 'vitest'
import { VAE } from '../../src/VAE'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

const LR = 1e-3
const INPUT = [0.2, 0.7, 0.5]
const RNG_SEED = 0xbeef

function build(): VAE {
  // input 3 -> encoder [4] -> latent 2 (mu, logVar: 4 outputs) -> decoder [4] -> 3
  return new VAE(3, [4], 2, [4])
}

const resetRng = () => seedRandom(RNG_SEED)

const mseOf = (x: number[], r: number[]) => x.reduce((s, v, i) => s + (v - r[i]) ** 2, 0) / x.length
const klOf = (mu: number[], lv: number[]) =>
  mu.reduce((s, m, i) => s - 0.5 * (1 + lv[i] - m * m - Math.exp(lv[i])), 0)

// Reference loss: MSE + KL with the eps drawn after resetRng.
function elbo(vae: VAE, x: number[]): { total: number; recon: number; kl: number } {
  const { reconstruction, mu, logVar } = vae.forward(x)
  const recon = mseOf(x, reconstruction)
  const kl = klOf(mu, logVar)
  return { total: recon + kl, recon, kl }
}

// Encoder layout (3 -> 4 -> 4): hidden layer 4 * (3 + 1) = 16 params, then output neurons of
// 5 params each: mu = neurons 0-1 (params 16..25), logVar = neurons 2-3 (params 26..35).
const MU: [number, number] = [16, 26]
const LV: [number, number] = [26, 36]

interface Slice {
  get(vae: VAE): number[]
  set(vae: VAE, p: number[]): void
}

function encSlice(range: [number, number]): Slice {
  const [from, to] = range
  return {
    get: (v) => v.encoder.getWeights().slice(from, to),
    set: (v, p) => {
      const w = v.encoder.getWeights()
      w.splice(from, to - from, ...p)
      v.encoder.setWeights(w)
    },
  }
}
const encAll: Slice = {
  get: (v) => v.encoder.getWeights(),
  set: (v, p) => v.encoder.setWeights(p),
}
const decAll: Slice = {
  get: (v) => v.decoder.getWeights(),
  set: (v, p) => v.decoder.setWeights(p),
}
// Encoder + decoder concatenated (encoder first).
const both: Slice = {
  get: (v) => [...v.encoder.getWeights(), ...v.decoder.getWeights()],
  set: (v, p) => {
    const n = v.encoder.getWeights().length
    v.encoder.setWeights(p.slice(0, n))
    v.decoder.setWeights(p.slice(n))
  },
}

function cosine(slice: Slice, lossOf: (v: VAE) => number): number {
  const vae = build()
  return gradientOracle({
    getParams: () => slice.get(vae),
    setParams: (p) => slice.set(vae, p),
    loss: () => lossOf(vae),
    step: (lr) => {
      vae.train(INPUT, lr)
    },
    lr: LR,
    resetRng,
  }).cosine
}

const total = (v: VAE) => elbo(v, INPUT).total
const klOnly = (v: VAE) => elbo(v, INPUT).kl
const reconOnly = (v: VAE) => elbo(v, INPUT).recon

describe('VAE oracle', () => {
  it('smoke: sizes are right, train returns finite losses consistent with MSE + KL', () => {
    const vae = build()
    const { mu, logVar } = vae.encode(INPUT)
    expect(mu).toHaveLength(2)
    expect(logVar).toHaveLength(2)
    const f = vae.forward(INPUT)
    expect(f.reconstruction).toHaveLength(3)
    expect(f.z).toHaveLength(2)
    expect([...f.reconstruction, ...f.mu, ...f.logVar, ...f.z].every(Number.isFinite)).toBe(true)
    expect(vae.generate()).toHaveLength(3)

    resetRng()
    const expected = elbo(vae, INPUT)
    resetRng()
    const r = vae.train(INPUT, LR)
    expect(Number.isFinite(r.totalLoss)).toBe(true)
    expect(r.reconLoss).toBeCloseTo(expected.recon, 12)
    expect(r.klLoss).toBeCloseTo(expected.kl, 12)
    expect(r.totalLoss).toBeCloseTo(expected.total, 12)
  })

  // AUDIT C2: measured cosine 0.987. The encoder gets only the KL gradient, so the recon
  // part of dL/d(encoder) is missing; see the recon-term block below for the isolated defect.
  // Measured cosine 0.987 (AUDIT C2 plus scale defects). Each block alone points the right way
  // (cosines 0.9999 / 0.9998 / 0.99998 / 0.99996) but their relative scales are wrong: the
  // decoder delta lacks the factor 2 of d(MSE)/dx_hat and the sigmoid derivative of the output
  // layer, and the encoder never receives dMSE/dz through z = mu + sigma * eps. Because the KL
  // gradient dominates the encoder step, the encoder blocks look fine in isolation.
  it.fails('oracle: global (encoder + decoder) follows -dL/dw of MSE + KL (cosine > 0.999)', () => {
    expect(cosine(both, total)).toBeGreaterThan(0.999)
  })
})

describe('VAE oracle (building blocks)', () => {
  it('decoder alone follows -dL/dw of MSE + KL (cosine > 0.999)', () => {
    expect(cosine(decAll, total)).toBeGreaterThan(0.999)
  })

  it('encoder (all) follows -dL/dw of MSE + KL (cosine > 0.999)', () => {
    expect(cosine(encAll, total)).toBeGreaterThan(0.999)
  })

  it('encoder mu neurons follow -dL/dw of MSE + KL (cosine > 0.999)', () => {
    expect(cosine(encSlice(MU), total)).toBeGreaterThan(0.999)
  })

  it('encoder logVar neurons follow -dL/dw of MSE + KL (cosine > 0.999)', () => {
    expect(cosine(encSlice(LV), total)).toBeGreaterThan(0.999)
  })

  // The reconstruction term alone is only a valid target for the decoder: the encoder step also
  // contains the KL gradient, so an encoder-vs-MSE-only cosine would not isolate anything.
  it('recon term alone: the decoder follows -dMSE/dw (cosine > 0.999)', () => {
    expect(cosine(decAll, reconOnly)).toBeGreaterThan(0.999)
  })

  it('KL term alone: encoder step follows -dKL/dw (cosine > 0.999)', () => {
    expect(cosine(encAll, klOnly)).toBeGreaterThan(0.999)
  })
})

describe('VAE learns', () => {
  const data: number[][] = [
    [0.9, 0.1, 0.5],
    [0.1, 0.9, 0.5],
    [0.5, 0.5, 0.1],
    [0.5, 0.5, 0.9],
  ]

  // Measured: MSE drops only to 0.91 of its start (plateau at the mean, 2000 epochs, lr 0.5).
  // Measured: MSE falls only to 0.91 of its start (plateau at the mean; 2000 epochs, lr 0.5).
  it.fails('learns: reconstruction MSE beats predicting the mean', () => {
    const vae = build()
    const meanVec = [0, 1, 2].map((j) => data.reduce((s, x) => s + x[j], 0) / data.length)
    const baseline = data.reduce((s, x) => s + mseOf(x, meanVec), 0) / data.length
    const loss = () => {
      // Deterministic eps so the before/after losses are comparable.
      resetRng()
      return data.reduce((s, x) => s + mseOf(x, vae.forward(x).reconstruction), 0) / data.length
    }
    // expectLearns needs final < 0.5 * initial; the extra check below compares with the mean.
    expectLearns({ loss, step: () => data.forEach((x) => vae.train(x, 0.5)), steps: 2000 })
    expect(loss()).toBeLessThan(0.5 * baseline)
  })

  // Measured: mu spread across inputs is ~2e-5 after training (needs > 0.05).
  // Measured: spread of mu across inputs ~2e-5 after training (needs > 0.05).
  it.fails('learns: mu varies with the input', () => {
    const vae = build()
    for (let i = 0; i < 2000; i++) data.forEach((x) => vae.train(x, 0.5))
    const mus = data.map((x) => vae.encode(x).mu)
    const spread = [0, 1].map((k) => Math.max(...mus.map((m) => m[k])) - Math.min(...mus.map((m) => m[k])))
    expect(Math.max(...spread)).toBeGreaterThan(0.05)
  })
})
