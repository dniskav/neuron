// Gradient-oracle test for VAE (feature 002 T025; fixed in feature 004 T008, C2 + N8).
//
//   1. smoke   - encode/forward/generate have the right sizes, forward returns eps, train returns
//                finite losses that match an independent computation.
//   2. oracle  - reference loss = MSE(x, x_hat) + klWeight * KL(N(mu, sigma^2) || N(0, I)) with the
//                reparametrisation noise fixed: resetRng reseeds Math.random before every
//                loss() and step(); forward() draws the same eps values in the same order.
//                Global and per block (decoder, encoder, mu, logVar), for 5 init seeds and
//                lr x0.97 / x1.03: cosine > 0.999 AND ratio in [0.9, 1.1] (the ratio is what
//                detects a wrong constant such as the missing factor 2 or dfn of the output).
//                The optimizer is the default SGD, so no Adam patch (Q1).
//   3. learns  - Q5: the encoder carries information about the input, causally:
//                MSE(x, decode(mu(x))) < 0.5 * MSE(x, decode(mu(x'))) for another input x', and
//                MSE(x, decode(mu(x))) < 0.5 * MSE of predicting the mean. The spread of mu is
//                only informative. Robust over 5 seeds x lr {0.7, 1, 1.3} (base lr 1.0).
//
// Defects fixed (HANDOFF N8): the encoder now receives -dMSE/dz through z = mu + sigma * eps,
// the decoder/encoder deltas carry the output derivative, and the decoder delta carries the
// factor 2 of dMSE/dx_hat. The encoder output is linear (Q3).
//
// Q4 (measured): with klWeight = 1 (MSE averaged over n, KL summed over the latent) the
// posterior collapses: mu is constant (spread ~3e-3) and the loss stays at the mean predictor
// (self/mean ~1.0) for lr 0.35..0.65 and 5 seeds. "learns" therefore uses klWeight = 0.003
// (measured: self/cross 0.14-0.16 and self/mean ~0.34 for 5 seeds x lr 0.7..1.3).
import { describe, it, expect } from 'vitest'
import { VAE } from '../../src/VAE'
import { NetworkN } from '../../src/NetworkN'
import { gradientOracle } from '../helpers/gradientOracle'
import { seedRandom } from '../helpers/seed'

const LR = 1e-3
const INPUT = [0.2, 0.7, 0.5]
const RNG_SEED = 0xbeef
const SEEDS = [1, 2, 3, 4, 5]
const LR_FACTORS = [0.97, 1, 1.03]

function build(seed = 0x5eed, klWeight = 1): VAE {
  // input 3 -> encoder [4] -> latent 2 (mu, logVar: 4 outputs) -> decoder [4] -> 3
  seedRandom(seed)
  return new VAE(3, [4], 2, [4], undefined, klWeight)
}

const resetRng = () => seedRandom(RNG_SEED)

const mseOf = (x: number[], r: number[]) => x.reduce((s, v, i) => s + (v - r[i]) ** 2, 0) / x.length
const klOf = (mu: number[], lv: number[]) =>
  mu.reduce((s, m, i) => s - 0.5 * (1 + lv[i] - m * m - Math.exp(lv[i])), 0)

// Reference loss: MSE + klWeight * KL with the eps drawn after resetRng.
function elbo(vae: VAE, x: number[], beta = 1): { total: number; recon: number; kl: number } {
  const { reconstruction, mu, logVar } = vae.forward(x)
  const recon = mseOf(x, reconstruction)
  const kl = klOf(mu, logVar)
  return { total: recon + beta * kl, recon, kl }
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

function measure(
  slice: Slice,
  lossOf: (v: VAE) => number,
  seed: number,
  lrFactor: number,
  beta = 1,
  prepare?: (v: VAE) => void,
  baseLr = LR,
) {
  const vae = build(seed, beta)
  prepare?.(vae)
  return gradientOracle({
    getParams: () => slice.get(vae),
    setParams: (p) => slice.set(vae, p),
    loss: () => lossOf(vae),
    step: (lr) => {
      vae.train(INPUT, lr)
    },
    lr: baseLr * lrFactor,
    resetRng,
  })
}

// Asserts cosine > 0.999 and ratio in [0.9, 1.1] for 5 seeds x lr x0.97/x1.03.
function expectFollowsGradient(
  slice: Slice,
  lossOf: (v: VAE, beta: number) => number,
  beta = 1,
  prepare?: (v: VAE) => void,
  baseLr = LR,
) {
  for (const seed of SEEDS) {
    for (const f of LR_FACTORS) {
      const r = measure(slice, (v) => lossOf(v, beta), seed, f, beta, prepare, baseLr)
      const tag = `seed ${seed}, lr x${f}: cosine ${r.cosine}, ratio ${r.ratio}`
      expect(r.cosine, tag).toBeGreaterThan(0.999)
      expect(r.ratio, tag).toBeGreaterThan(0.9)
      expect(r.ratio, tag).toBeLessThan(1.1)
    }
  }
}

const total = (v: VAE, beta: number) => elbo(v, INPUT, beta).total
const reconOnly = (v: VAE) => elbo(v, INPUT).recon
const klOnly = (v: VAE) => elbo(v, INPUT).kl

describe('VAE oracle', () => {
  it('smoke: sizes are right, train returns finite losses consistent with MSE + KL', () => {
    const vae = build()
    const { mu, logVar } = vae.encode(INPUT)
    expect(mu).toHaveLength(2)
    expect(logVar).toHaveLength(2)
    const f = vae.forward(INPUT)
    expect(f.reconstruction).toHaveLength(3)
    expect(f.z).toHaveLength(2)
    expect(f.eps).toHaveLength(2)
    expect([...f.reconstruction, ...f.mu, ...f.logVar, ...f.z, ...f.eps].every(Number.isFinite)).toBe(true)
    // z = mu + sigma * eps with the returned eps
    f.z.forEach((z, i) => expect(z).toBeCloseTo(f.mu[i] + Math.exp(0.5 * f.logVar[i]) * f.eps[i], 12))
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

  it('encoder output is linear (Q3): mu and logVar are not confined to (0, 1)', () => {
    const vae = build()
    const w = vae.encoder.getWeights()
    // large output-layer biases push the outputs far outside (0, 1) only if the layer is linear
    for (let n = 0; n < 4; n++) w[16 + n * 5 + 4] = n % 2 === 0 ? 3 : -3
    vae.encoder.setWeights(w)
    const { mu, logVar } = vae.encode(INPUT)
    expect(Math.abs(mu[0])).toBeGreaterThan(1.5)
    expect(logVar[1]).toBeLessThan(-1.5)
  })

  it('rejects an invalid klWeight', () => {
    expect(() => new VAE(3, [4], 2, [4], undefined, -1)).toThrow()
    expect(() => new VAE(3, [4], 2, [4], undefined, Number.NaN)).toThrow()
  })

  it('oracle: global (encoder + decoder) follows -dL/dw of MSE + KL (cosine > 0.999, ratio 0.9-1.1)', () => {
    expectFollowsGradient(both, total)
  })

  // A single SGD step is exactly lr * (-grad) when every delta is computed at the initial point,
  // so a LARGE lr is a valid oracle and exposes O(lr) errors that lr = 1e-3 hides: here, gz
  // computed with the decoder already updated (N1-style ordering bug).
  it('oracle: encoder, pure reconstruction (klWeight = 0) at a large lr (5) still equals lr * (-grad): gz uses the decoder BEFORE its update', () => {
    expectFollowsGradient(encAll, total, 0, undefined, 5)
  })

  // klWeight = 0 isolates the reconstruction path: the logVar gradient is then ONLY
  // gz * 0.5 * sigma * eps (the KL gradient of logVar is also ~0 at logVar ~ 0, but not exactly).
  it('oracle: pure reconstruction (klWeight = 0), encoder logVar neurons follow -dMSE/dw', () => {
    expectFollowsGradient(encSlice(LV), total, 0)
  })

  it('oracle: pure reconstruction (klWeight = 0), encoder mu neurons follow -dMSE/dw', () => {
    expectFollowsGradient(encSlice(MU), total, 0)
  })

  it('oracle: global with klWeight = 2.5 follows -dL/dw of MSE + 2.5 KL', () => {
    expectFollowsGradient(both, total, 2.5)
  })
})

describe('VAE oracle (building blocks)', () => {
  it('decoder alone follows -dL/dw of MSE + KL', () => {
    expectFollowsGradient(decAll, total)
  })

  it('encoder (all) follows -dL/dw of MSE + KL (includes the path through z)', () => {
    expectFollowsGradient(encAll, total)
  })

  it('encoder mu neurons follow -dL/dw of MSE + KL', () => {
    expectFollowsGradient(encSlice(MU), total)
  })

  it('encoder logVar neurons follow -dL/dw of MSE + KL (the 0.5*sigma*eps path)', () => {
    expectFollowsGradient(encSlice(LV), total)
  })

  it('encoder with a SIGMOID output (replaced by hand) still follows -dL/dw: dfn of the output is applied', () => {
    const swap = (v: VAE) => {
      ;(v as unknown as { encoder: NetworkN }).encoder = new NetworkN([3, 4, 4], {})
    }
    expectFollowsGradient(encAll, total, 1, swap)
  })

  // The reconstruction term alone is only a valid target for the decoder: the encoder step also
  // contains the KL gradient.
  it('recon term alone: the decoder follows -dMSE/dw', () => {
    expectFollowsGradient(decAll, (v) => reconOnly(v))
  })

  // Replaces the 002 test 'KL term alone: encoder step follows -dKL/dw', which is no longer true
  // in general (the encoder also receives the reconstruction gradient). With the decoder weights
  // set to zero, dMSE/dz = 0 exactly, so the encoder step is purely the KL gradient and the
  // test still isolates the KL block.
  it('KL term alone (decoder weights zeroed so dMSE/dz = 0): encoder step follows -dKL/dw', () => {
    const zeroDecoder = (v: VAE) => v.decoder.setWeights(v.decoder.getWeights().map(() => 0))
    expectFollowsGradient(encAll, (v) => klOnly(v), 1, zeroDecoder)
  })
})

describe('VAE learns', () => {
  const data: number[][] = [
    [0.9, 0.1, 0.5],
    [0.1, 0.9, 0.5],
    [0.5, 0.5, 0.1],
    [0.5, 0.5, 0.9],
  ]
  const BETA = 0.003
  const EPOCHS = 3000
  const BASE_LR = 1
  const meanVec = [0, 1, 2].map((j) => data.reduce((s, x) => s + x[j], 0) / data.length)
  const baseline = data.reduce((s, x) => s + mseOf(x, meanVec), 0) / data.length

  function trained(seed: number, lr: number): VAE {
    const vae = build(seed, BETA)
    seedRandom(100 + seed)
    for (let i = 0; i < EPOCHS; i++) data.forEach((x) => vae.train(x, lr))
    return vae
  }

  const variants = SEEDS.flatMap((seed) => [0.7, 1, 1.3].map((f) => ({ seed, f })))

  it('learns: reconstruction from mu beats predicting the mean by a wide margin (5 seeds x lr x0.7/x1.3)', () => {
    for (const { seed, f } of variants) {
      const vae = trained(seed, BASE_LR * f)
      const self = data.reduce((s, x) => s + mseOf(x, vae.decode(vae.encode(x).mu)), 0) / data.length
      // criterion < 0.5 x baseline; a 30 % margin means <= 0.35 x baseline
      expect(self, `seed ${seed}, lr x${f}: self ${self}, baseline ${baseline}`).toBeLessThan(0.35 * baseline)
    }
  })

  it('learns: mu varies with the input, causally (MSE with its own mu < 0.5 x MSE with another input\'s mu)', () => {
    for (const { seed, f } of variants) {
      const vae = trained(seed, BASE_LR * f)
      const own = data.map((x) => mseOf(x, vae.decode(vae.encode(x).mu)))
      const other = data.map((x, i) => mseOf(x, vae.decode(vae.encode(data[(i + 1) % data.length]).mu)))
      const sOwn = own.reduce((a, b) => a + b, 0) / own.length
      const sOther = other.reduce((a, b) => a + b, 0) / other.length
      // 0.35 = 0.5 with a 30 % margin
      expect(sOwn, `seed ${seed}, lr x${f}: own ${sOwn}, other ${sOther}`).toBeLessThan(0.35 * sOther)
    }
  })
})
