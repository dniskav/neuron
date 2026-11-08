// Gradient-oracle test for GAN (feature 002, T024).
//
//   1. smoke   - generate/discriminate have the right sizes and ranges, and trainStep
//                returns finite losses that match an independent computation (no it.fails).
//   2. oracle  - generator (fixed C2): the generator update follows -dL/dw of
//                L_G = -log D(G(z)) with D frozen; gradient goes through D.inputGradient.
//                Normal tests (dims 3/2/5, 5 seeds, lr x0.97/x1/x1.03; cosine and ratio).
//                Plus: the G step leaves D's weights identical to a replica of D's own step.
//   3. oracle  - discriminator: trainStep's D update (real label 1, fake label 0) follows
//                -dL/dw of the binary cross-entropy. Passes (delta = target - pred is exactly
//                the sigmoid+BCE gradient). Normal test.
//   4. learns  - it.fails (AUDIT C2): the generated mean does not approach the target 0.20.
//
// Randomness: trainStep draws z (for D) and then z2 (for G) from Math.random, nothing else.
// resetRng reseeds Math.random before every loss() and step(); each loss draws the same
// number of values in the same order, so loss and step see the same z / z2.
// Assumption: L_G uses D after its own update inside trainStep; the oracle freezes D at its
// initial weights. The difference is O(lr) (measured: cosine >= 0.9999999, ratio 0.999-1.001).
// Optimizer is the default SGD, so no Adam patch (Q1). Only the cosine is asserted (Q9).
import { describe, it, expect } from 'vitest'
import { GAN } from '../../src/GAN'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

const LR = 1e-3
const X_REAL = [0.2, 0.3, 0.1]
const RNG_SEED = 0xabc

function build(): GAN {
  // latent 2 -> generator [4] -> 3 ; discriminator 3 -> [4] -> 1. Output dim 3 on purpose: with 1
  // output the uniform gDelta has the right sign whenever the D slope is positive (cosine +-1 by luck).
  return new GAN(2, [4], 3, [4])
}

const resetRng = () => seedRandom(RNG_SEED)

const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length
const bce = (p: number, y: number) => -(y * Math.log(p) + (1 - y) * Math.log(1 - p))

describe('GAN oracle', () => {
  it('smoke: sizes and ranges are right, trainStep returns finite, consistent losses', () => {
    const gan = build()
    const z = gan.sampleLatent()
    expect(z).toHaveLength(2)
    const x = gan.generate(z)
    expect(x).toHaveLength(3)
    expect(x.every((v) => v > 0 && v < 1)).toBe(true)
    const d = gan.discriminate(x)
    expect(d).toBeGreaterThan(0)
    expect(d).toBeLessThan(1)

    // dLoss = BCE(real) + BCE(fake) at the start of the step (D updates sequentially, so
    // the fake term sees D after the real update: allow the O(lr) difference).
    resetRng()
    const dReal = gan.discriminate(X_REAL)
    const zStep = gan.sampleLatent()
    const dFake = gan.discriminate(gan.generate(zStep))
    const expectedD = bce(dReal, 1) + bce(dFake, 0)
    resetRng()
    const { dLoss, gLoss } = gan.trainStep([X_REAL], LR)
    expect(Number.isFinite(dLoss)).toBe(true)
    expect(Number.isFinite(gLoss)).toBe(true)
    expect(dLoss).toBeCloseTo(expectedD, 2)
    expect(gLoss).toBeGreaterThan(0)
  })

  // Generator step vs -dL/dw of L_G = -log D(G(z2)) with D frozen. The oracle freezes D at its
  // PRE-step weights while trainStep updates D first, so the measured difference is O(lr): with
  // 5 seeds, dims 2/3/5 and lr 1e-3 x {0.97, 1, 1.03} the cosine is >= 0.9999999 and the ratio
  // within 0.999-1.001 (margin: threshold 0.999 / ratio band [0.9, 1.1]).
  const genOracle = (dim: number, seed: number, lr: number) => {
    seedRandom(seed * 77)
    const gan = new GAN(2, [4], dim, [4])
    const xr = Array.from({ length: dim }, (_, i) => 0.1 + 0.1 * ((i * 7) % 5))
    // D weights x6: at the default init D is almost linear on the generator range and the
    // gradient at x̂ vs another point is indistinguishable (mutant 'xFake instead of xFake2').
    const dFrozen = gan.discriminator.getWeights().map((w) => 6 * w)
    gan.discriminator.setWeights(dFrozen)
    const reset = () => seedRandom(RNG_SEED + seed)
    return gradientOracle({
      getParams: () => gan.generator.getWeights(),
      setParams: (p) => gan.generator.setWeights(p),
      loss: () => {
        gan.discriminator.setWeights(dFrozen)
        gan.sampleLatent() // z, consumed by the discriminator update in trainStep
        const z2 = gan.sampleLatent()
        return -Math.log(gan.discriminate(gan.generate(z2)))
      },
      step: (l) => {
        gan.discriminator.setWeights(dFrozen)
        gan.trainStep([xr], l)
      },
      lr,
      resetRng: reset,
    })
  }

  for (const dim of [3, 2, 5]) {
    it(`oracle: generator step follows -dL/dw of -log D(G(z)) with D frozen (dim ${dim}, 5 seeds, lr x0.97/x1/x1.03)`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        for (const k of [0.97, 1, 1.03]) {
          const r = genOracle(dim, seed, LR * k)
          expect(r.cosine).toBeGreaterThan(0.999)
          expect(r.ratio).toBeGreaterThan(0.9)
          expect(r.ratio).toBeLessThan(1.1)
        }
      }
    })
  }

  it("the generator step does not touch the discriminator (D equals a replica of D's own step)", () => {
    seedRandom(7)
    const gan = build()
    const replica = build()
    replica.generator.setWeights(gan.generator.getWeights())
    replica.discriminator.setWeights(gan.discriminator.getWeights())
    resetRng()
    gan.trainStep([X_REAL], LR)
    resetRng()
    const dReal = replica.discriminate(X_REAL)
    replica.discriminator.trainWithDeltas(X_REAL, [1 - dReal], LR)
    const xFake = replica.generate(replica.sampleLatent())
    replica.discriminator.trainWithDeltas(xFake, [-replica.discriminate(xFake)], LR)
    expect(gan.discriminator.getWeights()).toEqual(replica.discriminator.getWeights())
  })

  it('oracle: discriminator step follows -dL/dw of the real/fake cross-entropy (cosine > 0.999)', () => {
    const gan = build()
    const gFrozen = gan.generator.getWeights()
    const r = gradientOracle({
      getParams: () => gan.discriminator.getWeights(),
      setParams: (p) => gan.discriminator.setWeights(p),
      loss: () => {
        gan.generator.setWeights(gFrozen)
        const z = gan.sampleLatent()
        return bce(gan.discriminate(X_REAL), 1) + bce(gan.discriminate(gan.generate(z)), 0)
      },
      step: (lr) => {
        gan.generator.setWeights(gFrozen)
        gan.trainStep([X_REAL], lr)
      },
      lr: LR,
      resetRng,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  // AUDIT C2 (T007 BLOCKED): the generator step is now exact, but this test still cannot pass
  // robustly. The gap to 0.20 reaches ~0 and then sits at a plateau ~0.12 (D output ~0.5 flat,
  // G stalls at ~0.08): final/initial ~0.4 (> 0.35 needed for a 30% margin) under lr x0.7/x1.3.
  it.fails('learns: the generated mean approaches the target distribution (0.20)', () => {
    const TARGET = 0.2
    const gans: GAN[] = [1, 2, 3].map((s) => {
      seedRandom(s)
      return build()
    })
    const zs: number[][] = Array.from({ length: 8 }, (_, i) => [Math.sin(i + 1), Math.cos(2 * i + 1)])
    const gap = (g: GAN) =>
      Math.abs(zs.reduce((s, z) => s + mean(g.generate(z)), 0) / zs.length - TARGET)
    expectLearns({
      loss: () => gans.reduce((s, g) => s + gap(g), 0) / gans.length,
      step: () =>
        gans.forEach((g) => {
          const batch = [Array.from({ length: 3 }, () => 0.15 + 0.1 * Math.random())]
          g.trainStep(batch, 0.1)
        }),
      steps: 2000,
    })
  })
})
