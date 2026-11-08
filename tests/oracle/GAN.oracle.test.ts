// Test de oráculo de gradiente para GAN.
//
//   1. smoke   - generate/discriminate have the right sizes and ranges, and trainStep
//                returns finite losses that match an independent computation.
//   2. oracle  - generator: la actualización de G sigue -dL/dw de L_G = -log D(G(z)) con D
//                congelado; el gradiente pasa por D.inputGradient. Pruebas normales (dims 3/2/5,
//                5 semillas, lr x0.97/x1/x1.03; coseno y ratio). Además, el paso de G deja los
//                pesos de D idénticos a una réplica del propio paso de D.
//   3. oracle  - discriminador: la actualización de D en trainStep (etiqueta real 1, falsa 0)
//                sigue -dL/dw de la entropía cruzada binaria. delta = objetivo - predicción es
//                exactamente el gradiente sigmoide + BCE. Prueba normal.
//   4. learns  - (a) el acercamiento al objetivo se verifica contra un D FIJO escrito a mano,
//                porque el entrenamiento conjunto no converge de forma robusta a 0.20 (ver
//                'learns (b)'); (b) el comportamiento conjunto se documenta con una propiedad
//                acotada y finita.
//
// Randomness: trainStep draws z (for D) and then z2 (for G) from Math.random, nothing else.
// resetRng reseeds Math.random before every loss() and step(); each loss draws the same
// number of values in the same order, so loss and step see the same z / z2.
// Assumption: L_G uses D after its own update inside trainStep; the oracle freezes D at its
// initial weights. The difference is O(lr) (measured: cosine >= 0.9999999, ratio 0.999-1.001).
// Optimizer is the default SGD, so no Adam patch (Q1). Only the cosine is asserted (Q9).
import { describe, it, expect } from 'vitest'
import { GAN } from '../../src/GAN'
import { SGD } from '../../src/optimizers'
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

  // learns (a): generator against a FIXED discriminator that knows the target. D is written by
  // hand (3 inputs -> 4 hidden -> 1, sigmoid): two hidden units give a bump on mean(x) centred
  // at 0.2 (steepness 20), output = sigmoid(12 (h1 - h2)); D(0.2) ~ 1.00, D(0.5) ~ 0.55.
  // The G step is the REAL trainStep path; D is restored after every call (trainStep also
  // updates D first, which is undone), so G only ever sees the frozen D up to O(lr).
  // Measured (6 seeds, lr 0.1 x{0.97,1,1.03}, 1000 steps): initial gap 0.26-0.38, final
  // 0.004-0.014, final/initial 0.01-0.05 (criterion < 0.35: margin > 85%).
  const fixedD = (): number[] => {
    const a = 20
    const c = 12
    return [
      a / 3, a / 3, a / 3, -a * 0.1,
      a / 3, a / 3, a / 3, -a * 0.3,
      0, 0, 0, 0,
      0, 0, 0, 0,
      c, -c, 0, 0, 0,
    ]
  }
  const ZS: number[][] = Array.from({ length: 8 }, (_, i) => [Math.sin(i + 1), Math.cos(2 * i + 1)])
  const meanGap = (g: GAN, target: number) =>
    Math.abs(ZS.reduce((s, z) => s + mean(g.generate(z)), 0) / ZS.length - target)

  it('learns: the generator approaches the target (0.20) against a FIXED discriminator (5 seeds, lr x0.7..x1.3)', () => {
    const dW = fixedD()
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const k of [0.7, 0.97, 1, 1.03, 1.3]) {
        seedRandom(seed)
        const g = new GAN(2, [4], 3, [4])
        g.discriminator.setWeights(dW)
        expect(g.discriminate([0.2, 0.2, 0.2])).toBeGreaterThan(0.99)
        const initial = meanGap(g, 0.2)
        for (let i = 0; i < 1000; i++) {
          g.trainStep([[0.2, 0.2, 0.2]], 0.1 * k)
          g.discriminator.setWeights(dW)
        }
        const final = meanGap(g, 0.2)
        expect(Number.isFinite(final)).toBe(true)
        expect(final).toBeLessThan(0.35 * initial)
      }
    }
  })

  // learns (b): joint adversarial training is DOCUMENTED, not required to converge to 0.20.
  // Measured (G[4], D[4], lr 0.02 x{0.7,1,1.3}, 6000 steps, 8 seeds): the gap goes to ~0 around
  // step 1500-3000 and then stays on a plateau/oscillation of 0.07-0.14 (max seen 0.139); D
  // flattens to ~0.5 everywhere (D(0.1)~0.495, D(0.2)~0.499, D(0.3)~0.496) so the signal
  // through D to G vanishes. That is known adversarial behaviour with equal SGD steps, not a
  // gradient defect (the generator oracle above is exact). Robust property asserted: outputs
  // and D stay finite and in (0,1), and from step 3000 to 6000 the gap stays below 0.2 (observed max 0.139, margin 30%; initial gaps are 0.22-0.38).
  it('joint training: stays bounded and finite (gap <= 0.2, D in (0,1)); no convergence to 0.20 required', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const k of [0.7, 1, 1.3]) {
        seedRandom(seed)
        const g = new GAN(2, [4], 3, [4])
        for (let i = 0; i < 6000; i++) {
          g.trainStep([Array.from({ length: 3 }, () => 0.15 + 0.1 * Math.random())], 0.02 * k)
          if ((i + 1) % 250 === 0) {
            const gap = meanGap(g, 0.2)
            expect(Number.isFinite(gap)).toBe(true)
            if (i + 1 >= 3000) expect(gap).toBeLessThan(0.2) // after the initial transient
            const d = g.discriminate([0.2, 0.2, 0.2])
            expect(d).toBeGreaterThan(0)
            expect(d).toBeLessThan(1)
          }
        }
      }
    }
  })
})

// T013 gap 2: the gLoss value that trainStep returns is -log D(G(z2)) (non-saturating), not
// -log(1 - D). z is drawn for the discriminator's fake sample, z2 for the generator, in this
// order, per sample.
describe('GAN trainStep gLoss / dLoss values (T013)', () => {
  const mkTwin = () => {
    seedRandom(0x5eed)
    return build()
  }

  it('batch of 1: gLoss = -log D(G(z2)) with D after its own update (hand-replicated step)', () => {
    const gan = mkTwin()
    const twin = mkTwin()

    // Hand replica of the D phase on the twin (same weights, same Math.random order).
    resetRng()
    const dRealPre = twin.discriminate(X_REAL)
    twin.discriminator.trainWithDeltas(X_REAL, [1 - dRealPre], LR)
    const xFake = twin.generate(twin.sampleLatent())
    const dFakePre = twin.discriminate(xFake)
    twin.discriminator.trainWithDeltas(xFake, [0 - dFakePre], LR)
    const xFake2 = twin.generate(twin.sampleLatent())
    const dScore = twin.discriminate(xFake2)
    const expectedG = -Math.log(dScore)
    const wrongG = -Math.log(1 - dScore)
    expect(Math.abs(expectedG - wrongG)).toBeGreaterThan(0.05) // the case discriminates

    resetRng()
    const { gLoss, dLoss } = gan.trainStep([X_REAL], LR)
    expect(gLoss).toBeCloseTo(expectedG, 12)
    expect(dLoss).toBeCloseTo(-Math.log(dRealPre) - Math.log(1 - dFakePre), 12)
  })

  it('batch of 3 with lr = 0 (nothing moves): gLoss and dLoss are the means of the per-sample values', () => {
    const gan = mkTwin()
    const batch = [X_REAL, [0.4, 0.1, 0.6], [0.9, 0.8, 0.2]]
    resetRng()
    let g = 0
    let d = 0
    for (const xr of batch) {
      const z = gan.sampleLatent()
      const z2 = gan.sampleLatent()
      g += -Math.log(gan.discriminate(gan.generate(z2)))
      d += -Math.log(gan.discriminate(xr)) - Math.log(1 - gan.discriminate(gan.generate(z)))
    }
    resetRng()
    const r = gan.trainStep(batch, 0)
    expect(r.gLoss).toBeCloseTo(g / batch.length, 10)
    expect(r.dLoss).toBeCloseTo(d / batch.length, 10)
  })
})

// ─── T015 (Q9): one dropout mask per forward ─────────────────────────────────
// Golden (dropoutRate = 0): 5 trainStep() calls (batch of 1) with seedRandom(42) on
// GAN(2,[4],3,[4]) with SGD in both networks, lr 0.5. Constants captured on the base BEFORE
// touching src (digest of each block: count, sum, sum of w_i*(i+1), sum of squares; 1e-9) plus
// the number of Math.random calls (20 = z and z2, one Box-Muller pair each, x 5): same
// trajectory, same order and number of draws as before when there is no dropout.
describe('GAN dropoutRate = 0 is unchanged by reuseMasks (T015 golden)', () => {
  const XS = [
    [0.2, 0.7, 0.5],
    [0.9, 0.1, 0.4],
    [0.3, 0.3, 0.8],
    [0.6, 0.5, 0.2],
    [0.1, 0.9, 0.7],
  ]
  const digest = (w: number[]) => ({
    n: w.length,
    sum: w.reduce((s, v) => s + v, 0),
    wsum: w.reduce((s, v, i) => s + v * (i + 1), 0),
    sq: w.reduce((s, v) => s + v * v, 0),
  })
  const GOLDEN = {
    g: { n: 27, sum: 0.16277477056674836, wsum: -7.383982925164016, sq: 1.510797761301129 },
    d: { n: 21, sum: -2.52194595899105, wsum: -22.785984726599814, sq: 2.1207875814106156 },
    calls: 20,
  }
  it('weights after 5 steps equal the base constants (1e-9) and Math.random is called 20 times', () => {
    seedRandom(42)
    const sgd = { optimizer: () => new SGD() }
    const gan = new GAN(2, [4], 3, [4], { generatorOptions: sgd, discriminatorOptions: sgd })
    const original = Math.random
    let calls = 0
    Math.random = () => {
      calls++
      return original()
    }
    try {
      for (let i = 0; i < 5; i++) gan.trainStep([XS[i]], 0.5)
    } finally {
      Math.random = original
    }
    for (const [name, got, want] of [
      ['generator', digest(gan.generator.getWeights()), GOLDEN.g],
      ['discriminator', digest(gan.discriminator.getWeights()), GOLDEN.d],
    ] as const) {
      expect(got.n, name).toBe(want.n)
      expect(got.sum, name).toBeCloseTo(want.sum, 9)
      expect(got.wsum, name).toBeCloseTo(want.wsum, 9)
      expect(got.sq, name).toBeCloseTo(want.sq, 9)
    }
    expect(calls).toBe(GOLDEN.calls)
  })
})

// Discriminator [3,4,1] with dropoutRate 0.3 (4 draws per D forward), generator without dropout.
// The draws of one trainStep, in order: D(xReal) 4, z 2, D(xFake) 4, z2 2, D(xFake2) 4. The
// reference losses repeat that order after the same reseed, so loss() and step() see the same
// masks. Before T015 the scores were computed without dropout and D.trainWithDeltas /
// D.inputGradient drew their own masks. D is frozen at its initial weights x6 as in the
// generator oracle above (the difference with D updated first inside trainStep is O(lr)).
describe('GAN with dropout in the discriminator: one mask per forward (T015, Q9)', () => {
  const MASK_SEEDS = [31, 32]
  const mkDropout = (seed: number) => {
    seedRandom(seed * 77)
    const sgd = { optimizer: () => new SGD() }
    const gan = new GAN(2, [4], 3, [4], {
      generatorOptions: sgd,
      discriminatorOptions: { ...sgd, dropoutRate: 0.3 },
    })
    return gan
  }
  const xr = [0.2, 0.3, 0.1]

  it('generator step follows -dL/dw of -log D(G(z2)) with D in the mask of that step (5 seeds x 2 masks x lr x0.97/x1.03)', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const maskSeed of MASK_SEEDS) {
        for (const k of [0.97, 1.03]) {
          const gan = mkDropout(seed)
          const dFrozen = gan.discriminator.getWeights().map((w) => 6 * w)
          const r = gradientOracle({
            getParams: () => gan.generator.getWeights(),
            setParams: (p) => gan.generator.setWeights(p),
            loss: () => {
              gan.discriminator.setWeights(dFrozen)
              gan.discriminator.predict(xr, true) // D(xReal): consumes its 4 mask draws
              gan.sampleLatent() // z
              gan.discriminator.predict(xr, true) // D(xFake): 4 draws (only the stream matters here)
              const z2 = gan.sampleLatent()
              return -Math.log(gan.discriminator.predict(gan.generate(z2), true)[0])
            },
            step: (lr) => {
              gan.discriminator.setWeights(dFrozen)
              gan.trainStep([xr], lr)
            },
            lr: LR * k,
            resetRng: () => seedRandom(maskSeed),
          })
          const tag = `init ${seed} mask ${maskSeed} x${k}: cosine ${r.cosine}, ratio ${r.ratio}`
          expect(r.cosine, tag).toBeGreaterThan(0.999)
          expect(r.ratio, tag).toBeGreaterThan(0.9)
          expect(r.ratio, tag).toBeLessThan(1.1)
        }
      }
    }
  })

  it('discriminator step follows -dL/dw of the real/fake cross-entropy with the masks of its two updates (cosine > 0.999)', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const maskSeed of MASK_SEEDS) {
        const gan = mkDropout(seed)
        const gFrozen = gan.generator.getWeights()
        const r = gradientOracle({
          getParams: () => gan.discriminator.getWeights(),
          setParams: (p) => gan.discriminator.setWeights(p),
          loss: () => {
            gan.generator.setWeights(gFrozen)
            const dReal = gan.discriminator.predict(xr, true)[0]
            const z = gan.sampleLatent()
            const dFake = gan.discriminator.predict(gan.generate(z), true)[0]
            return bce(dReal, 1) + bce(dFake, 0)
          },
          step: (lr) => {
            gan.generator.setWeights(gFrozen)
            gan.trainStep([xr], lr)
          },
          lr: LR,
          resetRng: () => seedRandom(maskSeed),
        })
        expect(r.cosine, `init ${seed} mask ${maskSeed}: cosine ${r.cosine}`).toBeGreaterThan(0.999)
      }
    }
  })

  // Dropout in the generator too (G [2,4,3]: 4 draws per G forward). The G forward of the G step
  // runs in training mode and G.trainWithDeltas reuses its mask. Draws of trainStep: D(xReal) 4,
  // z 2, D(xFake) 4 (G generates xFake in inference mode), z2 2, G(z2) 4, D(xFake2) 4.
  it('generator step with dropout in G and D follows -dL/dw of -log D(G(z2)) with both masks of the step', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const maskSeed of MASK_SEEDS) {
        for (const k of [0.97, 1.03]) {
          seedRandom(seed * 77)
          const sgd = { optimizer: () => new SGD() }
          const gan = new GAN(2, [4], 3, [4], {
            generatorOptions: { ...sgd, dropoutRate: 0.3 },
            discriminatorOptions: { ...sgd, dropoutRate: 0.3 },
          })
          const dFrozen = gan.discriminator.getWeights().map((w) => 6 * w)
          const r = gradientOracle({
            getParams: () => gan.generator.getWeights(),
            setParams: (p) => gan.generator.setWeights(p),
            loss: () => {
              gan.discriminator.setWeights(dFrozen)
              gan.discriminator.predict(xr, true) // D(xReal)
              gan.sampleLatent() // z
              gan.discriminator.predict(xr, true) // D(xFake): only the stream matters
              const z2 = gan.sampleLatent()
              const xFake2 = gan.generator.predict(z2, true)
              return -Math.log(gan.discriminator.predict(xFake2, true)[0])
            },
            step: (lr) => {
              gan.discriminator.setWeights(dFrozen)
              gan.trainStep([xr], lr)
            },
            lr: LR * k,
            resetRng: () => seedRandom(maskSeed),
          })
          const tag = `init ${seed} mask ${maskSeed} x${k}: cosine ${r.cosine}, ratio ${r.ratio}`
          expect(r.cosine, tag).toBeGreaterThan(0.999)
          expect(r.ratio, tag).toBeGreaterThan(0.9)
          expect(r.ratio, tag).toBeLessThan(1.1)
        }
      }
    }
  })

  it('trainStep draws one mask per D forward: 4 + 2 + 4 + 2 + 4 values', () => {
    const gan = mkDropout(3)
    const original = Math.random
    let calls = 0
    Math.random = () => {
      calls++
      return original()
    }
    try {
      gan.trainStep([xr], 0.01)
    } finally {
      Math.random = original
    }
    expect(calls).toBe(16)
  })
})
