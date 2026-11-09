// Test de oráculo de gradiente para Autoencoder.
//
// Same structure as NetworkN.oracle.test.ts (composition of NetworkN):
//   1. smoke   - encode/decode/reconstruct have the right sizes, train returns the
//                reconstruction MSE it reports (the loss train optimizes).
//   2. oracle  - the update train() really performs (encoder + decoder weights,
//                concatenated) follows -dL/dw from central finite differences.
//                Loss = MSE(x, reconstruct(x)), the same loss train() returns (the MEAN
//                over the d components). Asserts the cosine (> 0.999) AND the ratio
//                |step| / (lr |grad|) in [0.9, 1.1] (plain SGD), so a missing 2/d factor
//                and a duplicated act' are caught. Run with d = 3, 4 and 5 (NOT d = 2:
//                2/d = 1 would hide the factor), with hidden layers ([4]/[4]) and without
//                ([]/[]), and with tanh in hidden+latent layers so act' varies. Per block
//                (encoder, decoder) too.
//   3. learns  - expectLearns: reconstruction MSE on a fixed mini-set drops below half.
// The optimizer is plain SGD.
import { describe, it, expect } from 'vitest'
import { Autoencoder } from '../../src/Autoencoder'
import { SGD } from '../../src/optimizers'
import { tanh, sigmoid } from '../../src/activations'
import { mse } from '../../src/losses'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom, restoreRandom } from '../helpers/seed'

const INPUT = [0.2, 0.7, 0.5]
const LR = 1e-3
// lr 0.75 = 0.5 x d/2 (d = 3): train() sigue el gradiente de la MSE MEDIA (factor 2/d), d/2
// veces menor que el de la suma. Medido final/inicial (presupuesto 10000 pasadas; la red se
// queda al principio en una meseta hacia la reconstrucción media ~0.83 y sale en un instante que
// depende de la semilla; 3000 pasadas NO bastan para todas las semillas: 0.34 con la semilla por
// defecto, 0.78-0.95 con otras):
//   default seed, lr x0.7 / x1 / x1.3: 1e-4 / 0 / 0   (threshold 0.5, 30 % margin = 0.35)
//   seeds 1..7:   worst 0.10 at lr x0.7 and x1; at x1.3 worst 0.75 (seed 7 stays on the plateau).
const LEARN_LR = 0.75
const LEARN_STEPS = 10000

function build(): Autoencoder {
  // input 3 -> encoder [4] -> latent 2 -> decoder [4] -> output 3
  return new Autoencoder(3, [4], 2, [4], { optimizer: () => new SGD() })
}

// Flat view of all weights: encoder first, then decoder.
function getParams(ae: Autoencoder): number[] {
  return [...ae.encoder.getWeights(), ...ae.decoder.getWeights()]
}

function setParams(ae: Autoencoder, p: number[]): void {
  const nEnc = ae.encoder.getWeights().length
  ae.encoder.setWeights(p.slice(0, nEnc))
  ae.decoder.setWeights(p.slice(nEnc))
}

type Block = 'all' | 'encoder' | 'decoder'

// Oracle over all weights, or only the encoder / decoder block.
function oracle(ae: Autoencoder, x: number[], block: Block, lr = LR) {
  const nEnc = ae.encoder.getWeights().length
  const slice = (p: number[]) =>
    block === 'encoder' ? p.slice(0, nEnc) : block === 'decoder' ? p.slice(nEnc) : p
  return gradientOracle({
    getParams: () => slice(getParams(ae)),
    setParams: (p) =>
      setParams(
        ae,
        block === 'encoder'
          ? [...p, ...ae.decoder.getWeights()]
          : block === 'decoder'
            ? [...ae.encoder.getWeights(), ...p]
            : p,
      ),
    loss: () => mse(ae.reconstruct(x), x),
    step: (lr) => {
      ae.train(x, lr)
    },
    lr,
  })
}

function expectFollowsGradient(ae: Autoencoder, x: number[], block: Block, lr = LR) {
  const r = oracle(ae, x, block, lr)
  expect(r.cosine).toBeGreaterThan(0.999)
  // El paso es el gradiente de la MSE media que devuelve train() (factor 2/d).
  expect(r.ratio).toBeGreaterThan(0.9)
  expect(r.ratio).toBeLessThan(1.1)
}

const sgd = { optimizer: () => new SGD() }
const X3 = [0.2, 0.7, 0.5]
const X4 = [0.2, 0.7, 0.5, 0.9]
const X5 = [0.2, 0.7, 0.5, 0.9, 0.05]
const CASES: { name: string; x: number[]; make: () => Autoencoder }[] = [
  { name: 'd=3 [4]/[4]', x: X3, make: () => new Autoencoder(3, [4], 2, [4], sgd) },
  { name: 'd=4 [4]/[4]', x: X4, make: () => new Autoencoder(4, [4], 2, [4], sgd) },
  { name: 'd=5 [4]/[4]', x: X5, make: () => new Autoencoder(5, [4], 2, [4], sgd) },
  { name: 'd=3 []/[]', x: X3, make: () => new Autoencoder(3, [], 2, [], sgd) },
  { name: 'd=4 []/[]', x: X4, make: () => new Autoencoder(4, [], 2, [], sgd) },
  { name: 'd=5 []/[]', x: X5, make: () => new Autoencoder(5, [], 2, [], sgd) },
  {
    // tanh in hidden + latent (act' varies with the value), sigmoid output
    name: 'd=4 [4]/[4] tanh',
    x: X4,
    make: () => new Autoencoder(4, [4], 2, [4], { ...sgd, activations: [tanh, tanh, tanh, sigmoid] }),
  },
]

describe('Autoencoder oracle', () => {
  it('smoke: sizes are right, outputs are finite, train returns the reconstruction MSE', () => {
    const ae = build()
    const z = ae.encode(INPUT)
    expect(z).toHaveLength(2)
    expect(z.every(Number.isFinite)).toBe(true)

    const xHat = ae.reconstruct(INPUT)
    expect(xHat).toHaveLength(3)
    expect(xHat.every((v) => v > 0 && v < 1)).toBe(true)

    // The value train() returns is the MSE of the reconstruction it started from.
    const before = mse(ae.reconstruct(INPUT), INPUT)
    const reported = ae.train(INPUT, LR)
    expect(Number.isFinite(reported)).toBe(true)
    expect(reported).toBeCloseTo(before, 12)
  })

  for (const c of CASES) {
    it(`oracle ${c.name}: train() follows -dL/dw (cosine > 0.999, ratio in [0.9, 1.1])`, () => {
      expectFollowsGradient(c.make(), c.x, 'all')
    })
  }

  it('learns: reconstruction MSE on a fixed mini-set drops below half', () => {
    const ae = build()
    const data: number[][] = [
      [0.2, 0.7, 0.5],
      [0.9, 0.1, 0.4],
      [0.5, 0.5, 0.8],
    ]
    const loss = () => data.reduce((s, x) => s + mse(ae.reconstruct(x), x), 0) / data.length
    expectLearns({
      loss,
      step: () => data.forEach((x) => ae.train(x, LEARN_LR)),
      steps: LEARN_STEPS,
    })
  })
})

describe('Autoencoder oracle (building blocks)', () => {
  // Per-block check. The full-vector cosine is dominated by the decoder, so a sign
  // error or a duplicated act' in the encoder barely moves it; these blocks isolate each side.
  for (const c of CASES) {
    it(`the encoder alone follows -dL/dw (${c.name})`, () => {
      expectFollowsGradient(c.make(), c.x, 'encoder')
    })
    it(`the decoder alone follows -dL/dw (${c.name})`, () => {
      expectFollowsGradient(c.make(), c.x, 'decoder')
    })
  }
})

// act'(z[k]) of the encoder is taken per latent unit (dfn(z[k]), a different unit each time).
// Latent of 3 units with large weights and very different biases in the latent layer, so z
// (tanh) is saturated in some units and not in others and act'(z[k]) differs a lot per unit.
// A `dfn(z[0])` mutant (the same factor for every unit) breaks the encoder block.
describe("Autoencoder oracle (latent act' per unit, T013)", () => {
  const mk = () => {
    seedRandom(0x5eed)
    const ae = new Autoencoder(4, [4], 3, [4], {
      ...sgd,
      activations: [tanh, tanh, tanh, sigmoid],
    })
    restoreRandom()
    // Latent layer = encoder layer 1: 3 neurons x (4 weights + bias), after the 4x(4+1) hidden.
    const w = ae.encoder.getWeights()
    const base = 4 * (4 + 1)
    const biases = [0.1, 1.2, -2.5]
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < 4; j++) w[base + k * 5 + j] *= 3
      w[base + k * 5 + 4] = biases[k]
    }
    ae.encoder.setWeights(w)
    return ae
  }

  it("the latent units have very different act' (the case is sensitive)", () => {
    const z = mk().encode(X4)
    const d = z.map((v) => 1 - v * v) // tanh' on the post-activation value
    expect(Math.max(...d) / Math.min(...d)).toBeGreaterThan(5)
  })

  for (const block of ['encoder', 'decoder', 'all'] as const) {
    it(`oracle ${block} with per-unit latent act' (cosine > 0.999, ratio in [0.9, 1.1])`, () => {
      expectFollowsGradient(mk(), X4, block)
    })
  }
})

// Residual and dropout in the decoder: train() takes dL/dz from decoder.inputGradient, so
// both are honoured in the gradient, as in VAE.
describe('Autoencoder oracle (decoder with residual / dropout, T013)', () => {
  const SEEDS = [1, 2, 3, 4, 5]
  const LR_FACTORS = [0.97, 1.03]

  const mkResidual = (seed: number) => {
    seedRandom(seed)
    const ae = new Autoencoder(4, [3], 2, [2], { ...sgd, residual: true })
    restoreRandom()
    return ae
  }

  for (const block of ['all', 'decoder', 'encoder'] as const) {
    it(`residual in the decoder, ${block}: cosine > 0.999, ratio ~ 1 (5 seeds, lr x0.97 / x1.03)`, () => {
      for (const seed of SEEDS) {
        for (const f of LR_FACTORS) {
          expectFollowsGradient(mkResidual(seed), X4, block, LR * f)
        }
      }
    })
  }

  // Decoder [2,2,2,4]: two hidden layers (2 -> 2) with a dropout after each (and residual in
  // the 'dropout + residual' case), 4 Math.random draws per forward. The encoder has no hidden layer, so it has no dropout
  // and the decoder's draws are the only ones. Math.random is replaced by a cycle of period
  // 4: predict(z, true), inputGradient and trainWithDeltas then draw the SAME mask, and the
  // cycle restarts before every loss()/step() (fixed mask).
  const PATTERNS = [
    [0.1, 0.9, 0.8, 0.2],
    [0.7, 0.3, 0.95, 0.6],
  ]
  const cycle = (pattern: number[]) => {
    let i = 0
    Math.random = () => pattern[i++ % 4]
  }
  for (const [name, residual] of [
    ['dropout', false],
    ['dropout + residual', true],
  ] as const) {
    PATTERNS.forEach((pattern, pi) => {
      it(`${name} in the decoder with a fixed mask (pattern ${pi + 1}): cosine > 0.999, ratio ~ 1`, () => {
        try {
          for (const seed of SEEDS) {
            for (const f of LR_FACTORS) {
              seedRandom(seed)
              const ae = new Autoencoder(4, [], 2, [2, 2], { ...sgd, residual, dropoutRate: 0.4 })
              const x = X4
              const nEnc = ae.encoder.getWeights().length
              // 'all' = encoder + decoder; 'encoder' isolates the gradient that flows back
              // through the decoder (the decoder block dominates the full cosine).
              for (const block of ['all', 'encoder'] as const) {
                const r = gradientOracle({
                  getParams: () => (block === 'all' ? getParams(ae) : ae.encoder.getWeights()),
                  setParams: (p) => {
                    if (block === 'all') {
                      ae.encoder.setWeights(p.slice(0, nEnc))
                      ae.decoder.setWeights(p.slice(nEnc))
                    } else {
                      ae.encoder.setWeights(p)
                    }
                  },
                  loss: () => {
                    cycle(pattern)
                    return mse(ae.decoder.predict(ae.encoder.predict(x), true), x)
                  },
                  step: (lr) => {
                    cycle(pattern)
                    ae.train(x, lr)
                  },
                  lr: LR * f,
                })
                expect(r.cosine, `${block} seed ${seed} x${f}`).toBeGreaterThan(0.999)
                expect(r.ratio, `${block} seed ${seed} x${f}`).toBeGreaterThan(0.9)
                expect(r.ratio, `${block} seed ${seed} x${f}`).toBeLessThan(1.1)
              }
            }
          }
        } finally {
          restoreRandom()
        }
      })
    })
  }

  it('residual on the LAST decoder layer throws a clear error and leaves every weight untouched (Q2, like VAE/GAN)', () => {
    // decoder [2,4,4]: the last layer is 4 -> 4, so the residual applies to it
    const ae = new Autoencoder(4, [3], 2, [4], { ...sgd, residual: true })
    const before = getParams(ae)
    expect(() => ae.train(X4, 0.1)).toThrow(/residual on the last layer/)
    expect(getParams(ae)).toEqual(before)
  })

  it('residual on the LAST encoder layer also throws before any weight changes', () => {
    // encoder [4,2,2]: the latent layer is 2 -> 2, so the residual applies; decoder [2,2,4] is fine
    const ae = new Autoencoder(4, [2], 2, [2], { ...sgd, residual: true })
    const before = getParams(ae)
    expect(() => ae.train(X4, 0.1)).toThrow(/residual on the last layer/)
    expect(getParams(ae)).toEqual(before)
  })
})

// ─── One dropout mask per step ───────────────────────────────────────────────
// Golden (dropoutRate = 0): 5 train() steps with seedRandom(42) on a [3,[4],2,[4]] SGD model,
// lr 0.5. The constants are a digest of each block (count, sum, sum of w_i*(i+1) and sum of
// squares; tolerance 1e-9), plus the number of Math.random calls inside the 5 steps (0: the
// autoencoder itself draws nothing). With dropoutRate = 0 the trajectory and the generator use
// must be unchanged.
describe('Autoencoder dropoutRate = 0 is unchanged by reuseMasks (T015 golden)', () => {
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
    enc: { n: 26, sum: 0.2629851007419487, wsum: -4.9523582939721535, sq: 1.3932334391974033 },
    dec: { n: 27, sum: -2.7538057495101333, wsum: -30.99434873892924, sq: 2.9263587797612196 },
    calls: 0,
  }
  it('weights after 5 steps equal the base constants (1e-9) and Math.random is not called', () => {
    seedRandom(42)
    const ae = new Autoencoder(3, [4], 2, [4], { optimizer: () => new SGD() })
    const original = Math.random
    let calls = 0
    Math.random = () => {
      calls++
      return original()
    }
    try {
      for (const x of XS) ae.train(x, 0.5)
    } finally {
      Math.random = original
    }
    for (const [name, got, want] of [
      ['encoder', digest(ae.encoder.getWeights()), GOLDEN.enc],
      ['decoder', digest(ae.decoder.getWeights()), GOLDEN.dec],
    ] as const) {
      expect(got.n, name).toBe(want.n)
      expect(got.sum, name).toBeCloseTo(want.sum, 9)
      expect(got.wsum, name).toBeCloseTo(want.wsum, 9)
      expect(got.sq, name).toBeCloseTo(want.sq, 9)
    }
    expect(calls).toBe(GOLDEN.calls)
  })
})

// Dropout in BOTH sub-networks (the options are shared): encoder [4,[3],2] has 3 draws, decoder
// [2,[3,3],4] has 6, in the order encoder.predict(x, true), decoder.predict(z, true). The loss is
// the reconstruction MSE with the mask of THAT step: reseeding before loss() and step() makes
// both see the same masks. train() reuses one mask per sub-network for the loss, the decoder
// gradient and both updates. Init seeds 1..5, mask seeds 11 and 12, lr x0.97 / x1.03.
describe('Autoencoder with dropout in encoder and decoder: one mask per step (T015, Q9)', () => {
  const X4b = [0.2, 0.7, 0.5, 0.3]
  const SEEDS = [1, 2, 3, 4, 5]
  const MASK_SEEDS = [11, 12]
  const LR_FACTORS = [0.97, 1.03]

  for (const block of ['all', 'encoder', 'decoder'] as const) {
    it(`${block}: cosine > 0.999 and ratio in [0.9, 1.1] against the loss with the step's mask`, () => {
      for (const seed of SEEDS) {
        for (const maskSeed of MASK_SEEDS) {
          for (const f of LR_FACTORS) {
            seedRandom(seed)
            const ae = new Autoencoder(4, [3], 2, [3, 3], { optimizer: () => new SGD(), dropoutRate: 0.3 })
            const nEnc = ae.encoder.getWeights().length
            const slice = (p: number[]) =>
              block === 'encoder' ? p.slice(0, nEnc) : block === 'decoder' ? p.slice(nEnc) : p
            const r = gradientOracle({
              getParams: () => slice(getParams(ae)),
              setParams: (p) =>
                setParams(
                  ae,
                  block === 'encoder'
                    ? [...p, ...ae.decoder.getWeights()]
                    : block === 'decoder'
                      ? [...ae.encoder.getWeights(), ...p]
                      : p,
                ),
              loss: () => mse(ae.decoder.predict(ae.encoder.predict(X4b, true), true), X4b),
              step: (lr) => {
                ae.train(X4b, lr)
              },
              lr: LR * f,
              resetRng: () => seedRandom(maskSeed),
            })
            const tag = `${block} init ${seed} mask ${maskSeed} x${f}: cosine ${r.cosine} ratio ${r.ratio}`
            expect(r.cosine, tag).toBeGreaterThan(0.999)
            expect(r.ratio, tag).toBeGreaterThan(0.9)
            expect(r.ratio, tag).toBeLessThan(1.1)
          }
        }
      }
    })
  }

  it('train() draws exactly one mask per sub-network per step (3 + 6 values)', () => {
    seedRandom(3)
    const ae = new Autoencoder(4, [3], 2, [3, 3], { optimizer: () => new SGD(), dropoutRate: 0.3 })
    const original = Math.random
    let calls = 0
    Math.random = () => {
      calls++
      return original()
    }
    try {
      ae.train(X4b, 0.1)
    } finally {
      Math.random = original
    }
    expect(calls).toBe(9)
  })
})
