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
// The optimizer is plain SGD, so no Adam patch (Q1) is needed.
import { describe, it, expect } from 'vitest'
import { Autoencoder } from '../../src/Autoencoder'
import { SGD } from '../../src/optimizers'
import { tanh, sigmoid } from '../../src/activations'
import { mse } from '../../src/losses'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

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
function oracle(ae: Autoencoder, x: number[], block: Block) {
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
    lr: LR,
  })
}

function expectFollowsGradient(ae: Autoencoder, x: number[], block: Block) {
  const r = oracle(ae, x, block)
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
