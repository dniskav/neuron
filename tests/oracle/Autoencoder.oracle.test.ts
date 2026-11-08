// Gradient-oracle test for Autoencoder (feature 002, T018).
//
// Same structure as NetworkN.oracle.test.ts (composition of NetworkN):
//   1. smoke   - encode/decode/reconstruct have the right sizes, train returns the
//                reconstruction MSE it reports (the loss train optimizes).
//   2. oracle  - the update train() really performs (encoder + decoder weights,
//                concatenated) follows -dL/dw from central finite differences.
//                Loss = MSE(x, reconstruct(x)), the same loss train() returns.
//                Only the cosine is asserted (> 0.999), never the ratio (Q9).
//   3. learns  - expectLearns: reconstruction MSE on a fixed mini-set drops below half.
// The optimizer is plain SGD, so no Adam patch (Q1) is needed.
import { describe, it, expect } from 'vitest'
import { Autoencoder } from '../../src/Autoencoder'
import { SGD } from '../../src/optimizers'
import { mse } from '../../src/losses'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT = [0.2, 0.7, 0.5]
const LR = 1e-3

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

  it('oracle: train() follows -dL/dw from finite differences (cosine > 0.999)', () => {
    const ae = build()
    const r = gradientOracle({
      getParams: () => getParams(ae),
      setParams: (p) => setParams(ae, p),
      loss: () => mse(ae.reconstruct(INPUT), INPUT),
      step: (lr) => {
        ae.train(INPUT, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('oracle: same with no hidden layers (encoder [] and decoder [])', () => {
    const ae = new Autoencoder(3, [], 2, [], { optimizer: () => new SGD() })
    const r = gradientOracle({
      getParams: () => getParams(ae),
      setParams: (p) => setParams(ae, p),
      loss: () => mse(ae.reconstruct(INPUT), INPUT),
      step: (lr) => {
        ae.train(INPUT, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: reconstruction MSE on a fixed mini-set drops below half', () => {
    const ae = build()
    const data: number[][] = [
      [0.2, 0.7, 0.5],
      [0.9, 0.1, 0.4],
      [0.5, 0.5, 0.8],
    ]
    const loss = () =>
      data.reduce((s, x) => s + mse(ae.reconstruct(x), x), 0) / data.length
    expectLearns({
      loss,
      step: () => data.forEach((x) => ae.train(x, 0.5)),
      // Measured curve (final/initial): 0.83 at 300 steps and 0.83 at 1000 (plateau at
      // the mean reconstruction), 0.02 at 3000, ~0 at 10000. 3000 is the budget that learns.
      steps: 3000,
    })
  })
})

describe('Autoencoder oracle (building blocks)', () => {
  // Per-block cosine. The full-vector cosine is dominated by the decoder, so a sign
  // error in the encoder barely moves it; these blocks isolate each side.
  it('the encoder alone follows -dL/dw (cosine > 0.999)', () => {
    const ae = build()
    const nEnc = ae.encoder.getWeights().length
    const r = gradientOracle({
      getParams: () => getParams(ae).slice(0, nEnc),
      setParams: (p) => setParams(ae, [...p, ...ae.decoder.getWeights()]),
      loss: () => mse(ae.reconstruct(INPUT), INPUT),
      step: (lr) => {
        ae.train(INPUT, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('the decoder alone follows -dL/dw (cosine > 0.999)', () => {
    const ae = build()
    const nEnc = ae.encoder.getWeights().length
    const r = gradientOracle({
      getParams: () => getParams(ae).slice(nEnc),
      setParams: (p) => setParams(ae, [...ae.encoder.getWeights(), ...p]),
      loss: () => mse(ae.reconstruct(INPUT), INPUT),
      step: (lr) => {
        ae.train(INPUT, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })
})
