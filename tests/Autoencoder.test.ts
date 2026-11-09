// Tests de comportamiento de Autoencoder (src/Autoencoder.ts). El gradiente y el
// "aprende" ya están en tests/oracle/Autoencoder.oracle.test.ts; aquí no se repiten.
//
// Red de referencia (activaciones lineales, pesos fijados con setWeights):
//   encoder [2 -> 1]: z = 0.5·x1 − 1·x2 + 0.25      (getWeights: [0.5, −1, 0.25])
//   decoder [1 -> 2]: x̂1 = 2·z + 0.1, x̂2 = −1·z + 0  (getWeights: [2, 0.1, −1, 0])
// Cálculos a mano:
//   x = [1, 2]: z = 0.5 − 2 + 0.25 = −1.25; x̂ = [2·(−1.25) + 0.1, 1.25] = [−2.4, 1.25]
//     MSE = ((1 + 2.4)² + (2 − 1.25)²) / 2 = (11.56 + 0.5625) / 2 = 6.06125
//   x = [0, 0]: z = 0.25; x̂ = [0.6, −0.25]
//     MSE = ((0 − 0.6)² + (0 + 0.25)²) / 2 = (0.36 + 0.0625) / 2 = 0.21125
//   Media de los dos con lr = 0 (los pesos no cambian): (6.06125 + 0.21125) / 2 = 3.13625
//
// Red sigmoide con todos los pesos a 0: sigmoid(0) = 0.5 en cada salida, así que
//   x = [1, 0]: MSE = ((1 − 0.5)² + (0 − 0.5)²) / 2 = 0.25
//
// No existe un método reconstructionError en src/Autoencoder.ts: el MSE de la
// reconstrucción se comprueba con el valor que devuelve train() y con trainBatch().
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach.

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { Autoencoder } from '../src/Autoencoder'
import { SGD } from '../src/optimizers'
import { linear, sigmoid } from '../src/activations'
import { seedRandom, restoreRandom } from './helpers/seed'

beforeEach(() => seedRandom())
afterEach(() => restoreRandom())

const sgd = { optimizer: () => new SGD() }

/** Autoencoder 2 -> [] -> 1 -> [] -> 2 con los pesos de la cabecera. */
function linearNet(): Autoencoder {
  const ae = new Autoencoder(2, [], 1, [], { ...sgd, activations: [linear, linear] })
  ae.encoder.setWeights([0.5, -1, 0.25])
  ae.decoder.setWeights([2, 0.1, -1, 0])
  return ae
}

describe('Autoencoder constructor', () => {
  it('rejects inputSize < 1 with a clear message', () => {
    expect(() => new Autoencoder(0, [], 1, [])).toThrow(/Autoencoder: inputSize must be ≥ 1, got 0/)
    expect(() => new Autoencoder(-2, [], 1, [])).toThrow(/inputSize must be ≥ 1, got -2/)
  })

  it('rejects latentSize < 1 with a clear message', () => {
    expect(() => new Autoencoder(3, [], 0, [])).toThrow(/Autoencoder: latentSize must be ≥ 1, got 0/)
  })

  it('builds encoder [input, ...encHidden, latent] and decoder [latent, ...decHidden, input]', () => {
    const ae = new Autoencoder(3, [4], 2, [5, 6])
    expect(ae.encoder.structure).toEqual([3, 4, 2])
    expect(ae.decoder.structure).toEqual([2, 5, 6, 3])
  })

  it('latentSize >= inputSize is accepted: no throw and no console.warn (no warning exists in src)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const ae = new Autoencoder(2, [], 3, [])
      expect(ae.encoder.structure).toEqual([2, 3])
      expect(ae.decoder.structure).toEqual([3, 2])
      expect(warn).not.toHaveBeenCalled()
    } finally {
      warn.mockRestore()
    }
  })

  it('activations are split: first encoder layers, then decoder layers', () => {
    // encoder [2,3,1] has 2 layers, decoder [1,3,2] has 2 layers: 4 activations in total.
    const ae = new Autoencoder(2, [3], 1, [3], { activations: [linear, linear, linear, sigmoid] })
    expect(ae.encoder.layers.map((l) => l.neurons[0].activation)).toEqual([linear, linear])
    expect(ae.decoder.layers.map((l) => l.neurons[0].activation)).toEqual([linear, sigmoid])
  })

  it('too few activations: the override is dropped and both sub-networks use the sigmoid default', () => {
    // encoder [2,3,1] needs 2 activations and decoder [1,3,2] needs 2; only 1 is given, so
    // neither sub-network receives the override.
    const ae = new Autoencoder(2, [3], 1, [3], { activations: [linear] })
    expect(ae.encoder.layers.map((l) => l.neurons[0].activation)).toEqual([sigmoid, sigmoid])
    expect(ae.decoder.layers.map((l) => l.neurons[0].activation)).toEqual([sigmoid, sigmoid])
  })
})

describe('Autoencoder encode / decode / reconstruct', () => {
  it('encode gives the hand value z = −1.25 for x = [1, 2]', () => {
    const z = linearNet().encode([1, 2])
    expect(z).toHaveLength(1)
    expect(z[0]).toBeCloseTo(-1.25, 12)
  })

  it('decode gives the hand value [−2.4, 1.25] for z = −1.25', () => {
    const xHat = linearNet().decode([-1.25])
    expect(xHat).toHaveLength(2)
    expect(xHat[0]).toBeCloseTo(-2.4, 12)
    expect(xHat[1]).toBeCloseTo(1.25, 12)
  })

  it('reconstruct(x) gives [−2.4, 1.25] for x = [1, 2] (through the latent code)', () => {
    const xHat = linearNet().reconstruct([1, 2])
    expect(xHat[0]).toBeCloseTo(-2.4, 12)
    expect(xHat[1]).toBeCloseTo(1.25, 12)
  })

  it('decode(encode(x)) is identical to reconstruct(x)', () => {
    const ae = linearNet()
    for (const x of [[1, 2], [0, 0], [-3, 0.5]]) {
      expect(ae.decode(ae.encode(x))).toEqual(ae.reconstruct(x))
    }
  })

  it('output sizes follow latentSize and inputSize (5 -> 4 -> 2 -> 3 -> 5)', () => {
    const ae = new Autoencoder(5, [4], 2, [3])
    const z = ae.encode([0.1, 0.2, 0.3, 0.4, 0.5])
    expect(z).toHaveLength(2)
    const xHat = ae.decode(z)
    expect(xHat).toHaveLength(5)
    expect(ae.reconstruct([0.1, 0.2, 0.3, 0.4, 0.5])).toHaveLength(5)
  })

  it('sigmoid network with zero weights: encode = [0.5], reconstruct = [0.5, 0.5]', () => {
    const ae = new Autoencoder(2, [], 1, [])
    ae.encoder.setWeights([0, 0, 0])
    ae.decoder.setWeights([0, 0, 0, 0])
    expect(ae.encode([5, -3])[0]).toBeCloseTo(0.5, 12)
    const xHat = ae.reconstruct([1, 0])
    expect(xHat[0]).toBeCloseTo(0.5, 12)
    expect(xHat[1]).toBeCloseTo(0.5, 12)
  })

  it('encode rejects an input of the wrong length', () => {
    expect(() => linearNet().encode([1, 2, 3])).toThrow(
      /Autoencoder\.encode: expected input of length 2, got 3/,
    )
    expect(() => linearNet().encode([1])).toThrow(/Autoencoder\.encode: expected input of length 2, got 1/)
  })

  it('reconstruct rejects an input of the wrong length (through encode)', () => {
    expect(() => linearNet().reconstruct([1, 2, 3])).toThrow(/Autoencoder\.encode: expected input of length 2/)
  })

  it('decode rejects a latent vector of the wrong length', () => {
    expect(() => linearNet().decode([1, 2])).toThrow(
      /Autoencoder\.decode: expected latent vector of length 1, got 2/,
    )
    expect(() => linearNet().decode([])).toThrow(/Autoencoder\.decode: expected latent vector of length 1, got 0/)
  })
})

describe('Autoencoder train / trainBatch (reconstruction MSE by hand)', () => {
  it('train returns the MSE of the reconstruction BEFORE the update: 6.06125 for x = [1, 2]', () => {
    const ae = linearNet()
    const loss = ae.train([1, 2], 0.1)
    expect(loss).toBeCloseTo(6.06125, 12)
  })

  it('train with the sigmoid network returns 0.25 for x = [1, 0] and zero weights', () => {
    const ae = new Autoencoder(2, [], 1, [])
    ae.encoder.setWeights([0, 0, 0])
    ae.decoder.setWeights([0, 0, 0, 0])
    expect(ae.train([1, 0], 0.1)).toBeCloseTo(0.25, 12)
  })

  it('trainBatch with lr = 0 (weights fixed) returns the mean MSE: (6.06125 + 0.21125) / 2 = 3.13625', () => {
    const ae = linearNet()
    const mean = ae.trainBatch([[1, 2], [0, 0]], 0)
    expect(mean).toBeCloseTo(3.13625, 12)
    // With lr = 0 the network is unchanged: the same reconstruction as before.
    expect(ae.reconstruct([1, 2])[0]).toBeCloseTo(-2.4, 12)
  })

  it('trainBatch rejects an empty batch', () => {
    expect(() => linearNet().trainBatch([], 0.1)).toThrow(/Autoencoder\.trainBatch: batch X must be non-empty/)
  })

  it('train rejects an input of the wrong length and leaves every weight untouched', () => {
    const ae = linearNet()
    const encBefore = ae.encoder.getWeights()
    const decBefore = ae.decoder.getWeights()
    expect(() => ae.train([1, 2, 3], 0.1)).toThrow(/Autoencoder\.train: expected input of length 2, got 3/)
    expect(ae.encoder.getWeights()).toEqual(encBefore)
    expect(ae.decoder.getWeights()).toEqual(decBefore)
  })
})
