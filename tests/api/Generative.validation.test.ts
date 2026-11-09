// Validación de entradas de los modelos generativos y contrastivos (T041, tarea 006-coverage).
//
// Cubre Autoencoder.train, VAE (train y options.activations), GAN (trainStep y el contrato de
// salida sigmoide de D) y ContrastiveLearning (trainStep y computeLoss).
//
// Criterio de cada caso: las entradas inválidas lanzan con el nombre del método ANTES de tocar
// pesos. Por eso cada caso compara los pesos antes y después del intento fallido. Los casos
// con batch parcialmente inválido fijan que no hay entrenamiento a medias.
//
// lr: se acepta cualquier número finito >= 0. lr = 0 es un no-op válido (medir sin entrenar) y lo
// usan tests existentes (tests/Autoencoder.test.ts, tests/GAN.test.ts, tests/oracle/GAN.oracle.test.ts).
// NaN, ±Infinity y negativos lanzan.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach (tests/helpers/seed.ts).
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { Autoencoder } from '../../src/Autoencoder'
import { VAE } from '../../src/VAE'
import { GAN } from '../../src/GAN'
import { ContrastiveLearning } from '../../src/ContrastiveLearning'
import { relu, sigmoid, tanh } from '../../src/activations'
import { seedRandom, restoreRandom } from '../helpers/seed'

beforeEach(() => {
  seedRandom(0x5eed)
})

afterEach(() => {
  restoreRandom()
})

// Instantánea de todos los pesos de una red (NetworkN.getWeights devuelve un array plano).
const weightsOf = (...nets: { getWeights(): number[] }[]): string =>
  JSON.stringify(nets.map((n) => n.getWeights()))

const BAD_LR: [string, number][] = [
  ['negativo', -0.1],
  ['NaN', Number.NaN],
  ['Infinity', Number.POSITIVE_INFINITY],
]

describe('Autoencoder.train: validación de entrada y lr', () => {
  it('lr = 0 es un no-op válido: devuelve la pérdida y no cambia ningún peso', () => {
    const ae = new Autoencoder(3, [2], 2, [2])
    const before = weightsOf(ae.encoder, ae.decoder)
    const loss = ae.train([0.1, 0.5, 0.9], 0)
    expect(Number.isFinite(loss)).toBe(true)
    expect(weightsOf(ae.encoder, ae.decoder)).toBe(before)
  })

  it.each(BAD_LR)('lr %s lanza y no toca los pesos', (_name, lr) => {
    const ae = new Autoencoder(3, [2], 2, [2])
    const before = weightsOf(ae.encoder, ae.decoder)
    expect(() => ae.train([0.1, 0.5, 0.9], lr)).toThrow(
      /^Autoencoder\.train: lr must be a finite number >= 0, got /,
    )
    expect(weightsOf(ae.encoder, ae.decoder)).toBe(before)
  })
})

describe('VAE.train: validación de entrada y lr', () => {
  it('lista vacía o de longitud errónea lanza con el nombre VAE.train', () => {
    const vae = new VAE(3, [2], 1, [2])
    expect(() => vae.train([], 0.1)).toThrow(/^VAE\.train: x: expected array of length 3, got 0$/)
    expect(() => vae.train([0.1, 0.2], 0.1)).toThrow(/^VAE\.train: x: expected array of length 3, got 2$/)
  })

  it('una entrada con NaN lanza y no toca los pesos', () => {
    const vae = new VAE(3, [2], 1, [2])
    const before = weightsOf(vae.encoder, vae.decoder)
    expect(() => vae.train([Number.NaN, 0.2, 0.3], 0.1)).toThrow(/^VAE\.train: x: invalid value at index 0: NaN$/)
    expect(weightsOf(vae.encoder, vae.decoder)).toBe(before)
  })

  it.each(BAD_LR)('lr %s lanza y no toca los pesos', (_name, lr) => {
    const vae = new VAE(3, [2], 1, [2])
    const before = weightsOf(vae.encoder, vae.decoder)
    expect(() => vae.train([0.1, 0.5, 0.9], lr)).toThrow(/^VAE\.train: lr must be a finite number >= 0, got /)
    expect(weightsOf(vae.encoder, vae.decoder)).toBe(before)
  })
})

describe('VAE: options.activations', () => {
  it('acepta una lista de longitud decoderHidden + 1 que cubre las capas ocultas del encoder', () => {
    expect(() => new VAE(1, [1], 1, [1], { activations: [relu, sigmoid] })).not.toThrow()
    expect(() => new VAE(1, [1, 1], 1, [1], { activations: [relu, sigmoid] })).not.toThrow()
  })

  it('lanza con un mensaje VAE si la longitud no es la de las capas del decoder (antes: "Expected 2 activations")', () => {
    expect(() => new VAE(1, [1], 1, [1], { activations: [relu] })).toThrow(
      /^VAE: options\.activations needs 2 entries \(one per decoder layer/,
    )
    expect(() => new VAE(1, [1], 1, [1], { activations: [relu, sigmoid, tanh] })).toThrow(
      /^VAE: options\.activations needs 2 entries/,
    )
  })

  it('lanza si no alcanza para las capas ocultas del encoder', () => {
    expect(() => new VAE(1, [1, 1, 1], 1, [], { activations: [relu] })).toThrow(
      /must cover the 3 encoder hidden layers; got 1$/,
    )
  })
})

describe('GAN.trainStep: validación de realBatch y lr', () => {
  it('lote vacío lanza (antes: devolvía NaN)', () => {
    const gan = new GAN(2, [3], 2, [3])
    expect(() => gan.trainStep([], 0.05)).toThrow(/^GAN\.trainStep: realBatch must be a non-empty array$/)
  })

  it('fila de ancho distinto de la salida de G lanza con el índice', () => {
    const gan = new GAN(2, [3], 2, [3])
    expect(() => gan.trainStep([[1, 2, 3]], 0.05)).toThrow(
      /^GAN\.trainStep: realBatch\[0\]: expected array of length 2, got 3$/,
    )
  })

  it('valor no finito en una fila lanza', () => {
    const gan = new GAN(2, [3], 2, [3])
    expect(() => gan.trainStep([[Number.NaN, 0]], 0.05)).toThrow(
      /^GAN\.trainStep: realBatch\[0\]: invalid value at index 0: NaN$/,
    )
  })

  it.each(BAD_LR)('lr %s lanza y no toca G ni D', (_name, lr) => {
    const gan = new GAN(2, [3], 2, [3])
    const before = weightsOf(gan.generator, gan.discriminator)
    expect(() => gan.trainStep([[0.1, 0.2]], lr)).toThrow(/^GAN\.trainStep: lr must be a finite number >= 0, got /)
    expect(weightsOf(gan.generator, gan.discriminator)).toBe(before)
  })

  it('una fila inválida al final del lote no deja el entrenamiento a medias', () => {
    const gan = new GAN(2, [3], 2, [3])
    const before = weightsOf(gan.generator, gan.discriminator)
    expect(() => gan.trainStep([[0.1, 0.2], [1, 2, 3]], 0.05)).toThrow(/realBatch\[1\]/)
    expect(weightsOf(gan.generator, gan.discriminator)).toBe(before)
  })
})

describe('GAN: contrato de salida sigmoide de D (antes: documentado, no validado)', () => {
  it('la D por defecto (sigmoide) se construye sin error', () => {
    expect(() => new GAN(2, [3], 2, [3])).not.toThrow()
    expect(() => new GAN(2, [3], 2, [3], { discriminatorOptions: { activations: [relu, sigmoid] } })).not.toThrow()
  })

  it('una salida de D distinta de sigmoide lanza en el constructor', () => {
    expect(
      () => new GAN(2, [3], 2, [3], { discriminatorOptions: { activations: [relu, tanh] } }),
    ).toThrow(/^GAN: the discriminator output must use the sigmoid activation/)
  })
})

describe('ContrastiveLearning: validación de pairs y lr', () => {
  it('trainStep([]) y computeLoss([]) lanzan (antes: NaN y pesos NaN)', () => {
    const c = new ContrastiveLearning(2, [3], 2)
    const before = weightsOf(c.encoder, c.projectionHead)
    expect(() => c.computeLoss([])).toThrow(/^ContrastiveLearning\.computeLoss: pairs must be a non-empty array$/)
    expect(() => c.trainStep([], 0.01)).toThrow(/^ContrastiveLearning\.trainStep: pairs must be a non-empty array$/)
    expect(weightsOf(c.encoder, c.projectionHead)).toBe(before)
  })

  it('una vista con ancho distinto de la entrada del encoder lanza con el índice', () => {
    const c = new ContrastiveLearning(2, [3], 2)
    expect(() => c.trainStep([[[1, 2, 3], [1, 2]]], 0.01)).toThrow(
      /^ContrastiveLearning\.trainStep: pairs\[0\]\[0\]: expected array of length 2, got 3$/,
    )
  })

  it('un valor NaN en una vista lanza', () => {
    const c = new ContrastiveLearning(2, [3], 2)
    expect(() => c.computeLoss([[[Number.NaN, 0], [1, 2]]])).toThrow(
      /^ContrastiveLearning\.computeLoss: pairs\[0\]\[0\]: invalid value at index 0: NaN$/,
    )
  })

  it('un par que no tiene exactamente dos vistas lanza', () => {
    const c = new ContrastiveLearning(2, [3], 2)
    expect(() => c.trainStep([[[1, 2]] as unknown as [number[], number[]]], 0.01)).toThrow(
      /^ContrastiveLearning\.trainStep: pairs\[0\] must be a pair of 2 views$/,
    )
  })

  it.each(BAD_LR)('lr %s lanza y no toca los pesos', (_name, lr) => {
    const c = new ContrastiveLearning(2, [3], 2)
    const before = weightsOf(c.encoder, c.projectionHead)
    expect(() => c.trainStep([[[0.1, 0.2], [0.3, 0.4]]], lr)).toThrow(
      /^ContrastiveLearning\.trainStep: lr must be a finite number >= 0, got /,
    )
    expect(weightsOf(c.encoder, c.projectionHead)).toBe(before)
  })

  it('un par inválido al final no deja el entrenamiento a medias', () => {
    const c = new ContrastiveLearning(2, [3], 2)
    const before = weightsOf(c.encoder, c.projectionHead)
    expect(() =>
      c.trainStep([[[0.1, 0.2], [0.3, 0.4]], [[1, 2, 3], [1, 2]]], 0.01),
    ).toThrow(/pairs\[1\]\[0\]/)
    expect(weightsOf(c.encoder, c.projectionHead)).toBe(before)
  })
})
