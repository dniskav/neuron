// Tests de comportamiento de GAN (T030, tarea 006-coverage).
//
// Complementan tests/oracle/GAN.oracle.test.ts sin repetirlo: el oráculo fija el coseno de los
// gradientes de G y D, dLoss/gLoss frente a una réplica y el aprendizaje de G contra un D fijo.
// Aquí se fijan con valores a mano: el número de parámetros y las capas de G y D, la salida
// sigmoide, la salida de G para un z explícito, el muestreo Box-Muller con (u1, u2) fijados, la
// distribución de z, la regla de actualización de D en un paso, qué pesos cambia trainStep según
// lr, y el aprendizaje de D con G congelado.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach (tests/helpers/seed.ts).
// withStub entrega exactamente los valores de Math.random que se le dan y lanza si se pide más,
// así que además fija cuántos números consume cada método.
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { GAN } from '../src/GAN'
import { sigmoid, type Activation } from '../src/activations'
import { expectLearns } from './helpers/learns'
import { seedRandom, restoreRandom } from './helpers/seed'

beforeEach(() => {
  seedRandom(0x5eed)
})

afterEach(() => {
  restoreRandom()
})

// Math.random que devuelve `values` en orden y lanza si se pide otro valor.
// Devuelve el resultado de `fn` y el número de valores consumidos.
function withStub<T>(values: number[], fn: () => T): { value: T; draws: number } {
  const previous = Math.random
  let i = 0
  Math.random = () => {
    if (i >= values.length) throw new Error(`Math.random pedido más de ${values.length} veces`)
    return values[i++]
  }
  try {
    const value = fn()
    return { value, draws: i }
  } finally {
    Math.random = previous
  }
}

function expectClose(actual: number[], expected: number[], digits: number): void {
  expect(actual).toHaveLength(expected.length)
  expected.forEach((e, i) => expect(actual[i]).toBeCloseTo(e, digits))
}

const zeros = (n: number): number[] => new Array(n).fill(0)

describe('GAN: construcción', () => {
  it('G y D tienen la estructura, las capas y el número de parámetros pedidos', () => {
    const gan = new GAN(2, [4], 3, [4])
    expect(gan.latentDim).toBe(2)
    expect(gan.generator.structure).toEqual([2, 4, 3])
    expect(gan.discriminator.structure).toEqual([3, 4, 1])
    expect(gan.generator.layers.map((l) => l.neurons.length)).toEqual([4, 3])
    expect(gan.discriminator.layers.map((l) => l.neurons.length)).toEqual([4, 1])
    // Parámetros = Σ_capas (entradas + 1 sesgo) · neuronas.
    // G: (2+1)·4 + (4+1)·3 = 12 + 15 = 27.  D: (3+1)·4 + (4+1)·1 = 16 + 5 = 21.
    expect(gan.generator.getWeights()).toHaveLength(27)
    expect(gan.discriminator.getWeights()).toHaveLength(21)
  })

  it('G y D usan sigmoide en todas sus capas, también en la salida de D', () => {
    const gan = new GAN(2, [4], 3, [4])
    for (const net of [gan.generator, gan.discriminator]) {
      for (const layer of net.layers) {
        for (const neuron of layer.neurons) expect(neuron.activation).toBe(sigmoid)
      }
    }
  })
})

describe('GAN: salidas con pesos fijados a mano', () => {
  it('con pesos de G y D a cero, cada salida es σ(0) = 1/2 y D devuelve un escalar', () => {
    const gan = new GAN(2, [4], 3, [4])
    gan.generator.setWeights(zeros(27))
    gan.discriminator.setWeights(zeros(21))
    // Con todos los pesos y sesgos a 0, cada neurona calcula σ(0) = 1/(1+e⁰) = 1/2 en cada capa.
    expect(gan.generate([0.3, -1])).toEqual([0.5, 0.5, 0.5])
    expect(typeof gan.discriminate([0.2, 0.3, 0.1])).toBe('number')
    expect(gan.discriminate([0.2, 0.3, 0.1])).toBe(0.5)
  })

  it('D con una neurona y pesos (1, 0) y sesgo 0 calcula σ(x₀)', () => {
    const gan = new GAN(1, [], 2, [])
    // D tiene estructura [2, 1]: pesos [w₀, w₁] y sesgo. Aquí [1, 0, 0].
    gan.discriminator.setWeights([1, 0, 0])
    // D([0.5, 0.9]) = σ(1·0.5 + 0·0.9 + 0) = σ(0.5) = 1/(1 + e^(−0.5)) ≈ 0.6224593312018546
    expect(gan.discriminate([0.5, 0.9])).toBeCloseTo(0.6224593312018546, 12)
    // D([−1, 7]) = σ(−1) = 1/(1 + e) ≈ 0.2689414213699951 (el segundo valor no pesa)
    expect(gan.discriminate([-1, 7])).toBeCloseTo(0.2689414213699951, 12)
  })

  it('G con z explícito: G(z) = [σ(2z), σ(1 − 2z)]', () => {
    const gan = new GAN(1, [], 2, [])
    // G tiene estructura [1, 2]. Neurona 0: peso 2, sesgo 0 → σ(2z). Neurona 1: peso −2, sesgo 1 → σ(1 − 2z).
    gan.generator.setWeights([2, 0, -2, 1])
    // z = 0.5: [σ(1), σ(0)] = [0.7310585786300049, 0.5]
    expectClose(gan.generate([0.5]), [0.7310585786300049, 0.5], 12)
    // z = −1: [σ(−2), σ(3)] = [1/(1+e²), 1/(1+e⁻³)] ≈ [0.11920292202211755, 0.9525741268224334]
    expectClose(gan.generate([-1]), [0.11920292202211755, 0.9525741268224334], 12)
  })
})

describe('GAN.sampleLatent: Box-Muller con valores fijados', () => {
  it('latentDim 2, (u1, u2) = (0.5, 0.25): z = [r·cos(π/2), r·sin(π/2)] = [0, √(2 ln 2)]', () => {
    const gan = new GAN(2, [], 1, [])
    const { value: z, draws } = withStub([0.5, 0.25], () => gan.sampleLatent())
    // r = √(−2 ln 0.5) = √(2 ln 2) ≈ 1.1774100225154747; θ = 2π·0.25 = π/2, así cos θ = 0 y sen θ = 1.
    expect(draws).toBe(2)
    expectClose(z, [0, 1.1774100225154747], 12)
  })

  it('latentDim 2, (u1, u2) = (e^(−1/2), 0): r = 1 y θ = 0, así que z = [1, 0]', () => {
    const gan = new GAN(2, [], 1, [])
    const { value: z, draws } = withStub([Math.exp(-0.5), 0], () => gan.sampleLatent())
    // r = √(−2 ln e^(−1/2)) = √1 = 1; θ = 0, así cos θ = 1 y sen θ = 0.
    expect(draws).toBe(2)
    expectClose(z, [1, 0], 12)
  })

  it('latentDim 3: el último par solo aporta el coseno y se consumen 4 valores', () => {
    const gan = new GAN(3, [], 1, [])
    const { value: z, draws } = withStub([Math.exp(-0.5), 0, 0.5, 0], () => gan.sampleLatent())
    // Par 1 (e^(−1/2), 0) → [r·cos 0, r·sen 0] = [1, 0].
    // Par 2 (0.5, 0): r = √(2 ln 2), θ = 0 → cos = 1; el seno no se guarda porque i+1 = 3 no es < 3.
    expect(z).toHaveLength(3)
    expectClose(z, [1, 0, 1.1774100225154747], 12)
    expect(draws).toBe(4)
  })

  it('distribución: z ~ N(0, 1), con media 0 y E[z²] = 1, en 5 semillas', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      seedRandom(seed)
      const gan = new GAN(2, [], 1, [])
      let sum = 0
      let sq = 0
      let n = 0
      for (let i = 0; i < 20000; i++) {
        for (const v of gan.sampleLatent()) {
          sum += v
          sq += v * v
          n++
        }
      }
      // Esperado: E[z] = 0 y E[z²] = 1. Con n = 40 000 valores, el error típico de la media es
      // 1/√n = 0.005 y el de E[z²] es √(2/n) = 0.007. Umbral 0.03 (≈ 6 y 4 errores típicos).
      // Peor caso medido en estas 5 semillas: |media| 0.0089 y |E[z²] − 1| 0.0118, ambos por
      // debajo de 0.7 × 0.03 = 0.021.
      expect(Math.abs(sum / n)).toBeLessThan(0.03)
      expect(Math.abs(sq / n - 1)).toBeLessThan(0.03)
    }
  })
})

describe('GAN.generate', () => {
  it('sin argumento muestrea z con sampleLatent; con z = [1] da [σ(2), σ(−1)]', () => {
    const gan = new GAN(1, [], 2, [])
    gan.generator.setWeights([2, 0, -2, 1])
    // latentDim 1: un par (e^(−1/2), 0) → z = [r·cos 0] = [1].
    const { value: x, draws } = withStub([Math.exp(-0.5), 0], () => gan.generate())
    // G([1]) = [σ(2·1 + 0), σ(−2·1 + 1)] = [σ(2), σ(−1)] ≈ [0.8807970779778823, 0.2689414213699951]
    expect(draws).toBe(2)
    expectClose(x, [0.8807970779778823, 0.2689414213699951], 12)
  })

  it('con la misma semilla, generate() da la misma muestra; con otra semilla, otra', () => {
    const gan = new GAN(2, [4], 3, [4])
    seedRandom(11)
    const a = gan.generate()
    seedRandom(11)
    const b = gan.generate()
    seedRandom(12)
    const c = gan.generate()
    expect(b).toEqual(a)
    expect(c).not.toEqual(a)
  })
})

describe('GAN.trainStep: qué pesos cambian', () => {
  const batch = [
    [0.2, 0.3, 0.1],
    [0.9, 0.1, 0.4],
  ]

  it('lr = 0: ningún peso de G ni de D cambia; las dos pérdidas son finitas y positivas', () => {
    const gan = new GAN(2, [4], 3, [4])
    const gBefore = gan.generator.getWeights()
    const dBefore = gan.discriminator.getWeights()
    const { dLoss, gLoss } = gan.trainStep(batch, 0)
    expect(gan.generator.getWeights()).toEqual(gBefore)
    expect(gan.discriminator.getWeights()).toEqual(dBefore)
    // dLoss = −log D(x) − log(1 − D(x̂)) > 0 y gLoss = −log D(G(z₂)) > 0, porque D está en (0, 1).
    expect(Number.isFinite(dLoss) && dLoss > 0).toBe(true)
    expect(Number.isFinite(gLoss) && gLoss > 0).toBe(true)
  })

  it('lr > 0: en el mismo paso cambian a la vez los pesos de G y los de D', () => {
    const gan = new GAN(2, [4], 3, [4])
    const gBefore = gan.generator.getWeights()
    const dBefore = gan.discriminator.getWeights()
    gan.trainStep(batch, 0.1)
    const maxDiff = (after: number[], before: number[]) =>
      Math.max(...after.map((v, i) => Math.abs(v - before[i])))
    expect(maxDiff(gan.generator.getWeights(), gBefore)).toBeGreaterThan(1e-6)
    expect(maxDiff(gan.discriminator.getWeights(), dBefore)).toBeGreaterThan(1e-6)
  })
})

describe('GAN: regla de actualización de D en un paso (a mano)', () => {
  it('batch [[1]], G y D a cero, lr 0.5: w = (1 − σ(3/8))/4 y b = 1/4 − σ(3/8)/2', () => {
    const gan = new GAN(1, [], 1, [])
    gan.generator.setWeights([0, 0])
    gan.discriminator.setWeights([0, 0])
    gan.trainStep([[1]], 0.5)
    // Real (x = 1): D(1) = σ(0) = 1/2, delta = 1 − 1/2 = 1/2.
    //   w += lr·delta·x = 0.5·(1/2)·1 = 1/4;   b += lr·delta = 1/4.
    // Falso: G(z) = σ(0) = 1/2 para cualquier z, así que x̂ = 1/2. D ya actualizado:
    //   D(1/2) = σ((1/4)·(1/2) + 1/4) = σ(3/8), delta = 0 − σ(3/8).
    //   w += 0.5·(−σ(3/8))·(1/2) = −σ(3/8)/4;   b += 0.5·(−σ(3/8)) = −σ(3/8)/2.
    // Resultado: w = 1/4 − σ(3/8)/4 y b = 1/4 − σ(3/8)/2.
    const s = 1 / (1 + Math.exp(-0.375)) // σ(3/8) ≈ 0.5927
    const [w, b] = gan.discriminator.getWeights()
    expect(w).toBeCloseTo((1 - s) / 4, 12)
    expect(b).toBeCloseTo(0.25 - s / 2, 12)
  })
})

describe('GAN: el discriminador aprende con G congelado', () => {
  it('la pérdida real/falso de D cae a menos de 0.1 × la inicial (5 semillas, lr ×0.7/×1/×1.3)', () => {
    // G de estructura [1, 3] con pesos 0 y sesgo b = ln(0.1/0.9): σ(b) = 0.1 para todo z.
    const b = Math.log(0.1 / 0.9)
    const gFrozen = [0, b, 0, b, 0, b]
    const real = [0.9, 0.9, 0.9]
    let worst = 0
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const k of [0.7, 1, 1.3]) {
        seedRandom(seed)
        const gan = new GAN(1, [], 3, [])
        gan.generator.setWeights(gFrozen)
        // Premisa: G da [0.1, 0.1, 0.1] para cualquier z, así que el falso fijo es ese vector.
        expectClose(gan.generate(), [0.1, 0.1, 0.1], 12)
        const r = expectLearns({
          // L_D = −log D(real) − log(1 − D(falso)), la pérdida que minimiza el paso de D.
          loss: () => -Math.log(gan.discriminate(real)) - Math.log(1 - gan.discriminate([0.1, 0.1, 0.1])),
          step: () => {
            gan.trainStep([real], 0.5 * k)
            gan.generator.setWeights(gFrozen)
          },
          steps: 300,
          ratio: 0.1,
        })
        worst = Math.max(worst, r.ratio)
      }
    }
    // Peor caso medido: 0.032 (final/inicial). Margen exigido ≥ 30 % sobre el umbral: < 0.07.
    expect(worst).toBeLessThan(0.07)
  })
})

describe('GAN: sigmoide equivalente por valor (T065)', () => {
  // Sigmoide propia: distinto objeto que `sigmoid`, mismo valor en todo el dominio.
  const own: Activation = { fn: (x) => 1 / (1 + Math.exp(-x)), dfn: (o) => o * (1 - o) }
  const batches = [
    [
      [0.2, 0.3, 0.1],
      [0.9, 0.1, 0.4],
    ],
    [[0.5, 0.5, 0.5]],
    [
      [0.1, 0.9, 0.2],
      [0.4, 0.4, 0.8],
      [0.7, 0.2, 0.3],
    ],
  ]

  function trainWith(outAct: Activation) {
    seedRandom(0x5eed)
    const gan = new GAN(2, [4], 3, [4], { discriminatorOptions: { activations: [sigmoid, outAct] } })
    const losses = batches.map((b) => gan.trainStep(b, 0.1))
    return { gan, losses }
  }

  it('entrena igual que con la sigmoide exportada: mismos pesos y pérdidas tras 3 trainStep (1e-12)', () => {
    const exported = trainWith(sigmoid)
    const equivalent = trainWith(own)
    const expectSame = (a: number[], b: number[]) => {
      expect(a).toHaveLength(b.length)
      a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThanOrEqual(1e-12))
    }
    expectSame(equivalent.gan.generator.getWeights(), exported.gan.generator.getWeights())
    expectSame(equivalent.gan.discriminator.getWeights(), exported.gan.discriminator.getWeights())
    equivalent.losses.forEach((l, i) => {
      expect(Math.abs(l.dLoss - exported.losses[i].dLoss)).toBeLessThanOrEqual(1e-12)
      expect(Math.abs(l.gLoss - exported.losses[i].gLoss)).toBeLessThanOrEqual(1e-12)
    })
  })

  it('aceptar la sigmoide propia no consume Math.random: el siguiente valor coincide con el de la exportada', () => {
    seedRandom(7)
    new GAN(2, [4], 3, [4], { discriminatorOptions: { activations: [sigmoid, sigmoid] } })
    const nextExported = Math.random()
    seedRandom(7)
    new GAN(2, [4], 3, [4], { discriminatorOptions: { activations: [sigmoid, own] } })
    expect(Math.random()).toBe(nextExported)
  })
})
