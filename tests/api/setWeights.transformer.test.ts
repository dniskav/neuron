import { describe, it, expect } from 'vitest'
import { NetworkTransformer } from '../../src/NetworkTransformer'
import { NetworkTransformerRL } from '../../src/NetworkTransformerRL'

// I2 (T003): setWeights / setWeightsFlat / setWeightsStructured validan la
// longitud total ANTES de escribir. Redes pequeñas: dModel 4, 1 bloque, 2 cabezas.

const makeTF = () =>
  new NetworkTransformer(3, { vocabSize: 5, d_model: 4, nHeads: 2, d_ff: 8, nBlocks: 1, nClasses: 3 })

const makeRL = () =>
  new NetworkTransformerRL(3, 2, { d_model: 4, nHeads: 2, d_ff: 8, nBlocks: 1, nActions: 2 })

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

interface FlatModel {
  expected: number          // longitud total válida (desde la estructura)
  read(): number[]          // getWeights / getWeightsFlat
  write(w: unknown): void   // setter bajo prueba
}

// Batería T001 aplicada a un setter plano: corto, largo, NaN, Infinity,
// undefined y no-array. Cada error debe lanzar con el nombre del método y
// dejar el modelo exactamente igual.
function flatErrorBattery(method: string, make: () => FlatModel) {
  it('rechaza corto, largo, NaN, Infinity, undefined y no-array sin escribir nada', () => {
    const m = make()
    const good = m.read()
    const N = m.expected
    expect(good.length).toBe(N)

    const cases: [string, unknown, RegExp][] = [
      ['corto', good.slice(0, -1),
        new RegExp(`${esc(method)}: expected array of length ${N}, got ${N - 1}`)],
      ['largo', [...good, 0],
        new RegExp(`${esc(method)}: expected array of length ${N}, got ${N + 1}`)],
      ['NaN', good.map((v, i) => (i === 2 ? NaN : v)),
        new RegExp(`${esc(method)}: invalid value at index 2`)],
      ['Infinity', good.map((v, i) => (i === 1 ? Infinity : v)),
        new RegExp(`${esc(method)}: invalid value at index 1`)],
      ['undefined', undefined,
        new RegExp(`${esc(method)}: expected array, got undefined`)],
      ['no-array', 42,
        new RegExp(`${esc(method)}: expected array, got number`)],
    ]

    for (const [label, bad, re] of cases) {
      expect(() => m.write(bad), label).toThrow(re)
      expect(m.read(), `estado tras error: ${label}`).toEqual(good)
    }
  })

  it('set(get()) sigue funcionando y set(get() + 1) escribe exactamente', () => {
    const m = make()
    const before = m.read()

    m.write(before.slice())
    expect(m.read()).toEqual(before)

    const shifted = before.map(v => v + 1)
    m.write(shifted)
    expect(m.read()).toEqual(shifted)
  })
}

describe('NetworkTransformer.setWeights (I2, T003)', () => {
  const make = (): FlatModel => {
    const net = makeTF()
    return {
      expected: net.getWeights().length,
      read: () => net.getWeights(),
      write: (w) => net.setWeights(w as number[]),
    }
  }

  flatErrorBattery('NetworkTransformer.setWeights', make)
})

describe('NetworkTransformerRL.setWeights (I2, T003)', () => {
  const make = (): FlatModel => {
    const net = makeRL()
    return {
      expected: net.getWeights().length,
      read: () => net.getWeights(),
      write: (w) => net.setWeights(w as number[]),
    }
  }

  flatErrorBattery('NetworkTransformerRL.setWeights', make)
})

describe('NetworkTransformerRL.setWeightsFlat (I2, T003)', () => {
  const make = (): FlatModel => {
    const net = makeRL()
    return {
      expected: net.getWeightsFlat().length,
      read: () => net.getWeightsFlat(),
      write: (w) => net.setWeightsFlat(w as number[]),
    }
  }

  flatErrorBattery('NetworkTransformerRL.setWeightsFlat', make)
})

// ─── setWeightsStructured (aplanado) ──────────────────────────────────────────

const clone = <T>(x: T): T => structuredClone(x)

describe('NetworkTransformerRL.setWeightsStructured (I2, T003)', () => {
  const method = 'NetworkTransformerRL.setWeightsStructured'

  it('rechaza una longitud aplanada corta, larga o con NaN sin escribir nada', () => {
    const net = makeRL()
    const N = net.getWeightsFlat().length
    const before = net.getWeightsStructured()
    const beforeFlat = net.getWeightsFlat()

    const short = clone(before)
    short.outputBias = short.outputBias.slice(0, -1)

    const long = clone(before)
    long.outputBias = [...long.outputBias, 0]

    const nan = clone(before)
    nan.blocks[0].b1[0] = NaN

    expect(() => net.setWeightsStructured(short)).toThrow(
      new RegExp(`${esc(method)}: expected array of length ${N}, got ${N - 1}`),
    )
    expect(() => net.setWeightsStructured(long)).toThrow(
      new RegExp(`${esc(method)}: expected array of length ${N}, got ${N + 1}`),
    )
    expect(() => net.setWeightsStructured(nan)).toThrow(
      new RegExp(`${esc(method)}: invalid value at index`),
    )

    expect(net.getWeightsStructured()).toEqual(before)
    expect(net.getWeightsFlat()).toEqual(beforeFlat)
  })

  it('round trip válido y escritura exacta con el mismo orden de pesos', () => {
    const net = makeRL()
    const before = net.getWeightsStructured()
    const beforeFlat = net.getWeightsFlat()

    net.setWeightsStructured(clone(before))
    expect(net.getWeightsFlat()).toEqual(beforeFlat)

    // Cada hoja numérica +1: el aplanado de getWeightsFlat debe reflejar +1 en
    // las mismas posiciones (el orden de escritura no cambia).
    const shifted = mapNumbers(before, v => v + 1)
    net.setWeightsStructured(shifted)
    expect(net.getWeightsFlat()).toEqual(beforeFlat.map(v => v + 1))
  })
})

function mapNumbers(x: any, f: (n: number) => number): any {
  if (typeof x === 'number') return f(x)
  if (Array.isArray(x)) return x.map(v => mapNumbers(v, f))
  if (x && typeof x === 'object') {
    return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, mapNumbers(v, f)]))
  }
  return x
}
