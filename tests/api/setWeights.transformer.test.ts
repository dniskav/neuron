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

    const nOut = before.outputBias.length
    expect(N).toBeGreaterThan(0)
    expect(() => net.setWeightsStructured(short)).toThrow(
      new RegExp(`${esc(method)}\\(outputBias\\): expected array of length ${nOut}, got ${nOut - 1}`),
    )
    expect(() => net.setWeightsStructured(long)).toThrow(
      new RegExp(`${esc(method)}\\(outputBias\\): expected array of length ${nOut}, got ${nOut + 1}`),
    )
    expect(() => net.setWeightsStructured(nan)).toThrow(
      new RegExp(`${esc(method)}\\(blocks\\[0\\]\\.b1\\): invalid value at index 0`),
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

describe('NetworkTransformerRL.setWeightsStructured atomico y con forma (T020)', () => {
  const method = 'NetworkTransformerRL.setWeightsStructured'
  type S = ReturnType<NetworkTransformerRL['getWeightsStructured']>

  it('mover un elemento de b1 al final de ff1[0] lanza y deja el modelo intacto', () => {
    const net = makeRL()
    const before = net.getWeightsStructured()
    const flat = net.getWeightsFlat()
    const bad = clone(before)
    bad.blocks[0].ff1[0].push(bad.blocks[0].b1.pop() as number)
    expect(() => net.setWeightsStructured(bad)).toThrow(new RegExp(esc(method)))
    expect(net.getWeights()).toEqual(flat)
    expect(net.getWeightsStructured()).toEqual(before)
  })

  it('reformar Wq de 4x2 a 2x4 (mismo total) lanza y deja el modelo intacto', () => {
    const net = makeRL()
    const before = net.getWeightsStructured()
    const flat = net.getWeightsFlat()
    const bad = clone(before)
    const wq = bad.blocks[0].attn.heads[0].Wq
    const [r, c] = [wq.length, wq[0].length]
    expect(r).not.toBe(c)
    const f = wq.flat()
    // Mismo total, filas y columnas intercambiadas (p. ej. 2x4 -> 4x2)
    bad.blocks[0].attn.heads[0].Wq = Array.from({ length: c }, (_, i) => f.slice(i * r, (i + 1) * r))
    expect(bad.blocks[0].attn.heads[0].Wq.flat().length).toBe(f.length)
    expect(() => net.setWeightsStructured(bad)).toThrow(new RegExp(esc(method) + '\\(blocks\\[0\\]\\.heads\\[0\\]\\.Wq\\)'))
    expect(net.getWeights()).toEqual(flat)
  })

  // Un fallo de forma en cada tipo de matriz/vector no escribe nada
  const mutations: [string, (d: S) => void][] = [
    ['inputProj', d => { d.inputProj[0].pop() }],
    ['Wk', d => { d.blocks[0].attn.heads[1].Wk.pop() }],
    ['Wv', d => { d.blocks[0].attn.heads[0].Wv[0].push(1) }],
    ['Wo', d => { d.blocks[0].attn.Wo[1].pop() }],
    ['ff1', d => { d.blocks[0].ff1[0].push(d.blocks[0].ff1[1].pop() as number) }],
    ['ff2', d => { d.blocks[0].ff2.pop() }],
    ['norm1.gamma', d => { d.blocks[0].norm1.gamma.pop() }],
    ['norm2.beta', d => { d.blocks[0].norm2.beta.push(0) }],
    ['b2', d => { d.blocks[0].b2.pop() }],
    ['outputProj', d => { d.outputProj[0].pop() }],
    ['outputBias', d => { d.outputBias.pop() }],
    ['numero de cabezas', d => { d.blocks[0].attn.heads.pop() }],
    ['numero de bloques', d => { d.blocks.pop() }],
  ]
  for (const [label, mutate] of mutations) {
    it(`forma invalida en ${label}: lanza y no cambia nada`, () => {
      const net = makeRL()
      // Pesos distintos en TODAS partes: si algo se escribiera antes del fallo, se vería.
      const shifted = mapNumbers(net.getWeightsStructured(), (v: number) => v + 1) as S
      const flat = net.getWeightsFlat()
      mutate(shifted)
      expect(() => net.setWeightsStructured(shifted), label).toThrow(new RegExp(esc(method)))
      expect(net.getWeightsFlat()).toEqual(flat)
    })
  }

  it('el fallo en el ultimo elemento (outputBias) tampoco deja escrito el principio', () => {
    const net = makeRL()
    const flat = net.getWeightsFlat()
    const bad = mapNumbers(net.getWeightsStructured(), (v: number) => v + 1) as S
    bad.outputBias.pop()
    expect(() => net.setWeightsStructured(bad)).toThrow(/outputBias/)
    expect(net.getWeightsFlat()).toEqual(flat)
  })

  it('setWeightsStructured(getWeightsStructured()) es identidad', () => {
    const net = makeRL()
    net.train([[1, 2], [3, 4], [5, 6]], [1, 0], 0.01)
    const flat = net.getWeightsFlat()
    const st = net.getWeightsStructured()
    net.setWeightsStructured(st)
    expect(net.getWeightsFlat()).toEqual(flat)
    expect(net.getWeightsStructured()).toEqual(st)
  })

  it('una fila de 200000 elementos no desborda la pila', () => {
    const big = new NetworkTransformerRL(1, 200000, { d_model: 2, nHeads: 1, d_ff: 2, nBlocks: 1, nActions: 1 })
    const st = big.getWeightsStructured()
    expect(st.inputProj[0].length).toBe(200000)
    const shifted = clone(st)
    shifted.inputProj[0][199999] = 7
    big.setWeightsStructured(shifted)
    expect(big.getWeightsStructured().inputProj[0][199999]).toBe(7)
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
