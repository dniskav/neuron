// T001 (I2): setWeights plano de WeightMatrix, BiasVector, EmbeddingMatrix,
// LayerNorm y BatchNorm valida la longitud ANTES de escribir.
//
// Para cada clase: array corto, largo, con NaN, con Infinity, con undefined
// (elemento o argumento) y que no sea array lanzan con el nombre
// `Clase.setWeights` y las longitudes; getWeights() queda idéntico; y
// set(get()) y set(get() + 1) siguen funcionando.

import { describe, it, expect } from 'vitest'
import { WeightMatrix, BiasVector, EmbeddingMatrix } from '../../src/MatMul'
import { LayerNorm } from '../../src/LayerNorm'
import { BatchNorm } from '../../src/BatchNorm'

interface Settable {
  getWeights(): number[]
  setWeights(weights: number[]): void
}

// Entradas inválidas construidas a partir del estado actual. Los valores
// desplazados (+0.5) DIFIEREN del estado, así que una escritura anterior a la
// validación se detecta en la comprobación de estado intacto.
function invalidCases(
  name: string,
  w: number[],
  n: number,
): Array<{ label: string; value: unknown; message: string }> {
  return [
    {
      label: 'array corto',
      value: w.slice(0, -1).map(v => v + 0.5),
      message: `${name}: expected array of length ${n}, got ${n - 1}`,
    },
    {
      label: 'array largo',
      value: [...w.map(v => v + 0.5), 0.5],
      message: `${name}: expected array of length ${n}, got ${n + 1}`,
    },
    {
      label: 'NaN',
      value: w.map((v, i) => (i === 1 ? NaN : v + 0.5)),
      message: `${name}: invalid value at index 1: NaN`,
    },
    {
      label: 'Infinity',
      value: w.map((v, i) => (i === 2 ? Infinity : v + 0.5)),
      message: `${name}: invalid value at index 2: Infinity`,
    },
    {
      label: 'elemento undefined',
      value: w.map((v, i) => (i === 0 ? undefined : v + 0.5)),
      message: `${name}: invalid value at index 0: undefined`,
    },
    {
      label: 'argumento undefined',
      value: undefined,
      message: `${name}: expected array, got undefined`,
    },
    {
      label: 'no es un array',
      value: 42,
      message: `${name}: expected array, got number`,
    },
  ]
}

function battery(name: string, make: () => Settable, n: number): void {
  describe(name, () => {
    it(`tiene ${n} pesos en total (sanity del tamaño usado en el test)`, () => {
      expect(make().getWeights()).toHaveLength(n)
    })

    for (const probe of invalidCases(name, make().getWeights(), n)) {
      it(`lanza con ${probe.label} y nombra ${name} con las longitudes`, () => {
        const m = make()
        expect(() => m.setWeights(probe.value as number[])).toThrow(probe.message)
      })

      it(`deja getWeights() idéntico tras ${probe.label}`, () => {
        const m = make()
        const before = m.getWeights()
        expect(() => m.setWeights(probe.value as number[])).toThrow()
        expect(m.getWeights()).toEqual(before)
      })
    }

    it('set(get()) deja los pesos idénticos', () => {
      const m = make()
      const w = m.getWeights()
      m.setWeights(m.getWeights())
      expect(m.getWeights()).toEqual(w)
    })

    it('set(get() + 1) escribe exactamente los valores nuevos', () => {
      const m = make()
      const w = m.getWeights()
      m.setWeights(w.map(v => v + 1))
      expect(m.getWeights()).toEqual(w.map(v => v + 1))
    })
  })
}

battery('WeightMatrix.setWeights', () => new WeightMatrix(2, 3), 6)
battery('BiasVector.setWeights', () => new BiasVector(4), 4)
battery('EmbeddingMatrix.setWeights', () => new EmbeddingMatrix(3, 2), 6)
battery('LayerNorm.setWeights', () => new LayerNorm(3), 6)
battery('BatchNorm.setWeights', () => new BatchNorm(3), 6)

// La longitud esperada de LayerNorm y BatchNorm es 2·dim (gamma y beta), no dim.
describe('LayerNorm y BatchNorm esperan gamma y beta (2·dim)', () => {
  it('LayerNorm rechaza un array de dim elementos', () => {
    const ln = new LayerNorm(3)
    const before = ln.getWeights()
    expect(() => ln.setWeights([0.5, 0.5, 0.5])).toThrow(
      'LayerNorm.setWeights: expected array of length 6, got 3',
    )
    expect(ln.getWeights()).toEqual(before)
  })

  it('BatchNorm rechaza un array de dim elementos', () => {
    const bn = new BatchNorm(3)
    const before = bn.getWeights()
    expect(() => bn.setWeights([0.5, 0.5, 0.5])).toThrow(
      'BatchNorm.setWeights: expected array of length 6, got 3',
    )
    expect(bn.getWeights()).toEqual(before)
  })
})
