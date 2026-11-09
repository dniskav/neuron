// T010 (I2 + validación): NetworkN.setWeights valida la longitud total ANTES de
// resetear máscaras o escribir; NetworkN.trainWithDeltas valida inputs y
// outputDeltas con validateArray ANTES de consumir Math.random o tocar pesos.
//
// Golden: un trainWithDeltas válido da los mismos pesos que antes (constantes
// capturadas en la base, 1e-9) y coincide con train() usando sus deltas.

import { describe, it, expect, vi, afterEach } from 'vitest'
import { NetworkN } from '../../src/NetworkN'
import { Dropout } from '../../src/Dropout'

afterEach(() => {
  vi.restoreAllMocks()
})

// Estado no trivial y determinista (sin azar de Xavier).
function pattern(n: number): number[] {
  return Array.from({ length: n }, (_, i) => 0.05 * ((i % 7) - 3))
}

function patterned(structure: number[], options?: ConstructorParameters<typeof NetworkN>[1]): NetworkN {
  const net = new NetworkN(structure, options)
  net.setWeights(pattern(net.getWeights().length))
  return net
}

// Longitud total esperada: Σ (structure[i] + 1)·structure[i+1].
function expectedLength(structure: number[]): number {
  let total = 0
  for (let i = 0; i < structure.length - 1; i++) {
    total += (structure[i] + 1) * structure[i + 1]
  }
  return total
}

function maxAbsDiff(a: number[], b: number[]): number {
  expect(a.length).toBe(b.length)
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

// ─── setWeights ──────────────────────────────────────────────────────────────

describe('NetworkN.setWeights validation (I2)', () => {
  it('getWeights length equals Σ (structure[i] + 1)·structure[i+1]', () => {
    expect(new NetworkN([3, 4, 2]).getWeights().length).toBe(26)
    expect(new NetworkN([4, 8, 2]).getWeights().length).toBe(58)
    expect(new NetworkN([3, 4, 4, 2]).getWeights().length).toBe(46)
  })

  function battery(structure: number[], options?: ConstructorParameters<typeof NetworkN>[1]): void {
    const expected = expectedLength(structure)
    const method = 'NetworkN.setWeights'

    function rejectedAndIntact(bad: unknown, msg: string): void {
      const net = patterned(structure, options)
      const before = net.getWeights()
      expect(() => net.setWeights(bad as number[])).toThrow(msg)
      expect(net.getWeights()).toEqual(before)
    }

    describe(`${method} on [${structure.join(', ')}] (${expected} valores)`, () => {
      it('set(get()) keeps the weights identical', () => {
        const net = patterned(structure, options)
        const w = net.getWeights()
        net.setWeights(net.getWeights())
        expect(net.getWeights()).toEqual(w)
      })

      it('set(get() + 1) writes exactly the given values', () => {
        const net = patterned(structure, options)
        const shifted = net.getWeights().map(v => v + 1)
        net.setWeights(shifted)
        expect(net.getWeights()).toEqual(shifted)
      })

      it('rejects a short array with name and both lengths', () => {
        const net = patterned(structure, options)
        const short = net.getWeights().slice(0, expected - 1)
        const before = net.getWeights()
        expect(() => net.setWeights(short)).toThrow(
          `${method}: expected array of length ${expected}, got ${expected - 1}`,
        )
        expect(net.getWeights()).toEqual(before)
      })

      it('rejects a long array with name and both lengths', () => {
        rejectedAndIntact(
          [...patterned(structure, options).getWeights(), 0.5],
          `${method}: expected array of length ${expected}, got ${expected + 1}`,
        )
      })

      it('rejects NaN inside a correctly sized array', () => {
        const bad = patterned(structure, options).getWeights()
        bad[bad.length - 1] = NaN
        rejectedAndIntact(bad, `${method}: invalid value at index ${expected - 1}`)
      })

      it('rejects Infinity inside a correctly sized array', () => {
        const bad = patterned(structure, options).getWeights()
        bad[0] = Infinity
        rejectedAndIntact(bad, `${method}: invalid value at index 0`)
      })

      it('rejects undefined', () => {
        rejectedAndIntact(undefined, `${method}: expected array, got undefined`)
      })

      it('rejects a value that is not an array', () => {
        rejectedAndIntact({ length: expected }, `${method}: expected array, got object`)
      })
    })
  }

  battery([3, 4, 2])
  battery([3, 4, 4, 2], { residual: true })
  battery([4, 8, 2], { dropoutRate: 0.5 })

  it('an invalid setWeights does not reset dropout masks (validates before resetMask)', () => {
    const net = patterned([4, 8, 2], { dropoutRate: 0.5 })
    const w = net.getWeights()
    const spy = vi.spyOn(Dropout.prototype, 'resetMask')

    expect(() => net.setWeights(pattern(5))).toThrow(
      'NetworkN.setWeights: expected array of length 58, got 5',
    )
    expect(spy).not.toHaveBeenCalled()

    net.setWeights(w)
    expect(spy).toHaveBeenCalledTimes(1)
  })
})

// ─── trainWithDeltas ─────────────────────────────────────────────────────────

describe('NetworkN.trainWithDeltas validation (FR-011)', () => {
  const method = 'NetworkN.trainWithDeltas'

  it('rejects short inputs with name and both lengths; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([0.1, 0.2], [0.3, -0.2], 0.5)).toThrow(
      `${method}: expected array of length 3, got 2`,
    )
    expect(net.getWeights()).toEqual(before)
  })

  it('rejects long inputs with name and both lengths; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([0.1, 0.2, 0.3, 0.4], [0.3, -0.2], 0.5)).toThrow(
      `${method}: expected array of length 3, got 4`,
    )
    expect(net.getWeights()).toEqual(before)
  })

  it('rejects NaN inputs; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([0.1, NaN, 0.3], [0.3, -0.2], 0.5)).toThrow(
      `${method}: invalid value at index 1: NaN`,
    )
    expect(net.getWeights()).toEqual(before)
  })

  it('rejects undefined inputs; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() =>
      net.trainWithDeltas(undefined as unknown as number[], [0.3, -0.2], 0.5),
    ).toThrow(`${method}: expected array, got undefined`)
    expect(net.getWeights()).toEqual(before)
  })

  it('rejects short outputDeltas with name and both lengths; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([0.1, 0.2, 0.3], [0.3], 0.5)).toThrow(
      `${method}: expected array of length 2, got 1`,
    )
    expect(net.getWeights()).toEqual(before)
  })

  it('rejects long outputDeltas with name and both lengths; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([0.1, 0.2, 0.3], [0.3, -0.2, 0.1], 0.5)).toThrow(
      `${method}: expected array of length 2, got 3`,
    )
    expect(net.getWeights()).toEqual(before)
  })

  it('checks outputDeltas against the output size, not structure[0]', () => {
    // [4, 8, 2]: structure[0] = 4 but the output has 2 values.
    const net = patterned([4, 8, 2])
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], 0.5)).toThrow(
      `${method}: expected array of length 2, got 4`,
    )
    expect(net.getWeights()).toEqual(before)
  })

  it('rejects NaN outputDeltas; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([0.1, 0.2, 0.3], [NaN, -0.2], 0.5)).toThrow(
      `${method}: invalid value at index 0: NaN`,
    )
    expect(net.getWeights()).toEqual(before)
  })

  it('rejects undefined outputDeltas; weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() =>
      net.trainWithDeltas([0.1, 0.2, 0.3], undefined as unknown as number[], 0.5),
    ).toThrow(`${method}: expected array, got undefined`)
    expect(net.getWeights()).toEqual(before)
  })

  it('does not consume Math.random when the arguments are invalid (dropout on)', () => {
    const net = patterned([4, 8, 2], { dropoutRate: 0.5 })
    const before = net.getWeights()
    const spy = vi.spyOn(Math, 'random')

    expect(() => net.trainWithDeltas([1, 2, 3], [0.1, 0.2], 0.5)).toThrow(method)
    expect(() => net.trainWithDeltas([1, 2, 3, 4], [0.1], 0.5)).toThrow(method)
    expect(() => net.trainWithDeltas([1, 2, 3, 4], [0.1, NaN], 0.5)).toThrow(method)
    expect(spy).not.toHaveBeenCalled()
    expect(net.getWeights()).toEqual(before)

    // Sanity: a valid call does consume Math.random through the dropout mask.
    net.trainWithDeltas([1, 2, 3, 4], [0.1, 0.2], 0.5)
    expect(spy).toHaveBeenCalled()
  })

  it('inputGradient already rejects bad lengths before drawing a mask (regression guard)', () => {
    const net = patterned([4, 8, 2], { dropoutRate: 0.5 })
    const spy = vi.spyOn(Math, 'random')
    expect(() => net.inputGradient([1, 2, 3], [0.1, 0.2])).toThrow(
      'NetworkN.inputGradient: expected array of length 4, got 3',
    )
    expect(() => net.inputGradient([1, 2, 3, 4], [0.1])).toThrow(
      'NetworkN.inputGradient: expected array of length 2, got 1',
    )
    expect(spy).not.toHaveBeenCalled()
  })
})

// ─── golden: valid trainWithDeltas is unchanged ──────────────────────────────

describe('trainWithDeltas golden (valid inputs unchanged)', () => {
  // Constantes capturadas en la base (antes de T010): setWeights(pattern) y
  // trainWithDeltas([0.2, -0.4, 0.9], [0.3, -0.2], 0.5).
  const BASE_RESIDUAL = [-0.15148250740779737, -0.09703498518440533, -0.05667128333508802, -0.007412537038986689, 0.05039200077815909, 0.09921599844368183, 0.15176400350171593, -0.14803999610920457, -0.09952666112498278, -0.050946677750034454, 0.0021300249375775103, 0.05236669437508613, 0.10058761570764746, 0.1488247685847051, -0.14735572931558646, -0.09706192146176272, -0.05364985030976548, -0.0036312835080644958, 0.04619229439068748, 0.09679274386247806, 0.142570292775273, -0.14908112104264304, -0.09908579538321606, -0.049041379710601816, 0.0008074523406447129, 0.05187048811559714, 0.10122743169666006, 0.1512211877471884, -0.14871948212672845, -0.09892141115158368, -0.04750141584707344, 0.001535123960740757, 0.05152731477961099, 0.10160151776654047, 0.1513489692253322, -0.1468750714100656, 0.05594845909013402, 0.09463370150578819, 0.15008976888724876, 0.18939348034779002, 0.25, 0.046034360606577326, -0.2464224676705255, -0.2000598459248325, -0.14292898689852668, -0.1]

  const BASE_PLAIN = [-0.14962511482030683, -0.1007497703593864, -0.04831301669138061, 0.001874425898465987, 0.05049974696040519, 0.09900050607918963, 0.15224886132182336, -0.1475012651979741, -0.09937539046229721, -0.05124921907540558, 0.0028107429196625603, 0.05312304768851396, 0.1007359971395726, 0.14852800572085484, -0.14668801287192335, -0.09632001430213703, 0.02368763396796396, 0.07331278470797177, 0.12687460947263154, 0.16475200247826277, 0.30000000000000004, -0.199125089311976, -0.14887518980531453, -0.10124973964842104, -0.04316800165217519, -0.05]

  it('residual on a hidden layer: weights equal the base constants (1e-9)', () => {
    const net = patterned([3, 4, 4, 2], { residual: true })
    net.trainWithDeltas([0.2, -0.4, 0.9], [0.3, -0.2], 0.5)
    expect(maxAbsDiff(net.getWeights(), BASE_RESIDUAL)).toBeLessThan(1e-9)
  })

  it('no residual: weights equal the base constants (1e-9)', () => {
    const net = patterned([3, 4, 2])
    net.trainWithDeltas([0.2, -0.4, 0.9], [0.3, -0.2], 0.5)
    expect(maxAbsDiff(net.getWeights(), BASE_PLAIN)).toBeLessThan(1e-9)
  })

  // Réplica con train(): train usa g = t − p y delta = g·dfn(p). Con deltas
  // (t − p)·p·(1 − p) (sigmoide, sin residual en la última capa) el paso debe ser el mismo.
  function replicaCheck(options?: ConstructorParameters<typeof NetworkN>[1]): void {
    const structure = [3, 4, 4, 2]
    const x = [0.2, -0.4, 0.9]
    const t = [0.7, 0.1]
    const lr = 0.5

    const viaTrain = patterned(structure, options)
    viaTrain.train(x, t, lr)

    const viaDeltas = patterned(structure, options)
    const p = viaDeltas.predict(x)
    const deltas = t.map((ti, i) => (ti - p[i]) * p[i] * (1 - p[i]))
    viaDeltas.trainWithDeltas(x, deltas, lr)

    expect(maxAbsDiff(viaDeltas.getWeights(), viaTrain.getWeights())).toBeLessThan(1e-9)
  }

  it('replica: trainWithDeltas with sigmoid deltas equals train() (residual on hidden)', () => {
    replicaCheck({ residual: true })
  })

  it('replica: trainWithDeltas with sigmoid deltas equals train() (no residual)', () => {
    replicaCheck()
  })
})
