// Tests de comportamiento de src/PositionalEncoding.ts (feature 006, T017).
//
// Sinusoidal (Vaswani 2017), segun el archivo:
//   PE[pos][2i]   = sin( pos / 10000^(2i/d) )
//   PE[pos][2i+1] = cos( pos / 10000^(2i/d) )
// Derivaciones a mano para d = 4: 10000^(0/4) = 1 y 10000^(2/4) = 100.
// Los literales de sin/cos se calcularon fuera del modulo con series de Taylor
// (reduccion mod 2pi), no copiando la salida de Math.sin del codigo.
//
// Learned: W[pos][d] (tabla), apply suma W; update hace w + lr·dW (dW = -dL/dw).
//
// Los casos invalidos (pos negativo, dModel impar) se fijan como COMPORTAMIENTO
// ACTUAL en un describe aparte; T042 los cambiara. Son hallazgos (ver el informe).
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { LearnedPositionalEncoding, PositionalEncoding } from '../src/PositionalEncoding'
import { restoreRandom, seedRandom } from './helpers/seed'

/** Compara vectores elemento a elemento con 12 cifras decimales. */
function expectClose(actual: number[], expected: number[], digits = 12): void {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], digits))
}

describe('PositionalEncoding.encode (sinusoidal)', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  it('pos 0 da [0, 1, 0, 1, ...] porque sin 0 = 0 y cos 0 = 1 en todos los pares', () => {
    // Cada par (2i, 2i+1) con angulo 0 / freq = 0.
    expectClose(PositionalEncoding.encode(0, 4), [0, 1, 0, 1])
    expectClose(PositionalEncoding.encode(0, 6), [0, 1, 0, 1, 0, 1])
  })

  it('pos 1, d 4: el primer par usa frecuencia 1 y el segundo 10000^(2/4) = 100', () => {
    // i=0: ángulo 1/1 = 1      -> [sin 1, cos 1]
    // i=1: ángulo 1/100 = 0.01 -> [sin 0.01, cos 0.01]
    // Literales: sin 1 = 0.8414709848078965, cos 1 = 0.5403023058681398,
    //            sin 0.01 = 0.0099998333341667, cos 0.01 = 0.9999500004166653
    expectClose(PositionalEncoding.encode(1, 4), [
      0.8414709848078965,
      0.5403023058681398,
      0.009999833334166664,
      0.9999500004166653,
    ])
  })

  it('pos 2, d 2: un solo par con frecuencia 10000^0 = 1', () => {
    // [sin 2, cos 2] = [0.9092974268256817, -0.4161468365471424]
    expectClose(PositionalEncoding.encode(2, 2), [0.9092974268256817, -0.4161468365471424])
  })

  it('pos 100, d 4: 100/1 = 100 rad en el primer par y 100/100 = 1 rad en el segundo', () => {
    // sin 100 = -0.50636564110976, cos 100 = 0.86231887228768 (100 rad reducido mod 2pi)
    const pe = PositionalEncoding.encode(100, 4)
    expect(pe[0]).toBeCloseTo(-0.50636564110976, 10)
    expect(pe[1]).toBeCloseTo(0.86231887228768, 10)
    // Segundo par: 100/100 = 1 -> [sin 1, cos 1]
    expect(pe[2]).toBeCloseTo(0.8414709848078965, 12)
    expect(pe[3]).toBeCloseTo(0.5403023058681398, 12)
  })

  it('cada par (sin, cos) tiene norma 1 en todas las posiciones probadas', () => {
    // Identidad sin² + cos² = 1 por par, independiente de la frecuencia.
    for (let pos = 0; pos < 20; pos++) {
      const pe = PositionalEncoding.encode(pos, 8)
      for (let i = 0; i < 4; i++) {
        const norm = pe[2 * i] ** 2 + pe[2 * i + 1] ** 2
        expect(norm).toBeCloseTo(1, 12)
      }
    }
  })

  it('encodeSequence(3, 4) produce 3 filas iguales a encode(pos, 4)', () => {
    const seq = PositionalEncoding.encodeSequence(3, 4)
    expect(seq.length).toBe(3)
    for (let pos = 0; pos < 3; pos++) {
      expectClose(seq[pos], PositionalEncoding.encode(pos, 4))
    }
    // Fila 2 a mano: [sin 2, cos 2, sin 0.02, cos 0.02]
    expectClose(seq[2], [0.9092974268256817, -0.4161468365471424, 0.01999866669333308, 0.9998000066665778])
  })

  it('encodeSequence(0, 4) devuelve una lista vacia', () => {
    expect(PositionalEncoding.encodeSequence(0, 4)).toEqual([])
  })
})

describe('PositionalEncoding.apply (sinusoidal)', () => {
  it('suma la codificacion fila a fila y no modifica la entrada', () => {
    const emb = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]
    const copy = emb.map((row) => [...row])
    const out = PositionalEncoding.apply(emb)
    // Fila 0: PE(0,4) = [0,1,0,1] -> [1, 3, 3, 5]
    expectClose(out[0], [1, 3, 3, 5])
    // Fila 1: PE(1,4) = [sin1, cos1, sin0.01, cos0.01]
    expectClose(out[1], [
      5 + 0.8414709848078965,
      6 + 0.5403023058681398,
      7 + 0.009999833334166664,
      8 + 0.9999500004166653,
    ])
    expect(emb).toEqual(copy)
  })

  it('seqLen igual a la longitud no cambia el resultado', () => {
    const emb = [
      [1, 2],
      [3, 4],
    ]
    expect(PositionalEncoding.apply(emb, 2)).toEqual(PositionalEncoding.apply(emb))
  })
})

describe('LearnedPositionalEncoding', () => {
  // Tabla inicial fija: MAX = 4 posiciones, DIM = 3.
  const W0: number[][] = [
    [0.2, -0.1, 0.05],
    [-0.3, 0.25, 0.1],
    [0.15, 0.4, -0.2],
    [-0.05, 0.3, 0.35],
  ]
  const ZERO: number[][] = Array.from({ length: 4 }, () => [0, 0, 0])

  function build(w: number[][]): LearnedPositionalEncoding {
    const pe = new LearnedPositionalEncoding(4, 3)
    pe.weights = w.map((row) => [...row])
    return pe
  }

  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  it('el constructor crea maxSeqLen x dModel pesos en ±sqrt(1/dModel) y es determinista con la semilla', () => {
    // dModel 4 -> cota sqrt(1/4) = 0.5
    const a = new LearnedPositionalEncoding(3, 4)
    expect(a.weights.length).toBe(3)
    expect(a.weights.every((row) => row.length === 4)).toBe(true)
    expect(a.weights.flat().every((v) => Math.abs(v) <= 0.5)).toBe(true)

    seedRandom(0x5eed)
    const b = new LearnedPositionalEncoding(3, 4)
    expect(b.weights).toEqual(a.weights)
  })

  it('getEncoding(1) devuelve la fila 1 y una copia (mutarla no cambia la tabla)', () => {
    const pe = build(W0)
    const row = pe.getEncoding(1)
    expectClose(row, [-0.3, 0.25, 0.1])
    row[0] = 999
    expect(pe.weights[1][0]).toBe(-0.3)
  })

  it('getEncoding(4) con maxSeqLen 4 lanza un error explicito', () => {
    expect(() => build(W0).getEncoding(4)).toThrow(/Position 4 exceeds maxSeqLen=4/)
  })

  it('apply suma la fila de cada posicion: X = [[0,0,0],[1,1,1]] -> [W0[0], [0.7, 1.25, 1.1]]', () => {
    // Fila 1: 1 + (-0.3) = 0.7; 1 + 0.25 = 1.25; 1 + 0.1 = 1.1
    const out = build(W0).apply([
      [0, 0, 0],
      [1, 1, 1],
    ])
    expectClose(out[0], [0.2, -0.1, 0.05])
    expectClose(out[1], [0.7, 1.25, 1.1])
  })

  it('apply con mas filas que maxSeqLen (5 > 4) lanza un error explicito', () => {
    const X = Array.from({ length: 5 }, () => [0, 0, 0])
    expect(() => build(W0).apply(X)).toThrow(/Sequence length 5 exceeds maxSeqLen=4/)
  })

  it('update hace w + lr·dW: con W = 0, dW = [1, -1, 0.5] y lr 0.5 -> [0.5, -0.5, 0.25]', () => {
    const pe = build(ZERO)
    const dW = [
      [1, -1, 0.5],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ]
    pe.update(dW, 0.5)
    expectClose(pe.weights[0], [0.5, -0.5, 0.25])
    // Segundo paso: [0.5, -0.5, 0.25] + 0.5·[1, -1, 0.5] = [1, -1, 0.5]
    pe.update(dW, 0.5)
    expectClose(pe.weights[0], [1, -1, 0.5])
    // Filas con dW = 0 no cambian.
    expect(pe.weights[1]).toEqual([0, 0, 0])
  })

  it('update con lr 0 deja la tabla intacta', () => {
    const pe = build(W0)
    pe.update(
      [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ],
      0,
    )
    expect(pe.weights).toEqual(W0)
  })
})

// ─── COMPORTAMIENTO ACTUAL (T042 lo cambiara) ────────────────────────────────
// Estos casos son invalidos segun la spec (pos negativo, dModel impar). Hoy no
// lanzan: se fijan tal como estan para que T042 los reescriba. Son hallazgos.
describe('degenerate inputs (COMPORTAMIENTO ACTUAL; T042 will change these)', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  it('pos negativo no lanza: encode(-1, 4) = [sin(-1), cos(-1), sin(-0.01), cos(-0.01)]', () => {
    // sin(-x) = -sin x; cos(-x) = cos x.
    expectClose(PositionalEncoding.encode(-1, 4), [
      -0.8414709848078965,
      0.5403023058681398,
      -0.009999833334166664,
      0.9999500004166653,
    ])
  })

  it('dModel impar no lanza: en encode(1, 3) la ultima dimension recibe solo seno con 10000^(-2/3)', () => {
    // i = floor(3/2) = 1; 10000^(2/3) = 464.1588833612780, por tanto x = 10^(-8/3) = 0.00215443469003.
    // sin x = x - x³/6 = 0.00215443469003 - 1e-8/6 = 0.00215443302336 (x³ = 10^(-8)).
    const pe = PositionalEncoding.encode(1, 3)
    expect(pe.length).toBe(3)
    expectClose(pe, [0.8414709848078965, 0.5403023058681398, 0.002154433023365604])
  })
})
