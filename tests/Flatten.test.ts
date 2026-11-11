// Tests de comportamiento de Flatten (src/Flatten.ts), feature 006 T014.
//
// Convención del módulo: flat[h·W·C + w·C + c] = input[h][w][c] (fila, columna, canal).
// Valores esperados deducidos a mano en cada comentario; no se copian de la salida.
// Flatten no usa azar; la semilla se fija solo por convención de la suite (Constitución IV).
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Flatten } from '../src/Flatten'
import { restoreRandom, seedRandom } from './helpers/seed'

beforeEach(() => seedRandom(1234))
afterEach(() => restoreRandom())

// Entrada 2×2×2 (H=2, W=2, C=2).
const X222 = [
  [
    [1, 2],
    [3, 4],
  ],
  [
    [5, 6],
    [7, 8],
  ],
]

// Entrada 2×3×2 (H=2, W=3, C=2), no cuadrada para detectar intercambios de ejes.
// x[h][w][c] = 6h + 2w + c + 1. Con k = h·W·C + w·C + c = 6h + 2w + c, el valor es k + 1,
// así que la salida plana es [1, 2, ..., 12].
const X232 = [
  [
    [1, 2],
    [3, 4],
    [5, 6],
  ],
  [
    [7, 8],
    [9, 10],
    [11, 12],
  ],
]

describe('Flatten.forward', () => {
  it('aplana 2×2×2 en orden fila, columna, canal', () => {
    // (h,w)=(0,0) → 1,2; (0,1) → 3,4; (1,0) → 5,6; (1,1) → 7,8.
    expect(new Flatten().forward(X222)).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
  })

  it('aplana 2×3×2 (no cuadrada) con salida 1..12', () => {
    // flat[k] = k + 1 porque x[h][w][c] = k + 1 con k = 6h + 2w + c. Longitud H·W·C = 12.
    const flat = new Flatten().forward(X232)
    expect(flat).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    expect(flat).toHaveLength(12)
  })

  it('guarda la forma [H, W, C] de la última entrada', () => {
    const f = new Flatten()
    f.forward(X232)
    // H = 2 filas, W = 3 columnas, C = 2 canales.
    expect(f.inputShape).toEqual([2, 3, 2])
  })

  it('con una forma distinta, la salida sigue el orden del nuevo tensor', () => {
    const f = new Flatten()
    f.forward(X222)
    // H = 1, W = 3, C = 1: el orden es solo por columna → [9, 8, 7].
    expect(f.forward([[[9], [8], [7]]])).toEqual([9, 8, 7])
    expect(f.inputShape).toEqual([1, 3, 1])
  })
})

describe('Flatten.backward', () => {
  it('devuelve la forma original con cada gradiente en su posición', () => {
    const f = new Flatten()
    f.forward(X232)
    // dOut_k = 2k + 1 para k = 0..11 → 1, 3, 5, …, 23.
    // dInput[h][w][c] = dOut[6h + 2w + c] = 2(6h + 2w + c) + 1:
    //   h=0: w0 → [1, 3], w1 → [5, 7], w2 → [9, 11]
    //   h=1: w0 → [13, 15], w1 → [17, 19], w2 → [21, 23]
    const dOut = Array.from({ length: 12 }, (_, k) => 2 * k + 1)
    expect(f.backward(dOut)).toEqual([
      [
        [1, 3],
        [5, 7],
        [9, 11],
      ],
      [
        [13, 15],
        [17, 19],
        [21, 23],
      ],
    ])
  })

  it('es la inversa exacta de forward (ida y vuelta)', () => {
    const f2 = new Flatten()
    expect(f2.backward(f2.forward(X222))).toEqual(X222)
    const f3 = new Flatten()
    expect(f3.backward(f3.forward(X232))).toEqual(X232)
  })

  it('lanza un error claro si no hubo forward previo', () => {
    expect(() => new Flatten().backward([1])).toThrow(/call forward\(\) first/)
  })

  it('lanza si la longitud del gradiente no es H·W·C', () => {
    const f = new Flatten()
    f.forward(X222) // H·W·C = 8
    // Esperado: "expected gradient of length 8, got 7" y "... got 9".
    expect(() => f.backward([1, 2, 3, 4, 5, 6, 7])).toThrow(
      /expected gradient of length 8, got 7/,
    )
    expect(() => f.backward([1, 2, 3, 4, 5, 6, 7, 8, 9])).toThrow(
      /expected gradient of length 8, got 9/,
    )
  })

  it('usa la forma del último forward', () => {
    const f = new Flatten()
    f.forward(X232)
    f.forward([[[9], [8], [7]]]) // H = 1, W = 3, C = 1
    // Longitud 3 con la forma nueva; con la antigua (12) lanzaría.
    expect(f.backward([1, 2, 3])).toEqual([[[1], [2], [3]]])
  })
})
