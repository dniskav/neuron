// Validación de entradas de preprocesado (feature 006, T043; principio V).
//
// Cada error es un `Error` con mensaje «Clase.método: …» y se lanza ANTES de
// modificar el estado del objeto. Los casos válidos (golden) deducidos a mano
// no deben cambiar.
//
// Permutación de shuffle con Math.random = [0, 0.34, 0.99] sobre 4 elementos:
//   indices = [0,1,2,3]; i=3: j = floor(0·4) = 0 → [3,1,2,0]
//   i=2: j = floor(0.34·3) = 1 → [3,2,1,0]; i=1: j = floor(0.99·2) = 1 → sin cambio.
//   Resultado [3,2,1,0].
//
// MaxPool2D sobre X4 (4×4×1, poolSize 2): máximos de bloque 9, 8, 6, 10 en
//   (1,1), (1,3), (2,0), (3,3). Con dOutput = 1 en las 4 salidas, dInput tiene
//   1 en esas posiciones y 0 en el resto.

import { afterEach, describe, expect, it } from 'vitest'
import { DataAugmentation } from '../../src/DataAugmentation'
import { Flatten } from '../../src/Flatten'
import { MaxPool2D } from '../../src/MaxPool2D'
import { Tokenizer } from '../../src/Tokenizer'

const originalRandom = Math.random

function stubRandom(values: number[]): void {
  let i = 0
  Math.random = () => {
    if (i >= values.length) throw new Error('stubRandom: secuencia agotada')
    return values[i++]
  }
}

afterEach(() => {
  Math.random = originalRandom
})

const X4: number[][][] = [
  [[3], [0], [7], [1]],
  [[2], [9], [4], [8]],
  [[6], [5], [0.5], [-1]],
  [[-2], [4.5], [2.5], [10]],
]

// Gradiente de salida con 1 en cada ventana y su dInput esperado (argmax a mano).
const D_OUT_2x2: number[][][] = [
  [[1], [1]],
  [[1], [1]],
]
const D_IN_X4: number[][][] = [
  [[0], [0], [0], [0]],
  [[0], [1], [0], [1]],
  [[1], [0], [0], [0]],
  [[0], [0], [0], [1]],
]

describe('DataAugmentation.shuffle (T043)', () => {
  it('lanza si X tiene más filas que y', () => {
    expect(() => DataAugmentation.shuffle([[1], [2], [3]], [10, 20])).toThrow(
      /^DataAugmentation\.shuffle: /,
    )
  })

  it('lanza si y tiene más etiquetas que X', () => {
    expect(() => DataAugmentation.shuffle([[1], [2]], [10, 20, 30])).toThrow(
      /^DataAugmentation\.shuffle: /,
    )
  })

  it('golden: con longitudes iguales X e y reciben la misma permutación [3,2,1,0]', () => {
    stubRandom([0, 0.34, 0.99])
    const out = DataAugmentation.shuffle([[0], [1], [2], [3]], [10, 11, 12, 13])
    expect(out.X).toEqual([[3], [2], [1], [0]])
    expect(out.y).toEqual([13, 12, 11, 10])
  })
})

describe('DataAugmentation.normalize / standardize con X vacía (T043)', () => {
  it('normalize([]) lanza', () => {
    expect(() => DataAugmentation.normalize([])).toThrow(/^DataAugmentation\.normalize: /)
  })

  it('standardize([]) lanza', () => {
    expect(() => DataAugmentation.standardize([])).toThrow(/^DataAugmentation\.standardize: /)
  })

  it('golden: normalize([[0,10],[5,20],[10,30]]) → [[0,0],[0.5,0.5],[1,1]]', () => {
    const { normalized } = DataAugmentation.normalize([[0, 10], [5, 20], [10, 30]])
    expect(normalized).toEqual([[0, 0], [0.5, 0.5], [1, 1]])
  })
})

describe('MaxPool2D.forward con entrada menor que poolSize (T043)', () => {
  it('altura menor que poolSize lanza', () => {
    expect(() => new MaxPool2D(3).forward([[[1], [2]], [[3], [4]]])).toThrow(
      /^MaxPool2D\.forward: /,
    )
  })

  it('anchura menor que poolSize lanza', () => {
    expect(() => new MaxPool2D(2).forward([[[1]], [[2]]])).toThrow(/^MaxPool2D\.forward: /)
  })

  it('entrada vacía lanza', () => {
    expect(() => new MaxPool2D(2).forward([])).toThrow(/^MaxPool2D\.forward: /)
  })

  it('golden: forward de X4 con poolSize 2 → [[9,8],[6,10]]', () => {
    expect(new MaxPool2D(2).forward(X4)).toEqual([[[9], [8]], [[6], [10]]])
  })
})

describe('MaxPool2D.backward con forma de gradiente errónea (T043)', () => {
  it('menos filas que la salida lanza', () => {
    const p = new MaxPool2D(2)
    p.forward(X4)
    expect(() => p.backward([[[1], [1]]])).toThrow(/^MaxPool2D\.backward: /)
  })

  it('número de canales distinto lanza', () => {
    const p = new MaxPool2D(2)
    p.forward(X4)
    expect(() =>
      p.backward([
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
      ]),
    ).toThrow(/^MaxPool2D\.backward: /)
  })

  it('golden: backward con forma correcta enruta cada 1 al argmax de su ventana', () => {
    const p = new MaxPool2D(2)
    p.forward(X4)
    expect(p.backward(D_OUT_2x2)).toEqual(D_IN_X4)
  })

  it('un forward fallido no altera el estado: el backward posterior sigue siendo válido', () => {
    const p = new MaxPool2D(2)
    p.forward(X4)
    expect(() => p.forward([[[1]]])).toThrow(/^MaxPool2D\.forward: /)
    expect(p.backward(D_OUT_2x2)).toEqual(D_IN_X4)
  })
})

describe('Flatten.forward con entrada vacía o irregular (T043)', () => {
  it('entrada vacía lanza con mensaje claro', () => {
    expect(() => new Flatten().forward([])).toThrow(/^Flatten\.forward: /)
  })

  it('primera fila sin columnas lanza', () => {
    expect(() => new Flatten().forward([[]])).toThrow(/^Flatten\.forward: /)
  })

  it('filas de distinta anchura lanza indicando la fila', () => {
    expect(() => new Flatten().forward([[[1]], [[1], [2]]])).toThrow(
      /Flatten\.forward: row 1 has 2 columns, expected 1/,
    )
  })

  it('canales irregulares lanzan indicando la celda', () => {
    expect(() => new Flatten().forward([[[1, 2], [3]]])).toThrow(
      /Flatten\.forward: \[0\]\[1\] has 1 channels, expected 2/,
    )
  })

  it('golden: 2×2×2 → [1..8] en orden fila, columna, canal', () => {
    const X222 = [
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ]
    expect(new Flatten().forward(X222)).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
  })

  it('un forward fallido no altera la forma guardada: backward usa la del último forward válido', () => {
    const f = new Flatten()
    const X222 = [
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ]
    f.forward(X222)
    expect(() => f.forward([])).toThrow(/^Flatten\.forward: /)
    expect(f.backward([1, 2, 3, 4, 5, 6, 7, 8])).toEqual(X222)
  })
})

describe('Tokenizer.decode con id desconocido (decisión Q3: sigue siendo "<UNK>")', () => {
  it('un id fuera del vocabulario se decodifica como "<UNK>" y no lanza', () => {
    const tok = new Tokenizer({ mode: 'char' }).fit(['ab'])
    expect(tok.decode([999])).toBe('<UNK>')
    expect(tok.decode([999], false)).toBe('<UNK>')
  })

  it('un id desconocido mezclado con uno conocido conserva el resto', () => {
    const tok = new Tokenizer({ mode: 'char' }).fit(['ab'])
    const a = tok.tokenToId('a')
    expect(a).toBeDefined()
    expect(tok.decode([a!, 999])).toBe('a<UNK>')
  })
})
