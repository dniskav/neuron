// Tests de MaxPool2D (feature 006, T015). Valores esperados deducidos a mano;
// no se copia la salida del código.
//
// Convención del código: el máximo es el PRIMER valor en orden de escaneo
// (ph antes que pw) porque la comparación es estricta (`val > maxVal`).
//
// T055: con ventanas solapadas (stride < poolSize) el gradiente de cada ventana
// va SOLO a su propio argmax (guardado por ventana en forward) y se acumula con
// `+=` si una posición es el máximo de varias ventanas.
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { MaxPool2D } from '../src/MaxPool2D'
import { restoreRandom, seedRandom } from './helpers/seed'

beforeEach(() => seedRandom(2024))
afterEach(() => restoreRandom())

// Entrada 4×4×1 con valores distintos y sin empates dentro de cada bloque 2×2.
// Máximos de bloque (a mano):
//   (0,0) {3,0,2,9} → 9 en (1,1)   (0,1) {7,1,4,8} → 8 en (1,3)
//   (1,0) {6,5,-2,4.5} → 6 en (2,0) (1,1) {0.5,-1,2.5,10} → 10 en (3,3)
const X4: number[][][] = [
  [[3], [0], [7], [1]],
  [[2], [9], [4], [8]],
  [[6], [5], [0.5], [-1]],
  [[-2], [4.5], [2.5], [10]],
]

describe('MaxPool2D constructor and outputShape', () => {
  it('stride por defecto = poolSize (ventanas no solapadas)', () => {
    expect(new MaxPool2D(3).stride).toBe(3)
    expect(new MaxPool2D(2).poolSize).toBe(2)
  })

  it('stride explícito se respeta', () => {
    expect(new MaxPool2D(2, 1).stride).toBe(1)
  })

  it('poolSize no positivo lanza', () => {
    expect(() => new MaxPool2D(0)).toThrow(/poolSize must be positive/)
    expect(() => new MaxPool2D(-2)).toThrow(/poolSize must be positive/)
  })

  it('outputShape: floor((H − p)/s) + 1 por eje, canales sin cambios', () => {
    // 4×4, p 2, s 2: floor(2/2)+1 = 2 → [2, 2, 3]
    expect(new MaxPool2D(2).outputShape(4, 4, 3)).toEqual([2, 2, 3])
    // 4×4, p 2, s 1: floor(2/1)+1 = 3 → [3, 3, 1]
    expect(new MaxPool2D(2, 1).outputShape(4, 4, 1)).toEqual([3, 3, 1])
    // 7×6, p 3, s 2: floor(4/2)+1 = 3 (alto), floor(3/2)+1 = 2 (ancho) → [3, 2, 2]
    expect(new MaxPool2D(3, 2).outputShape(7, 6, 2)).toEqual([3, 2, 2])
    // 5×5, p 2, s 2: floor(3/2)+1 = 2 → [2, 2, 1] (la última fila/columna sobrante se descarta)
    expect(new MaxPool2D(2).outputShape(5, 5, 1)).toEqual([2, 2, 1])
  })
})

describe('MaxPool2D forward', () => {
  it('4×4×1, p 2, s 2: máximo de cada bloque (valores a mano)', () => {
    // Ver comentario de X4: [[9, 8], [6, 10]]
    expect(new MaxPool2D(2).forward(X4)).toEqual([
      [[9], [8]],
      [[6], [10]],
    ])
  })

  it('stride 1 (solapado): 3×3 salidas, cada una el máximo de su ventana 2×2 (a mano)', () => {
    // Ventanas (fila, columna superior izquierda):
    // (0,0) {3,0,2,9}→9  (0,1) {0,7,9,4}→9  (0,2) {7,1,4,8}→8
    // (1,0) {2,9,6,5}→9  (1,1) {9,4,5,0.5}→9 (1,2) {4,8,0.5,-1}→8
    // (2,0) {6,5,-2,4.5}→6 (2,1) {5,0.5,4.5,2.5}→5 (2,2) {0.5,-1,2.5,10}→10
    expect(new MaxPool2D(2, 1).forward(X4)).toEqual([
      [[9], [9], [8]],
      [[9], [9], [8]],
      [[6], [5], [10]],
    ])
  })

  it('stride 2 con poolSize 3 (distinto de poolSize): valor 5h+w, máximo en la esquina inferior derecha', () => {
    // 5×5, v(h,w) = 5h + w (sin empates). Ventana (oh, ow) cubre filas 2oh..2oh+2
    // y columnas 2ow..2ow+2; su máximo está en (2oh+2, 2ow+2) = 5(2oh+2) + (2ow+2):
    // (0,0) → 12, (0,1) → 14, (1,0) → 22, (1,1) → 24.
    const X5 = Array.from({ length: 5 }, (_, h) =>
      Array.from({ length: 5 }, (_, w) => [5 * h + w]),
    )
    expect(new MaxPool2D(3, 2).forward(X5)).toEqual([
      [[12], [14]],
      [[22], [24]],
    ])
  })

  it('canales independientes: cada canal elige su propio máximo', () => {
    // X[0][0] = [1, 10], X[0][1] = [4, 3], X[1][0] = [2, 0], X[1][1] = [3, 7].
    // Canal 0: {1,4,2,3} → 4. Canal 1: {10,3,0,7} → 10.
    const X = [
      [[1, 10], [4, 3]],
      [[2, 0], [3, 7]],
    ]
    expect(new MaxPool2D(2).forward(X)).toEqual([[[4, 10]]])
  })
})

describe('MaxPool2D backward', () => {
  it('ventanas no solapadas: el gradiente va solo al máximo de cada bloque', () => {
    // dOut = [[1, 2], [3, 4]]. Máximos en (1,1)←1, (1,3)←2, (2,0)←3, (3,3)←4.
    const p = new MaxPool2D(2)
    p.forward(X4)
    expect(p.backward([[[1], [2]], [[3], [4]]])).toEqual([
      [[0], [0], [0], [0]],
      [[0], [1], [0], [2]],
      [[3], [0], [0], [0]],
      [[0], [0], [0], [4]],
    ])
  })

  it('canales: el gradiente de cada canal va a su propio máximo', () => {
    // Forward: canal 0 máximo en (0,1); canal 1 máximo en (0,0).
    const X = [
      [[1, 10], [4, 3]],
      [[2, 0], [3, 7]],
    ]
    const p = new MaxPool2D(2)
    p.forward(X)
    expect(p.backward([[[1, 1]]])).toEqual([
      [[0, 1], [1, 0]],
      [[0, 0], [0, 0]],
    ])
  })

  it('solapamiento con el mismo máximo en ambas ventanas: los gradientes se SUMAN en esa posición', () => {
    // H 2, W 3, p 2, s 1. X = [[1, 9, 2], [4, 3, 5]].
    // Ventana (0,0) {1,9,4,3} → 9 en (0,1). Ventana (0,1) {9,2,3,5} → 9 en (0,1).
    // Forward = [[9, 9]]. Con dOut = [[0.5, 2]]: dX(0,1) = 0.5 + 2 = 2.5, resto 0.
    const p = new MaxPool2D(2, 1)
    const out = p.forward([
      [[1], [9], [2]],
      [[4], [3], [5]],
    ])
    expect(out).toEqual([[[9], [9]]])
    expect(p.backward([[[0.5], [2]]])).toEqual([
      [[0], [2.5], [0]],
      [[0], [0], [0]],
    ])
  })

  it('T055 ventanas solapadas, máximo de una ventana pero no de otra: cada gradiente va a SU argmax', () => {
    // H 2, W 3, p 2, s 1. X = [[1, 5, 9], [0, 3, 4]].
    // Ventana (0,0) {1,5,0,3} → 5 en (0,1). Ventana (0,1) {5,9,3,4} → 9 en (0,2).
    // Con dOut = [[1, 2]]: dX(0,1) = 1, dX(0,2) = 2, resto 0.
    // (5 es máximo de la primera ventana pero no de la segunda: no recibe el 2.)
    const p = new MaxPool2D(2, 1)
    expect(p.forward([[[1], [5], [9]], [[0], [3], [4]]])).toEqual([[[5], [9]]])
    expect(p.backward([[[1], [2]]])).toEqual([
      [[0], [1], [2]],
      [[0], [0], [0]],
    ])
  })

  it('T055 poolSize 3, stride 1, sin empates: gradiente por ventana a su argmax', () => {
    // H 3, W 4, p 3, s 1. X = [[1,2,3,0],[4,9,5,8],[6,7,0,1]] (sin empates).
    // Ventana (0,0): cols 0..2, máx 9 en (1,1). Ventana (0,1): cols 1..3, máx 9 en (1,1).
    // Ambas apuntan a (1,1): dOut = [[1, 10]] → dX(1,1) = 11. Resto 0.
    // Segundo caso: X2 = [[1,2,3,0],[4,6,5,8],[6,7,0,1]]:
    // ventana (0,0) {1,2,3,4,6,5,6,7,0}: máx 7 en (2,1); ventana (0,1) {2,3,0,6,5,8,7,0,1}: máx 8 en (1,3).
    // (6 aparece dos veces en la primera ventana, pero 7 es mayor: sin empate en el máximo.)
    const p = new MaxPool2D(3, 1)
    p.forward([[[1], [2], [3], [0]], [[4], [9], [5], [8]], [[6], [7], [0], [1]]])
    expect(p.backward([[[1], [10]]])).toEqual([
      [[0], [0], [0], [0]],
      [[0], [11], [0], [0]],
      [[0], [0], [0], [0]],
    ])
    p.forward([[[1], [2], [3], [0]], [[4], [6], [5], [8]], [[6], [7], [0], [1]]])
    expect(p.backward([[[1], [10]]])).toEqual([
      [[0], [0], [0], [0]],
      [[0], [0], [0], [10]],
      [[0], [1], [0], [0]],
    ])
  })

  it('T055 empate con ventanas solapadas: primer máximo de cada ventana, acumulado', () => {
    // H 2, W 3, p 2, s 1. X = [[3, 3, 3], [0, 0, 0]].
    // Ventana (0,0) {3,3,0,0}: primer 3 en (0,0). Ventana (0,1) {3,3,0,0}: primer 3 en (0,1).
    // dOut = [[1, 2]] → dX(0,0) = 1, dX(0,1) = 2.
    const p = new MaxPool2D(2, 1)
    p.forward([[[3], [3], [3]], [[0], [0], [0]]])
    expect(p.backward([[[1], [2]]])).toEqual([
      [[1], [2], [0]],
      [[0], [0], [0]],
    ])
  })

  it('empate: el gradiente va al primer máximo en orden de escaneo (fila, luego columna)', () => {
    // Ventana [[3, 3], [1, 0]]: los dos 3 empatan. Con `>` estricto gana (0,0),
    // el primero que se encuentra al escanear. dOut = 5 → dX(0,0) = 5.
    const p = new MaxPool2D(2)
    expect(p.forward([[[3], [3]], [[1], [0]]])).toEqual([[[3]]])
    expect(p.backward([[[5]]])).toEqual([
      [[5], [0]],
      [[0], [0]],
    ])
  })

  it('backward sin forward previo lanza', () => {
    expect(() => new MaxPool2D(2).backward([[[1]]])).toThrow(/call forward\(\) first/)
  })
})

describe('MaxPool2D current behavior (T043 changes this)', () => {
  it('entrada menor que poolSize devuelve [] (se cambiará a un error en T043)', () => {
    // H = 2, p = 3: floor((2 − 3)/3) + 1 = 0 → sin ventanas.
    expect(new MaxPool2D(3).forward([[[1], [2]], [[3], [4]]])).toEqual([])
  })
})
