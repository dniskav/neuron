// Validación de entradas de SOM (src/SOM.ts). Feature 006, T056, principio V.
//
// Reproducciones de los hallazgos de T024:
//   (a) quantizationError([]) devolvía NaN en silencio.
//   (b) train con NaN/Infinity en X corrompía los pesos (quedaban NaN).
//   (c) el constructor truncaba enteros no válidos (new SOM(1.5, 2, 1) creaba 1 fila).
//   (d) train solo validaba X[0]; una fila posterior errónea lanzaba con los pesos
//       ya modificados (entrenamiento parcial).
//
// Regla de los errores: lanzan ANTES de tocar pesos. Los tests (b) y (d) comparan
// los pesos antes y después de la llamada fallida.
//
// Golden (T056, regla "no cambia"): el entrenamiento válido con semilla fija se
// capturó con el código base antes de tocar src/SOM.ts (2×2, 20 puntos en 4 grupos,
// 4 épocas, semillas 1 y 2 de seedRandom). Los valores se comparan con tolerancia
// 1e-12. No son cifras derivadas a mano: son la salida de la base, y su función es
// detectar cambios de comportamiento en las entradas válidas.
//
// Cifra a mano de la única entrada de QE de este archivo (ver el test de golden):
// rejilla 2×2 con w = {(0,0)=[0,0], (0,1)=[2,0], (1,0)=[0,2], (1,1)=[2,2]} y
// X = [[1,1],[3,3],[0.5,0.2]]:
//   [1,1]: d² a los cuatro prototipos = 2, 2, 2, 2 → empate, gana (0,0), d = √2.
//   [3,3]: d² = 18, 10, 10, 2 → gana (1,1), d = √2.
//   [0.5,0.2]: d² = 0.29, 2.29, 3.49, 5.49 → gana (0,0), d = √0.29.
//   QE = (√2 + √2 + √0.29)/3 = (2√2 + √0.29)/3 ≈ 1.1223145.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { SOM } from '../../src/SOM'
import { mulberry32, seedRandom, restoreRandom } from '../helpers/seed'

/** Copia profunda de los pesos; structuredClone conserva NaN, JSON no. */
function snapshot(som: SOM): number[][][] {
  return structuredClone(som.weights)
}

/** Cuatro grupos de 5 puntos alrededor de (5,5), (9,1), (1,9) y (9,9) (mulberry32(99)). */
function blobs(): number[][] {
  const centers = [[5, 5], [9, 1], [1, 9], [9, 9]]
  const rnd = mulberry32(99)
  const X: number[][] = []
  for (const c of centers) {
    for (let i = 0; i < 5; i++) X.push([c[0] + rnd() - 0.5, c[1] + rnd() - 0.5])
  }
  return X
}

const FAST = { initialLr: 0.5, finalLr: 0.5, initialSigma: 1, finalSigma: 1 }

describe('SOM validation (T056)', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  describe('(a) quantizationError([]) lanza', () => {
    it('X vacía lanza un error explícito en lugar de devolver NaN', () => {
      const som = new SOM(1, 2, 1)
      expect(() => som.quantizationError([])).toThrow(/SOM\.quantizationError: dataset X must be non-empty/)
    })
  })

  describe('(b) train con valores no finitos lanza antes de tocar pesos', () => {
    it('NaN en la única fila: lanza y los pesos quedan intactos', () => {
      const som = new SOM(1, 2, 1, FAST)
      const before = snapshot(som)
      expect(() => som.train([[NaN]], 1)).toThrow(/SOM\.train: X\[0\]/)
      expect(som.weights).toEqual(before)
    })

    it('Infinity en una fila: lanza y los pesos quedan intactos', () => {
      const som = new SOM(1, 2, 1, FAST)
      const before = snapshot(som)
      expect(() => som.train([[Infinity]], 1)).toThrow(/SOM\.train: X\[0\]/)
      expect(som.weights).toEqual(before)
    })

    it('NaN en una fila posterior: lanza con el índice y no entrena las filas válidas', () => {
      const som = new SOM(2, 2, 1, FAST)
      const before = snapshot(som)
      const X = [[0.2], [0.7], [NaN], [0.9]]
      expect(() => som.train(X, 3)).toThrow(/SOM\.train: X\[2\]/)
      expect(som.weights).toEqual(before)
    })
  })

  describe('(c) el constructor rechaza enteros no válidos', () => {
    const msg = /rows, cols and inputSize must be positive integers/

    it.each([
      ['rows 1.5', 1.5, 2, 1],
      ['cols 2.5', 2, 2.5, 1],
      ['inputSize 1.5', 2, 2, 1.5],
      ['rows NaN', NaN, 2, 1],
      ['cols Infinity', 2, Infinity, 1],
    ])('%s lanza', (_name, rows, cols, inputSize) => {
      expect(() => new SOM(rows, cols, inputSize)).toThrow(msg)
    })

    it('enteros válidos siguen construyendo la rejilla', () => {
      const som = new SOM(1, 2, 1)
      expect(som.weights.length).toBe(1)
      expect(som.weights[0].length).toBe(2)
    })
  })

  describe('(d) train valida todas las filas antes de entrenar', () => {
    it('una fila posterior de ancho erróneo lanza con X[k] y no modifica pesos', () => {
      // 40 filas válidas, la 41.ª (índice 40) tiene ancho 1 en una SOM de entrada 2.
      const X: number[][] = Array.from({ length: 40 }, (_, i) => [i / 40, 1 - i / 40])
      X.push([1])
      const som = new SOM(3, 3, 2)
      const before = snapshot(som)
      expect(() => som.train(X, 2)).toThrow(/SOM\.train: X\[40\]/)
      expect(som.weights).toEqual(before)
    })

    it('la primera fila de ancho correcto sigue dando el mensaje existente', () => {
      const som = new SOM(2, 2, 2)
      expect(() => som.train([[1, 2, 3]], 1)).toThrow(/SOM\.train: expected input size 2, got 3/)
    })
  })

  describe('golden: entradas válidas no cambian (capturado con la base)', () => {
    it('entrenamiento 2×2, semilla 1, 4 épocas: pesos, QE y BMU', () => {
      seedRandom(1)
      const X = blobs()
      const som = new SOM(2, 2, 2, { initialLr: 0.4, finalLr: 0.05, initialSigma: 1.5, finalSigma: 0.5 })
      const qe0 = som.quantizationError(X)
      som.train(X, 4)
      const golden = [
        [[5.1217711415317035, 5.3263237333340525], [3.642972821631913, 7.754805747277076]],
        [[7.574704585450199, 3.6978641820943707], [7.159919550033381, 7.1153710775075085]],
      ]
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          for (let i = 0; i < 2; i++) {
            expect(som.weights[r][c][i]).toBeCloseTo(golden[r][c][i], 12)
          }
        }
      }
      expect(qe0).toBeCloseTo(8.442985115785557, 12)
      expect(som.quantizationError(X)).toBeCloseTo(2.314167657000586, 12)
      expect(X.map((x) => som.getBMU(x))[0]).toEqual([0, 0])
      expect(X.map((x) => som.getBMU(x))[19]).toEqual([1, 1])
    })

    it('QE con pesos fijados: (2√2 + √0.29)/3 (cuenta a mano en la cabecera)', () => {
      const som = new SOM(2, 2, 2)
      som.weights = [
        [[0, 0], [2, 0]],
        [[0, 2], [2, 2]],
      ]
      const expected = (2 * Math.SQRT2 + Math.sqrt(0.29)) / 3
      expect(som.quantizationError([[1, 1], [3, 3], [0.5, 0.2]])).toBeCloseTo(expected, 12)
    })
  })
})
