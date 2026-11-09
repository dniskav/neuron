// Tests de EarlyStopping (T010, specs/006-coverage).
//
// Algoritmo de referencia (src/EarlyStopping.ts):
//   min: mejora si value <  best − minDelta
//   max: mejora si value >  best + minDelta
//   Mejora → counter = 0, best y bestEpoch actualizados (y snapshot si restoreBest).
//   No mejora → counter++; si counter >= patience → stopped = true y update devuelve true.
//
// Todos los valores de frontera son binarios exactos (0.25, 0.5, 0.75 …) para que
// la comparación estricta no dependa de redondeo. El módulo no usa aleatoriedad;
// la semilla se fija igualmente, como en el resto de tests.
//
// Las entradas inválidas (NaN, patience no entera, minDelta negativo, …) no se
// fijan aquí: son hallazgos reportados en la tarea T010 (ver informe).

import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { EarlyStopping } from '../src/EarlyStopping'
import { restoreRandom, seedRandom } from './helpers/seed'

beforeEach(() => seedRandom(0x5eed))
afterEach(restoreRandom)

/** Alimenta la serie y devuelve el índice (0-based) de la llamada que pide parar, o -1. */
function stopIndex(es: EarlyStopping, values: number[]): number {
  for (let i = 0; i < values.length; i++) {
    if (es.update(values[i], i)) return i
  }
  return -1
}

describe('EarlyStopping', () => {
  describe('valores por defecto', () => {
    it('patience 10, minDelta 1e-4, mode min, sin restoreBest y estado inicial limpio', () => {
      // Valores por defecto declarados en el constructor (?? 10, ?? 1e-4, ?? 'min').
      const es = new EarlyStopping()
      expect(es.patience).toBe(10)
      expect(es.minDelta).toBe(1e-4)
      expect(es.mode).toBe('min')
      expect(es.restoreBest).toBe(false)
      expect(es.bestValue).toBe(Infinity)
      expect(es.counter).toBe(0)
      expect(es.stopped).toBe(false)
      expect(es.bestWeights).toBeNull()
    })

    it('mode max arranca con bestValue = -Infinity', () => {
      const es = new EarlyStopping({ mode: 'max' })
      expect(es.bestValue).toBe(-Infinity)
    })
  })

  describe('modo min', () => {
    it('serie [1, 0.9, 0.95, 0.96, 0.97] con patience 2 para en la 4.ª llamada (índice 3)', () => {
      // 1    : 1 < Infinity → mejora. best 1, c 0.
      // 0.9  : 0.9 < 1 − 1e-4 → mejora. best 0.9, bestEpoch 1, c 0.
      // 0.95 : 0.95 < 0.9 − 1e-4 falso → c 1 (1 < 2, sigue).
      // 0.96 : no mejora → c 2 ≥ 2 → para. Es la 4.ª llamada (índice 3). 0.97 no se alimenta.
      const es = new EarlyStopping({ patience: 2, mode: 'min' })
      expect(stopIndex(es, [1, 0.9, 0.95, 0.96, 0.97])).toBe(3)
      expect(es.stopped).toBe(true)
      expect(es.counter).toBe(2)
      expect(es.bestValue).toBe(0.9)
      expect(es.bestEpoch).toBe(1)
    })

    it('update devuelve false en las llamadas previas a la parada y true en la de parada', () => {
      const es = new EarlyStopping({ patience: 2 })
      expect(es.update(1, 0)).toBe(false)
      expect(es.update(0.9, 1)).toBe(false)
      expect(es.update(0.95, 2)).toBe(false) // c 1
      expect(es.update(0.96, 3)).toBe(true)  // c 2 ≥ 2
    })

    it('una mejora reinicia el contador: patience 2, minDelta 0, serie [1, 2, 0.5, 0.6, 0.7] para en índice 4', () => {
      // 1   : mejora, c 0.
      // 2   : 2 < 1 falso → c 1.
      // 0.5 : 0.5 < 1 → mejora, c 0 (best 0.5, bestEpoch 2).
      // 0.6 : c 1.
      // 0.7 : c 2 ≥ 2 → para en índice 4.
      // Si la mejora no reiniciara el contador, pararía en el índice 3.
      const es = new EarlyStopping({ patience: 2, minDelta: 0 })
      expect(stopIndex(es, [1, 2, 0.5, 0.6, 0.7])).toBe(4)
      expect(es.bestValue).toBe(0.5)
      expect(es.bestEpoch).toBe(2)
    })

    it('frontera de minDelta estricta: value = best − minDelta no es mejora', () => {
      // minDelta 0.25 (exacto). best 1 → umbral 0.75. 0.75 < 0.75 es falso → c 1.
      const es = new EarlyStopping({ patience: 5, minDelta: 0.25 })
      es.update(1, 0)
      es.update(0.75, 1)
      expect(es.counter).toBe(1)
      expect(es.bestValue).toBe(1)
      expect(es.bestEpoch).toBe(0)
    })

    it('value justo por debajo del umbral de minDelta sí es mejora', () => {
      // best 1, minDelta 0.25 → umbral 0.75. 0.7 < 0.75 → mejora, c 0.
      const es = new EarlyStopping({ patience: 5, minDelta: 0.25 })
      es.update(1, 0)
      es.update(0.7, 1)
      expect(es.counter).toBe(0)
      expect(es.bestValue).toBe(0.7)
      expect(es.bestEpoch).toBe(1)
    })

    it('una mejora menor que minDelta no reinicia el contador: [1, 0.9, 0.8, 0.7], minDelta 0.25, patience 2, para en índice 2', () => {
      // Umbral 0.75. 0.9 < 0.75 falso → c 1. 0.8 < 0.75 falso → c 2 ≥ 2 → para en índice 2.
      // (0.9 y 0.8 sí bajan respecto de 1, pero no superan minDelta.)
      const es = new EarlyStopping({ patience: 2, minDelta: 0.25 })
      expect(stopIndex(es, [1, 0.9, 0.8, 0.7])).toBe(2)
    })

    it('con minDelta 0 la misma serie mejora en cada llamada y no para', () => {
      // 0.9 < 1, 0.8 < 0.9, 0.7 < 0.8 → mejora siempre; c nunca llega a 2. Índice −1.
      const es = new EarlyStopping({ patience: 2, minDelta: 0 })
      expect(stopIndex(es, [1, 0.9, 0.8, 0.7])).toBe(-1)
      expect(es.bestValue).toBe(0.7)
      expect(es.bestEpoch).toBe(3)
    })

    it('un valor igual al mejor no es mejora (comparación estricta)', () => {
      // minDelta 0: 1 < 1 − 0 falso → c 1 con patience 5.
      const es = new EarlyStopping({ patience: 5, minDelta: 0 })
      es.update(1, 0)
      es.update(1, 1)
      expect(es.counter).toBe(1)
      expect(es.bestEpoch).toBe(0)
    })

    it('patience 3 cuenta tres no-mejoras: [1, 2, 2] sigue y [1, 2, 2, 2] para en índice 3', () => {
      // 1 mejora (c 0). 2 → c 1, 2 → c 2 (< 3, sigue). Índice 2 devuelve false.
      // 2 → c 3 ≥ 3 → para en índice 3.
      expect(stopIndex(new EarlyStopping({ patience: 3 }), [1, 2, 2])).toBe(-1)
      expect(stopIndex(new EarlyStopping({ patience: 3 }), [1, 2, 2, 2])).toBe(3)
    })

    it('patience 1 para en la primera no-mejora (índice 1 de [1, 2])', () => {
      // 1 mejora (c 0). 2 → c 1 ≥ 1 → para.
      expect(stopIndex(new EarlyStopping({ patience: 1 }), [1, 2])).toBe(1)
    })
  })

  describe('modo max', () => {
    it('serie [0.5, 0.7, 0.69, 0.68] con patience 2 para en índice 3; best 0.7 en época 1', () => {
      // 0.5  : 0.5 > −Infinity + 1e-4 → mejora.
      // 0.7  : 0.7 > 0.5 + 1e-4 → mejora, bestEpoch 1, c 0.
      // 0.69 : 0.69 > 0.7001 falso → c 1.
      // 0.68 : c 2 ≥ 2 → para en índice 3.
      const es = new EarlyStopping({ patience: 2, mode: 'max' })
      expect(stopIndex(es, [0.5, 0.7, 0.69, 0.68])).toBe(3)
      expect(es.bestValue).toBe(0.7)
      expect(es.bestEpoch).toBe(1)
      expect(es.counter).toBe(2)
    })

    it('frontera estricta en max: value = best + minDelta no es mejora, value por encima sí', () => {
      // minDelta 0.25 (exacto). best 0.5 → umbral 0.75. 0.75 > 0.75 falso → c 1.
      const es = new EarlyStopping({ patience: 5, mode: 'max', minDelta: 0.25 })
      es.update(0.5, 0)
      es.update(0.75, 1)
      expect(es.counter).toBe(1)
      expect(es.bestValue).toBe(0.5)
      // 0.8 > 0.75 → mejora.
      es.update(0.8, 2)
      expect(es.counter).toBe(0)
      expect(es.bestValue).toBe(0.8)
      expect(es.bestEpoch).toBe(2)
    })
  })

  describe('restoreBest y snapshot de pesos', () => {
    it('sin restoreBest no guarda pesos aunque se pasen', () => {
      const es = new EarlyStopping({ patience: 2, restoreBest: false })
      es.update(1, 0, [1, 2])
      expect(es.bestWeights).toBeNull()
    })

    it('con restoreBest guarda los pesos de la época óptima, no los de la última llamada', () => {
      // patience 2, min:
      // (1, [1,2])  : mejora → snapshot [1,2].
      // (2, [9,9])  : no mejora → c 1, snapshot sigue [1,2].
      // (3, [7,7])  : no mejora → c 2 → para. Snapshot sigue [1,2], no [7,7].
      const es = new EarlyStopping({ patience: 2, restoreBest: true })
      expect(es.update(1, 0, [1, 2])).toBe(false)
      expect(es.update(2, 1, [9, 9])).toBe(false)
      expect(es.update(3, 2, [7, 7])).toBe(true)
      expect(es.bestWeights).toEqual([1, 2])
      expect(es.bestEpoch).toBe(0)
    })

    it('el snapshot se actualiza en cada nueva mejora', () => {
      // (1, [1,2]) mejora; (0.5, [3,4]) mejora → snapshot [3,4].
      const es = new EarlyStopping({ patience: 2, restoreBest: true })
      es.update(1, 0, [1, 2])
      es.update(0.5, 1, [3, 4])
      expect(es.bestWeights).toEqual([3, 4])
      expect(es.bestEpoch).toBe(1)
    })

    it('guarda una copia: modificar el array original no altera bestWeights', () => {
      const w = [1, 2]
      const es = new EarlyStopping({ patience: 2, restoreBest: true })
      es.update(1, 0, w)
      w[0] = 100
      expect(es.bestWeights).toEqual([1, 2])
      expect(es.bestWeights).not.toBe(w)
    })
  })

  describe('estado tras parar', () => {
    it('después de parar, update devuelve true sin modificar contador, mejor ni época', () => {
      // Tras la serie [1, 0.9, 0.95, 0.96] con patience 2: stopped, c 2, best 0.9 en época 1.
      const es = new EarlyStopping({ patience: 2 })
      stopIndex(es, [1, 0.9, 0.95, 0.96])
      expect(es.update(0.001, 4)).toBe(true)
      expect(es.counter).toBe(2)
      expect(es.bestValue).toBe(0.9)
      expect(es.bestEpoch).toBe(1)
    })
  })

  describe('reset', () => {
    it('limpia contador, parada, mejor, época y snapshot, conserva la configuración y permite reutilizar', () => {
      const es = new EarlyStopping({ patience: 2, restoreBest: true, minDelta: 0 })
      // Racha a medias: 1 mejora, 0.9 mejora, 0.95 → c 1. Contador a 1 antes del reset.
      es.update(1, 0, [1, 2])
      es.update(0.9, 1, [3, 4])
      es.update(0.95, 2, [5, 6])
      expect(es.counter).toBe(1)

      es.reset()
      expect(es.counter).toBe(0)
      expect(es.stopped).toBe(false)
      expect(es.bestValue).toBe(Infinity)
      expect(es.bestEpoch).toBe(0)
      expect(es.bestWeights).toBeNull()
      // La configuración no cambia.
      expect(es.patience).toBe(2)
      expect(es.minDelta).toBe(0)
      expect(es.restoreBest).toBe(true)
      expect(es.mode).toBe('min')

      // Tras el reset, la serie completa para de nuevo en el índice 3.
      expect(stopIndex(es, [1, 0.9, 0.95, 0.96])).toBe(3)
    })

    it('en modo max, reset devuelve bestValue a -Infinity', () => {
      const es = new EarlyStopping({ mode: 'max' })
      es.update(0.8, 0)
      es.reset()
      expect(es.bestValue).toBe(-Infinity)
    })
  })
})
