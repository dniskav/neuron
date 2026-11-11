// Validación de entradas de EarlyStopping, LRScheduler, DataLoader,
// PositionalEncoding y Word2Vec (T042, specs/006-coverage).
//
// Cada grupo tiene:
//   - un golden con valores válidos y semilla fija (idéntico antes y después
//     del cambio; los valores se capturaron con el código previo a T042), y
//   - tests de error: lanzan `Clase.método: …` ANTES de tocar el estado.
//
// Decisiones (ver informe de T042):
//   - EarlyStopping.patience: entero ≥ 1, como exige Trainer.ts. patience 0
//     pararía en la primera no-mejora sin ningún margen y ningún uso lo pide.
//   - Word2Vec.train lr: finito ≥ 0. lr = 0 se mantiene válido porque
//     tests/Word2Vec.test.ts y tests/oracle/Word2Vec.oracle.test.ts lo usan
//     como no-op (fuera de los archivos de T042).

import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { EarlyStopping } from '../../src/EarlyStopping'
import { LRScheduler } from '../../src/LRScheduler'
import { DataLoader } from '../../src/DataLoader'
import { PositionalEncoding } from '../../src/PositionalEncoding'
import { Word2Vec } from '../../src/Word2Vec'
import { restoreRandom, seedRandom } from '../helpers/seed'

beforeEach(() => seedRandom(0x5eed))
afterEach(restoreRandom)

// ─── Golden (valores válidos, capturados antes del cambio) ───────────────────

describe('golden: entradas válidas no cambian', () => {
  it('EarlyStopping: serie [1, 0.9, 0.8, 0.7], patience 2, minDelta 0.25', () => {
    // 1 mejora (umbral Infinity); 0.9 no supera 1 − 0.25 = 0.75 → c 1;
    // 0.8 → c 2 ≥ 2 → para en el índice 2; 0.7 ya llega parado.
    const es = new EarlyStopping({ patience: 2, mode: 'min', minDelta: 0.25 })
    const flags = [1, 0.9, 0.8, 0.7].map((v, i) => es.update(v, i))
    expect(flags).toEqual([false, false, true, true])
    expect(es.bestValue).toBe(1)
    expect(es.counter).toBe(2)
    expect(es.stopped).toBe(true)
    expect(es.bestEpoch).toBe(0)
  })

  it('LRScheduler: stepDecay, exponentialDecay y cosineAnnealing con valores a mano', () => {
    const L = new LRScheduler()
    // stepDecay: 0.8 · 0.5^floor(7/3) = 0.8 · 0.25 = 0.2
    // exponentialDecay: 2 · 0.5³ = 0.25
    // cosineAnnealing(1, 2, 10, 0.1): 0.1 + 0.5·0.9·(1 + cos(0.2π)) = 0.914057647…
    // cosineAnnealing(1, 0, 10): 0 + 0.5·1·(1 + 1) = 1
    expect(L.stepDecay(0.8, 7, 0.5, 3)).toBeCloseTo(0.2, 12)
    expect(L.exponentialDecay(2, 3, 0.5)).toBeCloseTo(0.25, 12)
    expect(L.cosineAnnealing(1, 2, 10, 0.1)).toBeCloseTo(0.9140576474687263, 12)
    expect(L.cosineAnnealing(1, 0, 10)).toBe(1)
  })

  it('DataLoader: semilla 42, 5 muestras, batchSize 2, validationSplit 0.2', () => {
    seedRandom(42)
    const data = { inputs: [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], targets: [[1], [2], [3], [4], [5]] }
    const dl = new DataLoader(data, 2, 0.2)
    const batches = []
    while (dl.hasNext()) batches.push(dl.next())
    expect(dl.length).toBe(4)
    expect(dl.validationLength).toBe(1)
    expect(batches).toEqual([
      { inputs: [[1, 0], [5, 0]], targets: [[1], [5]] },
      { inputs: [[3, 0], [2, 0]], targets: [[3], [2]] },
    ])
    expect(dl.getValidationData()).toEqual({ inputs: [[4, 0]], targets: [[4]] })
  })

  it('PositionalEncoding: encodeSequence(3, 4) y encode(5, 6) con dModel par', () => {
    // pos 1, dModel 4: sin(1), cos(1), sin(0.01), cos(0.01) (ver tests/PositionalEncoding.test.ts)
    expect(PositionalEncoding.encodeSequence(3, 4)).toEqual([
      [0, 1, 0, 1],
      [0.8414709848078965, 0.5403023058681398, 0.009999833334166664, 0.9999500004166653],
      [0.9092974268256817, -0.4161468365471424, 0.01999866669333308, 0.9998000066665778],
    ])
    // Tolerancia 1e-12, no igualdad exacta: Math.pow/Math.sin difieren en 1 ULP
    // entre versiones de V8 (0.2300017116647674 en Node 20, …746 en Node 26).
    const pe = PositionalEncoding.encode(5, 6)
    const esperado = [
      -0.9589242746631385, 0.28366218546322625, 0.23000171166476746,
      0.9731902242785205, 0.010771965118034833, 0.9999419807006283,
    ]
    expect(pe.length).toBe(esperado.length)
    esperado.forEach((v, i) => expect(pe[i]).toBeCloseTo(v, 12))
  })

  it('Word2Vec: semilla 7, skip-gram por defecto, lr 0.1, 2 épocas', () => {
    seedRandom(7)
    const m = new Word2Vec(3, { windowSize: 1 })
    const losses = m.train([['a', 'b', 'c', 'a']], 0.1, 2)
    expect(losses).toEqual([1.0809033832926496, 1.0681076810497692])
    expect(m.getEmbedding('a')).toEqual([-0.6046690760577088, -0.5280206759085605, 0.5437279414738643])
  })
})

// ─── EarlyStopping ───────────────────────────────────────────────────────────

describe('EarlyStopping: validación', () => {
  it.each([NaN, 1.5, -1, 0])('patience %s lanza (debe ser entero >= 1)', (patience) => {
    expect(() => new EarlyStopping({ patience })).toThrow(
      /^EarlyStopping\.constructor: patience must be an integer >= 1/,
    )
  })

  it.each([NaN, -1, Infinity])('minDelta %s lanza (debe ser finito >= 0)', (minDelta) => {
    expect(() => new EarlyStopping({ minDelta })).toThrow(/^EarlyStopping\.constructor: minDelta/)
  })

  it('patience 1 y minDelta 0 son válidos (frontera)', () => {
    expect(() => new EarlyStopping({ patience: 1, minDelta: 0 })).not.toThrow()
  })

  it('update(NaN) lanza y no modifica el estado', () => {
    const es = new EarlyStopping({ patience: 2 })
    expect(() => es.update(NaN, 0)).toThrow(/^EarlyStopping\.update: value must not be NaN/)
    expect(es.counter).toBe(0)
    expect(es.bestValue).toBe(Infinity)
    expect(es.bestEpoch).toBe(0)
    expect(es.stopped).toBe(false)
    // Tras el error, la serie válida se comporta como en una instancia nueva.
    expect(es.update(1, 1)).toBe(false)
    expect(es.bestValue).toBe(1)
  })
})

// ─── LRScheduler ─────────────────────────────────────────────────────────────

describe('LRScheduler: validación', () => {
  const s = new LRScheduler()

  it.each([0, -1, NaN])('stepDecay con epochsDrop %s lanza', (epochsDrop) => {
    expect(() => s.stepDecay(1, 1, 0.5, epochsDrop)).toThrow(/^LRScheduler\.stepDecay: epochsDrop/)
  })

  it.each([NaN, Infinity, -Infinity])('exponentialDecay con decayRate %s lanza', (decayRate) => {
    expect(() => s.exponentialDecay(1, 1, decayRate)).toThrow(/^LRScheduler\.exponentialDecay: decayRate/)
  })

  it.each([0, -1, NaN])('cosineAnnealing con maxEpochs %s lanza', (maxEpochs) => {
    expect(() => s.cosineAnnealing(1, 0, maxEpochs)).toThrow(/^LRScheduler\.cosineAnnealing: maxEpochs/)
  })
})

// ─── DataLoader ──────────────────────────────────────────────────────────────

describe('DataLoader: validación', () => {
  const data = { inputs: [[1], [2]], targets: [[1], [2]] }

  it.each([0, -1, 1.5, NaN])('batchSize %s lanza (entero >= 1)', (batchSize) => {
    expect(() => new DataLoader(data, batchSize)).toThrow(/^DataLoader\.constructor: batchSize/)
  })

  it('batchSize 1 (por defecto) y 2 son válidos', () => {
    expect(() => new DataLoader(data)).not.toThrow()
    expect(() => new DataLoader(data, 2)).not.toThrow()
  })
})

// ─── PositionalEncoding ──────────────────────────────────────────────────────

describe('PositionalEncoding: validación', () => {
  it.each([-1, 1.5, NaN])('encode con pos %s lanza (entero >= 0)', (pos) => {
    expect(() => PositionalEncoding.encode(pos, 4)).toThrow(/^PositionalEncoding\.encode: pos/)
  })

  it.each([3, 0, -2, 1.5])('encode con dModel %s lanza (par, entero >= 2)', (dModel) => {
    expect(() => PositionalEncoding.encode(1, dModel)).toThrow(/^PositionalEncoding\.encode: dModel/)
  })

  it.each([-1, 2.5, NaN])('encodeSequence con seqLen %s lanza (entero >= 0)', (seqLen) => {
    expect(() => PositionalEncoding.encodeSequence(seqLen, 4)).toThrow(
      /^PositionalEncoding\.encodeSequence: seqLen/,
    )
  })

  it('encodeSequence con dModel impar lanza', () => {
    expect(() => PositionalEncoding.encodeSequence(2, 3)).toThrow(/^PositionalEncoding\.encodeSequence: dModel/)
  })

  it('apply hereda la validación de dModel (apply usa encodeSequence)', () => {
    expect(() => PositionalEncoding.apply([[0, 0, 0]])).toThrow(/dModel/)
  })

  it('encodeSequence(0, 4) es válido y devuelve []', () => {
    expect(PositionalEncoding.encodeSequence(0, 4)).toEqual([])
  })
})

// ─── Word2Vec.train ──────────────────────────────────────────────────────────

describe('Word2Vec.train: validación', () => {
  const sents = [['a', 'b', 'c', 'a']]

  it.each([NaN, Infinity, -0.1])('lr %s lanza (finito >= 0)', (lr) => {
    const m = new Word2Vec(3)
    expect(() => m.train(sents, lr, 1)).toThrow(/^Word2Vec\.train: lr/)
  })

  it('lr 0 sigue siendo válido (no-op usado por tests de Word2Vec fuera de T042)', () => {
    const m = new Word2Vec(3)
    expect(() => m.train(sents, 0, 1)).not.toThrow()
  })

  it.each([0, -1, 1.5, NaN])('epochs %s lanza (entero >= 1)', (epochs) => {
    const m = new Word2Vec(3)
    expect(() => m.train(sents, 0.1, epochs)).toThrow(/^Word2Vec\.train: epochs/)
  })

  it('un train rechazado no construye el vocabulario ni cambia los embeddings', () => {
    const m = new Word2Vec(3)
    const before = m.embeddings.map((r) => [...r])
    expect(() => m.train(sents, NaN, 1)).toThrow()
    expect(() => m.train(sents, 0.1, 0)).toThrow()
    expect(m.vocabSize).toBe(0)
    expect(m.embeddings).toEqual(before)
  })

  it('getEmbedding de palabra desconocida ya lanza', () => {
    const m = new Word2Vec(3)
    m.buildVocab(sents)
    expect(() => m.getEmbedding('zz')).toThrow(/unknown word "zz"/)
  })
})
