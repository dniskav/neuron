// T048 (FR-008 / Q5, barrido total de `push(...)` y `Math.max/min(...)`):
// ningún spread de argumentos sobre arrays que superan el límite de la pila
// (~124 669 argumentos en este worktree, Node 26; ver T035).
//
// Parte A (getWeights): Network, NetworkN, Conv1D, Conv2D y WeightInspector.
// Parte B (argmax/min/max): softmax de MatMul, SoftmaxRegression.predict,
//   Trainer (argmax de las métricas), plateauDecay de LRScheduler y
//   Word2Vec (_softmax sobre el vocabulario).
//
// Dimensiones de los casos grandes (deducidas a mano):
//   - Network(130000, 1, 1): neurona oculta con 130000 pesos + sesgo, salida
//     con 1 peso + sesgo. Longitud = 130001 + 2 = 130003.
//   - NetworkN([130000, 1]): una neurona con 130000 pesos + sesgo = 130001.
//   - Conv1D(10, 1, 130000): núcleos 130000·1·1 + sesgos 130000 = 260000.
//   - Conv2D(2, 2, 1, 1, 130000): núcleos 130000·1·1·1 + sesgos 130000 = 260000.
//   - softmax de 200000 valores iguales a 0.5: cada salida = 1/200000 exacta.
//   - SoftmaxRegression con 200000 clases: pesos de una columna, la única
//     puntuación máxima está en la clase 123456.
//
// Los valores grandes se fijan con setWeights (valores distintos) y se
// comparan con un bucle que devuelve el primer índice distinto: un toEqual
// sobre cientos de miles de números tarda minutos al fallar.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { Network } from '../../src/Network'
import { NetworkN } from '../../src/NetworkN'
import { Conv1D } from '../../src/Conv1D'
import { Conv2D } from '../../src/Conv2D'
import { WeightInspector } from '../../src/WeightInspector'
import { Trainer } from '../../src/Trainer'
import { SoftmaxRegression } from '../../src/LogisticRegression'
import { Word2Vec } from '../../src/Word2Vec'
import { softmax } from '../../src/MatMul'
import { LRScheduler } from '../../src/LRScheduler'
import { seedRandom, restoreRandom } from '../helpers/seed'

const SEED = 0x5eed

beforeEach(() => {
  seedRandom(SEED)
})

afterEach(() => {
  restoreRandom()
})

// Índice de la primera diferencia (-1 si son idénticos, -2 si difiere la longitud).
function firstMismatch(actual: number[], expected: number[]): number {
  if (actual.length !== expected.length) return -2
  for (let i = 0; i < actual.length; i++) if (actual[i] !== expected[i]) return i
  return -1
}

// Array de longitud n con valores distintos: i + 1 (sin repetir, exacto en double).
function ramp(n: number): number[] {
  const a = new Array<number>(n)
  for (let i = 0; i < n; i++) a[i] = i + 1
  return a
}

// ── Parte A: getWeights de tamaño grande ─────────────────────────────────────

describe('T048 parte A: getWeights sin RangeError por spread de argumentos', () => {
  it('Network(130000, 1, 1): longitud 130003 y round-trip exacto', () => {
    const net = new Network(130000, 1, 1)
    const flat = net.getWeights()
    expect(flat.length).toBe(130000 + 1 + 1 + 1)
    const src = ramp(flat.length)
    net.setWeights(src)
    expect(firstMismatch(net.getWeights(), src)).toBe(-1)
  })

  it('NetworkN([130000, 1]): longitud 130001 y round-trip exacto', () => {
    const net = new NetworkN([130000, 1])
    const flat = net.getWeights()
    expect(flat.length).toBe(130000 + 1)
    const src = ramp(flat.length)
    net.setWeights(src)
    expect(firstMismatch(net.getWeights(), src)).toBe(-1)
  })

  it('Conv1D(10, 1, 130000): núcleos y 130000 sesgos, longitud 260000, round-trip exacto', () => {
    const conv = new Conv1D(10, 1, 130000)
    const flat = conv.getWeights()
    expect(flat.length).toBe(130000 + 130000)
    const src = ramp(flat.length)
    conv.setWeights(src)
    expect(firstMismatch(conv.getWeights(), src)).toBe(-1)
  })

  it('Conv2D(2, 2, 1, 1, 130000): núcleos y 130000 sesgos, longitud 260000, round-trip exacto', () => {
    const conv = new Conv2D(2, 2, 1, 1, 130000)
    const flat = conv.getWeights()
    expect(flat.length).toBe(130000 + 130000)
    const src = ramp(flat.length)
    conv.setWeights(src)
    expect(firstMismatch(conv.getWeights(), src)).toBe(-1)
  })

  it('WeightInspector.inspect y inspectAll sobre NetworkN([130000, 1]) con pesos a mano', () => {
    // Pesos = 1 y sesgo = 0 (último elemento). Derivación a mano:
    //   n = 130001; media = 130000/130001; min = 0; max = 1;
    //   muertos con umbral 1e-3: solo el sesgo (|0| < 1e-3) → 1.
    const net = new NetworkN([130000, 1])
    const src = new Array<number>(130001).fill(1)
    src[130000] = 0
    net.setWeights(src)

    const stats = WeightInspector.inspect(net)
    expect(stats.length).toBe(1)
    expect(stats[0].totalParams).toBe(130001)
    expect(stats[0].mean).toBeCloseTo(130000 / 130001, 12)
    expect(stats[0].min).toBe(0)
    expect(stats[0].max).toBe(1)
    expect(stats[0].deadCount).toBe(1)

    const all = WeightInspector.inspectAll(net)
    expect(all.totalParams).toBe(130001)
    expect(all.deadCount).toBe(1)
  })
})

// ── Parte B: argmax / min / max sobre arrays grandes ────────────────────────

describe('T048 parte B: softmax, argmax y mínimo sin spread de argumentos', () => {
  it('softmax de 200000 valores iguales a 0.5: cada salida es 1/200000', () => {
    // exp(0.5 − 0.5) = 1 para todos; suma = 200000; salida = 1/200000.
    const out = softmax(new Array<number>(200000).fill(0.5))
    expect(out.length).toBe(200000)
    const expected = 1 / 200000
    let bad = -1
    for (let i = 0; i < out.length; i++) {
      if (out[i] !== expected) {
        bad = i
        break
      }
    }
    expect(bad).toBe(-1)
  })

  it('SoftmaxRegression.predict con 200000 clases: la clase 123456 gana', () => {
    // Pesos: 1 solo en la clase 123456 (con x = [1]); sesgos 0.
    // La puntuación máxima es única, así que argmax = 123456.
    const model = new SoftmaxRegression()
    const K = 200000
    model.weights = Array.from({ length: K }, (_, k) => [k === 123456 ? 1 : 0])
    model.biases = new Array<number>(K).fill(0)
    expect(model.predict([1])).toBe(123456)
  })

  it('Trainer (métricas binarias): predicción de 200000 valores con máximo en 150000 → clase 1', () => {
    // Objetivo [0] → clase real 0. Argmax = 150000, recortado a nClasses − 1 = 1
    // → clase predicha 1 → una equivocación → accuracy 0.
    const pred = new Array<number>(200000).fill(0)
    pred[150000] = 1
    const fake = {
      predict: () => pred,
      train: () => 0,
      getWeights: () => [0],
      setWeights: () => {},
    }
    const trainer = new Trainer(fake, { epochs: 1, lr: 0.1, computeMetrics: true })
    trainer.train({ inputs: [[0]], targets: [[0]] })
    expect(trainer.getMetrics()[0].accuracy).toBe(0)
  })

  it('LRScheduler.plateauDecay con historial de 200000 valores: mínimo 0.2 → decae', () => {
    // history[i] = 1 salvo el último = 0.2. patience = 200000 → se mira todo.
    // 0.5 >= 0.2 → lr · factor = 0.1 · 0.5.
    const history = new Array<number>(200000).fill(1)
    history[199999] = 0.2
    const sch = new LRScheduler()
    expect(sch.plateauDecay(0.1, 0.5, history, 200000, 0.5)).toBe(0.1 * 0.5)
  })

  it('Word2Vec con vocabulario de 130000 palabras: train devuelve una pérdida finita y positiva', () => {
    // El softmax de skip-gram recorre todo el vocabulario (130000 puntuaciones).
    const words = Array.from({ length: 130000 }, (_, i) => `w${i}`)
    const w2v = new Word2Vec(2)
    w2v.buildVocab([words])
    const losses = w2v.train([['w0', 'w1']], 0.025, 1)
    expect(losses.length).toBe(1)
    expect(Number.isFinite(losses[0])).toBe(true)
    expect(losses[0]).toBeGreaterThan(0)
  })
})

// ── Casos límite: fijados con la semántica de Math.max / Math.min ───────────
// Math.max(...[]) = -Infinity; Math.max con cualquier NaN = NaN;
// indexOf(NaN) = -1; indexOf(-Infinity) en [] = -1.

describe('T048 casos límite (semántica de Math.max / Math.min / indexOf)', () => {
  it('softmax([]) = []', () => {
    expect(softmax([])).toEqual([])
  })

  it('softmax con NaN: todas las salidas son NaN', () => {
    const out = softmax([NaN, 1])
    expect(out.length).toBe(2)
    expect(Number.isNaN(out[0])).toBe(true)
    expect(Number.isNaN(out[1])).toBe(true)
  })

  it('SoftmaxRegression.predict con puntuación NaN devuelve -1', () => {
    // Puntuación de la clase 0: NaN·1 + 0 = NaN → máximo NaN → indexOf = -1.
    const model = new SoftmaxRegression()
    model.weights = [[NaN], [0]]
    model.biases = [0, 0]
    expect(model.predict([1])).toBe(-1)
  })

  it('Trainer con predicción NaN: argmax -1 se recorta a la clase 0', () => {
    // pred = [NaN, 0.5] → argmax -1 → recorte → 0. Objetivo [1] (one-hot de
    // longitud 2) → clase real 1 → equivocación → accuracy 0.
    const fake = {
      predict: () => [NaN, 0.5],
      train: () => 0,
      getWeights: () => [0],
      setWeights: () => {},
    }
    const trainer = new Trainer(fake, { epochs: 1, lr: 0.1, computeMetrics: true })
    trainer.train({ inputs: [[0]], targets: [[0, 1]] })
    expect(trainer.getMetrics()[0].accuracy).toBe(0)
  })

  it('Trainer con predicción vacía: argmax -1 → clase 0 (no lanza)', () => {
    const fake = {
      predict: () => [] as number[],
      train: () => 0,
      getWeights: () => [0],
      setWeights: () => {},
    }
    const trainer = new Trainer(fake, { epochs: 1, lr: 0.1, computeMetrics: true })
    expect(() => trainer.train({ inputs: [[0]], targets: [[1]] })).not.toThrow()
  })

  it('Trainer one-hot con empate: el primer índice del máximo gana', () => {
    // pred = [0.2, 0.9, 0.9] → argmax 1 (primero). Objetivo [0, 0, 1] → clase 2.
    // 1 ≠ 2 → accuracy 0. Si el argmax devolviera el último índice, daría 1.
    const fake = {
      predict: () => [0.2, 0.9, 0.9],
      train: () => 0,
      getWeights: () => [0],
      setWeights: () => {},
    }
    const trainer = new Trainer(fake, { epochs: 1, lr: 0.1, computeMetrics: true })
    trainer.train({ inputs: [[0]], targets: [[0, 0, 1]] })
    expect(trainer.getMetrics()[0].accuracy).toBe(0)
  })

  it('LRScheduler.plateauDecay con historial NaN: el mínimo es NaN y no decae', () => {
    // min([NaN, 1]) = NaN → 0.5 >= NaN es falso → se mantiene lr = 0.1.
    const sch = new LRScheduler()
    expect(sch.plateauDecay(0.1, 0.5, [NaN, 1], 2, 0.5)).toBe(0.1)
  })
})

describe('T048 golden de tamaño pequeño (semilla fija)', () => {
  it('NetworkN([3, 4, 2]) y Network(3, 4, 2) devuelven la longitud esperada', () => {
    // NetworkN([3,4,2]): 4·(3+1) + 2·(4+1) = 16 + 10 = 26.
    expect(new NetworkN([3, 4, 2]).getWeights().length).toBe(26)
    // Network(3,4,2): 4·(3+1) + 2·(4+1) = 26.
    expect(new Network(3, 4, 2).getWeights().length).toBe(26)
  })
})
