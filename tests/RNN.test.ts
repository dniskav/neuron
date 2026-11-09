// Tests de comportamiento de RNN (feature 006, T027).
//
// Complementan tests/oracle/RNN.oracle.test.ts (humo, oráculo de gradiente y
// "aprende" de 002) sin repetirlo. Aquí van: valores exactos de forward con
// pesos fijados a mano, reset del estado oculto, la pérdida MSE a mano, la regla
// de un paso con T = 1 con cada gradiente escrito a mano, el caso de secuencia
// vacía, la validación del constructor y un test de aprendizaje con margen.
//
// Referencia de tanh: Math.tanh, implementación independiente de la del módulo
// (que usa (e^{2x} − 1)/(e^{2x} + 1)). Los valores intermedios se escriben como
// fórmula en cada test, no como salida copiada del código.
//
// Aprendizaje (tarea de memoria: la salida en t es x_{t−1}, con x_{−1} = 0):
//   RNN(1, 4, 1), 6 pasos, 50 llamadas a backward, lr base 0.1.
//   Umbral: pérdida final < 0.5 · inicial. Exigido en el peor caso sobre
//   semillas 1..5 y lr ×0.7 y ×1.3: ratio < 0.7 · 0.5 = 0.35 (margen ≥ 30 %).
//   Peor caso medido en el momento de escribir el test: 0.108 (semilla 3, lr 0.07).
//   Con mapeo directo (salida = f(x_t), sin memoria) el peor caso fue 0.44: por
//   eso la tarea usa memoria, que sí exige la recurrencia.
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { RNN } from '../src/RNN'
import { seedRandom, restoreRandom, withSeed } from './helpers/seed'

beforeEach(() => seedRandom())
afterEach(() => restoreRandom())

// tanh(1) ≈ 0.7615941559557649, tanh(0.6) ≈ 0.5370495669980353 (referencia Math.tanh).
const T1 = Math.tanh(1)

/** Red 1-1-1 con pesos y sesgos fijados a mano (los campos son públicos). */
function net111(wxh: number, whh: number, why: number, bh = 0, by = 0): RNN {
  const n = new RNN(1, 1, 1)
  n.Wxh = [[wxh]]
  n.Whh = [[whh]]
  n.Why = [[why]]
  n.bh = [bh]
  n.by = [by]
  return n
}

/** MSE sobre la secuencia con forward() solamente, desde estado cero. */
function mse(net: RNN, seq: number[][], tgt: number[][]): number {
  net.reset()
  const { outputs } = net.forward(seq)
  let s = 0
  outputs.forEach((o, t) => o.forEach((v, k) => (s += (v - tgt[t][k]) ** 2)))
  return s / (seq.length * outputs[0].length)
}

// Tarea de memoria para el test de aprendizaje.
const SEQ_M: number[][] = [[1], [0], [-1], [0.5], [0.8], [-0.6]]
const TGT_M: number[][] = [[0], [1], [0], [-1], [0.5], [0.8]]

describe('RNN constructor', () => {
  it('rejects non-positive sizes with the documented message', () => {
    const msg = 'RNN: all sizes must be positive'
    expect(() => new RNN(0, 2, 1)).toThrow(msg)
    expect(() => new RNN(2, 0, 1)).toThrow(msg)
    expect(() => new RNN(2, 2, 0)).toThrow(msg)
    expect(() => new RNN(-1, 2, 1)).toThrow(msg)
  })

  it('builds matrices with the documented shapes and Xavier limits sqrt(2/fan_in)', () => {
    // RNN(2, 3, 2): Wxh 3×2 (fan_in 2), Whh 3×3 (fan_in 3), Why 2×3 (fan_in 3).
    // Límites: Wxh → sqrt(2/2) = 1; Whh y Why → sqrt(2/3) ≈ 0.8165.
    const net = new RNN(2, 3, 2)
    expect(net.Wxh).toHaveLength(3)
    net.Wxh.forEach((row) => expect(row).toHaveLength(2))
    expect(net.Whh).toHaveLength(3)
    net.Whh.forEach((row) => expect(row).toHaveLength(3))
    expect(net.Why).toHaveLength(2)
    net.Why.forEach((row) => expect(row).toHaveLength(3))
    expect(net.bh).toEqual([0, 0, 0])
    expect(net.by).toEqual([0, 0])

    const limXH = Math.sqrt(2 / 2)
    const limHH = Math.sqrt(2 / 3)
    const flatXH = net.Wxh.flat()
    const flatHH = net.Whh.flat()
    const flatHY = net.Why.flat()
    for (const w of flatXH) expect(Math.abs(w)).toBeLessThanOrEqual(limXH)
    for (const w of flatHH) expect(Math.abs(w)).toBeLessThanOrEqual(limHH)
    for (const w of flatHY) expect(Math.abs(w)).toBeLessThanOrEqual(limHH)
    // Con semilla fija no todos los pesos son cero.
    expect(flatHH.some((w) => w !== 0)).toBe(true)
  })
})

describe('RNN forward (valores a mano)', () => {
  it('tanh(0) = 0: zero input and zero bias give a zero hidden state and output', () => {
    // h_1 = tanh(1·0 + 1·0 + 0) = tanh(0) = 0; o_1 = 1·0 + 0 = 0.
    // (Una sigmoide daría 0.5 aquí: el test distingue tanh de sigmoid.)
    const net = net111(1, 1, 1)
    const { hiddens, outputs } = net.forward([[0]])
    expect(hiddens[0][0]).toBe(0)
    expect(outputs[0][0]).toBe(0)
  })

  it('one step with two hidden units: h = tanh(Wxh·x + bh), o = by + Why·h', () => {
    // RNN(1, 2, 1), x = 1, h_0 = 0 (Whh = 0):
    //   hRaw_1 = 0.5·1 + 0.1 = 0.6            → h_1 = tanh(0.6)
    //   hRaw_2 = −1·1 + 0    = −1             → h_2 = tanh(−1) = −tanh(1)
    //   o = −0.2 + 2·h_1 + (−1)·h_2 = −0.2 + 2·tanh(0.6) + tanh(1)
    const net = new RNN(1, 2, 1)
    net.Wxh = [[0.5], [-1]]
    net.Whh = [[0, 0], [0, 0]]
    net.Why = [[2, -1]]
    net.bh = [0.1, 0]
    net.by = [-0.2]
    const { hiddens, outputs } = net.forward([[1]])
    expect(hiddens[0][0]).toBeCloseTo(Math.tanh(0.6), 12)
    expect(hiddens[0][1]).toBeCloseTo(-Math.tanh(1), 12)
    expect(outputs[0][0]).toBeCloseTo(-0.2 + 2 * Math.tanh(0.6) + Math.tanh(1), 12)
  })

  it('recurrence: h_2 = tanh(Wxh·x_2 + Whh·h_1 + bh), so h_2 ≠ 0 although x_2 = 0', () => {
    // Wxh = Whh = Why = 1, bh = by = 0, x = [1, 0]:
    //   h_1 = tanh(1·1 + 1·0) = tanh(1)
    //   h_2 = tanh(1·0 + 1·h_1) = tanh(tanh(1))  ≈ 0.6420149920119997
    const net = net111(1, 1, 1)
    const { hiddens, outputs } = net.forward([[1], [0]])
    expect(hiddens[0][0]).toBeCloseTo(T1, 12)
    expect(hiddens[1][0]).toBeCloseTo(Math.tanh(Math.tanh(1)), 12)
    expect(hiddens[1][0]).not.toBe(0)
    expect(outputs[1][0]).toBeCloseTo(Math.tanh(Math.tanh(1)), 12)
  })

  it('returns T outputs of length outputSize and T hiddens of length hiddenSize', () => {
    const net = new RNN(2, 3, 2)
    const seq = [[1, 0], [0, 1], [0.5, -0.5], [0, 0]]
    const { outputs, hiddens } = net.forward(seq)
    expect(outputs).toHaveLength(4)
    expect(hiddens).toHaveLength(4)
    outputs.forEach((o) => expect(o).toHaveLength(2))
    hiddens.forEach((h) => {
      expect(h).toHaveLength(3)
      h.forEach((v) => expect(Math.abs(v)).toBeLessThan(1))
    })
  })

  it('reset() clears the hidden state; without reset the state carries over', () => {
    const net = net111(1, 1, 1)
    net.forward([[1], [0]]) // deja h = tanh(tanh(1)) como estado actual
    const carried = net.forward([[0]]).hiddens[0][0]
    // Sin reset: h = tanh(0 + 1·tanh(tanh(1))) ≠ 0.
    expect(carried).toBeCloseTo(Math.tanh(Math.tanh(Math.tanh(1))), 12)
    expect(carried).not.toBe(0)

    net.reset()
    // Con estado limpio: h = tanh(1·0 + 1·0) = 0 exactamente.
    expect(net.forward([[0]]).hiddens[0][0]).toBe(0)
  })
})

describe('RNN backward (valores a mano)', () => {
  it('returns the MSE over T·outputSize: L = (1 + tanh²(1)) / 2', () => {
    // Wxh = 1, Whh = 0, Why = 1, b = 0; x = [0, 1]:
    //   h_1 = tanh(0) = 0 → o_1 = 0, diff_1 = 0 − 1 = −1
    //   h_2 = tanh(1) → o_2 = t,   diff_2 = t − 0 = t,  t = tanh(1)
    //   L = (1/(T·O)) Σ diff² = (1 + t²) / (2·1)
    const net = net111(1, 0, 1)
    const loss = net.backward([[0], [1]], [[1], [0]], 0)
    expect(loss).toBeCloseTo((1 + T1 * T1) / 2, 12)
  })

  it('discards an earlier forward state: the loss is the one from a fresh state', () => {
    // Whh = 1 para que el estado previo sí influyera si no se descartara.
    // Con lr = 0 los pesos no cambian; el valor debe ser el de (1 + t²)/2 de arriba.
    const net = net111(1, 1, 1)
    net.forward([[5], [5], [5]])
    const loss = net.backward([[0], [1]], [[1], [0]], 0)
    // Desde estado cero: h_1 = tanh(0) = 0, h_2 = tanh(1·1 + 1·0) = t,
    // o = [0, t], targets = [1, 0] → L = (1 + t²)/2.
    expect(loss).toBeCloseTo((1 + T1 * T1) / 2, 12)
  })

  it('one step with T = 1 matches the hand-derived update of every parameter', () => {
    // Red 1-1-1, x = 1, h_0 = 0, y = 0, Wxh = 1, Whh = 0, Why = 2, bh = by = 0, lr = 0.01.
    // Con t = tanh(1):
    //   h = t;  o = 2t;  diff = 2t;  L = diff² = 4t²
    //   dL/do = 2·diff = 4t  →  dOutputs = −4t   (el optimizador hace w + lr·g)
    //   dWhy = dOutputs·h = −4t²                 → Why = 2 − 4·lr·t²
    //   dh = Why·dOutputs = −8t;  dhRaw = dh·(1 − t²) = −8t(1 − t²)
    //   dWxh = dhRaw·x = −8t(1 − t²)             → Wxh = 1 − 8·lr·t(1 − t²)
    //   dbh = dhRaw                              → bh  = −8·lr·t(1 − t²)
    //   dby = dOutputs                           → by  = −4·lr·t
    //   dWhh = dhRaw·h_prev = 0                  → Whh = 0
    const lr = 0.01
    const net = net111(1, 0, 2)
    const loss = net.backward([[1]], [[0]], lr)
    expect(loss).toBeCloseTo(4 * T1 * T1, 12)
    expect(net.Why[0][0]).toBeCloseTo(2 - 4 * lr * T1 * T1, 12)
    expect(net.Wxh[0][0]).toBeCloseTo(1 - 8 * lr * T1 * (1 - T1 * T1), 12)
    expect(net.bh[0]).toBeCloseTo(-8 * lr * T1 * (1 - T1 * T1), 12)
    expect(net.by[0]).toBeCloseTo(-4 * lr * T1, 12)
    expect(net.Whh[0][0]).toBeCloseTo(0, 12)
  })

  it('an empty sequence returns 0 and leaves every weight unchanged', () => {
    const net = new RNN(2, 3, 2)
    const before = JSON.stringify([net.Wxh, net.Whh, net.Why, net.bh, net.by])
    expect(net.backward([], [], 0.1)).toBe(0)
    expect(JSON.stringify([net.Wxh, net.Whh, net.Why, net.bh, net.by])).toBe(before)
  })
})

describe('RNN aprende (tarea de memoria, semilla fija)', () => {
  it('one small step (lr 0.01) lowers the loss for each of 5 seeds', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      withSeed(seed, () => {
        const net = new RNN(1, 4, 1)
        const before = mse(net, SEQ_M, TGT_M)
        net.backward(SEQ_M, TGT_M, 0.01)
        const after = mse(net, SEQ_M, TGT_M)
        expect(after).toBeLessThan(before)
      })
    }
  })

  it('50 backward steps: worst final/initial < 0.7 · 0.5 over seeds 1..5 and lr ×0.7, ×1.3', () => {
    // Umbral 0.5 (la pérdida baja a la mitad). Margen ≥ 30 %: peor caso < 0.35.
    // Medido: peor caso 0.108 (semilla 3, lr 0.07). Los valores iniciales son > 0.2
    // en todas las combinaciones, así que el cociente no es trivial.
    const baseLr = 0.1
    let worst = 0
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const f of [0.7, 1.3]) {
        withSeed(seed, () => {
          const net = new RNN(1, 4, 1)
          const initial = mse(net, SEQ_M, TGT_M)
          expect(initial).toBeGreaterThan(0.2)
          for (let i = 0; i < 50; i++) net.backward(SEQ_M, TGT_M, baseLr * f)
          const final = mse(net, SEQ_M, TGT_M)
          expect(Number.isFinite(final)).toBe(true)
          worst = Math.max(worst, final / initial)
        })
      }
    }
    expect(worst).toBeLessThan(0.7 * 0.5)
  })
})
