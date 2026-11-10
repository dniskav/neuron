import { describe, it, expect } from 'vitest'
import { GRULayer } from '../src/GRU'
import { Adam } from '../src/optimizers'

describe('GRULayer', () => {
  it('creates with correct dimensions', () => {
    const gru = new GRULayer(3, 4)
    expect(gru.inputSize).toBe(3)
    expect(gru.hSize).toBe(4)
  })

  it('predict returns array of hidden size', () => {
    const gru = new GRULayer(3, 4)
    const h = gru.predict([1, 2, 3])
    expect(h.length).toBe(4)
  })

  it('predict returns finite values', () => {
    const gru = new GRULayer(3, 4)
    const h = gru.predict([0.5, -0.5, 1.0])
    h.forEach(v => expect(isFinite(v)).toBe(true))
  })

  it('maintains state across calls', () => {
    const gru = new GRULayer(2, 3)
    const h1 = gru.predict([1, 0])
    const h2 = gru.predict([0, 1])
    const differs = h1.some((v, i) => v !== h2[i])
    expect(differs).toBe(true)
  })

  it('reset clears state', () => {
    const gru = new GRULayer(2, 3)
    gru.predict([1, 0])
    gru.predict([0, 1])
    gru.reset()
    gru.h.forEach(v => expect(v).toBe(0))
  })

  it('validates input size', () => {
    const gru = new GRULayer(3, 4)
    expect(() => gru.predict([1, 2])).toThrow()
    expect(() => gru.predict([1, 2, 3, 4])).toThrow()
  })

  it('validates constructor parameters', () => {
    expect(() => new GRULayer(0, 4)).toThrow()
    expect(() => new GRULayer(3, 0)).toThrow()
  })

  it('backprop updates weights', () => {
    const gru = new GRULayer(2, 3)
    gru.predict([1, 0])
    gru.predict([0, 1])

    const wBefore = gru.getWeightsFlat()

    const dh_seq = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    gru.backprop(dh_seq, 0.01)

    const wAfter = gru.getWeightsFlat()
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
  })

  it('getWeightsFlat and setWeightsFlat work correctly', () => {
    const gru = new GRULayer(3, 4)
    const w = gru.getWeightsFlat()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    gru.setWeightsFlat(newW)
    expect(gru.getWeightsFlat()).toEqual(newW)
  })

  it('structured getWeights/setWeights work correctly', () => {
    const gru = new GRULayer(3, 4)
    const w = gru.getWeights()
    expect(w.resetGate).toBeDefined()
    expect(w.updateGate).toBeDefined()
    expect(w.newGate).toBeDefined()

    gru.setWeights(w)
    const w2 = gru.getWeights()
    expect(w2.resetGate.W).toEqual(w.resetGate.W)
  })

  it('works with Adam optimizer', () => {
    const gru = new GRULayer(2, 3, () => new Adam())
    gru.predict([1, 0])
    gru.predict([0, 1])

    const wBefore = gru.getWeightsFlat()
    const dh_seq = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    gru.backprop(dh_seq, 0.01)

    const wAfter = gru.getWeightsFlat()
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
    expect(wAfter.every(v => isFinite(v))).toBe(true)
  })

  it('Adam optimizer produces different updates than SGD', () => {
    const gruSGD = new GRULayer(2, 3)
    const gruAdam = new GRULayer(2, 3, () => new Adam())

    // Copy weights so they start the same
    gruSGD.setWeightsFlat(gruAdam.getWeightsFlat())

    // Run identical forward passes
    gruSGD.predict([1, 0]); gruSGD.predict([0, 1])
    gruAdam.predict([1, 0]); gruAdam.predict([0, 1])

    const dh_seq = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    gruSGD.backprop(dh_seq, 0.01)
    gruAdam.backprop(dh_seq, 0.01)

    const wSGD = gruSGD.getWeightsFlat()
    const wAdam = gruAdam.getWeightsFlat()
    const differs = wSGD.some((v, i) => v !== wAdam[i])
    expect(differs).toBe(true)
  })
})

// Pesos deterministas (sin Math.random) para fijar la salida de un forward.
const mkW = (rows: number, cols: number, k: number) =>
  Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => Math.sin(k + 1.7 * i + 0.9 * j) * 0.8))
const mkGate = (h: number, n: number, k: number) => ({
  W: mkW(h, n, k),
  b: Array.from({ length: h }, (_, i) => 0.1 * Math.cos(k + i)),
})
const GOLDEN_SEQ = [[1, -2, 0.5], [3, 0.1, -1], [20, -15, 8], [0.2, 0.3, 0.4]]

describe('GRULayer stable tanh (T059)', () => {
  // Valores calculados con la fórmula anterior (e^{2x}−1)/(e^{2x}+1) antes del
  // cambio; Math.tanh debe coincidir con tolerancia 1e-12.
  const GOLDEN = [
    [0.2982196362022136, 0.07103990232895635, -0.41653076355852947, 0.020181586895853018],
    [0.33377421132313445, -0.7253748069624101, -0.31086466203029284, 0.18457599149287143],
    [0.3340209746901341, -0.9616005552495405, -0.9553556746167102, 0.20467223586225994],
    [-0.11526234281061681, -0.44979700412527135, -0.23079990552005647, 0.20793638157036046],
  ]

  it('forward with fixed weights matches the pre-change golden values', () => {
    const g = new GRULayer(3, 4)
    g.setWeights({ resetGate: mkGate(4, 7, 1), updateGate: mkGate(4, 7, 2), newGate: mkGate(4, 7, 3) })
    GOLDEN_SEQ.forEach((x, t) => {
      const h = g.predict(x)
      h.forEach((v, i) => expect(Math.abs(v - GOLDEN[t][i])).toBeLessThan(1e-12))
    })
  })

  it.each([355, 400, 1000, -355, -400, -1000])('finite state when the candidate preactivation is %d', (x) => {
    const g = new GRULayer(1, 1)
    const zero = { W: [[0, 0]], b: [0] }
    // n_pre = 1·x + 0·(r⊙h) + 0 = x; update gate z = sigmoid(0) = 0.5; h_prev = 0.
    g.setWeights({ resetGate: zero, updateGate: zero, newGate: { W: [[1, 0]], b: [0] } })
    const h = g.predict([x])
    // h = (1 − 0.5)·tanh(x) + 0.5·0 = ±0.5
    expect(h[0]).toBeCloseTo(Math.sign(x) * 0.5, 12)
  })
})
