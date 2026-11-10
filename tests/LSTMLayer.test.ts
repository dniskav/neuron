import { describe, it, expect } from 'vitest'
import { LSTMLayer } from '../src/LSTMLayer'
import { Adam } from '../src/optimizers'

describe('LSTMLayer', () => {
  it('creates with correct dimensions', () => {
    const lstm = new LSTMLayer(3, 4)
    expect(lstm.inputSize).toBe(3)
    expect(lstm.hSize).toBe(4)
  })

  it('predict returns array of hidden size', () => {
    const lstm = new LSTMLayer(3, 4)
    const h = lstm.predict([1, 2, 3])
    expect(h.length).toBe(4)
  })

  it('predict returns finite values', () => {
    const lstm = new LSTMLayer(3, 4)
    const h = lstm.predict([0.5, -0.5, 1.0])
    h.forEach(v => expect(isFinite(v)).toBe(true))
  })

  it('maintains state across calls', () => {
    const lstm = new LSTMLayer(2, 3)
    const h1 = lstm.predict([1, 0])
    const h2 = lstm.predict([0, 1])
    // h2 should be different from h1 because state changed
    const differs = h1.some((v, i) => v !== h2[i])
    expect(differs).toBe(true)
  })

  it('reset clears state', () => {
    const lstm = new LSTMLayer(2, 3)
    lstm.predict([1, 0])
    lstm.predict([0, 1])
    lstm.reset()
    // After reset, h should be all zeros
    lstm.h.forEach(v => expect(v).toBe(0))
    lstm.c.forEach(v => expect(v).toBe(0))
  })

  it('validates input size', () => {
    const lstm = new LSTMLayer(3, 4)
    expect(() => lstm.predict([1, 2])).toThrow()
    expect(() => lstm.predict([1, 2, 3, 4])).toThrow()
  })

  it('validates constructor parameters', () => {
    expect(() => new LSTMLayer(0, 4)).toThrow()
    expect(() => new LSTMLayer(3, 0)).toThrow()
    expect(() => new LSTMLayer(-1, 4)).toThrow()
  })

  it('getWeightsFlat and setWeightsFlat work correctly', () => {
    const lstm = new LSTMLayer(3, 4)
    const w = lstm.getWeightsFlat()
    expect(w.length).toBeGreaterThan(0)
    expect(w.every(v => isFinite(v))).toBe(true)

    const newW = w.map(v => v + 0.01)
    lstm.setWeightsFlat(newW)
    expect(lstm.getWeightsFlat()).toEqual(newW)
  })

  it('backprop updates weights', () => {
    const lstm = new LSTMLayer(2, 3)
    // Run a few steps
    lstm.predict([1, 0])
    lstm.predict([0, 1])

    const wBefore = lstm.getWeightsFlat()

    // Backprop with dummy gradients
    const dh_seq = [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]
    lstm.backprop(dh_seq, 0.01)

    const wAfter = lstm.getWeightsFlat()
    // Weights should have changed
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
  })

  it('works with Adam optimizer', () => {
    const lstm = new LSTMLayer(2, 3, () => new Adam())
    // Run a few steps
    lstm.predict([1, 0])
    lstm.predict([0, 1])

    const wBefore = lstm.getWeightsFlat()

    // Backprop with dummy gradients
    const dh_seq = [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ]
    lstm.backprop(dh_seq, 0.01)

    const wAfter = lstm.getWeightsFlat()
    // Weights should have changed with Adam
    const changed = wBefore.some((v, i) => v !== wAfter[i])
    expect(changed).toBe(true)
    // After multiple steps, Adam should produce weight changes
    expect(wAfter.every(v => isFinite(v))).toBe(true)
  })

  it('Adam optimizer produces different updates than SGD', () => {
    const lstmSGD = new LSTMLayer(2, 3)
    const lstmAdam = new LSTMLayer(2, 3, () => new Adam())

    // Copy Adam's weights to SGD so they start the same
    lstmSGD.setWeightsFlat(lstmAdam.getWeightsFlat())

    // Run identical forward passes
    lstmSGD.predict([1, 0]); lstmSGD.predict([0, 1])
    lstmAdam.predict([1, 0]); lstmAdam.predict([0, 1])

    const dh_seq = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    lstmSGD.backprop(dh_seq, 0.01)
    lstmAdam.backprop(dh_seq, 0.01)

    const wSGD = lstmSGD.getWeightsFlat()
    const wAdam = lstmAdam.getWeightsFlat()
    // Adam and SGD should produce different weight updates
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

describe('LSTMLayer stable tanh (T059)', () => {
  // Valores calculados con la fórmula anterior (e^{2x}−1)/(e^{2x}+1) antes del
  // cambio; Math.tanh debe coincidir con tolerancia 1e-12.
  const GOLDEN = [
    [-0.09122480709131517, -0.061079392919465486, 0.24294666578662866, 0.023249792390182676],
    [-0.07888697764331842, 0.47490773905023936, 0.11026443292027016, -0.021790428943157143],
    [-0.005053580079612365, 0.40133151528466676, 0.02177587883448283, -0.0007375265995631974],
    [0.12453445632733004, 0.12140290779818255, 0.329742603639174, -0.09290595088752522],
  ]

  it('forward with fixed weights matches the pre-change golden values', () => {
    const l = new LSTMLayer(3, 4)
    l.setWeights({
      forgetGate: mkGate(4, 7, 4), inputGate: mkGate(4, 7, 5),
      cellGate: mkGate(4, 7, 6), outputGate: mkGate(4, 7, 7),
    })
    GOLDEN_SEQ.forEach((x, t) => {
      const h = l.predict(x)
      h.forEach((v, i) => expect(Math.abs(v - GOLDEN[t][i])).toBeLessThan(1e-12))
    })
  })

  it.each([355, 400, 1000, -355, -400, -1000])('finite state when the cell preactivation is %d', (x) => {
    const l = new LSTMLayer(1, 1)
    const zero = { W: [[0, 0]], b: [0] }
    // zg = x; i = o = sigmoid(0) = 0.5; c_prev = 0 → c = 0.5·tanh(x) = ±0.5;
    // h = 0.5·tanh(±0.5).
    l.setWeights({ forgetGate: zero, inputGate: zero, cellGate: { W: [[1, 0]], b: [0] }, outputGate: zero })
    const h = l.predict([x])
    const s = Math.sign(x)
    expect(l.c[0]).toBeCloseTo(s * 0.5, 12)
    expect(h[0]).toBeCloseTo(0.5 * Math.tanh(s * 0.5), 12)
  })
})
