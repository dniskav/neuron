// Gradient-oracle test for Neuron (feature 002, T007).
//
// Convencion (plantilla NetworkN.oracle.test.ts): train() hace w + lr*g con
// g = -dL/dw, asi que el paso real debe coincidir con -grad L (coseno ~ +1).
//
// Neuron tiene solo 2 parametros (weight, bias). El helper gradientOracle exige
// >= 3 gradientes numericos no nulos (FEW_GRADIENTS), por eso el coseno se calcula
// aqui en linea, con la misma formula del helper y diferencias finitas centrales:
//   cosine = (delta . -g) / (|delta| |g|)
//
//   1. humo: predice una salida finita en (0, 1) y train() no lanza.
//   2. oraculo: el paso real sigue -dL/dw (coseno > 0.999) en varios puntos.
//      Ademas, el paso coincide con la forma cerrada w + lr*(t-p)*p*(1-p)*x.
//   3. expectLearns: la perdida sobre datos fijos baja a menos de la mitad.
import { describe, it, expect } from 'vitest'
import { Neuron } from '../../src/Neuron'
import { expectLearns } from '../helpers/learns'

const LR = 1e-3
const EPS = 1e-5

function build(weight: number, bias: number): Neuron {
  const n = new Neuron()
  n.weight = weight
  n.bias = bias
  return n
}

// L = 0.5 (t - p)^2 con p = predict(x) usando los parametros dados.
function lossAt(p: [number, number], x: number, t: number): number {
  const n = build(p[0], p[1])
  return 0.5 * (t - n.predict(x)) ** 2
}

// Coseno entre el paso real de train() y -grad L (diferencias finitas centrales).
function cosineOf(w0: number, b0: number, x: number, t: number): number {
  const p0: [number, number] = [w0, b0]
  const g = p0.map((_, i) => {
    const plus: [number, number] = [...p0]
    const minus: [number, number] = [...p0]
    plus[i] += EPS
    minus[i] -= EPS
    return (lossAt(plus, x, t) - lossAt(minus, x, t)) / (2 * EPS)
  })
  const n = build(w0, b0)
  n.train(x, t, LR)
  const delta = [n.weight - w0, n.bias - b0]
  const nd = Math.hypot(delta[0], delta[1])
  const ng = Math.hypot(g[0], g[1])
  const dot = delta[0] * -g[0] + delta[1] * -g[1]
  return dot / (nd * ng)
}

// Puntos (w, b, x, t) elegidos a mano: signos distintos y gradiente no nulo.
const POINTS: [number, number, number, number][] = [
  [0.4, -0.2, 0.7, 0.9],
  [-0.6, 0.3, -1.2, 0.2],
  [1.1, 0.5, 0.3, 0.4],
]

describe('Neuron oraculo', () => {
  it('humo: predice una salida en (0, 1) y train no lanza', () => {
    const n = build(0.4, -0.2)
    const out = n.predict(0.7)
    expect(Number.isFinite(out)).toBe(true)
    expect(out).toBeGreaterThan(0)
    expect(out).toBeLessThan(1)
    expect(() => n.train(0.7, 0.9, LR)).not.toThrow()
    expect(Number.isFinite(n.weight) && Number.isFinite(n.bias)).toBe(true)
  })

  it('oraculo: el paso de train() sigue -dL/dw (coseno > 0.999)', () => {
    for (const [w, b, x, t] of POINTS) {
      expect(cosineOf(w, b, x, t)).toBeGreaterThan(0.999)
    }
  })

  it('oraculo: el paso coincide con la forma cerrada w + lr*(t-p)*p*(1-p)*x', () => {
    for (const [w, b, x, t] of POINTS) {
      const p = 1 / (1 + Math.exp(-(x * w + b)))
      const g = (t - p) * p * (1 - p)
      const n = build(w, b)
      n.train(x, t, LR)
      expect(n.weight).toBeCloseTo(w + LR * g * x, 12)
      expect(n.bias).toBeCloseTo(b + LR * g, 12)
    }
  })

  it('expectLearns: la perdida sobre datos fijos baja a menos de la mitad', () => {
    const n = build(0.1, 0.0)
    const data: [number, number][] = [
      [0.5, 0.9],
      [-0.4, 0.1],
      [0.9, 0.8],
    ]
    const loss = () => data.reduce((s, [x, t]) => s + 0.5 * (t - n.predict(x)) ** 2, 0)
    expectLearns({
      loss,
      step: () => data.forEach(([x, t]) => n.train(x, t, 0.5)),
      steps: 200,
    })
  })
})
