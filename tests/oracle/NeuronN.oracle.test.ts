// Gradient-oracle test for NeuronN (feature 002, T007).
//
// Plantilla NetworkN.oracle.test.ts: humo, oraculo y expectLearns.
// Convencion: train() hace w + lr*g con g = -dL/dw; un modelo correcto da coseno ~ +1.
// Parametros: weights (3) y bias, leidos y escritos con getParams/setParams.
// Perdida: L = 0.5 (t - predict(x))^2, con predict deterministico (sin dropout).
import { describe, it, expect } from 'vitest'
import { NeuronN } from '../../src/NeuronN'
import { sigmoid, tanh } from '../../src/activations'
import { SGD } from '../../src/optimizers'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT = [0.5, -0.3, 0.8]
const TARGET = 0.7

// Pesos iniciales fijos (independientes de Math.random) para que el test sea reproducible.
const W0 = [0.2, -0.4, 0.1]
const B0 = 0.05

function setup(activation = sigmoid) {
  const n = new NeuronN(3, activation, () => new SGD())
  n.weights = [...W0]
  n.bias = B0
  const loss = () => 0.5 * (TARGET - n.predict(INPUT)) ** 2
  return {
    n,
    loss,
    step: (lr: number) => n.train(INPUT, TARGET, lr),
    getParams: () => [...n.weights, n.bias],
    setParams: (p: number[]) => {
      n.weights = p.slice(0, 3)
      n.bias = p[3]
    },
  }
}

describe('NeuronN oraculo', () => {
  it('humo: predice una salida finita y train actualiza los pesos', () => {
    const { n } = setup()
    const out = n.predict(INPUT)
    expect(Number.isFinite(out)).toBe(true)
    expect(out).toBeGreaterThan(0)
    expect(out).toBeLessThan(1)
    const before = [...n.weights]
    n.train(INPUT, TARGET, 0.1)
    expect(n.weights.every(Number.isFinite)).toBe(true)
    expect(n.weights).not.toEqual(before)
  })

  it('oraculo: el paso sigue -dL/dw (sigmoid)', () => {
    expectGradientMatches(setup(sigmoid))
  })

  it('oraculo: el paso sigue -dL/dw (tanh)', () => {
    expectGradientMatches(setup(tanh))
  })

  it('expectLearns: la perdida baja a menos de la mitad', () => {
    const m = setup()
    expectLearns({ loss: m.loss, step: () => m.step(0.5), steps: 200 })
  })
})
