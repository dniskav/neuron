// Gradient-oracle test for Layer (feature 002, T007).
//
// Layer no tiene metodo train(): cada neurona NeuronN es independiente (comparte
// entradas, pero su gradiente depende solo de sus propios pesos). Por eso el paso
// de la capa es train() de cada neurona con su objetivo, y la perdida es la suma
// de 0.5 (t_i - p_i)^2 sobre las salidas.
// Parametros: por neurona, weights y bias, concatenados en un vector plano.
import { describe, it, expect } from 'vitest'
import { Layer } from '../../src/Layer'
import { SGD } from '../../src/optimizers'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT = [0.5, -0.3, 0.8]
const TARGETS = [0.7, 0.2]

// Pesos iniciales fijos (2 neuronas x (3 pesos + bias)).
const INIT = [
  [0.2, -0.4, 0.1, 0.05],
  [-0.3, 0.25, 0.6, -0.1],
]

function setup() {
  const layer = new Layer(2, 3, undefined, () => new SGD())
  layer.neurons.forEach((n, i) => {
    n.weights = INIT[i].slice(0, 3)
    n.bias = INIT[i][3]
  })
  const loss = () => {
    const p = layer.predict(INPUT)
    return p.reduce((s, v, i) => s + 0.5 * (TARGETS[i] - v) ** 2, 0)
  }
  return {
    layer,
    loss,
    step: (lr: number) => layer.neurons.forEach((n, i) => n.train(INPUT, TARGETS[i], lr)),
    getParams: () => layer.neurons.flatMap((n) => [...n.weights, n.bias]),
    setParams: (p: number[]) => {
      layer.neurons.forEach((n, i) => {
        const o = i * 4
        n.weights = p.slice(o, o + 3)
        n.bias = p[o + 3]
      })
    },
  }
}

describe('Layer oraculo', () => {
  it('humo: predice salidas finitas y el paso cambia los pesos', () => {
    const m = setup()
    const out = m.layer.predict(INPUT)
    expect(out).toHaveLength(2)
    expect(out.every(Number.isFinite)).toBe(true)
    const before = m.getParams()
    m.step(0.1)
    expect(m.getParams().every(Number.isFinite)).toBe(true)
    expect(m.getParams()).not.toEqual(before)
  })

  it('oraculo: el paso sigue -dL/dw en todas las neuronas', () => {
    expectGradientMatches(setup())
  })

  it('expectLearns: la perdida de la capa baja a menos de la mitad', () => {
    const m = setup()
    expectLearns({ loss: m.loss, step: () => m.step(0.5), steps: 200 })
  })
})
