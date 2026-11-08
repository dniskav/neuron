// Gradient-oracle test for BatchNorm (feature 002, T012).
//
// BatchNorm normaliza con estadisticas de ejecucion (runningMean / runningVar),
// y forward() las ACTUALIZA en cada llamada. Para que loss() sea determinista,
// loss() y step() restauran esas estadisticas a su valor inicial antes de actuar.
// Asi la perdida es funcion solo de (gamma, beta).
//
// Parametros del oraculo: gamma y beta (getWeights/setWeights, en ese orden).
// Paso real: forward(x) y luego trainParams(t - y, lr), que hace
//   gamma += lr * (t - y) * x_norm   y   beta += lr * (t - y),
// es decir, sigue -dL/dgamma y -dL/dbeta con L = 0.5 * sum (t - y)^2.
//
// La derivada respecto a la entrada se comprueba aparte: backward() devuelve
// dOut * gamma / std, que es la derivada con estadisticas tratadas como
// constantes. Con momentum = 1 las estadisticas no cambian nunca, asi que
// forward(x) es exactamente esa funcion y la comparacion con diferencias
// finitas es exacta.
import { describe, it, expect } from 'vitest'
import { BatchNorm } from '../../src/BatchNorm'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT = [0.5, -0.3, 0.8]
const TARGET = [0.7, 0.2, -0.4]
const INIT_GAMMA = [0.9, 1.2, 0.6]
const INIT_BETA = [0.1, -0.2, 0.05]
// Estadisticas de ejecucion iniciales no triviales (media y varianza por canal).
const RM0 = [0.2, -0.1, 0.3]
const RV0 = [0.5, 2, 1.2]

function sumSquares(y: number[]): number {
  return y.reduce((s, v, i) => s + 0.5 * (TARGET[i] - v) ** 2, 0)
}

function cosine(a: number[], b: number[]): number {
  const dot = a.reduce((s, v, i) => s + v * b[i], 0)
  const na = Math.sqrt(a.reduce((s, v) => s + v * v, 0))
  const nb = Math.sqrt(b.reduce((s, v) => s + v * v, 0))
  return dot / (na * nb)
}

function setup() {
  const bn = new BatchNorm(3)
  bn.runningMean = [...RM0]
  bn.runningVar = [...RV0]
  bn.setWeights([...INIT_GAMMA, ...INIT_BETA])
  const resetStats = () => {
    bn.runningMean = [...RM0]
    bn.runningVar = [...RV0]
  }
  const loss = () => {
    resetStats()
    return sumSquares(bn.forward(INPUT))
  }
  const step = (lr: number) => {
    resetStats()
    const y = bn.forward(INPUT)
    bn.trainParams(
      y.map((v, i) => TARGET[i] - v),
      lr,
    )
  }
  return {
    bn,
    loss,
    step,
    getParams: () => bn.getWeights(),
    setParams: (p: number[]) => bn.setWeights(p),
  }
}

describe('BatchNorm oraculo (gamma, beta)', () => {
  it('humo: predice salidas finitas y el paso cambia gamma y beta', () => {
    const m = setup()
    const y = m.bn.forward(INPUT)
    expect(y).toHaveLength(3)
    expect(y.every(Number.isFinite)).toBe(true)
    const before = m.getParams()
    m.step(0.1)
    const after = m.getParams()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after).not.toEqual(before)
  })

  it('oraculo: el paso sigue -dL/dgamma y -dL/dbeta', () => {
    expectGradientMatches(setup())
  })

  it('expectLearns: la perdida baja a menos de la mitad ajustando la salida a un objetivo fijo', () => {
    const m = setup()
    expectLearns({ loss: m.loss, step: () => m.step(0.5), steps: 200 })
  })
})

describe('BatchNorm oraculo (derivada respecto a la entrada)', () => {
  it('backward sigue -dL/dx frente a diferencias finitas de la entrada', () => {
    // momentum = 1: forward no cambia las estadisticas (0 * x no aporta nada).
    const bn = new BatchNorm(3, 1)
    bn.runningMean = [...RM0]
    bn.runningVar = [...RV0]
    bn.setWeights([...INIT_GAMMA, ...INIT_BETA])

    const lossAt = (x: number[]) => sumSquares(bn.forward(x))

    const y = bn.forward(INPUT)
    const dX = bn.backward(y.map((v, i) => TARGET[i] - v))

    const eps = 1e-5
    const negGradFD = INPUT.map((_, i) => {
      const plus = [...INPUT]
      plus[i] += eps
      const minus = [...INPUT]
      minus[i] -= eps
      return -(lossAt(plus) - lossAt(minus)) / (2 * eps)
    })

    expect(dX.every(Number.isFinite)).toBe(true)
    expect(cosine(dX, negGradFD)).toBeGreaterThan(0.999)
  })
})
