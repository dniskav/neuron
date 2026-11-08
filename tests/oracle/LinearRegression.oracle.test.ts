// Gradient-oracle test for LinearRegression.fitGD.
//
// fitGD minimiza MSE = (1/m) Σ (ŷ - y)^2 con ŷ = Xa·W, Xa = [X | 1] (sesgo en la
// última posición de W). Convención (constitución II): el paso real debe seguir
// -dL/dw, así que el coseno con el gradiente numérico es ~ +1.
//
// fitGD CONTINUA desde los pesos existentes si weights.length === nFeatures + 1
// (solo inicializa a cero si no hay pesos o la forma no coincide). Por eso el paso es un
// `fitGD(X, y, lr, 1)` real que parte del punto que fija el oráculo, y expectLearns acumula
// pasos de 1 época. fitNormal se usa como referencia de solución cerrada.
import { describe, it, expect } from 'vitest'
import { LinearRegression } from '../../src/LinearRegression'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

// Caso de 1 muestra (firma permitida): L = (w·[x,1] - y)^2.
const X1 = [[0.5, -1.5]]
const Y1 = [2.0]

// Caso de 4 muestras, datos exactos de y = 2·x1 - x2 + 0.5 (residuo cero).
// Solución de mínimos cuadrados a mano: W = [2, -1, 0.5].
const X4 = [
  [1, 0],
  [0, 1],
  [1, 1],
  [2, 1],
]
const Y4 = [2.5, -0.5, 1.5, 3.5]
const W_EXACT = [2, -1, 0.5]

function setup(X: number[][], y: number[]) {
  const model = new LinearRegression()
  // Estado inicial explícito: fitGD continúa desde estos pesos (punto de partida 0).
  model.weights = new Array(X[0].length + 1).fill(0)
  const nFeat = X[0].length
  // Pérdida MSE del objetivo que minimiza fitGD, leyendo los pesos actuales.
  const loss = () => {
    let s = 0
    for (let i = 0; i < X.length; i++) {
      let yhat = model.weights[nFeat]
      for (let j = 0; j < nFeat; j++) yhat += model.weights[j] * X[i][j]
      s += (yhat - y[i]) ** 2
    }
    return s / X.length
  }
  return {
    model,
    loss,
    fit: (lr: number, epochs: number) => void model.fitGD(X, y, lr, epochs),
    step: (lr: number) => void model.fitGD(X, y, lr, 1),
    getParams: () => [...model.weights],
    setParams: (p: number[]) => {
      model.weights = [...p]
    },
  }
}

describe('LinearRegression oraculo (fitGD)', () => {
  it('humo: la perdida por epoca es finita, baja y los pesos son finitos', () => {
    const model = new LinearRegression()
    const hist = model.fitGD(X4, Y4, 0.2, 50)
    expect(hist).toHaveLength(50)
    expect(hist.every(Number.isFinite)).toBe(true)
    expect(hist[49]).toBeLessThan(hist[0])
    expect(model.weights.every(Number.isFinite)).toBe(true)
  })

  it('oraculo: 1 muestra y 1 epoca, el paso sigue -dL/dw', () => {
    expectGradientMatches(setup(X1, Y1))
  })

  it('oraculo: 4 muestras y 1 epoca, la suma sobre muestras sigue -dL/dw', () => {
    expectGradientMatches(setup(X4, Y4))
  })

  it('expectLearns: fitGD en 2000 pasos de 1 epoca baja la perdida a menos de la mitad', () => {
    const m = setup(X4, Y4)
    expectLearns({ loss: m.loss, step: () => m.step(0.2), steps: 2000 })
  })

  it('convergencia: fitGD llega a la solucion de fitNormal (tolerancia 1e-6)', () => {
    const exact = new LinearRegression()
    exact.fitNormal(X4, Y4)
    // Comprobación contra la solución exacta calculada a mano.
    W_EXACT.forEach((w, i) => expect(exact.weights[i]).toBeCloseTo(w, 9))

    const m = setup(X4, Y4)
    m.fit(0.2, 2000)
    W_EXACT.forEach((w, i) => expect(m.model.weights[i]).toBeCloseTo(w, 6))
  })
})
