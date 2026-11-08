// T012 (feature 005, N7 y Q6): predict / predictProba / predict+getCoefficients validan
// contra la forma de los pesos PUBLICOS (weights.length, weights[0].length, weights.length - 1),
// no contra el campo privado _nFeatures, que solo fija train(). Con pesos asignados a mano
// (sin train) predicen; las formas inconsistentes lanzan con mensaje claro.
//
// Valores a mano en cada test. Golden de train(): constantes capturadas con la base ANTES del
// cambio (semilla 0x5eed, la misma de tests/setup.ts), tolerancia 1e-9.
import { describe, it, expect } from 'vitest'
import { LogisticRegression, SoftmaxRegression } from '../../src/LogisticRegression'
import { LinearRegression } from '../../src/LinearRegression'
import { seedRandom } from '../helpers/seed'

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

// softmax de referencia, independiente de src/
function softmaxRef(scores: number[]): number[] {
  const z = scores.reduce((acc, s) => acc + Math.exp(s), 0)
  return scores.map((s) => Math.exp(s) / z)
}

function expectClose(actual: number[], expected: number[], tol = 1e-9) {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i] - expected[i]), `indice ${i}`).toBeLessThanOrEqual(tol)
  }
}

// ─── LogisticRegression ──────────────────────────────────────────────────────

describe('LogisticRegression.predict con pesos asignados a mano (N7)', () => {
  it('sigma(w·x + b) a mano: w=[0.5,-1], b=0.25, x=[1,0.5]', () => {
    const m = new LogisticRegression()
    m.weights = [0.5, -1]
    m.bias = 0.25
    // z = 0.5·1 + (−1)·0.5 + 0.25 = 0.25
    expect(m.predict([1, 0.5])).toBeCloseTo(sigmoid(0.25), 12)
    expect(m.classify([1, 0.5])).toBe(1)
    // z = 0.5·(−1) + (−1)·0.5 + 0.25 = −0.75
    expect(m.classify([-1, 0.5])).toBe(0)
  })

  it('valida contra weights.length y no contra el numero de rasgos de un train() previo', () => {
    // train con 3 rasgos fija _nFeatures = 3; despues se asignan pesos de 2 rasgos a mano.
    const m = new LogisticRegression()
    m.train([[1, 0, 0]], [1], 0.1, 1)
    m.weights = [2, -1]
    m.bias = 0
    // z = 2·3 + (−1)·1 + 0 = 5
    expect(m.predict([3, 1])).toBeCloseTo(sigmoid(5), 12)
  })

  it('rechaza x de longitud distinta de weights.length con mensaje claro', () => {
    const m = new LogisticRegression()
    m.weights = [1, 2]
    expect(() => m.predict([1, 2, 3])).toThrow(
      /LogisticRegression\.predict: expected array of length 2, got 3/,
    )
  })

  it('sin pesos sigue lanzando "not been trained"', () => {
    expect(() => new LogisticRegression().predict([1, 2])).toThrow(/has not been trained yet/)
  })
})

// ─── SoftmaxRegression ───────────────────────────────────────────────────────

describe('SoftmaxRegression.predictProba con pesos asignados a mano (N7)', () => {
  const W = [
    [1, 0],
    [0, 1],
    [-1, -1],
  ]
  const B = [0, 0.5, 0]
  const x = [0.2, 0.4]

  function model(weights: number[][], biases: number[]) {
    const m = new SoftmaxRegression()
    m.weights = weights.map((r) => [...r])
    m.biases = [...biases]
    return m
  }

  it('coincide con softmax a mano y suma 1: scores = [0.2, 0.9, −0.6]', () => {
    // class0: 1·0.2 + 0·0.4 + 0   = 0.2
    // class1: 0·0.2 + 1·0.4 + 0.5 = 0.9
    // class2: −1·0.2 − 1·0.4 + 0  = −0.6
    const m = model(W, B)
    const p = m.predictProba(x)
    expectClose(p, softmaxRef([0.2, 0.9, -0.6]), 1e-12)
    expect(p.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12)
    expect(m.predict(x)).toBe(1)
  })

  it('rechaza biases de longitud distinta de weights.length', () => {
    expect(() => model(W, [0, 0.5]).predictProba(x)).toThrow(
      /SoftmaxRegression\.predictProba: expected 3 biases, got 2/,
    )
  })

  it('rechaza filas de pesos de anchos distintos', () => {
    expect(() =>
      model(
        [
          [1, 0],
          [0, 1, 5],
          [-1, -1],
        ],
        B,
      ).predictProba(x),
    ).toThrow(/SoftmaxRegression\.predictProba: weights row 1 has length 3, expected 2/)
  })

  it('rechaza x de longitud distinta de las columnas de los pesos', () => {
    expect(() => model(W, B).predictProba([1, 2, 3])).toThrow(
      /SoftmaxRegression\.predictProba: expected array of length 2, got 3/,
    )
  })

  it('sin pesos sigue lanzando "not been trained"', () => {
    expect(() => new SoftmaxRegression().predictProba(x)).toThrow(/has not been trained yet/)
  })
})

// ─── LinearRegression ────────────────────────────────────────────────────────
// weights = [w1, w2, bias] para 2 rasgos.

describe('LinearRegression.predict y getCoefficients con pesos asignados a mano (N7, Q6)', () => {
  it('predict a mano: 2·3 + (−1)·4 + 0.5 = 2.5', () => {
    const m = new LinearRegression()
    m.weights = [2, -1, 0.5]
    expect(m.predict([3, 4])).toBeCloseTo(2.5, 12)
  })

  it('el sesgo es el ultimo peso: con x = [0, 0] devuelve exactamente el sesgo', () => {
    const m = new LinearRegression()
    m.weights = [2, -1, 0.5]
    expect(m.predict([0, 0])).toBe(0.5)
  })

  it('getCoefficients separa pesos y sesgo', () => {
    const m = new LinearRegression()
    m.weights = [2, -1, 0.5]
    expect(m.getCoefficients()).toEqual({ weights: [2, -1], bias: 0.5 })
  })

  it('usa weights.length − 1 aunque un fitGD previo tuviera otro numero de rasgos', () => {
    const m = new LinearRegression()
    m.fitGD(
      [
        [1, 2, 3],
        [2, 0, 1],
        [0, 1, 0],
      ],
      [1, 2, 3],
      0.01,
      1,
    )
    m.weights = [1, 2, 0.5]
    // 1·1 + 2·1 + 0.5 = 3.5
    expect(m.predict([1, 1])).toBeCloseTo(3.5, 12)
    expect(m.getCoefficients()).toEqual({ weights: [1, 2], bias: 0.5 })
  })

  it('rechaza x de longitud distinta de weights.length − 1 con mensaje claro', () => {
    const m = new LinearRegression()
    m.weights = [2, -1, 0.5]
    expect(() => m.predict([1])).toThrow(/LinearRegression\.predict: expected 2 features, got 1/)
  })

  it('sin pesos sigue lanzando "not been fitted" en predict y getCoefficients', () => {
    expect(() => new LinearRegression().predict([1, 2])).toThrow(/not been fitted yet/)
    expect(() => new LinearRegression().getCoefficients()).toThrow(/not been fitted yet/)
  })
})

// ─── Golden: train / fitGD no cambian (base antes de T012) ──────────────────

const LX = [
  [0.5, -0.3, 0.8],
  [-0.2, 0.9, -0.4],
  [1.1, 0.2, -0.6],
  [-0.8, -0.5, 0.3],
  [0.3, 0.7, 0.9],
  [-1.0, 0.1, -0.2],
]
const LY = [1, 0, 1, 0, 1, 0]
const SX = [
  [1, 0.1],
  [0.9, -0.2],
  [-0.1, 1],
  [0.2, 0.8],
  [-1, -1],
  [-0.8, -1.2],
]
const SY = [0, 0, 1, 1, 2, 2]
const RX = [
  [1, 2],
  [2, 0.5],
  [3, 1],
  [0.5, 3],
  [2.5, 2.5],
]
const RY = [3.1, 2.4, 4.2, 3.9, 5.3]

describe('golden: train y fitGD con semilla fija no cambian (T012)', () => {
  it('LogisticRegression.train(lr 0.3, 5 epocas): perdidas, pesos y sesgo', () => {
    seedRandom(0x5eed)
    const m = new LogisticRegression()
    const losses = m.train(LX, LY, 0.3, 5)
    expectClose(losses, [
      0.5182314751544562, 0.395997398879023, 0.31848287086389326, 0.26621002113293896,
      0.22886286184738633,
    ])
    expectClose(m.weights, [2.0226946393458176, -0.20665595978656473, 1.1105031033628723])
    expectClose([m.bias], [-0.031236373120036486])
    // predict tras train usa los pesos entrenados (2 rasgos → 3 pesos).
    const z = m.weights.reduce((s, w, j) => s + w * LX[0][j], m.bias)
    expect(m.predict(LX[0])).toBeCloseTo(sigmoid(z), 12)
  })

  it('SoftmaxRegression.train(lr 0.5, 10 epocas): perdidas, pesos y sesgos', () => {
    seedRandom(0x5eed)
    const m = new SoftmaxRegression()
    const losses = m.train(SX, SY, 0.5, 10)
    expectClose(losses, [
      1.1005648851820797, 0.47077032290017545, 0.27562329946955605, 0.1893443401113588,
      0.14356726239482734, 0.1156589975723123, 0.09692731160824164, 0.08348934780288575,
      0.07337526219831796, 0.06548380887502557,
    ])
    expectClose(m.weights.flat(), [
      2.8899358959987835, -1.216140583342956, -0.20150656836344444, 1.9909475470385998,
      -1.607732142275444, -1.6904055056594884,
    ])
    expectClose(m.biases, [-0.0553005849875355, 0.32493207428749454, -0.269631489299959])
  })

  it('LinearRegression.fitGD(lr 0.05, 8 epocas): perdidas y pesos', () => {
    seedRandom(0x5eed)
    const m = new LinearRegression()
    const losses = m.fitGD(RX, RY, 0.05, 8)
    expectClose(losses, [
      15.262, 0.7821006999999989, 0.08071718885249982, 0.04512862902860319,
      0.04204729152238211, 0.040804398017829834, 0.039852089750804495, 0.03906551135287851,
    ])
    expectClose(m.weights, [0.8893997213486039, 0.9758577478703238, 0.45929062463399994])
  })
})
