// Gradient-oracle test for LogisticRegression and SoftmaxRegression (feature 002, T027).
//
// Convencion (constitucion II): train() hace w + lr*g con g = -dL/dw; un modelo correcto
// da coseno ~ +1 entre el paso real y -dL/dw por diferencias finitas. Con una muestra y
// una epoca, train() es exactamente UNA actualizacion.
//
// LogisticRegression (BCE: L = -[y log p + (1-y) log(1-p)], p = sigma(w.x + bias)):
//   - `weights` NO incluye el sesgo: el sesgo es el campo escalar `bias`. bindParams no
//     admite escalares, asi que getParams/setParams concatenan weights y bias a mano.
//   - train() solo reinicia pesos aleatorios si weights.length !== nFeatures. Con los
//     pesos iniciales ya fijados no consume Math.random.
//   - Gradiente: dL/dw = (p - y) x, dL/dbias = (p - y). El paso es w += lr (y - p) x.
//
// SoftmaxRegression (CE: L = -log softmax(scores)[y]):
//   - `weights[k][j]` y `biases[k]` son publicos.
//   - Desde T014 (Q7) train() CONTINUA desde los pesos existentes si su forma coincide con
//     los datos, asi que el paso es `train(X, y, lr, 1)` real, sin stub de Math.random, y el
//     oraculo mide desde cualquier punto (tambien con sesgos distintos de 0).
//   - Gradiente: dL/dW_k = (P_k - 1{k=y}) x, dL/db_k = (P_k - 1{k=y}).
//
// predict() y predictProba() validan contra la forma de `weights` (N7, feature 005), así
// que también funcionan con pesos asignados a mano (ver el bloque final de este archivo).
//
// Helpers usados sin modificar: gradientOracle (bindParams, expectGradientMatches) y
// expectLearns. La semilla global (tests/setup.ts) fija Math.random en cada test.
import { describe, it, expect } from 'vitest'
import { LogisticRegression, SoftmaxRegression } from '../../src/LogisticRegression'
import { bindParams, expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

// ─── Referencias independientes (no usan el codigo de src/) ─────────────────

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const dot = (w: number[], x: number[]) => w.reduce((s, wj, j) => s + wj * x[j], 0)

function bce(y: number, p: number): number {
  return -(y * Math.log(p) + (1 - y) * Math.log(1 - p))
}

function softmaxProbs(W: number[][], b: number[], x: number[]): number[] {
  const s = W.map((wk, k) => b[k] + dot(wk, x))
  const z = s.reduce((acc, v) => acc + Math.exp(v), 0)
  return s.map((v) => Math.exp(v) / z)
}

function softmaxCE(W: number[][], b: number[], x: number[], label: number): number {
  return -Math.log(softmaxProbs(W, b, x)[label])
}

// ─── LogisticRegression ──────────────────────────────────────────────────────

const LR_X = [0.5, -0.3, 0.8]
const LR_Y = 1
const LR_W0 = [0.2, -0.4, 0.1]
const LR_B0 = 0.05

function setupLogistic() {
  const m = new LogisticRegression()
  m.weights = [...LR_W0]
  m.bias = LR_B0
  const loss = () => bce(LR_Y, sigmoid(dot(m.weights, LR_X) + m.bias))
  return {
    m,
    loss,
    step: (lr: number) => {
      m.train([LR_X], [LR_Y], lr, 1)
    },
    getParams: () => [...bindParams(m.weights).getParams(), m.bias],
    setParams: (p: number[]) => {
      bindParams(m.weights).setParams(p.slice(0, 3))
      m.bias = p[3]
    },
  }
}

describe('LogisticRegression oraculo', () => {
  it('humo: predice una probabilidad finita y train actualiza pesos y sesgo', () => {
    const m = new LogisticRegression()
    m.train([LR_X], [LR_Y], 0.1, 1) // inicializa pesos (semilla fija)
    const p = m.predict(LR_X)
    expect(Number.isFinite(p)).toBe(true)
    expect(p).toBeGreaterThan(0)
    expect(p).toBeLessThan(1)
    const before = [...m.weights]
    const beforeBias = m.bias
    m.train([LR_X], [LR_Y], 0.1, 1)
    expect(m.weights.every(Number.isFinite)).toBe(true)
    expect(Number.isFinite(m.bias)).toBe(true)
    expect(m.weights).not.toEqual(before)
    expect(m.bias).not.toBe(beforeBias)
  })

  it('comportamiento: predict tras un paso desde w=0, b=0 coincide con sigma(0.23) a mano', () => {
    // Tras train([[2,-1]], [1], 0.2, 1): w = [0.2, -0.1], bias = 0.1 (ver test siguiente).
    // predict([0.5, -0.3]) = sigma(0.2*0.5 + (-0.1)(-0.3) + 0.1) = sigma(0.23).
    const m = new LogisticRegression()
    m.weights = [0, 0]
    m.bias = 0
    m.train([[2, -1]], [1], 0.2, 1)
    expect(m.predict([0.5, -0.3])).toBeCloseTo(sigmoid(0.23), 12)
  })

  it('comportamiento: un paso desde w=0, b=0 da el valor esperado a mano', () => {
    // x=[2,-1], y=1, lr=0.2: z=0, p=0.5, (y-p)=0.5.
    // w0 = 0 + 0.2*0.5*2 = 0.2 ; w1 = 0 + 0.2*0.5*(-1) = -0.1 ; bias = 0 + 0.2*0.5 = 0.1
    const m = new LogisticRegression()
    m.weights = [0, 0]
    m.bias = 0
    m.train([[2, -1]], [1], 0.2, 1)
    expect(m.weights[0]).toBeCloseTo(0.2, 12)
    expect(m.weights[1]).toBeCloseTo(-0.1, 12)
    expect(m.bias).toBeCloseTo(0.1, 12)
  })

  it('oraculo: el paso sigue -dL/dw (pesos y sesgo)', () => {
    expectGradientMatches(setupLogistic())
  })

  it('expectLearns: la perdida BCE sobre un conjunto separable baja a menos de la mitad', () => {
    const X = [
      [1, 0],
      [0, 1],
      [-1, 0],
      [0, -1],
    ]
    const Y = [1, 1, 0, 0]
    const m = new LogisticRegression()
    m.weights = [0, 0]
    m.bias = 0
    const meanBce = () => {
      let s = 0
      for (let i = 0; i < X.length; i++) s += bce(Y[i], sigmoid(dot(m.weights, X[i]) + m.bias))
      return s / X.length
    }
    expectLearns({ loss: meanBce, step: () => void m.train(X, Y, 0.5, 1), steps: 200 })
  })
})

// ─── SoftmaxRegression ───────────────────────────────────────────────────────

const SM_X = [0.5, -0.3, 0.8]
const SM_LABEL = 2
const SM_W0 = [
  [0.2, -0.4, 0.1],
  [-0.3, 0.25, 0.6],
  [0.5, 0.1, -0.5],
]
const SM_B0 = [0.1, -0.2, 0.05]

function setupSoftmax() {
  const m = new SoftmaxRegression()
  m.weights = SM_W0.map((r) => [...r])
  m.biases = [...SM_B0]
  const loss = () => softmaxCE(m.weights, m.biases, SM_X, SM_LABEL)
  return {
    m,
    loss,
    step: (lr: number) => {
      m.train([SM_X], [SM_LABEL], lr, 1)
    },
    getParams: () => bindParams(m.weights, m.biases).getParams(),
    setParams: (p: number[]) => bindParams(m.weights, m.biases).setParams(p),
  }
}

describe('SoftmaxRegression oraculo', () => {
  it('humo: predice probabilidades que suman 1 y train produce pesos finitos', () => {
    const m = new SoftmaxRegression()
    m.train([SM_X], [SM_LABEL], 0.1, 1) // inicializa pesos (semilla fija)
    const probs = m.predictProba(SM_X)
    expect(probs).toHaveLength(3)
    expect(probs.every(Number.isFinite)).toBe(true)
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(m.weights.flat().every(Number.isFinite)).toBe(true)
    expect(m.biases.every(Number.isFinite)).toBe(true)
  })

  it('comportamiento: predictProba tras un paso desde pesos cero coincide con softmax a mano', () => {
    // Tras el paso exacto de la prueba siguiente: W = [[-0.1,-0.2],[-0.1,-0.2],[0.2,0.4]],
    // b = [-0.1,-0.1,0.2]. Para x=[0.5,-0.3]: s = [-0.09, -0.09, 0.18] (a mano), y softmax de s.
    const m = new SoftmaxRegression()
    m.weights = [
      [0, 0],
      [0, 0],
      [0, 0],
    ]
    m.biases = [0, 0, 0]
    m.train([[1, 2]], [2], 0.3, 1)
    const W = [
      [-0.1, -0.2],
      [-0.1, -0.2],
      [0.2, 0.4],
    ]
    const b = [-0.1, -0.1, 0.2]
    const x = [0.5, -0.3]
    const expected = softmaxProbs(W, b, x)
    const got = m.predictProba(x)
    expect(got).toHaveLength(3)
    got.forEach((p, k) => expect(p).toBeCloseTo(expected[k], 12))
  })

  it('comportamiento: un paso desde pesos cero da el valor esperado a mano', () => {
    // 3 clases, 2 rasgos, x=[1,2], y=2, lr=0.3. Con pesos 0: P_k = 1/3.
    // delta_k = P_k - 1{k=2} = [1/3, 1/3, -2/3].  W_k -= lr * delta_k * x.
    // W0 = W1 = [-0.1, -0.2] ; W2 = [0.2, 0.4] ; b = [-0.1, -0.1, 0.2].
    const m = new SoftmaxRegression()
    m.weights = [
      [0, 0],
      [0, 0],
      [0, 0],
    ]
    m.biases = [0, 0, 0]
    m.train([[1, 2]], [2], 0.3, 1)
    expect(m.weights[0][0]).toBeCloseTo(-0.1, 12)
    expect(m.weights[0][1]).toBeCloseTo(-0.2, 12)
    expect(m.weights[1][0]).toBeCloseTo(-0.1, 12)
    expect(m.weights[1][1]).toBeCloseTo(-0.2, 12)
    expect(m.weights[2][0]).toBeCloseTo(0.2, 12)
    expect(m.weights[2][1]).toBeCloseTo(0.4, 12)
    expect(m.biases[0]).toBeCloseTo(-0.1, 12)
    expect(m.biases[1]).toBeCloseTo(-0.1, 12)
    expect(m.biases[2]).toBeCloseTo(0.2, 12)
  })

  it('oraculo: el paso sigue -dL/dw (pesos y sesgos)', () => {
    expectGradientMatches(setupSoftmax())
  })

  it('expectLearns: la perdida CE sobre un conjunto separable baja a menos de la mitad', () => {
    const X = [
      [1, 0],
      [0.9, 0.1],
      [0, 1],
      [0.1, 0.9],
      [-1, -1],
      [-0.9, -1.1],
    ]
    const Y = [0, 0, 1, 1, 2, 2]
    const m = new SoftmaxRegression()
    m.weights = [
      [0, 0],
      [0, 0],
      [0, 0],
    ]
    m.biases = [0, 0, 0]
    const meanCE = () => {
      let s = 0
      for (let i = 0; i < X.length; i++) s += softmaxCE(m.weights, m.biases, X[i], Y[i])
      return s / X.length
    }
    // train() continua desde los pesos (desde T014): 200 pasos de 1 epoca acumulan aprendizaje.
    expectLearns({ loss: meanCE, step: () => void m.train(X, Y, 0.5, 1), steps: 200 })
  })
})

// N7 (feature 005): predict/predictProba validan contra `weights`, así que un modelo con
// pesos asignados a mano (sin train) predice.
describe('API: pesos asignados sin train() (N7)', () => {
  it('LogisticRegression.predict funciona con weights/bias asignados a mano', () => {
    const m = new LogisticRegression()
    m.weights = [...LR_W0]
    m.bias = LR_B0
    expect(Number.isFinite(m.predict(LR_X))).toBe(true)
  })

  it('SoftmaxRegression.predictProba funciona con weights/biases asignados a mano', () => {
    const m = new SoftmaxRegression()
    m.weights = SM_W0.map((r) => [...r])
    m.biases = [...SM_B0]
    const p = m.predictProba(SM_X)
    expect(p.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10)
  })
})
