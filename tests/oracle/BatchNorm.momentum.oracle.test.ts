// Gradient-oracle test for BatchNorm.backward with any momentum (feature 004, T002).
//
// forward() actualiza runningMean / runningVar con la propia x y NORMALIZA con
// esas estadisticas ya actualizadas. backward() devuelve la derivada EXACTA de
// esa salida respecto a x (incluida la dependencia de la varianza):
//   dy/dx = gamma * (m/s - (1-m) * m^3 * u^2 / s^3),  u = x - media_vieja.
// Se compara por componente con -dL/dx por diferencias finitas sobre forward,
// partiendo siempre de las mismas estadisticas, para m en {0, 0.1, 0.5, 0.9, 1}.
// Sustituye al test de excepcion de 002 (excepcion N2: coseno > 0.9, razon 8-13).
import { describe, it, expect } from 'vitest'
import { BatchNorm } from '../../src/BatchNorm'
import { mulberry32 } from '../helpers/seed'

const DIM = 3

function norm(a: number[]): number {
  return Math.sqrt(a.reduce((s, v) => s + v * v, 0))
}

function cosine(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + v * b[i], 0) / (norm(a) * norm(b))
}

interface Case {
  input: number[]
  target: number[]
  gamma: number[]
  beta: number[]
  rm: number[]
  rv: number[]
}

// Mismas constantes que BatchNorm.oracle.test.ts.
const FIXED: Case = {
  input: [0.5, -0.3, 0.8],
  target: [0.7, 0.2, -0.4],
  gamma: [0.9, 1.2, 0.6],
  beta: [0.1, -0.2, 0.05],
  rm: [0.2, -0.1, 0.3],
  rv: [0.5, 2, 1.2],
}

function makeCase(seed: number): Case {
  const r = mulberry32(seed)
  const vec = (lo: number, hi: number) => Array.from({ length: DIM }, () => lo + (hi - lo) * r())
  return {
    input: vec(-1, 1),
    target: vec(-1, 1),
    gamma: vec(0.5, 1.5),
    beta: vec(-0.3, 0.3),
    rm: vec(-0.5, 0.5),
    rv: vec(0.5, 2),
  }
}

const MOMENTA = [0, 0.1, 0.5, 0.9, 1]

// backward() frente a -dL/dx por diferencias finitas, con L = 0.5*sum (t-y)^2.
function compare(c: Case, momentum: number) {
  const bn = new BatchNorm(DIM, momentum)
  const reset = () => {
    bn.runningMean = [...c.rm]
    bn.runningVar = [...c.rv]
  }
  bn.setWeights([...c.gamma, ...c.beta])
  const lossAt = (x: number[]) => {
    reset()
    const y = bn.forward(x)
    return y.reduce((s, v, i) => s + 0.5 * (c.target[i] - v) ** 2, 0)
  }

  reset()
  const y = bn.forward(c.input)
  const dX = bn.backward(y.map((v, i) => c.target[i] - v))

  const eps = 1e-5
  const negGradFD = c.input.map((_, i) => {
    const plus = [...c.input]
    plus[i] += eps
    const minus = [...c.input]
    minus[i] -= eps
    return -(lossAt(plus) - lossAt(minus)) / (2 * eps)
  })
  expect(dX.every(Number.isFinite)).toBe(true)
  expect(negGradFD.every(Number.isFinite)).toBe(true)
  return { dX, negGradFD, cos: cosine(dX, negGradFD), ratio: norm(dX) / norm(negGradFD) }
}

function expectExact(c: Case, momentum: number) {
  const { dX, negGradFD, cos, ratio } = compare(c, momentum)
  dX.forEach((v, i) => expect(Math.abs(v - negGradFD[i])).toBeLessThan(1e-6))
  // m = 0: la salida no depende de x (todo es 0), no hay direccion que comparar.
  if (momentum > 0) {
    expect(cos).toBeGreaterThan(0.999)
    expect(Math.abs(ratio - 1)).toBeLessThan(1e-4)
  }
}

// Extremos: entradas +-3 y varianzas muy distintas; |u|/s >= 2 en varios canales.
const EXTREME: Case = {
  input: [3, -3, 2.5],
  target: [0.7, 0.2, -0.4],
  gamma: [0.9, 1.2, 0.6],
  beta: [0.1, -0.2, 0.05],
  rm: [-0.5, 0.4, 0],
  rv: [0.05, 0.3, 5],
}

const CASES: [string, Case][] = [
  ['constantes de T012', FIXED],
  ...[1, 2, 3, 4, 5].map((s): [string, Case] => [`semilla ${s}`, makeCase(s)]),
  ['extremo +-3, varianzas distintas', EXTREME],
]

describe('BatchNorm.backward exacto (N2)', () => {
  describe.each(MOMENTA)('momentum %s', momentum => {
    it.each(CASES)('%s: coincide con -dL/dx por diferencias finitas', (_n, c) => {
      expectExact(c, momentum)
    })
  })

  it('caso |u|/s >= 2 donde el termino de la varianza supera el 5 % del total', () => {
    const m = 0.5
    const c = EXTREME
    const bn = new BatchNorm(DIM, m)
    bn.runningMean = [...c.rm]
    bn.runningVar = [...c.rv]
    // Canal 0: u = 3.5, var_nueva = m*0.05 + (1-m)*m^2*u^2, s^2 ~ 1.55
    const u = c.input[0] - c.rm[0]
    const varNew = m * c.rv[0] + (1 - m) * m * m * u * u
    const s = Math.sqrt(varNew + 1e-5)
    expect(Math.abs(u) / s).toBeGreaterThanOrEqual(2)
    const first = m / s
    const second = ((1 - m) * m ** 3 * u * u) / s ** 3
    expect(second / first).toBeGreaterThan(0.05)
    expectExact(c, m)
  })
})
