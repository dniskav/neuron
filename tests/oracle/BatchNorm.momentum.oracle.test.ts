// Gradient-oracle test for BatchNorm with momentum < 1 (feature 002, T041).
//
// AUDIT N2: con momentum m < 1 (0.1 por defecto) forward() actualiza
// runningMean / runningVar con la propia x y NORMALIZA con esas estadisticas ya
// actualizadas. Como x - mean_nuevo = m * (x - mean_viejo), la salida depende de
// x con pendiente ~ m * gamma / std, mientras que backward() devuelve
// dOut * gamma / std (estadisticas como constantes, sin el factor m).
//
// Medido (diferencias finitas sobre forward, partiendo siempre de las mismas
// estadisticas): con m = 0.1 la DIRECCION es casi correcta (coseno 0.9995-1.0000
// en las entradas de este test; 0.91 en entradas extremas) pero la MAGNITUD de
// backward es ~1/m = ~10x la real. El coseno por si solo no la ve, asi que
// el test fija las dos cosas: coseno > 0.9 y razon de normas en (8, 13).
//
// Es una aproximacion del gradiente (stop-gradient sobre las estadisticas), no
// un error de signo: excepcion justificada (Constitucion II). Con momentum = 1
// las estadisticas no cambian y backward es exacto (control abajo y en
// BatchNorm.oracle.test.ts). La correccion exacta (propagar por media/varianza o
// normalizar con las estadisticas previas) va a la feature 004; no se toca src/.
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

// backward() frente a -dL/dx por diferencias finitas, con L = 0.5*sum (t-y)^2.
function compareBackwardWithFD(c: Case, momentum: number): { cos: number; ratio: number } {
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
  return { cos: cosine(dX, negGradFD), ratio: norm(dX) / norm(negGradFD) }
}

const CASES: [string, Case][] = [
  ['constantes de T012', FIXED],
  ...[1, 2, 3, 4, 5].map((s): [string, Case] => [`semilla ${s}`, makeCase(s)]),
]

describe('BatchNorm momentum < 1 (AUDIT N2)', () => {
  it.each(CASES)('AUDIT N2: momentum 0.1, %s: direccion correcta, magnitud ~10x', (_name, c) => {
    const { cos, ratio } = compareBackwardWithFD(c, 0.1)
    expect(cos).toBeGreaterThan(0.9)
    // backward no es exacto: la magnitud se desvia ~1/m = ~10x.
    expect(ratio).toBeGreaterThan(8)
    expect(ratio).toBeLessThan(13)
  })

  it.each(CASES)('control: momentum 1, %s: coseno ~ 1 y razon ~ 1', (_name, c) => {
    const { cos, ratio } = compareBackwardWithFD(c, 1)
    expect(cos).toBeGreaterThan(0.99999)
    expect(Math.abs(ratio - 1)).toBeLessThan(1e-4)
  })
})
