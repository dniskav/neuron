// Oráculo de dx de Dropout (feature 006, T016, US4/FR-006).
//
// Convención (igual que el oráculo de Flatten): backward(dOut) con dOut = dL/dy
// devuelve dL/dx, sin negar. Dropout no tiene parámetros.
//
// Dropout es estocástico: forward(x, true) sortea Math.random por elemento.
// Con seedRandom(2024) la máscara queda fijada. Las pérdidas usan reuseMask(x),
// que reaplica la MISMA máscara sin sortear de nuevo, así que L(x) es una función
// determinista de x y las diferencias finitas miden la derivada de esa función.
//
// Notación: y_i = x_i · m_i, con m_i ∈ {0, 2} (rate 0.5 → escala 1/(1−0.5) = 2).
//   L1(x) = Σ_i c_i · y_i        →  dL1/dx_i = c_i · m_i
//   L2(x) = Σ_i c_i · y_i²       →  dL2/dx_i = 2 · c_i · m_i² · x_i
// Para L2, dOut_i = 2 · c_i · y_i (con y = reuseMask(x0)); backward da
// dOut_i · m_i = 2 · c_i · m_i² · x_i, que coincide con la derivada.
//
// Como la semilla no permite deducir a mano qué unidades se apagan, la máscara
// se lee de backward(1,…,1) = m y se comprueba que sus valores son {0, 2} con
// ambos presentes (el caso discrimina). El test con stub de Math.random fija
// una máscara calculada a mano.
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { Dropout } from '../../src/Dropout'
import { restoreRandom, seedRandom } from '../helpers/seed'

beforeEach(() => seedRandom(2024))
afterEach(() => restoreRandom())

const TOL = 1e-6
const EPS = 1e-5
const RATE = 0.5
const SCALE = 1 / (1 - RATE) // 2

// Entrada de 10 valores no enteros.
const X0 = [0.5, -1.25, 2.0, 0.75, -0.4, 1.1, 0.9, -0.3, -1.6, 0.2]
// Pesos fijos c_i de la pérdida.
const C = [0.5, -1, 2, 0.25, -0.75, 1.5, 3, -2, 0.1, -0.6]

/** Máscara guardada tras forward(x, true): reuseMask(1,…,1) = 1 ⊙ m = m. */
function maskOf(d: Dropout, n: number): number[] {
  return d.reuseMask(new Array(n).fill(1))
}

/** L1 = Σ c_i · y_i con y = reuseMask(x) (misma máscara). */
function L1(d: Dropout, x: number[]): number {
  return d.reuseMask(x).reduce((s, y, i) => s + C[i] * y, 0)
}

/** L2 = Σ c_i · y_i² con y = reuseMask(x) (no lineal). */
function L2(d: Dropout, x: number[]): number {
  return d.reuseMask(x).reduce((s, y, i) => s + C[i] * y * y, 0)
}

/** Derivada numérica dL/dx_i por diferencias finitas centrales. */
function finiteDiff(L: (x: number[]) => number, x: number[]): number[] {
  return x.map((_, i) => {
    const xp = [...x]
    const xm = [...x]
    xp[i] += EPS
    xm[i] -= EPS
    return (L(xp) - L(xm)) / (2 * EPS)
  })
}

function maxAbsDiff(a: number[], b: number[]): number {
  expect(a).toHaveLength(b.length)
  return Math.max(...a.map((v, i) => Math.abs(v - b[i])))
}

describe('Dropout dx oracle', () => {
  it('L1 lineal: backward(c) coincide con diferencias finitas de reuseMask por componente', () => {
    const d = new Dropout(RATE)
    d.forward(X0, true)
    // Máscara fijada por la semilla: solo valores {0, 2}, con ceros y no ceros.
    const m = maskOf(d, X0.length)
    expect(m.every((v) => v === 0 || v === SCALE)).toBe(true)
    expect(m.some((v) => v === 0)).toBe(true)
    expect(m.some((v) => v === SCALE)).toBe(true)

    const dx = d.backward(C)
    const fd = finiteDiff((x) => L1(d, x), X0)
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)
  })

  it('L1 lineal: el gradiente esperado a mano es c_i · m_i (0 donde se apagó la unidad)', () => {
    // dL1/dx_i = c_i · m_i. Con m_i = 0 en las unidades apagadas y m_i = 2 en las activas.
    // Ejemplo de la fórmula: si m_0 = 2, entonces dL1/dx_0 = 0.5 · 2 = 1.
    const d = new Dropout(RATE)
    d.forward(X0, true)
    const m = maskOf(d, X0.length)
    const expected = C.map((c, i) => c * m[i])
    const dx = d.backward(C)
    expect(maxAbsDiff(dx, expected)).toBeLessThan(TOL)
    expect(maxAbsDiff(finiteDiff((x) => L1(d, x), X0), expected)).toBeLessThan(TOL)
  })

  it('L2 no lineal: backward(2·c·y) coincide con diferencias finitas por componente', () => {
    const d = new Dropout(RATE)
    d.forward(X0, true)
    const y = d.reuseMask(X0)
    // dOut_i = 2 · c_i · y_i con y = m ⊙ x0.
    const dOut = C.map((c, i) => 2 * c * y[i])
    const dx = d.backward(dOut)
    const fd = finiteDiff((x) => L2(d, x), X0)
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)

    // Valor a mano: dL2/dx_i = 2 · c_i · m_i² · x_i (m_i² ∈ {0, 4}).
    const m = maskOf(d, X0.length)
    const expected = C.map((c, i) => 2 * c * m[i] * m[i] * X0[i])
    expect(maxAbsDiff(dx, expected)).toBeLessThan(TOL)
  })

  it('rate 0: no hay máscara y backward devuelve dOut tal cual', () => {
    const d = new Dropout(0)
    d.forward(X0, true)
    expect(d.hasMask()).toBe(false)
    // L = Σ c_i · forward(x, true)_i es la identidad: dL/dx_i = c_i.
    const dx = d.backward(C)
    expect(dx).toEqual(C)
    const fd = finiteDiff(
      (x) => d.forward(x, true).reduce((s, y, i) => s + C[i] * y, 0),
      X0,
    )
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)
  })

  it('inferencia: forward(x, false) borra la máscara y backward devuelve dOut tal cual', () => {
    const d = new Dropout(RATE)
    d.forward(X0, true) // deja una máscara guardada
    d.forward(X0, false) // la inferencia debe borrarla
    expect(d.hasMask()).toBe(false)
    // L = Σ c_i · x_i en inferencia: dL/dx_i = c_i.
    const dx = d.backward(C)
    expect(dx).toEqual(C)
    const fd = finiteDiff(
      (x) => d.forward(x, false).reduce((s, y, i) => s + C[i] * y, 0),
      X0,
    )
    expect(maxAbsDiff(dx, fd)).toBeLessThan(TOL)
  })

  it('máscara a mano con stub de Math.random: [0.0, 0.9] con rate 0.5 → [0, 2]; backward([1,1]) = [0, 2]', () => {
    // u_i > 0.5 ? 2 : 0. u_0 = 0.0 → 0; u_1 = 0.9 → 2.
    // forward([1,1]) = [1·0, 1·2] = [0, 2]. backward([1,1]) = [1·0, 1·2] = [0, 2].
    const draws = [0.0, 0.9]
    let calls = 0
    Math.random = () => draws[calls++]

    const d = new Dropout(RATE)
    expect(d.forward([1, 1], true)).toEqual([0, 2])
    expect(calls).toBe(2)
    expect(d.backward([1, 1])).toEqual([0, 2])
    // reuseMask no sortea: con [3, 5] → [3·0, 5·2] = [0, 10], y no consume Math.random.
    expect(d.reuseMask([3, 5])).toEqual([0, 10])
    expect(calls).toBe(2)
  })
})
