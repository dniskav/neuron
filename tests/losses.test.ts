import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { seedRandom, restoreRandom } from './helpers/seed'
import {
  mse,
  crossEntropy,
  mseDelta,
  crossEntropyDelta,
  crossEntropyDeltaRaw,
} from '../src/losses'

// losses es determinista; la semilla se fija igualmente (Constitución IV).
beforeEach(() => seedRandom(0x5eed))
afterEach(() => restoreRandom())

// Todas las cifras esperadas están deducidas a mano en el comentario de cada expect.

// ─── mse ──────────────────────────────────────────────────────────────────────

describe('mse', () => {
  it('media de los cuadrados de los errores', () => {
    // Errores (actual − predicho): 1−1 = 0 y 4−2 = 2. Cuadrados 0 y 4.
    // Suma 4, dividida entre 2 salidas: 2
    expect(mse([1, 2], [1, 4])).toBe(2)
  })

  it('divide entre el número de salidas', () => {
    // Cuatro salidas con error 2: cada cuadrado es 4, suma 16, / 4 = 4.
    // Si no se dividiera daría 16.
    expect(mse([0, 0, 0, 0], [2, 2, 2, 2])).toBe(4)
  })

  it('una sola salida: (a − p)²', () => {
    // (1 − 3)² = 4
    expect(mse([3], [1])).toBe(4)
    // (0.2 − 0.5)² = 0.09
    expect(mse([0.5], [0.2])).toBeCloseTo(0.09, 12)
  })

  it('el signo del error no importa', () => {
    // Error −2 y +2 dan el mismo cuadrado: (1−3)² = 4 y (3−1)² = 4
    expect(mse([1], [3])).toBe(mse([3], [1]))
  })

  it('predicción perfecta → 0', () => {
    // Todos los errores son 0
    expect(mse([0.2, 0.7], [0.2, 0.7])).toBe(0)
  })
})

// ─── crossEntropy ─────────────────────────────────────────────────────────────

describe('crossEntropy', () => {
  it('−ln(0.9) para [0.9, 0.1] frente a [1, 0]', () => {
    // Salida 0 (actual 1): 1·ln 0.9 = ln 0.9
    // Salida 1 (actual 0): (1−0)·ln(1 − 0.1) = ln 0.9
    // Suma 2·ln 0.9; negada y dividida entre 2: −ln 0.9 ≈ 0.1053605157
    expect(crossEntropy([0.9, 0.1], [1, 0])).toBeCloseTo(0.1053605156578263, 12)
  })

  it('promedia sobre las salidas', () => {
    // Salida 0 (actual 1): ln 0.9. Salida 1 (actual 0): ln(1 − 0.9) = ln 0.1.
    // Suma ln 0.09 ≈ −2.4079456087; negada y entre 2: ≈ 1.2039728043
    expect(crossEntropy([0.9, 0.9], [1, 0])).toBeCloseTo(1.2039728043259361, 12)
  })

  it('predicción perfecta → ~0', () => {
    // ln(1 − 1e−15) ≈ −1e−15, así que el resultado es ≈ 1e−15
    expect(crossEntropy([1, 0], [1, 0])).toBeCloseTo(0, 10)
  })

  it('recorte de eps: p = 0 con actual 1 no devuelve Infinity', () => {
    // p se recorta a eps = 1e−15: −ln(1e−15) = 15·ln 10 ≈ 34.5387763949
    const value = crossEntropy([0], [1])
    expect(Number.isFinite(value)).toBe(true)
    expect(value).toBeCloseTo(15 * Math.LN10, 9)
  })

  it('recorte de eps: p = 1 con actual 0 no devuelve Infinity', () => {
    // 1 − p se recorta a eps; en coma flotante 1 − (1 − 1e−15) ≈ 9.99e−16,
    // así que −ln ≈ 34.5396. Tolerancia de 0.05 para cubrir ese redondeo.
    const value = crossEntropy([1], [0])
    expect(Number.isFinite(value)).toBe(true)
    expect(value).toBeCloseTo(34.54, 1)
  })

  it('acierto exacto con actual 0 → ~0', () => {
    // p = 0 con actual 0: ln(1 − 1e−15) ≈ −1e−15, negado ≈ 1e−15
    expect(crossEntropy([0], [0])).toBeCloseTo(0, 10)
  })
})

// ─── mseDelta ─────────────────────────────────────────────────────────────────

describe('mseDelta', () => {
  it('es a − p (negativo del gradiente de MSE)', () => {
    // 1 − 0.3 = 0.7
    expect(mseDelta(0.3, 1)).toBeCloseTo(0.7, 12)
    // 0.2 − 0.8 = −0.6
    expect(mseDelta(0.8, 0.2)).toBeCloseTo(-0.6, 12)
  })

  it('sin error → 0', () => {
    // 0.5 − 0.5 = 0
    expect(mseDelta(0.5, 0.5)).toBe(0)
  })

  it('coherencia: −∂/∂p de (a − p)²/2 por diferencias finitas', () => {
    // L(p) = (a − p)²/2 tiene ∂L/∂p = −(a − p). Entonces −∂L/∂p = a − p.
    // Con a = 1, p = 0.3: −∂L/∂p = 0.7. Con a = 0.2, p = 0.8: −∂L/∂p = −0.6.
    const h = 1e-6
    const L = (a: number) => (p: number) => (a - p) ** 2 / 2
    const negDeriv = (f: (p: number) => number, p: number) =>
      -((f(p + h) - f(p - h)) / (2 * h))

    expect(negDeriv(L(1), 0.3)).toBeCloseTo(0.7, 6)
    expect(negDeriv(L(0.2), 0.8)).toBeCloseTo(-0.6, 6)
    expect(negDeriv(L(1), 0.3)).toBeCloseTo(mseDelta(0.3, 1), 6)
    expect(negDeriv(L(0.2), 0.8)).toBeCloseTo(mseDelta(0.8, 0.2), 6)
  })
})

// ─── crossEntropyDelta ────────────────────────────────────────────────────────

describe('crossEntropyDelta', () => {
  it('es actual − predicho (sigmoide + entropía cruzada)', () => {
    // 1 − 0.8 = 0.2
    expect(crossEntropyDelta(0.8, 1)).toBeCloseTo(0.2, 12)
    // 0 − 0.3 = −0.3
    expect(crossEntropyDelta(0.3, 0)).toBeCloseTo(-0.3, 12)
  })

  it('coherencia: −∂/∂z de CE(σ(z), a) por diferencias finitas', () => {
    // Con salida sigmoide p = σ(z) la derivada de CE respecto de z es p − a
    // (la σ' se cancela). Entonces −∂CE/∂z = a − p = crossEntropyDelta(p, a).
    const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
    const ce = (a: number) => (z: number) => crossEntropy([sigmoid(z)], [a])
    const h = 1e-6
    const negDeriv = (f: (z: number) => number, z: number) =>
      -((f(z + h) - f(z - h)) / (2 * h))

    const z = 0.4
    expect(negDeriv(ce(1), z)).toBeCloseTo(crossEntropyDelta(sigmoid(z), 1), 6)
    expect(negDeriv(ce(0), z)).toBeCloseTo(crossEntropyDelta(sigmoid(z), 0), 6)
  })
})

// ─── crossEntropyDeltaRaw ─────────────────────────────────────────────────────

describe('crossEntropyDeltaRaw', () => {
  it('actual = 1: 1 / p', () => {
    // 1 / 0.8 = 1.25
    expect(crossEntropyDeltaRaw(0.8, 1)).toBeCloseTo(1.25, 12)
    // 1 / 0.25 = 4
    expect(crossEntropyDeltaRaw(0.25, 1)).toBeCloseTo(4, 12)
  })

  it('actual = 0: −1 / (1 − p)', () => {
    // −1 / (1 − 0.5) = −2
    expect(crossEntropyDeltaRaw(0.5, 0)).toBeCloseTo(-2, 12)
    // −1 / (1 − 0.6) = −2.5
    expect(crossEntropyDeltaRaw(0.6, 0)).toBeCloseTo(-2.5, 12)
  })

  it('recorte de eps en los extremos: finito, no Infinity', () => {
    // p = 0 se recorta a 1e−15: 1 / 1e−15 = 1e15
    const hi = crossEntropyDeltaRaw(0, 1)
    expect(Number.isFinite(hi)).toBe(true)
    expect(hi).toBeCloseTo(1e15, -14)

    // p = 1 se recorta a 1 − 1e−15; 1 − p ≈ 1e−15, así que −1 / (1 − p) ≈ −1e15
    const lo = crossEntropyDeltaRaw(1, 0)
    expect(Number.isFinite(lo)).toBe(true)
    expect(lo).toBeCloseTo(-1e15, -14)
  })

  it('coherencia: −∂/∂p de CE(p, a) por diferencias finitas', () => {
    // CE(p) = −ln p con a = 1: ∂CE/∂p = −1/p, luego −∂CE/∂p = 1/p.
    // CE(p) = −ln(1 − p) con a = 0: ∂CE/∂p = 1/(1 − p), luego −∂CE/∂p = −1/(1 − p).
    const h = 1e-7
    const negDeriv = (f: (p: number) => number, p: number) =>
      -((f(p + h) - f(p - h)) / (2 * h))

    expect(negDeriv((p) => crossEntropy([p], [1]), 0.8)).toBeCloseTo(1.25, 5)
    expect(negDeriv((p) => crossEntropy([p], [0]), 0.5)).toBeCloseTo(-2, 5)
  })
})
