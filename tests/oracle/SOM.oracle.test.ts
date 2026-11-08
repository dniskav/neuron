// Test de SOM (feature 002, T030).
//
// EXCEPCION A LA CONSTITUCION II (justificada): el SOM no tiene oraculo de gradiente.
// Su regla w <- w + lr·h·(x − w) con h = exp(−d²/2σ²) y BMU = argmin ‖x − w‖ no es el
// descenso por gradiente de una perdida diferenciable (la eleccion de la BMU es
// discreta), asi que no existe dL/dw que comparar con diferencias finitas. Se sustituye por:
//   1. humo: forma de los pesos, BMU valida, error de cuantizacion finito;
//   2. regla exacta: BMU y vecindad con valores calculados a mano para un paso;
//   3. expectLearns: el error de cuantizacion medio sobre datos fijos baja.
// La semilla de tests/setup.ts fija Math.random (init de pesos y orden de presentacion).
import { describe, it, expect } from 'vitest'
import { SOM } from '../../src/SOM'
import { expectLearns } from '../helpers/learns'

// Cuatro cumulos fijos en 2D, lejos del rango [0,1) de inicializacion.
const DATA: number[][] = [
  [4.9, 5.1],
  [5.1, 4.9],
  [8.9, 1.1],
  [9.1, 0.9],
  [1.1, 8.9],
  [0.9, 9.1],
  [8.9, 8.9],
  [9.1, 9.1],
]

describe('SOM (sin oraculo de gradiente, ver cabecera)', () => {
  it('humo: pesos [rows][cols][inputSize], BMU en rango y QE finito', () => {
    const som = new SOM(3, 4, 2)
    expect(som.weights).toHaveLength(3)
    expect(som.weights[0]).toHaveLength(4)
    expect(som.weights[0][0]).toHaveLength(2)
    const [r, c] = som.getBMU([0.5, 0.5])
    expect(r).toBeGreaterThanOrEqual(0)
    expect(r).toBeLessThan(3)
    expect(c).toBeGreaterThanOrEqual(0)
    expect(c).toBeLessThan(4)
    expect(som.predict([0.5, 0.5])).toEqual([r, c])
    som.train(DATA, 2)
    expect(Number.isFinite(som.quantizationError(DATA))).toBe(true)
    for (const row of som.weights)
      for (const w of row) for (const v of w) expect(Number.isFinite(v)).toBe(true)
  })

  it('regla exacta: la BMU y sus vecinas se acercan al dato con h = exp(−d²/2σ²)', () => {
    // Rejilla 1×3, entrada 1-D, lr 0.5, σ 1. Con una muestra y 1 epoca, t = 0 ⇒ lr y σ iniciales.
    // Pesos [0], [0.4], [1]; x = 0.5. Distancias 0.5, 0.1, 0.5 ⇒ BMU = (0, 1).
    // BMU (d=0, h=1):      0.4 + 0.5·1·(0.5 − 0.4)           = 0.45
    // Vecina (0,0) (d²=1): h = e^(−1/2) = 0.6065306597126334
    //                      0 + 0.5·h·(0.5 − 0)               = 0.15163266492815836
    // Vecina (0,2) (d²=1): 1 + 0.5·h·(0.5 − 1)               = 0.8483673350718416
    const som = new SOM(1, 3, 1, { initialLr: 0.5, finalLr: 0.01, initialSigma: 1, finalSigma: 1 })
    som.weights = [[[0], [0.4], [1]]]
    expect(som.getBMU([0.5])).toEqual([0, 1])
    som.train([[0.5]], 1)
    expect(som.weights[0][1][0]).toBeCloseTo(0.45, 12)
    expect(som.weights[0][0][0]).toBeCloseTo(0.15163266492815836, 12)
    expect(som.weights[0][2][0]).toBeCloseTo(0.8483673350718416, 12)
  })

  it('regla exacta: una neurona fuera de la vecindad (h < 1e-6) no cambia', () => {
    // Con σ 0.1 y d²=1: h = exp(−1/0.02) = exp(−50) ≈ 1.9e-22 < 1e-6 ⇒ sin cambios.
    const som = new SOM(1, 3, 1, { initialLr: 0.5, finalLr: 0.5, initialSigma: 0.1, finalSigma: 0.1 })
    som.weights = [[[0], [0.4], [1]]]
    som.train([[0.5]], 1)
    expect(som.weights[0][0][0]).toBe(0)
    expect(som.weights[0][2][0]).toBe(1)
    expect(som.weights[0][1][0]).toBeCloseTo(0.45, 12)
  })

  it('BMU: elige la neurona con menor distancia euclidea (empate: la primera)', () => {
    const som = new SOM(2, 2, 2)
    som.weights = [
      [
        [0, 0],
        [10, 0],
      ],
      [
        [0, 10],
        [10, 10],
      ],
    ]
    expect(som.getBMU([9, 1])).toEqual([0, 1])
    expect(som.getBMU([1, 9])).toEqual([1, 0])
    expect(som.getBMU([9, 9])).toEqual([1, 1])
    expect(som.getBMU([5, 0])).toEqual([0, 0])
  })

  it('expectLearns: el error de cuantizacion medio sobre datos fijos baja', () => {
    const som = new SOM(2, 2, 2)
    const { initial, final } = expectLearns({
      loss: () => som.quantizationError(DATA),
      step: () => som.train(DATA, 1),
      steps: 30,
    })
    expect(initial).toBeGreaterThan(1)
    expect(final).toBeLessThan(initial)
  })
})
