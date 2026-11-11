// Tests de comportamiento de Perceptron (feature 006, T019).
//
// Complementan tests/oracle/Perceptron.oracle.test.ts (humo, regla exacta de un
// paso y AND) sin repetirlo: aquí van el borde estricto del escalón, las
// validaciones, OR con su convergencia exacta, los pesos finales de AND/OR
// con lr distinto y la cota de acierto de XOR.
//
// Sin oráculo de gradiente: el escalón z > 0 no es derivable (ver la cabecera
// del oráculo). El Perceptron no usa Math.random; la semilla de tests/setup.ts
// no cambia nada.
//
// Robustez frente a lr: con pesos iniciales 0, cada actualización es
// lr·error·x, así que w(lr) = lr·w(1) y z(lr) = lr·z(1). El signo de z no
// cambia, luego la trayectoria de errores es la misma para cualquier lr > 0.
// Los tests comprueban lr = 0.07, 0.1 y 0.13 (×0.7 y ×1.3 del valor base).
//
// Umbral de épocas para AND y OR: 20. Margen exigido del 30 %: converger en
// como mucho 0.7 · 20 = 14 épocas.
import { describe, it, expect } from 'vitest'
import { Perceptron } from '../src/Perceptron'

type Sample = [number[], 0 | 1]

const AND: Sample[] = [
  [[0, 0], 0],
  [[0, 1], 0],
  [[1, 0], 0],
  [[1, 1], 1],
]

const OR: Sample[] = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 1],
]

const XOR: Sample[] = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0],
]

const UMBRAL_EPOCAS = 20
const MARGEN_EPOCAS = 0.7 * UMBRAL_EPOCAS // 14

/** Una época: presenta cada muestra una vez; devuelve cuántas se clasificaron mal. */
function epoch(p: Perceptron, data: Sample[], lr: number): number {
  let errors = 0
  for (const [x, t] of data) {
    if (p.train(x, t, lr) !== 0) errors++
  }
  return errors
}

/** Entrena hasta una época sin errores; devuelve el número de época (1-based) o -1. */
function epochsToConverge(p: Perceptron, data: Sample[], lr: number, maxEpochs: number): number {
  for (let e = 1; e <= maxEpochs; e++) {
    if (epoch(p, data, lr) === 0) return e
  }
  return -1
}

function accuracy(p: Perceptron, data: Sample[]): number {
  return data.filter(([x, t]) => p.predict(x) === t).length / data.length
}

describe('Perceptron: constructor y validaciones', () => {
  it('nInputs debe ser un entero positivo', () => {
    for (const bad of [0, -1, 1.5, NaN]) {
      expect(() => new Perceptron(bad)).toThrow(/nInputs must be a positive integer/)
    }
  })

  it('un perceptron nuevo tiene todos los pesos y el sesgo a 0', () => {
    const p = new Perceptron(3)
    expect(p.weights).toEqual([0, 0, 0])
    expect(p.bias).toBe(0)
  })

  it('predict rechaza vectores de longitud distinta de nInputs', () => {
    const p = new Perceptron(2)
    expect(() => p.predict([1])).toThrow(/Perceptron.predict/)
    expect(() => p.predict([1, 2, 3])).toThrow(/Perceptron.predict/)
  })

  it('predict rechaza valores no finitos', () => {
    const p = new Perceptron(2)
    expect(() => p.predict([NaN, 1])).toThrow(/Perceptron.predict/)
    expect(() => p.predict([1, Infinity])).toThrow(/Perceptron.predict/)
  })

  it('train rechaza un target distinto de 0 y 1', () => {
    const p = new Perceptron(2)
    for (const bad of [2, -1, 0.5]) {
      expect(() => p.train([1, 1], bad as unknown as 0 | 1, 0.1)).toThrow(/target must be 0 or 1/)
    }
  })

  it('train rechaza un lr no positivo o no finito', () => {
    const p = new Perceptron(2)
    expect(() => p.train([1, 1], 1, 0)).toThrow(/learning rate must be positive/)
    expect(() => p.train([1, 1], 1, -0.1)).toThrow(/learning rate must be positive/)
    expect(() => p.train([1, 1], 1, NaN)).toThrow(/Perceptron.train/)
  })

  it('una llamada a train rechazada no cambia pesos ni sesgo', () => {
    const p = new Perceptron(2)
    p.weights = [0.25, -0.5]
    p.bias = 0.75
    expect(() => p.train([1, 1], 2 as unknown as 0 | 1, 0.1)).toThrow()
    expect(() => p.train([1, 1], 1, 0)).toThrow()
    expect(p.weights).toEqual([0.25, -0.5])
    expect(p.bias).toBe(0.75)
  })
})

describe('Perceptron: escalón de salida', () => {
  it('z exactamente 0 da salida 0 (la comparación es estricta, z > 0)', () => {
    // w = [1, −1], b = 0. Para x = [1, 1]: z = 1 − 1 + 0 = 0 → salida 0.
    // Para x = [2, 2]: z = 0 → salida 0. Para x = [1, 0]: z = 1 → salida 1.
    const p = new Perceptron(2)
    p.weights = [1, -1]
    p.bias = 0
    expect(p.predict([1, 1])).toBe(0)
    expect(p.predict([2, 2])).toBe(0)
    expect(p.predict([1, 0])).toBe(1)
  })

  it('con entradas 0 la salida es 1 si y solo si el sesgo es positivo', () => {
    // z = b cuando x = [0, 0].
    const p = new Perceptron(2)
    p.bias = 0.5
    expect(p.predict([0, 0])).toBe(1)
    p.bias = -0.5
    expect(p.predict([0, 0])).toBe(0)
    p.bias = 0
    expect(p.predict([0, 0])).toBe(0)
  })

  it('pesos negativos invierten la clase: w = [−2, 1], b = 1', () => {
    // x = [1, 0]: z = −2 + 1 = −1 → 0. x = [0, 0]: z = 1 → 1. x = [0, 1]: z = 2 → 1.
    const p = new Perceptron(2)
    p.weights = [-2, 1]
    p.bias = 1
    expect(p.predict([1, 0])).toBe(0)
    expect(p.predict([0, 0])).toBe(1)
    expect(p.predict([0, 1])).toBe(1)
  })
})

describe('Perceptron: regla de actualización en casos de borde', () => {
  it('sin error no cambia pesos ni sesgo', () => {
    // w = [0.5, −0.5], b = −1, x = [1, 1]: z = 0.5 − 0.5 − 1 = −1 → salida 0.
    // Target 0 → error 0 → devuelve 0 y no toca nada.
    const p = new Perceptron(2)
    p.weights = [0.5, -0.5]
    p.bias = -1
    expect(p.train([1, 1], 0, 0.3)).toBe(0)
    expect(p.weights).toEqual([0.5, -0.5])
    expect(p.bias).toBe(-1)
  })

  it('error +1 con entrada negativa mueve w en −x: lr 0.3, x = [−1, 2], target 1', () => {
    // w = [0, 0], b = 0 → z = 0 → salida 0 → error = 1 − 0 = 1.
    // w = [0 + 0.3·1·(−1), 0 + 0.3·1·2] = [−0.3, 0.6]; b = 0 + 0.3·1 = 0.3.
    const p = new Perceptron(2)
    expect(p.train([-1, 2], 1, 0.3)).toBe(1)
    expect(p.weights[0]).toBeCloseTo(-0.3, 12)
    expect(p.weights[1]).toBeCloseTo(0.6, 12)
    expect(p.bias).toBeCloseTo(0.3, 12)
  })

  it('error −1 resta lr·x y lr del sesgo: w = [1, 1], b = 0, x = [2, 0], lr 0.25', () => {
    // z = 2 → salida 1, target 0 → error = −1.
    // w = [1 − 0.25·2, 1 − 0] = [0.5, 1]; b = 0 − 0.25 = −0.25.
    const p = new Perceptron(2)
    p.weights = [1, 1]
    p.bias = 0
    expect(p.train([2, 0], 0, 0.25)).toBe(-1)
    expect(p.weights[0]).toBeCloseTo(0.5, 12)
    expect(p.weights[1]).toBeCloseTo(1, 12)
    expect(p.bias).toBeCloseTo(-0.25, 12)
  })

  it('un paso corrige la muestra: x = [1, 2], target 1, lr 1 → w = [1, 2], b = 1, z = 6', () => {
    // z = 0 → error +1 → w = [1, 2], b = 1. Nuevo z = 1 + 4 + 1 = 6 > 0.
    // (Cambio de z = lr·error·(‖x‖² + 1) = 1·1·(5 + 1) = 6, con z_antes = 0.)
    const p = new Perceptron(2)
    expect(p.train([1, 2], 1, 1)).toBe(1)
    expect(p.weights).toEqual([1, 2])
    expect(p.bias).toBe(1)
    expect(p.predict([1, 2])).toBe(1)
  })
})

describe('Perceptron: convergencia en datos separables (umbral 20 épocas, margen 30 %)', () => {
  it('AND converge en la época 6 con lr 1 (traza a mano abajo) y dentro del margen', () => {
    // Traza a mano con lr = 1, orden [0,0], [0,1], [1,0], [1,1]:
    //   Época 1: solo falla [1,1] (z = 0) → w = (1,1), b = 1.
    //   Época 2: [0,0] (z = 1) → b = 0; [0,1] (z = 1) → w = (1,0), b = −1;
    //            [1,1] (z = 0) → w = (2,1), b = 0. Fallos: 3.
    //   Época 3: [0,1] (z = 1) → w = (2,0), b = −1; [1,0] (z = 1) → w = (1,0), b = −2;
    //            [1,1] (z = −1) → w = (2,1), b = −1. Fallos: 3.
    //   Época 4: [1,0] (z = 1) → w = (1,1), b = −2; [1,1] (z = 0) → w = (2,2), b = −1. Fallos: 2.
    //   Época 5: [0,1] (z = 1) → w = (2,1), b = −2. Fallos: 1.
    //   Época 6: z = −2, −1, 0, 1 → todas correctas. Fallos: 0.
    const p = new Perceptron(2)
    const e = epochsToConverge(p, AND, 1, UMBRAL_EPOCAS)
    expect(e).toBe(6)
    expect(e).toBeLessThanOrEqual(MARGEN_EPOCAS)
    expect(p.weights).toEqual([2, 1])
    expect(p.bias).toBe(-2)
    expect(accuracy(p, AND)).toBe(1)
  })

  it('OR converge en la época 4 con lr 1 (traza a mano abajo) y dentro del margen', () => {
    // Traza a mano con lr = 1:
    //   Época 1: [0,1] (z = 0) → w = (0,1), b = 1. Resto correcto ([1,0] z = 1, [1,1] z = 2).
    //   Época 2: [0,0] (z = 1) → b = 0; [1,0] (z = 0) → w = (1,1), b = 1. Fallos: 2.
    //   Época 3: [0,0] (z = 1) → b = 0. Resto: [0,1] z = 1, [1,0] z = 1, [1,1] z = 2. Fallos: 1.
    //   Época 4: z = 0, 1, 1, 2 → todas correctas. Fallos: 0.
    const p = new Perceptron(2)
    const e = epochsToConverge(p, OR, 1, UMBRAL_EPOCAS)
    expect(e).toBe(4)
    expect(e).toBeLessThanOrEqual(MARGEN_EPOCAS)
    expect(p.weights).toEqual([1, 1])
    expect(p.bias).toBe(0)
    expect(accuracy(p, OR)).toBe(1)
  })

  it('la convergencia no depende de lr (×0.7, base, ×1.3): mismas épocas y pesos = lr·w(1)', () => {
    // Con pesos iniciales 0, w(lr) = lr·w(1). Para AND, w(1) = (2, 1), b(1) = −2.
    // Con lr = 0.07: w = (0.14, 0.07), b = −0.14. Con lr = 0.13: w = (0.26, 0.13), b = −0.26.
    for (const lr of [0.07, 0.1, 0.13]) {
      const p = new Perceptron(2)
      expect(epochsToConverge(p, AND, lr, UMBRAL_EPOCAS)).toBe(6)
      expect(p.weights[0]).toBeCloseTo(2 * lr, 12)
      expect(p.weights[1]).toBeCloseTo(1 * lr, 12)
      expect(p.bias).toBeCloseTo(-2 * lr, 12)
    }
  })

  it('OR: con lr ×0.7 y ×1.3 converge también en la época 4', () => {
    // w(1) = (1, 1), b(1) = 0 → con lr = 0.07 o 0.13 los pesos son lr·(1, 1) y b = 0.
    for (const lr of [0.07, 0.13]) {
      const p = new Perceptron(2)
      expect(epochsToConverge(p, OR, lr, UMBRAL_EPOCAS)).toBe(4)
      expect(p.weights[0]).toBeCloseTo(lr, 12)
      expect(p.weights[1]).toBeCloseTo(lr, 12)
      expect(p.bias).toBeCloseTo(0, 12)
    }
  })
})

describe('Perceptron: XOR no es linealmente separable', () => {
  it('ningún w, b clasifica los 4 puntos: la prueba por suma de desigualdades', () => {
    // Si los 4 fueran correctos: (0,0)→0 da b ≤ 0; (0,1)→1 da w2 + b > 0;
    // (1,0)→1 da w1 + b > 0; (1,1)→0 da w1 + w2 + b ≤ 0.
    // Sumando las dos del medio: w1 + w2 + 2b > 0. Sumando la primera y la última:
    // w1 + w2 + 2b ≤ 0. Contradicción. Por tanto la precisión máxima es 3/4.
    //
    // Y 3/4 se alcanza: w = (1, 1), b = −0.5 falla solo en (1,1) (z = 1.5 → 1).
    const p = new Perceptron(2)
    p.weights = [1, 1]
    p.bias = -0.5
    expect(accuracy(p, XOR)).toBe(0.75)
  })

  it('durante 1000 épocas la precisión nunca pasa de 0.75', () => {
    const p = new Perceptron(2)
    let maxAcc = 0
    for (let e = 0; e < 1000; e++) {
      epoch(p, XOR, 0.1)
      maxAcc = Math.max(maxAcc, accuracy(p, XOR))
    }
    expect(maxAcc).toBeLessThanOrEqual(0.75)
  })
})
