// Test de Perceptron (feature 002, T026).
//
// EXCEPCION A LA CONSTITUCION II (justificada): el Perceptron no tiene oraculo de
// gradiente. Su salida es el escalon z > 0 ? 1 : 0, que no es diferenciable, y su
// regla de aprendizaje (regla del perceptron, Rosenblatt 1957) no es el descenso
// por gradiente de ninguna perdida: w <- w + lr·(target − output)·x solo actualiza
// ante errores de clasificacion. Por tanto no existe dL/dw que comparar con
// diferencias finitas. Se sustituye por:
//   1. humo: prediccion binaria y train devuelve el error (target − output);
//   2. regla exacta: valores de w, bias y error calculados a mano para un paso;
//   3. expectLearns: el numero de errores por epoca sobre AND (linealmente separable)
//      baja a 0 y se mantiene en 0;
//   4. control de teoria: XOR (no separable) nunca llega a 0 errores.
// El Perceptron no usa Math.random; la semilla de tests/setup.ts no afecta.
import { describe, it, expect } from 'vitest'
import { Perceptron } from '../../src/Perceptron'
import { expectLearns } from '../helpers/learns'

type Sample = [number[], 0 | 1]

const AND: Sample[] = [
  [[0, 0], 0],
  [[0, 1], 0],
  [[1, 0], 0],
  [[1, 1], 1],
]

const XOR: Sample[] = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0],
]

/** Una epoca: presenta cada muestra una vez y devuelve cuantas se clasificaron mal. */
function epoch(p: Perceptron, data: Sample[], lr: number): number {
  let errors = 0
  for (const [x, t] of data) {
    if (p.train(x, t, lr) !== 0) errors++
  }
  return errors
}

function countErrors(p: Perceptron, data: Sample[]): number {
  return data.filter(([x, t]) => p.predict(x) !== t).length
}

describe('Perceptron (sin oraculo de gradiente, ver cabecera)', () => {
  it('humo: predice 0 o 1 y train devuelve target − output', () => {
    // Pesos a cero: z = 0, y z > 0 es falso, luego la salida es 0.
    const p = new Perceptron(2)
    expect(p.predict([1, 1])).toBe(0)
    // Target 1 con salida 0 ⇒ error +1.
    expect(p.train([1, 1], 1, 0.1)).toBe(1)
    // Otro perceptron fresco: target 0 con salida 0 ⇒ error 0 y sin cambios.
    const q = new Perceptron(2)
    expect(q.train([1, 1], 0, 0.1)).toBe(0)
    expect(q.weights).toEqual([0, 0])
    expect(q.bias).toBe(0)
  })

  it('regla exacta: un paso sobre el error actualiza w y bias como w + lr·error·x', () => {
    // Calculo a mano: w = [0, 0], b = 0. Entrada [1, 1], target 1, lr 0.5.
    // output = 0 (z = 0), error = 1 − 0 = 1.
    // w = [0 + 0.5·1·1, 0 + 0.5·1·1] = [0.5, 0.5]; b = 0 + 0.5·1 = 0.5.
    const p = new Perceptron(2)
    expect(p.train([1, 1], 1, 0.5)).toBe(1)
    expect(p.weights).toEqual([0.5, 0.5])
    expect(p.bias).toBe(0.5)
    // Ahora z = 0.5 + 0.5 + 0.5 = 1.5 > 0, output 1, target 1 ⇒ sin error y sin cambios.
    expect(p.train([1, 1], 1, 0.5)).toBe(0)
    expect(p.weights).toEqual([0.5, 0.5])
    expect(p.bias).toBe(0.5)
  })

  it('regla exacta: un error con target 0 resta lr·x', () => {
    // w = [0.5, 0.5], b = 0.5. Entrada [1, 0], target 0.
    // z = 0.5 + 0.5 = 1.0 > 0 ⇒ output 1, error = 0 − 1 = −1.
    // w = [0.5 − 0.2, 0.5 − 0] = [0.3, 0.5]; b = 0.5 − 0.2 = 0.3.
    const p = new Perceptron(2)
    p.weights = [0.5, 0.5]
    p.bias = 0.5
    expect(p.train([1, 0], 0, 0.2)).toBe(-1)
    expect(p.weights[0]).toBeCloseTo(0.3, 12)
    expect(p.weights[1]).toBeCloseTo(0.5, 12)
    expect(p.bias).toBeCloseTo(0.3, 12)
  })

  it('expectLearns: AND (separable) pasa de errores a 0 y no vuelve a fallar', () => {
    const p = new Perceptron(2)
    const errors = () => countErrors(p, AND)
    // Inicialmente (pesos a cero) hay 1 error: el caso [1,1] → 1.
    expect(errors()).toBe(1)
    expectLearns({ loss: errors, step: () => epoch(p, AND, 0.1), steps: 20 })
    // Tras aprender, cada epoca siguiente no tiene errores y no cambia los pesos.
    const w = [...p.weights]
    const b = p.bias
    expect(epoch(p, AND, 0.1)).toBe(0)
    expect(p.weights).toEqual(w)
    expect(p.bias).toBe(b)
  })

  it('el numero de errores por epoca llega a 0 en AND dentro de 10 epocas', () => {
    const p = new Perceptron(2)
    let reached = -1
    for (let e = 1; e <= 10; e++) {
      if (epoch(p, AND, 0.1) === 0) {
        reached = e
        break
      }
    }
    expect(reached).toBeGreaterThan(0)
    for (const [x, t] of AND) expect(p.predict(x)).toBe(t)
  })

  it('control de teoria: XOR (no separable) nunca llega a 0 errores en 1000 epocas', () => {
    // Minsky y Papert (1969): ninguna recta separa XOR, asi que el teorema de convergencia
    // no aplica y el error por epoca nunca es 0 de forma sostenida.
    const p = new Perceptron(2)
    let zeroEpochs = 0
    for (let e = 0; e < 1000; e++) {
      if (epoch(p, XOR, 0.1) === 0) zeroEpochs++
    }
    expect(zeroEpochs).toBe(0)
  })
})
