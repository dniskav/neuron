// Oraculo de NetworkFast: el paso REAL de train() se compara con diferencias finitas de la
// MSE media (coseno > 0.999, ratio en [0.99, 1.01]), como NetworkN. Ademas, inputGradient
// frente a diferencias finitas respecto a la entrada.
import { describe, it, expect } from 'vitest'
import { NetworkFast } from '../../src/NetworkFast'
import { SGD } from '../../src/optimizers'
import { relu, sigmoid, tanh } from '../../src/activations'
import { expectGradientMatches, gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

const INPUT = [0.5, -0.3, 0.8]
const TARGET = [0.7]

function setup(
  structure: number[],
  opts: { residual?: boolean | ((i: number) => boolean); dropoutRate?: number; relu?: boolean } = {},
  target: number[] = TARGET,
  training = false,
) {
  const n = structure.length - 1
  const activations = Array.from({ length: n }, (_, i) => (opts.relu && i < n - 1 ? relu : sigmoid))
  const net = new NetworkFast(structure, {
    optimizer: () => new SGD(), activations, residual: opts.residual, dropoutRate: opts.dropoutRate,
  })
  const loss = () => {
    const p = net.predict(INPUT, training)
    return p.reduce((s, v, i) => s + (target[i] - v) ** 2, 0) / p.length
  }
  return {
    net, loss,
    step: (lr: number) => void net.train(INPUT, target, lr),
    getParams: () => net.getWeights(),
    setParams: (p: number[]) => net.setWeights(p),
  }
}

describe('NetworkFast oraculo (sin residual)', () => {
  const S = [3, 4, 3, 1]

  it('oraculo: el paso sigue -dL/dw y la escala es 1', () => {
    for (const useRelu of [false, true]) {
      const r = expectGradientMatches(setup(S, { relu: useRelu }))
      expect(r.ratio).toBeGreaterThan(0.99)
      expect(r.ratio).toBeLessThan(1.01)
    }
  })

  const T_BY_N: Record<number, number[]> = {
    1: [0.7], 2: [0.7, 0.2], 3: [0.7, 0.2, -0.4], 4: [0.7, 0.2, -0.4, 0.5],
  }
  for (const n of [1, 2, 3, 4]) {
    it(`escala: ratio en [0.99, 1.01] con n = ${n} salidas, semillas 1..5, lr x0.97/x1.03`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        for (const k of [0.97, 1.03]) {
          seedRandom(seed)
          const m = setup([3, 4, 3, n], {}, T_BY_N[n])
          const r = gradientOracle({ ...m, lr: 1e-3 * k })
          expect(r.cosine, `n ${n} seed ${seed} x${k}`).toBeGreaterThan(0.999)
          expect(r.ratio, `n ${n} seed ${seed} x${k}`).toBeGreaterThan(0.99)
          expect(r.ratio, `n ${n} seed ${seed} x${k}`).toBeLessThan(1.01)
        }
      }
    })
  }

  it('expectLearns: la perdida baja a la mitad', () => {
    const m = setup(S)
    expectLearns({ loss: m.loss, step: () => m.step(0.5), steps: 200 })
  })
})

describe('NetworkFast oraculo (residual)', () => {
  const CASES: [string, number[], boolean | ((i: number) => boolean), number[]][] = [
    ['[3,3,3,1] residual: true', [3, 3, 3, 1], true, TARGET],
    ['[3,3,3,3] residual en la salida', [3, 3, 3, 3], true, [0.7, 0.2, -0.4]],
    ['[3,3,3,1] residual: i === 1', [3, 3, 3, 1], (i: number) => i === 1, TARGET],
    ['[3,3,3,3] solo la capa de salida', [3, 3, 3, 3], (i: number) => i === 2, [0.7, 0.2, -0.4]],
    ['[3,4,4,4,2] residual: true', [3, 4, 4, 4, 2], true, [0.7, 0.2]],
  ]
  for (const [name, st, residual, target] of CASES) {
    it(`oraculo robusto a lr x0.97/x1.03 y semillas 1..5 (${name})`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        for (const k of [0.97, 1, 1.03]) {
          seedRandom(seed)
          const m = setup(st, { residual, relu: true }, target)
          const r = gradientOracle({ ...m, lr: 1e-3 * k })
          expect(r.cosine, `${name} seed ${seed} x${k}`).toBeGreaterThan(0.999)
          expect(r.ratio, `${name} seed ${seed} x${k}`).toBeGreaterThan(0.99)
          expect(r.ratio, `${name} seed ${seed} x${k}`).toBeLessThan(1.01)
        }
      }
    })
  }
})

describe('NetworkFast oraculo (dropout con mascara fija)', () => {
  for (const [name, residual] of [['sin residual', false], ['con residual', true]] as const) {
    it(`oraculo (${name})`, () => {
      const m = setup([3, 3, 3, 1], { dropoutRate: 0.5, residual }, TARGET, true)
      const r = expectGradientMatches({ ...m, resetRng: () => seedRandom(7) })
      expect(r.ratio).toBeGreaterThan(0.99)
      expect(r.ratio).toBeLessThan(1.01)
    })
  }
})

describe('NetworkFast.inputGradient frente a diferencias finitas', () => {
  type Res = boolean | ((i: number) => boolean)
  const CASES: [string, number[], Res, boolean][] = [
    ['[3,4,3,1] sigmoide', [3, 4, 3, 1], false, false],
    ['[3,4,3,1] relu', [3, 4, 3, 1], false, true],
    ['[3,3,3,1] relu residual: true', [3, 3, 3, 1], true, true],
    ['[3,3,3,1] residual: i === 1', [3, 3, 3, 1], (i: number) => i === 1, false],
  ]
  for (const [name, st, residual, useRelu] of CASES) {
    it(`coincide, semillas 1..5 (${name})`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const m = setup(st, { residual, relu: useRelu })
        const x = [0.5, -0.3, 0.8]
        const t = [0.7]
        const p = m.net.predict(x)
        // delta = (t - p) * dfn(p) para L = 0.5 (t - p)^2 (contrato de trainWithDeltas)
        const deltas = p.map((v, i) => (t[i] - v) * v * (1 - v))
        const got = m.net.inputGradient(x, deltas)
        const L = (xx: number[]) => m.net.predict(xx).reduce((s, v, i) => s + 0.5 * (t[i] - v) ** 2, 0)
        const eps = 1e-6
        const want = x.map((_, j) => {
          const xp = [...x]; xp[j] += eps
          const xm = [...x]; xm[j] -= eps
          return -(L(xp) - L(xm)) / (2 * eps)
        })
        got.forEach((v, j) => expect(Math.abs(v - want[j])).toBeLessThan(1e-6))
      }
    })
  }
})

// ─── trainBatch frente a diferencias finitas de la MSE media del lote ─────────
// tanh (not relu) here: with bias 0 and dead units a pre-activation is exactly 0 and finite
// differences cross the ReLU kink, which says nothing about the batch gradient.
describe('NetworkFast.trainBatch oraculo (media del lote)', () => {
  const XS = [[0.5, -0.3, 0.8], [-0.6, 0.2, 0.1], [0.9, 0.9, -0.4], [0.0, -0.7, 0.3], [0.3, 0.5, -0.9]]
  const TS = [[0.7, 0.1], [0.2, 0.9], [0.5, 0.5], [0.1, 0.3], [0.8, 0.6]]
  const CASES: [string, number[], boolean | ((i: number) => boolean), boolean][] = [
    ['[3,4,3,2] sigmoide', [3, 4, 3, 2], false, false],
    ['[3,4,3,2] tanh', [3, 4, 3, 2], false, true],
    ['[3,3,3,2] tanh residual en capas ocultas', [3, 3, 3, 2], (i: number) => i === 1, true],
  ]
  for (const [name, st, residual, useAct] of CASES) {
    it(`el paso del lote sigue -dL/dw con escala 1 (${name}), semillas 1..5`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const n = st.length - 1
        const net = new NetworkFast(st, {
          optimizer: () => new SGD(), residual,
          activations: Array.from({ length: n }, (_, i) => (useAct && i < n - 1 ? tanh : sigmoid)),
        })
        const loss = () => XS.reduce((s, x, i) => {
          const p = net.predict(x)
          return s + p.reduce((q, v, k) => q + (TS[i][k] - v) ** 2, 0) / p.length
        }, 0) / XS.length
        const r = gradientOracle({
          getParams: () => net.getWeights(), setParams: p => net.setWeights(p), loss,
          step: lr => void net.trainBatch(XS, TS, lr), lr: 1e-3,
        })
        expect(r.cosine, `${name} seed ${seed}`).toBeGreaterThan(0.999)
        expect(r.ratio, `${name} seed ${seed}`).toBeGreaterThan(0.99)
        expect(r.ratio, `${name} seed ${seed}`).toBeLessThan(1.01)
      }
    })
  }

  it('dropout con mascaras fijas: el paso del lote sigue el gradiente', () => {
    seedRandom(3)
    const net = new NetworkFast([3, 4, 3, 2], { optimizer: () => new SGD(), dropoutRate: 0.3, activations: [tanh, tanh, sigmoid] })
    // Fixed masks: the loss draws a mask per example in order, as trainBatch does.
    const loss = () => XS.reduce((s, x, i) => {
      const p = net.predict(x, true)
      return s + p.reduce((q, v, k) => q + (TS[i][k] - v) ** 2, 0) / p.length
    }, 0) / XS.length
    const r = expectGradientMatches({
      getParams: () => net.getWeights(), setParams: p => net.setWeights(p), loss,
      step: lr => void net.trainBatch(XS, TS, lr), lr: 1e-4, resetRng: () => seedRandom(7),
    })
    expect(r.ratio).toBeGreaterThan(0.99)
    expect(r.ratio).toBeLessThan(1.01)
  })
})
