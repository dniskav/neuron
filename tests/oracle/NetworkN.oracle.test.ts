// PLANTILLA de archivo de oraculo de gradiente (feature 002, T005).
//
// Cada modelo entrenable tiene un archivo `tests/oracle/<Modelo>.oracle.test.ts`
// con este patron (copiar y cambiar modelo, perdida y datos):
//   1. humo: construir, predecir y entrenar un paso; todo finito.
//   2. oraculo: `expectGradientMatches` compara el paso REAL (`step`) con las
//      diferencias finitas de `loss`. Convencion: los optimizadores hacen
//      `w + lr*g`, asi que un modelo correcto da coseno ~ 1 con -dL/dw.
//   3. expectLearns: la perdida (medida fuera de `train`, sobre datos fijos)
//      baja a menos de la mitad tras N pasos.
//   4. Un bug conocido de la auditoria se marca con un test de fallo esperado
//      (ver specs/000-audit/AUDIT.md) hasta que se corrige. NetworkN ya no tiene
//      ninguno: I1 se corrigio en 004 (T001) y todos sus oraculos son normales.
// Para adaptar: `getParams/setParams` (getWeights/setWeights o `bindParams`);
// `loss` debe ser determinista (usar `predict`, sin dropout); `step` llama a
// `train` UNA vez. Si el modelo usa `WeightMatrix`/Adam, envolver con
// `withPlainSGD` (Q1). La semilla global (tests/setup.ts) fija los pesos iniciales.
import { describe, it, expect } from 'vitest'
import { NetworkN } from '../../src/NetworkN'
import { SGD } from '../../src/optimizers'
import { relu, sigmoid, tanh, leakyRelu } from '../../src/activations'
import { expectGradientMatches, gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

const INPUT = [0.5, -0.3, 0.8]
const TARGET = [0.7]

// L = 0.5 * sum (t - p)^2 ; train() usa g = t - p = -dL/d(salida) y delta = g * f'(f(z)).
function setup(
  structure: number[],
  opts: { residual?: boolean | ((i: number) => boolean); dropoutRate?: number } = {},
  target: number[] = TARGET,
  training = false,
) {
  const net = new NetworkN(structure, { optimizer: () => new SGD(), ...opts })
  const loss = () => {
    const p = net.predict(INPUT, training)
    return p.reduce((s, v, i) => s + 0.5 * (target[i] - v) ** 2, 0)
  }
  return {
    net,
    loss,
    step: (lr: number) => void net.train(INPUT, target, lr),
    getParams: () => net.getWeights(),
    setParams: (p: number[]) => net.setWeights(p),
  }
}

describe('NetworkN oraculo (sin residual)', () => {
  const S = [3, 4, 3, 1]

  it('humo: predice y entrena con salidas finitas', () => {
    const m = setup(S)
    expect(m.net.predict(INPUT).every(Number.isFinite)).toBe(true)
    expect(Number.isFinite(m.net.train(INPUT, TARGET, 0.1))).toBe(true)
  })

  it('oraculo: el paso sigue -dL/dw', () => {
    const r = expectGradientMatches(setup(S))
    // With plain SGD and one output the step is exactly lr * g. The scale is
    // checked here because a missing output-activation derivative is a global
    // scalar for a single output and leaves the cosine untouched (T038 review).
    expect(r.ratio).toBeCloseTo(1, 6)
  })

  it('expectLearns: la perdida baja a la mitad', () => {
    const m = setup(S)
    expectLearns({ loss: m.loss, step: () => m.step(0.5), steps: 200 })
  })
})

describe('NetworkN oraculo (residual)', () => {
  const S = [3, 3, 3, 1] // capas 0 y 1 cumplen entradas == salidas

  it('humo: predice y entrena con salidas finitas', () => {
    const m = setup(S, { residual: true })
    expect(m.net.predict(INPUT).every(Number.isFinite)).toBe(true)
    expect(Number.isFinite(m.net.train(INPUT, TARGET, 0.1))).toBe(true)
  })

  // I1 corregido: dfn se evalua sobre la salida de f antes del salto y el salto
  // propaga g = dL/d(salida), no el delta. Se comprueba coseno > 0.999 y ratio
  // en [0.99, 1.01] con lr x0.97/x1.03 y semillas 1..5.
  const CASES: [string, number[], boolean | ((i: number) => boolean), number[]][] = [
    ['[3,3,3,1] residual: true', S, true, TARGET],
    ['[3,3,3,3] residual en la salida', [3, 3, 3, 3], true, [0.7, 0.2, -0.4]],
    ['[3,3,3,1] residual: i === 1', S, (i: number) => i === 1, TARGET],
    ['[3,3,3,3] solo la capa de salida', [3, 3, 3, 3], (i: number) => i === 2, [0.7, 0.2, -0.4]],
    ['[3,4,4,4,2] residual: true', [3, 4, 4, 4, 2], true, [0.7, 0.2]],
  ]
  for (const [name, st, residual, target] of CASES) {
    it(`oraculo: el paso sigue -dL/dw (${name})`, () => {
      const r = expectGradientMatches(setup(st, { residual }, target))
      expect(r.ratio).toBeGreaterThan(0.99)
      expect(r.ratio).toBeLessThan(1.01)
    })

    it(`oraculo robusto a lr x0.97/x1.03 y semillas 1..5 (${name})`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        for (const k of [0.97, 1, 1.03]) {
          seedRandom(seed)
          const m = setup(st, { residual }, target)
          const r = gradientOracle({ ...m, lr: 1e-3 * k })
          expect(r.cosine).toBeGreaterThan(0.999)
          expect(r.ratio).toBeGreaterThan(0.99)
          expect(r.ratio).toBeLessThan(1.01)
        }
      }
    })
  }

  it('expectLearns con residual', () => {
    const m = setup(S, { residual: true })
    expectLearns({ loss: m.loss, step: () => m.step(0.5), steps: 200 })
  })

  it('trainWithDeltas lanza con residual en la ultima capa (Q2)', () => {
    const net = new NetworkN([3, 3, 3, 3], { residual: true })
    const before = net.getWeights()
    expect(() => net.trainWithDeltas(INPUT, [0.1, 0.1, 0.1], 0.1)).toThrow(/residual/)
    expect(net.getWeights()).toEqual(before)
  })

  it('trainWithDeltas sigue valido con residual en capas ocultas', () => {
    const net = new NetworkN([3, 3, 3, 1], { residual: true })
    expect(() => net.trainWithDeltas(INPUT, [0.1], 0.1)).not.toThrow()
  })
})

describe('NetworkN dropout (derivada sobre el valor correcto)', () => {
  // Con Math.random reiniciada antes de cada loss()/step(), la mascara es la
  // misma y determinista: el oraculo compara con diferencias finitas del
  // forward en modo entrenamiento.
  for (const [name, residual] of [['sin residual', false], ['con residual', true]] as const) {
    it(`oraculo con mascara fija (${name})`, () => {
      const m = setup([3, 3, 3, 1], { dropoutRate: 0.5, residual }, TARGET, true)
      const r = expectGradientMatches({ ...m, resetRng: () => seedRandom(7) })
      expect(r.ratio).toBeGreaterThan(0.99)
      expect(r.ratio).toBeLessThan(1.01)
    })
  }

  it('oraculo con mascara fija, residual y dropout, salida con residual', () => {
    const m = setup([3, 3, 3, 3], { dropoutRate: 0.5, residual: true }, [0.7, 0.2, -0.4], true)
    expectGradientMatches({ ...m, resetRng: () => seedRandom(11) })
  })
})

describe('NetworkN dropout (estabilidad)', () => {
  // Migrado desde GradientCheck.test.ts (Q6): entrenamiento estable.
  it('entrenamiento estable: perdida finita y sin crecer mas de 20x en 20 pasos', () => {
    const net = new NetworkN([3, 4, 4, 1], { optimizer: () => new SGD(), dropoutRate: 0.5 })
    const losses: number[] = []
    for (let i = 0; i < 20; i++) losses.push(net.train(INPUT, TARGET, 0.01))
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 20)
  })
})

// ─── inputGradient (T005, Q1/Q2) ─────────────────────────────────────────────
// inputGradient(x, deltas) devuelve -dL/dx con deltas = (t - p) * dfn(p) (el
// contrato de trainWithDeltas) y no modifica pesos. Se compara con diferencias
// finitas centrales de L = 0.5 * sum (t - p)^2 respecto a CADA componente de
// la entrada (|a - b| < 1e-6), ademas de coseno y ratio.
describe('NetworkN.inputGradient', () => {
  type Res = boolean | ((i: number) => boolean)
  const outDeltas = (net: NetworkN, x: number[], t: number[]) => {
    const p = net.predict(x)
    const dfn = net.layers[net.layers.length - 1].neurons[0].activation.dfn
    return p.map((v, i) => (t[i] - v) * dfn(v))
  }
  const fdInput = (net: NetworkN, x: number[], t: number[], training = false, reset?: () => void) => {
    const L = (xx: number[]) => {
      reset?.()
      return net.predict(xx, training).reduce((s, v, i) => s + 0.5 * (t[i] - v) ** 2, 0)
    }
    const eps = 1e-6
    return x.map((_, j) => {
      const xp = [...x]; xp[j] += eps
      const xm = [...x]; xm[j] -= eps
      return -(L(xp) - L(xm)) / (2 * eps)
    })
  }
  const cosine = (a: number[], b: number[]) => {
    const d = a.reduce((s, v, i) => s + v * b[i], 0)
    return d / (Math.hypot(...a) * Math.hypot(...b))
  }
  const ratio = (a: number[], b: number[]) => {
    const d = a.reduce((s, v, i) => s + v * b[i], 0)
    return d / b.reduce((s, v) => s + v * v, 0)
  }

  const CASES: [string, number[], Res, boolean][] = [
    ['[3,4,3,1] sigmoide', [3, 4, 3, 1], false, false],
    ['[3,4,3,1] relu', [3, 4, 3, 1], false, true],
    ['[3,3,3,1] sigmoide residual: true', [3, 3, 3, 1], true, false],
    ['[3,3,3,1] relu residual: true', [3, 3, 3, 1], true, true],
    ['[3,3,3,1] residual: i === 1', [3, 3, 3, 1], (i: number) => i === 1, false],
    ['[3,4,4,4,2] relu residual: true', [3, 4, 4, 4, 2], true, true],
  ]
  const target = (n: number) => (n === 1 ? [0.7] : [0.7, -0.2])

  for (const [name, st, residual, useRelu] of CASES) {
    const mk = () => {
      const n = st.length - 1
      const acts = Array.from({ length: n }, (_, i) => (useRelu && i < n - 1 ? relu : sigmoid))
      return new NetworkN(st, { activations: acts, residual, optimizer: () => new SGD() })
    }

    it(`coincide con diferencias finitas, semillas 1..5 (${name})`, () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const net = mk()
        const t = target(st[st.length - 1])
        const x = [0.5, -0.3, 0.8]
        const got = net.inputGradient(x, outDeltas(net, x, t))
        const want = fdInput(net, x, t)
        got.forEach((v, j) => expect(Math.abs(v - want[j])).toBeLessThan(1e-6))
        expect(cosine(got, want)).toBeGreaterThan(0.999)
        expect(ratio(got, want)).toBeGreaterThan(0.99)
        expect(ratio(got, want)).toBeLessThan(1.01)
      }
    })

    it(`no modifica pesos y es repetible (${name})`, () => {
      seedRandom(3)
      const net = mk()
      const t = target(st[st.length - 1])
      const x = [0.5, -0.3, 0.8]
      const d = outDeltas(net, x, t)
      const before = net.getWeights()
      const a = net.inputGradient(x, d)
      const b = net.inputGradient(x, d)
      expect(net.getWeights()).toEqual(before)
      expect(b).toEqual(a)
    })
  }

  for (const residual of [false, true]) {
    it(`con dropout y mascara fija coincide con diferencias finitas (residual ${residual})`, () => {
      for (const seed of [7, 8, 9]) {
        seedRandom(1)
        const net = new NetworkN([3, 6, 6, 1], {
          optimizer: () => new SGD(), dropoutRate: 0.3, residual,
          activations: [tanh, leakyRelu, sigmoid],
        })
        const x = [0.5, -0.3, 0.8]
        const t = [0.7]
        seedRandom(seed)
        const p = net.predict(x, true)
        const d = [(t[0] - p[0]) * p[0] * (1 - p[0])]
        seedRandom(seed)
        const got = net.inputGradient(x, d)
        const want = fdInput(net, x, t, true, () => seedRandom(seed))
        got.forEach((v, j) => expect(Math.abs(v - want[j])).toBeLessThan(1e-6))
        expect(Math.hypot(...want)).toBeGreaterThan(1e-4) // no degenerado
        expect(cosine(got, want)).toBeGreaterThan(0.999)
        expect(ratio(got, want)).toBeGreaterThan(0.99)
        expect(ratio(got, want)).toBeLessThan(1.01)
      }
    })
  }

  it('lanza con residual en la ultima capa sin tocar pesos (Q2)', () => {
    const net = new NetworkN([3, 3, 3, 3], { residual: true })
    const before = net.getWeights()
    expect(() => net.inputGradient(INPUT, [0.1, 0.1, 0.1])).toThrow(/residual/)
    expect(net.getWeights()).toEqual(before)
  })

  it('valida longitudes y valores', () => {
    const net = new NetworkN([3, 4, 1])
    expect(() => net.inputGradient([1, 2], [0.1])).toThrow(/inputGradient/)
    expect(() => net.inputGradient(INPUT, [0.1, 0.2])).toThrow(/inputGradient/)
    expect(() => net.inputGradient(INPUT, [NaN])).toThrow(/inputGradient/)
  })

  it('un paso en x a favor del resultado baja la perdida', () => {
    seedRandom(2)
    const net = new NetworkN([3, 4, 1])
    const t = [0.9]
    const loss = (x: number[]) => 0.5 * (t[0] - net.predict(x)[0]) ** 2
    const g = net.inputGradient(INPUT, outDeltas(net, INPUT, t))
    expect(loss(INPUT.map((v, j) => v + 0.1 * g[j]))).toBeLessThan(loss(INPUT))
  })
})
