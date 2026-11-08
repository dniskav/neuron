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
//   4. `it.fails` SOLO para un bug conocido de la auditoria, con comentario
//      `// AUDIT <id>` (ver specs/000-audit/AUDIT.md). Cuando se corrija el
//      bug, el `it.fails` se pone en rojo y hay que convertirlo en `it` normal.
// Para adaptar: `getParams/setParams` (getWeights/setWeights o `bindParams`);
// `loss` debe ser determinista (usar `predict`, sin dropout); `step` llama a
// `train` UNA vez. Si el modelo usa `WeightMatrix`/Adam, envolver con
// `withPlainSGD` (Q1). La semilla global (tests/setup.ts) fija los pesos iniciales.
import { describe, it, expect } from 'vitest'
import { NetworkN } from '../../src/NetworkN'
import { SGD } from '../../src/optimizers'
import { expectGradientMatches, gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT = [0.5, -0.3, 0.8]
const TARGET = [0.7]

// L = 0.5 * sum (t - p)^2 ; train() usa delta = (t - p) * f'(p) = -dL/dz.
function setup(structure: number[], opts: { residual?: boolean } = {}) {
  const net = new NetworkN(structure, { optimizer: () => new SGD(), ...opts })
  const loss = () => {
    const p = net.predict(INPUT)
    return p.reduce((s, v, i) => s + 0.5 * (TARGET[i] - v) ** 2, 0)
  }
  return {
    net,
    loss,
    step: (lr: number) => void net.train(INPUT, TARGET, lr),
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
    console.log('NetworkN cos', r.cosine, 'ratio', r.ratio)
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

  // AUDIT I1: dfn se evalua sobre el valor ya sumado y el salto mezcla deltas
  // previos a la activacion; coseno 0.86 en la auditoria, 0.887 con estos datos.
  it.fails('oraculo: el paso sigue -dL/dw', () => {
    expectGradientMatches(setup(S, { residual: true }))
  })

  it('el coseno del residual es parcial, como en la auditoria (0.8-0.95)', () => {
    const r = gradientOracle(setup(S, { residual: true }))
    console.log('NetworkN residual cos', r.cosine, 'ratio', r.ratio)
    expect(r.cosine).toBeGreaterThan(0.8)
    expect(r.cosine).toBeLessThan(0.95)
  })

  it('expectLearns con residual', () => {
    const m = setup(S, { residual: true })
    expectLearns({ loss: m.loss, step: () => m.step(0.5), steps: 200 })
  })
})

describe('NetworkN dropout', () => {
  // Antes en GradientCheck.test.ts (Q6): entrenamiento estable.
  it('entrenamiento estable: perdida finita y sin crecer mas de 20x en 20 pasos', () => {
    const net = new NetworkN([3, 4, 4, 1], { optimizer: () => new SGD(), dropoutRate: 0.5 })
    const losses: number[] = []
    for (let i = 0; i < 20; i++) losses.push(net.train(INPUT, TARGET, 0.01))
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 20)
  })
})
