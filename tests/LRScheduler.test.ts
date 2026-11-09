import { describe, it, expect } from 'vitest'
import { LRScheduler } from '../src/LRScheduler'

describe('LRScheduler', () => {
  const scheduler = new LRScheduler()

  it('stepDecay reduces lr at drop points', () => {
    const lr = scheduler.stepDecay(1.0, 0, 0.5, 100)
    expect(lr).toBe(1.0) // epoch 0: no drop

    const lr100 = scheduler.stepDecay(1.0, 100, 0.5, 100)
    expect(lr100).toBe(0.5) // epoch 100: drop by 0.5

    const lr200 = scheduler.stepDecay(1.0, 200, 0.5, 100)
    expect(lr200).toBe(0.25) // epoch 200: drop by 0.5^2
  })

  it('exponentialDecay reduces lr continuously', () => {
    const lr0 = scheduler.exponentialDecay(1.0, 0, 0.99)
    expect(lr0).toBe(1.0)

    const lr100 = scheduler.exponentialDecay(1.0, 100, 0.99)
    expect(lr100).toBeCloseTo(Math.pow(0.99, 100), 5)
  })

  it('plateauDecay reduces lr when loss plateaus', () => {
    const history = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84]
    const lr = scheduler.plateauDecay(0.1, 0.84, history, 5, 0.5)
    // Loss hasn't improved in last 5 epochs (all 0.84), so lr should be reduced
    expect(lr).toBeCloseTo(0.05, 10)
  })

  it('plateauDecay keeps lr when loss improves', () => {
    const history = [1.0, 0.9, 0.8, 0.7, 0.6]
    const lr = scheduler.plateauDecay(0.1, 0.5, history, 5, 0.5)
    // Loss improved, so lr stays the same
    expect(lr).toBe(0.1)
  })

  it('cosineAnnealing returns correct value', () => {
    // At epoch 0, should return maxLr
    const lr0 = scheduler.cosineAnnealing(1.0, 0, 100, 0)
    expect(lr0).toBeCloseTo(1.0, 5)

    // At epoch maxEpochs, should return minLr
    const lr100 = scheduler.cosineAnnealing(1.0, 100, 100, 0)
    expect(lr100).toBeCloseTo(0, 5)

    // At epoch maxEpochs/2, should be halfway
    const lr50 = scheduler.cosineAnnealing(1.0, 50, 100, 0)
    expect(lr50).toBeCloseTo(0.5, 5)
  })

  // ── Ampliación T019 (feature 006) ──────────────────────────────────────────
  // Bordes y la rama no cubierta de plateauDecay (historial más corto que
  // patience). Cada valor esperado está derivado a mano en su comentario.
  // LRScheduler no usa Math.random: la semilla de tests/setup.ts no influye.

  it('stepDecay: la caída ocurre exactamente en múltiplos de epochsDrop', () => {
    // floor(99/100) = 0 → sin caída. floor(150/100) = 1 → ×0.5.
    expect(scheduler.stepDecay(1.0, 99, 0.5, 100)).toBe(1.0)
    expect(scheduler.stepDecay(1.0, 150, 0.5, 100)).toBe(0.5)
  })

  it('stepDecay: lr 0.8, dropRate 0.5, epoch 300, epochsDrop 100 → 0.8·0.5³ = 0.1', () => {
    // floor(300/100) = 3.
    expect(scheduler.stepDecay(0.8, 300, 0.5, 100)).toBeCloseTo(0.1, 12)
  })

  it('exponentialDecay: lr 2, decayRate 0.5, epoch 3 → 2·0.5³ = 0.25', () => {
    expect(scheduler.exponentialDecay(2.0, 3, 0.5)).toBeCloseTo(0.25, 12)
  })

  it('exponentialDecay: decayRate 1 mantiene el lr en cualquier época', () => {
    // 1^500 = 1 → 0.3·1 = 0.3.
    expect(scheduler.exponentialDecay(0.3, 500, 1)).toBe(0.3)
  })

  it('plateauDecay: con menos historial que patience devuelve lr aunque la pérdida empeore', () => {
    // history.length = 2 < patience = 5 → return lr. (Sin esa guarda, 3.0 ≥ min 1.0
    // reduciría el lr a 0.05.) Esperado: 0.1.
    expect(scheduler.plateauDecay(0.1, 3.0, [1.0, 2.0], 5, 0.5)).toBe(0.1)
  })

  it('plateauDecay: solo mira las últimas `patience` pérdidas', () => {
    // Ventana slice(−2) = [0.5, 0.6] → mínimo 0.5. Pérdida actual 0.4 < 0.5 → mejora → lr 0.1.
    // Con todo el historial (mínimo 0.1), 0.4 ≥ 0.1 reduciría el lr a 0.05.
    expect(scheduler.plateauDecay(0.1, 0.4, [0.1, 0.5, 0.6], 2, 0.5)).toBe(0.1)
  })

  it('plateauDecay: una pérdida igual al mínimo reciente cuenta como estancamiento', () => {
    // Ventana slice(−2) = [0.4, 0.3] → mínimo 0.3. 0.3 ≥ 0.3 → decae: 0.1·0.5 = 0.05.
    expect(scheduler.plateauDecay(0.1, 0.3, [0.5, 0.4, 0.3], 2, 0.5)).toBeCloseTo(0.05, 12)
  })

  it('cosineAnnealing con minLr: extremos lr y minLr; mitad (lr + minLr)/2', () => {
    // lr 1, minLr 0.2. Época 0: 0.2 + 0.5·0.8·(1 + 1) = 1.0.
    // Época 100 de 100: 0.2 + 0.4·(1 − 1) = 0.2. Época 50: 0.2 + 0.4·(1 + 0) = 0.6.
    expect(scheduler.cosineAnnealing(1.0, 0, 100, 0.2)).toBeCloseTo(1.0, 12)
    expect(scheduler.cosineAnnealing(1.0, 100, 100, 0.2)).toBeCloseTo(0.2, 12)
    expect(scheduler.cosineAnnealing(1.0, 50, 100, 0.2)).toBeCloseTo(0.6, 12)
  })

  it('cosineAnnealing: un cuarto del recorrido vale 0.6 + 0.2·√2 ≈ 0.8828427125', () => {
    // Época 25 de 100: cos(π/4) = √2/2. Valor: 0.2 + 0.4·(1 + √2/2) = 0.6 + 0.2·√2.
    expect(scheduler.cosineAnnealing(1.0, 25, 100, 0.2)).toBeCloseTo(0.8828427125, 9)
  })

  it('cosineAnnealing: es decreciente de la época 0 a maxEpochs', () => {
    // Con lr 1, minLr 0.2: épocas 0, 25, 50, 75, 100 dan 1.0, 0.8828…, 0.6, 0.3171…, 0.2.
    const vals = [0, 25, 50, 75, 100].map((e) => scheduler.cosineAnnealing(1.0, e, 100, 0.2))
    for (let i = 1; i < vals.length; i++) expect(vals[i]).toBeLessThan(vals[i - 1])
  })

  it('cosineAnnealing: minLr por defecto es 0', () => {
    // Época 100 → 0 y época 50 con lr 0.4 → 0.2.
    expect(scheduler.cosineAnnealing(1.0, 100, 100)).toBeCloseTo(0, 12)
    expect(scheduler.cosineAnnealing(0.4, 50, 100)).toBeCloseTo(0.2, 12)
  })
})
