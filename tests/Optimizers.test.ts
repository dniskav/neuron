import { describe, it, expect } from 'vitest'
import { SGD, Momentum, Adam } from '../src/optimizers'

describe('SGD', () => {
  it('updates weight with gradient * lr', () => {
    const sgd = new SGD()
    const w = 1.0
    const g = 0.5
    const lr = 0.1
    expect(sgd.step(w, g, lr)).toBeCloseTo(1.05, 10)
  })

  it('is stateless', () => {
    const sgd = new SGD()
    const w1 = sgd.step(1.0, 0.5, 0.1)
    const w2 = sgd.step(1.0, 0.5, 0.1)
    expect(w1).toBeCloseTo(w2, 10)
  })
})

describe('Momentum', () => {
  it('accumulates velocity', () => {
    const mom = new Momentum(0.9)
    const w1 = mom.step(1.0, 1.0, 0.1)
    // First step: v = 0.9*0 + 0.1*1 = 0.1, w = 1.0 + 0.1 = 1.1
    expect(w1).toBeCloseTo(1.1, 10)

    const w2 = mom.step(w1, 1.0, 0.1)
    // Second step: v = 0.9*0.1 + 0.1*1 = 0.19, w = 1.1 + 0.19 = 1.29
    expect(w2).toBeCloseTo(1.29, 10)
  })
})

describe('Adam', () => {
  it('updates weight', () => {
    const adam = new Adam()
    const w = adam.step(1.0, 0.5, 0.01)
    // Adam should produce a valid update
    expect(typeof w).toBe('number')
    expect(isFinite(w)).toBe(true)
  })

  it('adapts learning rate based on gradient history', () => {
    const adam = new Adam()
    let w = 1.0
    // Run several steps with constant gradient
    for (let i = 0; i < 100; i++) {
      w = adam.step(w, 1.0, 0.01)
    }
    // Should have moved significantly from initial value
    expect(w).not.toBeCloseTo(1.0, 1)
  })

  it('handles zero gradient', () => {
    const adam = new Adam()
    const w = adam.step(1.0, 0.0, 0.01)
    // With zero gradient, weight should stay approximately the same
    // (Adam has bias correction but with zero gradient, m and v stay near zero)
    expect(w).toBeCloseTo(1.0, 5)
  })
})
