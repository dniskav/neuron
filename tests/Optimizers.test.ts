import { describe, it, expect } from 'vitest'
import { SGD, Momentum, Adam, ClipOptimizer, ClippedOptimizerFactory } from '../src/optimizers'

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

describe('ClipOptimizer', () => {
  it('clips gradient to specified value', () => {
    // A gradient of 100 should be clipped to 1.0
    const clip = new ClipOptimizer(new SGD(), 1.0)
    // With SGD: w + lr * clipped_gradient = 1.0 + 0.01 * 1.0 = 1.01
    const w = clip.step(1.0, 100.0, 0.01)
    expect(w).toBeCloseTo(1.01, 10)
  })

  it('does not clip gradient within bounds', () => {
    const clip = new ClipOptimizer(new SGD(), 1.0)
    const w = clip.step(1.0, 0.5, 0.01)
    expect(w).toBeCloseTo(1.005, 10)
  })

  it('clips negative gradients', () => {
    const clip = new ClipOptimizer(new SGD(), 1.0)
    const w = clip.step(1.0, -100.0, 0.01)
    expect(w).toBeCloseTo(0.99, 10)
  })

  it('works with Adam inner optimizer', () => {
    const clip = new ClipOptimizer(new Adam(), 1.0)
    const w = clip.step(1.0, 5.0, 0.01)
    expect(typeof w).toBe('number')
    expect(isFinite(w)).toBe(true)
    // Should have changed (but less than without clipping)
    expect(w).not.toBe(1.0)
  })

  it('ClippedOptimizerFactory creates clipping optimizers', () => {
    const factory = ClippedOptimizerFactory(() => new SGD(), 0.5)
    const opt = factory()
    expect(opt instanceof ClipOptimizer).toBe(true)

    // Gradient of 10 should be clipped to 0.5
    const w = opt.step(1.0, 10.0, 0.01)
    expect(w).toBeCloseTo(1.005, 10)
  })

  it('ClippedOptimizerFactory works with Adam', () => {
    const factory = ClippedOptimizerFactory(() => new Adam(), 1.0)
    const opt = factory()
    expect(opt instanceof ClipOptimizer).toBe(true)
    const w = opt.step(1.0, 3.0, 0.01)
    expect(isFinite(w)).toBe(true)
  })
})
