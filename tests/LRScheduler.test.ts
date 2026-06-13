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
})
