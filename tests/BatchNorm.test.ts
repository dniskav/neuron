import { describe, it, expect } from 'vitest'
import { BatchNorm } from '../src/BatchNorm'

describe('BatchNorm', () => {
  it('creates with correct dimensions', () => {
    const bn = new BatchNorm(4)
    expect(bn.dim).toBe(4)
  })

  it('gamma initialized to 1, beta to 0', () => {
    const bn = new BatchNorm(3)
    bn.gamma.forEach(g => expect(g).toBe(1))
    bn.beta.forEach(b => expect(b).toBe(0))
  })

  it('forward returns array of correct length', () => {
    const bn = new BatchNorm(3)
    const out = bn.forward([1, 2, 3])
    expect(out.length).toBe(3)
  })

  it('forward returns finite values', () => {
    const bn = new BatchNorm(3)
    const out = bn.forward([0.5, -0.5, 1.0])
    expect(out.every(v => isFinite(v))).toBe(true)
  })

  it('forward updates running statistics', () => {
    const bn = new BatchNorm(3)
    const origMean = [...bn.runningMean]
    bn.forward([1, 2, 3])
    // Running mean should have changed
    const changed = bn.runningMean.some((v, i) => v !== origMean[i])
    expect(changed).toBe(true)
  })

  it('forward validates input length', () => {
    const bn = new BatchNorm(3)
    expect(() => bn.forward([1, 2])).toThrow()
  })

  it('backward returns gradient w.r.t. input', () => {
    const bn = new BatchNorm(3)
    bn.forward([1, 2, 3])
    const dX = bn.backward([0.1, 0.2, 0.3])
    expect(dX.length).toBe(3)
    expect(dX.every(v => isFinite(v))).toBe(true)
  })

  it('trainParams updates gamma and beta', () => {
    const bn = new BatchNorm(3)
    bn.forward([1, 2, 3])
    const origGamma = [...bn.gamma]
    const origBeta = [...bn.beta]
    bn.trainParams([0.1, 0.2, 0.3], 0.01)
    const gammaChanged = bn.gamma.some((g, i) => g !== origGamma[i])
    const betaChanged = bn.beta.some((b, i) => b !== origBeta[i])
    expect(gammaChanged).toBe(true)
    expect(betaChanged).toBe(true)
  })

  it('getWeights returns gamma and beta', () => {
    const bn = new BatchNorm(3)
    const w = bn.getWeights()
    expect(w.length).toBe(6) // 3 gamma + 3 beta
  })

  it('setWeights restores gamma and beta', () => {
    const bn = new BatchNorm(3)
    const w = bn.getWeights()
    const newW = w.map((v, i) => v + i * 0.1)
    bn.setWeights(newW)
    expect(bn.getWeights()).toEqual(newW)
  })
})
