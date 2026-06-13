import { describe, it, expect } from 'vitest'
import { LayerNorm } from '../src/LayerNorm'

describe('LayerNorm', () => {
  it('creates with correct dimensions', () => {
    const norm = new LayerNorm(4)
    expect(norm.gamma.length).toBe(4)
    expect(norm.beta.length).toBe(4)
  })

  it('gamma initialized to 1, beta to 0', () => {
    const norm = new LayerNorm(3)
    norm.gamma.forEach(g => expect(g).toBe(1))
    norm.beta.forEach(b => expect(b).toBe(0))
  })

  it('predictOne normalizes input', () => {
    const norm = new LayerNorm(3)
    norm.resetCache(1)
    const out = norm.predictOne([1, 2, 3], 0)
    // Should be normalized (mean ~0, std ~1)
    const mean = out.reduce((s, v) => s + v, 0) / out.length
    expect(mean).toBeCloseTo(0, 5)
  })

  it('predictOne with different inputs gives different outputs when gamma != 1', () => {
    const norm = new LayerNorm(3)
    norm.gamma = [2, 1, 0.5] // Different gamma values
    norm.resetCache(2)
    const out1 = norm.predictOne([1, 2, 3], 0)
    const out2 = norm.predictOne([10, 2, 3], 1)  // Different distribution (outlier at index 0)
    // With different gamma, outputs should differ
    const differs = out1.some((v, i) => Math.abs(v - out2[i]) > 0.001)
    expect(differs).toBe(true)
  })

  it('backwardOne updates gamma and beta', () => {
    const norm = new LayerNorm(3)
    norm.resetCache(1)
    norm.predictOne([1, 2, 3], 0)

    const origGamma = [...norm.gamma]
    const origBeta = [...norm.beta]

    norm.backwardOne([0.1, 0.2, 0.3], 0, 0.01)

    // gamma and beta should have changed
    const gammaChanged = norm.gamma.some((g, i) => g !== origGamma[i])
    const betaChanged = norm.beta.some((b, i) => b !== origBeta[i])
    expect(gammaChanged).toBe(true)
    expect(betaChanged).toBe(true)
  })

  it('backwardOne returns gradient w.r.t. input', () => {
    const norm = new LayerNorm(3)
    norm.resetCache(1)
    norm.predictOne([1, 2, 3], 0)

    const dX = norm.backwardOne([1, 0, 0], 0, 0.01)
    expect(dX.length).toBe(3)
    expect(dX.every(v => isFinite(v))).toBe(true)
  })

  it('getWeights returns gamma and beta', () => {
    const norm = new LayerNorm(3)
    const w = norm.getWeights()
    expect(w.length).toBe(6) // 3 gamma + 3 beta
  })

  it('setWeights restores gamma and beta', () => {
    const norm = new LayerNorm(3)
    const w = norm.getWeights()
    const newW = w.map((v, i) => v + i * 0.1)
    norm.setWeights(newW)
    expect(norm.getWeights()).toEqual(newW)
  })
})
