import { afterEach, describe, expect, it } from 'vitest'
import { NeuronN } from '../../src/NeuronN'
import { mulberry32, restoreRandom, seedRandom, withSeed } from './seed'

const original = Math.random
const draw = (n: number) => Array.from({ length: n }, () => Math.random())

describe('seed helpers', () => {
  afterEach(restoreRandom)

  it('same seed gives the same sequence', () => {
    seedRandom(42)
    const a = draw(20)
    seedRandom(42)
    expect(draw(20)).toEqual(a)
    const g1 = mulberry32(7)
    const g2 = mulberry32(7)
    expect(Array.from({ length: 10 }, g1)).toEqual(Array.from({ length: 10 }, g2))
  })

  it('different seeds give different sequences', () => {
    seedRandom(1)
    const a = draw(20)
    seedRandom(2)
    expect(draw(20)).not.toEqual(a)
  })

  it('values lie in [0, 1)', () => {
    seedRandom()
    for (const v of draw(5000)) {
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })

  it('restoreRandom returns the original Math.random', () => {
    seedRandom(3)
    expect(Math.random).not.toBe(original)
    seedRandom(4)
    restoreRandom()
    expect(Math.random).toBe(original)
  })

  it('withSeed is deterministic and restores Math.random', () => {
    const before = Math.random
    const a = withSeed(9, () => draw(5))
    const b = withSeed(9, () => draw(5))
    expect(a).toEqual(b)
    expect(Math.random).toBe(before)
    expect(() =>
      withSeed(9, () => {
        throw new Error('x')
      })
    ).toThrow('x')
    expect(Math.random).toBe(before)
  })

  it('two NeuronN built after the same seed have identical weights', () => {
    seedRandom(123)
    const a = new NeuronN(3)
    seedRandom(123)
    const b = new NeuronN(3)
    expect(a.weights).toEqual(b.weights)
    expect(a.bias).toBe(b.bias)
  })
})
