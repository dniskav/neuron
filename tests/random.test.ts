import { describe, expect, it, afterEach } from 'vitest'
import { random, setRandom, getRandom, seededRandom, withRandom } from '../src/random'
import { mulberry32 } from './helpers/seed'

const originalRandom = Math.random

afterEach(() => {
  setRandom(null)
  Math.random = originalRandom
})

describe('random()', () => {
  it('uses Math.random at call time when no generator is set', () => {
    setRandom(null)
    Math.random = () => 0.25
    expect(random()).toBe(0.25)
    Math.random = () => 0.75
    expect(random()).toBe(0.75)
  })

  it('is the same as Math.random when called with the default generator', () => {
    Math.random = mulberry32(99)
    const expected = mulberry32(99)
    for (let i = 0; i < 5; i++) {
      expect(random()).toBe(expected())
    }
  })
})

describe('seededRandom', () => {
  it('produces the same sequence as mulberry32 from the test helper', () => {
    const fromSrc = seededRandom(42)
    const reference = mulberry32(42)
    for (let i = 0; i < 20; i++) {
      expect(fromSrc()).toBe(reference())
    }
  })

  it('returns values in [0, 1)', () => {
    const r = seededRandom(7)
    for (let i = 0; i < 1000; i++) {
      const v = r()
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })
})

describe('setRandom / getRandom', () => {
  it('getRandom returns null by default and the installed function after setRandom', () => {
    setRandom(null)
    expect(getRandom()).toBeNull()
    const fn = seededRandom(1)
    setRandom(fn)
    expect(getRandom()).toBe(fn)
  })

  it('setRandom(null) goes back to Math.random', () => {
    setRandom(seededRandom(3))
    setRandom(null)
    expect(getRandom()).toBeNull()
    Math.random = () => 0.5
    expect(random()).toBe(0.5)
  })

  it('random() uses the installed generator instead of Math.random', () => {
    Math.random = () => 0.9
    setRandom(seededRandom(5))
    const expected = seededRandom(5)
    expect(random()).toBe(expected())
  })
})

describe('withRandom', () => {
  it('runs body with the given generator and returns its value', () => {
    const out = withRandom(seededRandom(11), () => random())
    expect(out).toBe(seededRandom(11)())
  })

  it('restores the previous generator (null) after body returns', () => {
    setRandom(null)
    withRandom(seededRandom(11), () => random())
    expect(getRandom()).toBeNull()
  })

  it('restores the previous generator when body throws', () => {
    const previous = seededRandom(2)
    setRandom(previous)
    expect(() =>
      withRandom(seededRandom(11), () => {
        random()
        throw new Error('boom')
      }),
    ).toThrow('boom')
    expect(getRandom()).toBe(previous)
  })

  it('restores null when body throws and the previous generator was Math.random', () => {
    setRandom(null)
    expect(() =>
      withRandom(seededRandom(4), () => {
        throw new Error('fail')
      }),
    ).toThrow('fail')
    expect(getRandom()).toBeNull()
  })
})
