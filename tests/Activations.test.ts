import { describe, it, expect, expectTypeOf } from 'vitest'
import { sigmoid, relu, tanh, linear, leakyRelu, elu, makeLeakyRelu, makeElu, type Activation } from '../src/activations'
import { expectSameWeights, runPair, type Weighted } from './helpers/equivalence'

describe('sigmoid', () => {
  it('returns 0.5 for input 0', () => {
    expect(sigmoid.fn(0)).toBeCloseTo(0.5, 10)
  })

  it('returns ~1 for large positive input', () => {
    expect(sigmoid.fn(100)).toBeCloseTo(1, 5)
  })

  it('returns ~0 for large negative input', () => {
    expect(sigmoid.fn(-100)).toBeCloseTo(0, 5)
  })

  it('derivative is correct', () => {
    const x = 0.5
    const out = sigmoid.fn(x)
    const deriv = sigmoid.dfn(out)
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    expect(deriv).toBeCloseTo(out * (1 - out), 10)
  })
})

describe('relu', () => {
  it('returns 0 for negative input', () => {
    expect(relu.fn(-5)).toBe(0)
  })

  it('returns input for positive input', () => {
    expect(relu.fn(5)).toBe(5)
  })

  it('returns 0 for 0', () => {
    expect(relu.fn(0)).toBe(0)
  })

  it('derivative is 1 for positive output', () => {
    expect(relu.dfn(5)).toBe(1)
  })

  it('derivative is 0 for zero/negative output', () => {
    expect(relu.dfn(0)).toBe(0)
    expect(relu.dfn(-1)).toBe(0)
  })
})

describe('tanh', () => {
  it('returns 0 for input 0', () => {
    expect(tanh.fn(0)).toBeCloseTo(0, 10)
  })

  it('returns ~1 for large positive input', () => {
    expect(tanh.fn(100)).toBeCloseTo(1, 5)
  })

  it('returns ~-1 for large negative input', () => {
    expect(tanh.fn(-100)).toBeCloseTo(-1, 5)
  })

  it('derivative is correct', () => {
    const x = 0.5
    const out = tanh.fn(x)
    const deriv = tanh.dfn(out)
    expect(deriv).toBeCloseTo(1 - out * out, 10)
  })
})

describe('linear', () => {
  it('returns input unchanged', () => {
    expect(linear.fn(5)).toBe(5)
    expect(linear.fn(-3)).toBe(-3)
  })

  it('derivative is always 1', () => {
    expect(linear.dfn(100)).toBe(1)
    expect(linear.dfn(-100)).toBe(1)
  })
})

describe('leakyRelu', () => {
  it('returns input for positive', () => {
    expect(leakyRelu.fn(5)).toBe(5)
  })

  it('returns alpha * input for negative', () => {
    expect(leakyRelu.fn(-5)).toBeCloseTo(-0.05, 10)
  })

  it('derivative is 1 for positive', () => {
    expect(leakyRelu.dfn(5)).toBe(1)
  })

  it('derivative is alpha for negative', () => {
    expect(leakyRelu.dfn(-0.05)).toBeCloseTo(0.01, 10)
  })
})

describe('elu', () => {
  it('returns input for positive', () => {
    expect(elu.fn(5)).toBe(5)
  })

  it('returns alpha*(exp(x)-1) for negative', () => {
    const x = -1
    expect(elu.fn(x)).toBeCloseTo(Math.exp(x) - 1, 10)
  })

  it('derivative is 1 for positive', () => {
    expect(elu.dfn(5)).toBe(1)
  })

  it('derivative is out + alpha for negative', () => {
    const out = elu.fn(-1)
    expect(elu.dfn(out)).toBeCloseTo(out + 1, 10)
  })
})

describe('makeLeakyRelu', () => {
  it('creates custom alpha', () => {
    const custom = makeLeakyRelu(0.1)
    expect(custom.fn(-10)).toBeCloseTo(-1, 10)
    expect(custom.dfn(-1)).toBeCloseTo(0.1, 10)
  })
})

describe('makeElu', () => {
  it('creates custom alpha', () => {
    const custom = makeElu(2.0)
    expect(custom.fn(-100)).toBeCloseTo(-2, 5)
  })
})

describe('tanh stable for large |x| (T059)', () => {
  it.each([355, 400, 1000])('tanh.fn(%d) is exactly 1 (not NaN)', (x) => {
    expect(tanh.fn(x)).toBe(1)
    expect(tanh.fn(-x)).toBe(-1)
  })

  it('tanh.fn(354) and tanh.fn(-354) saturate', () => {
    expect(tanh.fn(354)).toBe(1)
    expect(tanh.fn(-354)).toBe(-1)
  })

  it('matches Math.tanh to 1e-12 across the pre-change range', () => {
    // Golden: values of the previous formula, fixed here.
    const GOLDEN: Array<[number, number]> = [
      [0.5, 0.46211715726000974],
      [1, 0.7615941559557649],
      [3, 0.9950547536867305],
      [-7, -0.9999983369439446],
      [100, 1],
      [-100, -1],
    ]
    for (const [x, y] of GOLDEN) expect(Math.abs(tanh.fn(x) - y)).toBeLessThan(1e-12)
    for (let x = -20; x <= 20; x += 0.5) expect(Math.abs(tanh.fn(x) - Math.tanh(x))).toBeLessThan(1e-12)
  })

  it('dfn at saturation is 0', () => {
    expect(tanh.dfn(tanh.fn(1000))).toBe(0)
  })
})

describe('activation name and alpha (T008)', () => {
  it('built-ins carry their literal name', () => {
    expect(sigmoid.name).toBe('sigmoid')
    expect(relu.name).toBe('relu')
    expect(tanh.name).toBe('tanh')
    expect(linear.name).toBe('linear')
    expect(leakyRelu.name).toBe('leakyRelu')
    expect(elu.name).toBe('elu')
  })

  it('built-in name is a readonly union of the built-in names, not string', () => {
    expectTypeOf(sigmoid.name).toEqualTypeOf<'sigmoid' | 'relu' | 'tanh' | 'linear' | 'leakyRelu' | 'elu'>()
    expectTypeOf(elu.name).toEqualTypeOf<'sigmoid' | 'relu' | 'tanh' | 'linear' | 'leakyRelu' | 'elu'>()
    expectTypeOf(makeElu(2).alpha).toEqualTypeOf<number>()
  })

  it('parametric built-ins expose alpha; the others do not', () => {
    expect(leakyRelu.alpha).toBe(0.01)
    expect(elu.alpha).toBe(1)
    expect(sigmoid.alpha).toBeUndefined()
    expect(relu.alpha).toBeUndefined()
  })

  it('factories expose the alpha they were built with', () => {
    expect(makeLeakyRelu(0.2).alpha).toBe(0.2)
    expect(makeElu(2).alpha).toBe(2)
    expect(makeLeakyRelu(0.2).name).toBe('leakyRelu')
    expect(makeElu(2).name).toBe('elu')
  })

  it('custom activations without name or alpha still satisfy Activation', () => {
    const own: Activation = { fn: (x) => x * 2, dfn: () => 2 }
    expect(own.name).toBeUndefined()
    expect(own.alpha).toBeUndefined()
    expect(own.fn(3)).toBe(6)
  })

  it('fn and dfn of the built-ins are unchanged by the new fields', () => {
    expect(leakyRelu.fn(-5)).toBeCloseTo(-0.05, 12)
    expect(leakyRelu.dfn(-0.05)).toBe(0.01)
    expect(elu.dfn(elu.fn(-1))).toBeCloseTo(elu.fn(-1) + 1, 12)
  })
})

describe('expectSameWeights helper (T008)', () => {
  it('accepts identical arrays with relTol 0 and with a tolerance', () => {
    expect(() => expectSameWeights([1, -2.5, 0], [1, -2.5, 0], 0)).not.toThrow()
    expect(() => expectSameWeights([1, 2], [1 + 1e-13, 2], 1e-12)).not.toThrow()
  })

  it('rejects arrays of equal length with different values (not a length-only check)', () => {
    expect(() => expectSameWeights([1, 2], [3, 4], 0)).toThrow()
    expect(() => expectSameWeights([1, 2], [1, 3], 1e-12)).toThrow()
  })

  it('rejects arrays of different length', () => {
    expect(() => expectSameWeights([1, 2], [1], 1e-12)).toThrow()
  })

  it('with relTol 0 uses Object.is: -0 and 0 differ, NaN equals NaN', () => {
    expect(() => expectSameWeights([0], [-0], 0)).toThrow()
    expect(() => expectSameWeights([Number.NaN], [Number.NaN], 0)).not.toThrow()
  })

  it('relTol is relative to max(1, |a|, |b|)', () => {
    expect(() => expectSameWeights([1e6], [1e6 + 1e-7], 1e-12)).not.toThrow()
    expect(() => expectSameWeights([1e6], [1e6 + 1], 1e-12)).toThrow()
  })
})

class Fake implements Weighted {
  w: number[]
  constructor() {
    this.w = [Math.random(), Math.random()]
  }
  getWeights(): number[] {
    return [...this.w]
  }
  bump(): void {
    this.w = this.w.map((v) => v + 1)
  }
}

describe('runPair helper (T008)', () => {
  it('builds and steps both nets with the same seed, so the weights match', () => {
    const { a, b } = runPair(() => new Fake(), () => new Fake(), (n) => (n as Fake).bump(), 7)
    expectSameWeights(a, b, 0)
    expect(a).toHaveLength(2)
  })

  it('is deterministic for a given seed and depends on the seed', () => {
    const r1 = runPair(() => new Fake(), () => new Fake(), () => {}, 7)
    const r2 = runPair(() => new Fake(), () => new Fake(), () => {}, 7)
    const r3 = runPair(() => new Fake(), () => new Fake(), () => {}, 8)
    expectSameWeights(r1.a, r2.a, 0)
    expect(() => expectSameWeights(r1.a, r3.a, 0)).toThrow()
  })

  it('detects a pair that is not equivalent', () => {
    const { a, b } = runPair(() => new Fake(), () => new Fake(), (n) => (n as Fake).bump(), 7)
    const other = runPair(() => new Fake(), () => new Fake(), () => {}, 7)
    expect(() => expectSameWeights(a, other.b, 0)).toThrow()
    expect(b).toHaveLength(2)
  })

  it('restores Math.random afterwards', () => {
    const before = Math.random
    runPair(() => new Fake(), () => new Fake(), () => {}, 7)
    expect(Math.random).toBe(before)
  })
})
