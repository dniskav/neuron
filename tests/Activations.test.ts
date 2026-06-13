import { describe, it, expect } from 'vitest'
import { sigmoid, relu, tanh, linear, leakyRelu, elu, makeLeakyRelu, makeElu } from '../src/activations'

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
