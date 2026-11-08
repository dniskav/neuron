import { describe, it, expect } from 'vitest'
import {
  gradientOracle,
  expectGradientMatches,
  bindParams,
  withPlainSGD,
  OracleError,
  type OracleOptions,
} from './gradientOracle'
import { WeightMatrix } from '../../src/MatMul'
import { Adam } from '../../src/optimizers'

// Toy linear regression: y_hat = w . x, L = mean 0.5 (y_hat - y)^2
const X = [
  [1, 2, -1, 0.5],
  [0.5, -1, 2, 1],
  [-2, 1, 0.5, 3],
  [1.5, 0.5, 1, -2],
]
const Y = [1, -1, 2, 0.5]

function trueGrad(w: number[]): number[] {
  const g = [0, 0, 0, 0]
  X.forEach((x, k) => {
    const e = x.reduce((a, xi, i) => a + xi * w[i], 0) - Y[k]
    for (let i = 0; i < 4; i++) g[i] += (e * x[i]) / X.length
  })
  return g
}

function toy(
  gradFn: (w: number[]) => number[] = trueGrad,
  sign = 1,
): OracleOptions & { w: number[] } {
  const w = [0.3, -0.2, 0.5, 0.1]
  const loss = () =>
    X.reduce((s, x, k) => {
      const e = x.reduce((a, xi, i) => a + xi * w[i], 0) - Y[k]
      return s + 0.5 * e * e
    }, 0) / X.length
  return {
    w,
    getParams: () => [...w],
    setParams: (p) => p.forEach((v, i) => (w[i] = v)),
    loss,
    // convention: optimizers add lr * (-dL/dw)
    step: (lr) => {
      const g = gradFn(w)
      for (let i = 0; i < w.length; i++) w[i] += sign * lr * -g[i]
    },
    lr: 1e-3,
  }
}

describe('gradientOracle', () => {
  it('(a) correct gradient gives cosine > 0.9999 and ratio ~ 1', () => {
    const r = gradientOracle(toy())
    expect(r.cosine).toBeGreaterThan(0.9999)
    expect(r.ratio).toBeCloseTo(1, 3)
    expect(r.n).toBe(4)
    expect(() => expectGradientMatches(toy())).not.toThrow()
  })

  it('restores the parameters after running', () => {
    const o = toy()
    const before = [...o.w]
    gradientOracle(o)
    expect(o.w).toEqual(before)
  })

  it('(b) inverted sign gives cosine < -0.9999 and expectGradientMatches throws', () => {
    const r = gradientOracle(toy(trueGrad, -1))
    expect(r.cosine).toBeLessThan(-0.9999)
    expect(() => expectGradientMatches(toy(trueGrad, -1))).toThrow(OracleError)
  })

  it('(c) a model omitting one gradient term gives cosine < 0.999 and throws', () => {
    const missing = (w: number[]) => {
      const g = trueGrad(w)
      g[2] = 0
      return g
    }
    const r = gradientOracle(toy(missing))
    expect(r.cosine).toBeLessThan(0.999)
    expect(() => expectGradientMatches(toy(missing))).toThrow(OracleError)
  })

  it('(d) a null step throws OracleError ZERO_STEP', () => {
    const o = { ...toy(), step: () => {} }
    expect(() => gradientOracle(o)).toThrow(OracleError)
    expect(() => gradientOracle(o)).toThrow(/ZERO_STEP/)
  })

  it('throws on NaN loss and on fewer than 3 non-zero gradients', () => {
    expect(() => gradientOracle({ ...toy(), loss: () => NaN })).toThrow(/NAN/)
    const w = [1, 2, 3, 4]
    const flat: OracleOptions = {
      getParams: () => [...w],
      setParams: (p) => p.forEach((v, i) => (w[i] = v)),
      loss: () => 5 * w[0] * w[0],
      step: (lr) => {
        w[0] -= lr
      },
    }
    expect(() => gradientOracle(flat)).toThrow(/FEW_GRADIENTS/)
  })

  it('calls resetRng before every loss() and step()', () => {
    const o = toy()
    const log: string[] = []
    const loss = o.loss
    const step = o.step
    gradientOracle({
      ...o,
      loss: () => (log.push('loss'), loss()),
      step: (lr) => (log.push('step'), step(lr)),
      resetRng: () => log.push('reset'),
    })
    for (let i = 0; i < log.length; i++) if (log[i] !== 'reset') expect(log[i - 1]).toBe('reset')
    expect(log.filter((x) => x === 'reset').length).toBe(log.filter((x) => x !== 'reset').length)
  })

  it('samples deterministically when there are more than maxParams parameters', () => {
    const n = 1000
    const w = Array.from({ length: n }, (_, i) => 0.1 + (i % 7) * 0.05)
    const mk = (): OracleOptions => ({
      getParams: () => [...w],
      setParams: (p) => p.forEach((v, i) => (w[i] = v)),
      loss: () => w.reduce((s, v) => s + v * v, 0),
      step: (lr) => w.forEach((v, i) => (w[i] = v + lr * -2 * v)),
    })
    const a = gradientOracle(mk())
    const b = gradientOracle(mk())
    expect(a.n).toBe(400)
    expect(a).toEqual(b)
    expect(a.cosine).toBeGreaterThan(0.9999)
  })
})

describe('bindParams', () => {
  it('(e) reads and writes nested arrays in place', () => {
    const m = [
      [1, 2],
      [3, 4],
    ]
    const row0 = m[0]
    const v = [5, 6]
    const b = bindParams(m, v)
    expect(b.getParams()).toEqual([1, 2, 3, 4, 5, 6])
    b.setParams([10, 20, 30, 40, 50, 60])
    expect(m).toEqual([
      [10, 20],
      [30, 40],
    ])
    expect(m[0]).toBe(row0)
    expect(v).toEqual([50, 60])
    expect(() => b.setParams([1, 2])).toThrow(OracleError)
  })

  it('handles 3D tensors', () => {
    const t = [[[1], [2]], [[3], [4]]]
    const b = bindParams(t)
    expect(b.getParams()).toEqual([1, 2, 3, 4])
    b.setParams([4, 3, 2, 1])
    expect(t).toEqual([[[4], [3]], [[2], [1]]])
  })
})

describe('withPlainSGD', () => {
  it('(f) makes WeightMatrix.update move W exactly lr*g and restores Adam', () => {
    const before = Adam.prototype.step
    const wm = new WeightMatrix(2, 3)
    const w0 = wm.W.map((r) => [...r])
    const g = [
      [1, -2, 3],
      [0.5, 0, -4],
    ]
    const lr = 0.1
    withPlainSGD(() => wm.update(g, lr))
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 3; j++) expect(wm.W[i][j]).toBeCloseTo(w0[i][j] + lr * g[i][j], 12)
    expect(Adam.prototype.step).toBe(before)
  })

  it('restores Adam even if fn throws and returns fn result', () => {
    const before = Adam.prototype.step
    expect(() =>
      withPlainSGD(() => {
        throw new Error('boom')
      }),
    ).toThrow('boom')
    expect(Adam.prototype.step).toBe(before)
    expect(withPlainSGD(() => 7)).toBe(7)
  })
})
