import { describe, it, expect } from 'vitest'
import { expectLearns, LearnsError } from './learns'

/** Regresion y = a*x con gradiente descendente sobre MSE fijo. */
function regression(lr: number, a0 = 0) {
  const xs = [1, 2, 3]
  const ys = [2, 4, 6]
  let a = a0
  const loss = () => xs.reduce((s, x, i) => s + (a * x - ys[i]) ** 2, 0) / xs.length
  const step = () => {
    const g = xs.reduce((s, x, i) => s + 2 * (a * x - ys[i]) * x, 0) / xs.length
    a -= lr * g
  }
  return { loss, step }
}

describe('expectLearns', () => {
  it('acepta una regresion que converge', () => {
    const { loss, step } = regression(0.05)
    const r = expectLearns({ loss, step, steps: 50 })
    expect(r.final).toBeLessThan(0.5 * r.initial)
  })

  it('rechaza un modelo que no cambia', () => {
    const { loss } = regression(0.05)
    expect(() => expectLearns({ loss, step: () => {}, steps: 10 })).toThrow(LearnsError)
  })

  it('rechaza un modelo que diverge', () => {
    const { loss, step } = regression(1.0) // lr demasiado grande
    expect(() => expectLearns({ loss, step, steps: 20 })).toThrow(LearnsError)
  })

  it('rechaza una perdida NaN', () => {
    let calls = 0
    const loss = () => (calls++ === 0 ? 1 : NaN)
    expect(() => expectLearns({ loss, step: () => {}, steps: 3 })).toThrow(/no finita/)
    expect(() => expectLearns({ loss: () => NaN, step: () => {}, steps: 3 })).toThrow(LearnsError)
  })

  it('rechaza una bajada del 10 % con ratio 0.5', () => {
    let l = 1
    expect(() =>
      expectLearns({
        loss: () => l,
        step: () => {
          l *= 0.9
        },
        steps: 1
      })
    ).toThrow(LearnsError)
  })

  it('acepta esa misma bajada del 10 % con un ratio mas laxo', () => {
    let l = 1
    expect(() =>
      expectLearns({
        loss: () => l,
        step: () => {
          l *= 0.9
        },
        steps: 1,
        ratio: 0.95
      })
    ).not.toThrow()
  })

  it('llama a step exactamente `steps` veces y valida los argumentos', () => {
    let n = 0
    let l = 1
    expectLearns({
      loss: () => l,
      step: () => {
        n++
        l /= 10
      },
      steps: 4
    })
    expect(n).toBe(4)
    expect(() => expectLearns({ loss: () => 1, step: () => {}, steps: 0 })).toThrow(LearnsError)
  })
})
