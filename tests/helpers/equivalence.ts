import { expect } from 'vitest'
import { withSeed } from './seed'

/** Anything whose trainable state can be read as a flat array of numbers. */
export interface Weighted {
  getWeights(): number[]
}

/**
 * Element-by-element comparison of two weight vectors.
 * - relTol = 0: exact, using Object.is (so -0 and 0 differ, NaN equals NaN).
 * - relTol > 0: |a - b| <= relTol * max(1, |a|, |b|).
 */
export function expectSameWeights(a: number[], b: number[], relTol: number): void {
  expect(b.length, 'weight count').toBe(a.length)
  for (let i = 0; i < a.length; i++) {
    if (relTol === 0) {
      expect(Object.is(a[i], b[i]), `index ${i}: ${a[i]} vs ${b[i]}`).toBe(true)
    } else {
      const bound = relTol * Math.max(1, Math.abs(a[i]), Math.abs(b[i]))
      expect(Math.abs(a[i] - b[i]) <= bound, `index ${i}: ${a[i]} vs ${b[i]} (bound ${bound})`).toBe(true)
    }
  }
}

/**
 * Builds and trains two networks under the same seed (withSeed), so both see
 * the same Math.random stream, and returns the final weights of each.
 * `steps` is applied to each freshly built net in turn.
 */
export function runPair<A extends Weighted, B extends Weighted>(
  makeA: () => A,
  makeB: () => B,
  steps: (net: A | B) => void,
  seed: number,
): { a: number[]; b: number[] } {
  const a = withSeed(seed, () => {
    const net = makeA()
    steps(net)
    return net.getWeights()
  })
  const b = withSeed(seed, () => {
    const net = makeB()
    steps(net)
    return net.getWeights()
  })
  return { a, b }
}
