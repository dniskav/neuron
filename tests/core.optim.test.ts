import { describe, it, expect } from 'vitest'
import {
  VecSGD,
  VecMomentum,
  VecAdam,
  VecClip,
  ScalarArrayOptimizer,
  vecFromFactory,
  type VecOptimizer,
} from '../src/core/optim'
import {
  SGD,
  Momentum,
  Adam,
  ClipOptimizer,
  ClippedOptimizerFactory,
  type Optimizer,
  type OptimizerFactory,
} from '../src/optimizers'
import { mulberry32 } from './helpers/seed'

const STEPS = 200

// Gradients with zeros, negatives and values larger than any clip.
function makeGrad(rand: () => number): number {
  const r = rand()
  if (r < 0.1) return 0
  if (r < 0.2) return (rand() - 0.5) * 20 // up to +-10, above clip 0.5
  return (rand() - 0.5) * 2
}

function expectIdentical(a: ArrayLike<number>, b: ArrayLike<number>, label: string): void {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) {
      throw new Error(`${label}: element ${i} differs: ${a[i]} vs ${b[i]}`)
    }
  }
}

// Runs the vector optimizer and N scalar instances side by side.
function compare(
  makeVec: (size: number) => VecOptimizer,
  makeScalar: () => Optimizer,
  size: number,
  seed: number,
  asFloat64 = false,
): void {
  const rand = mulberry32(seed)
  const vec = makeVec(size)
  const scalars = Array.from({ length: size }, makeScalar)
  const init = Array.from({ length: size }, () => rand() - 0.5)
  const pv: number[] | Float64Array = asFloat64 ? Float64Array.from(init) : init.slice()
  const ps = init.slice()
  for (let s = 0; s < STEPS; s++) {
    const g = Array.from({ length: size }, () => makeGrad(rand))
    const lr = 0.001 + rand() * 0.2
    vec.step(pv, asFloat64 ? Float64Array.from(g) : g, lr)
    for (let i = 0; i < size; i++) ps[i] = scalars[i].step(ps[i], g[i], lr)
    expectIdentical(pv, ps, `step ${s}`)
  }
}

const cases: Array<[string, (n: number) => VecOptimizer, () => Optimizer]> = [
  ['SGD', (n) => new VecSGD(n), () => new SGD()],
  ['Momentum 0.9', (n) => new VecMomentum(n), () => new Momentum()],
  ['Momentum 0.5', (n) => new VecMomentum(n, 0.5), () => new Momentum(0.5)],
  ['Adam default', (n) => new VecAdam(n), () => new Adam()],
  ['Adam 0.8/0.99/1e-6', (n) => new VecAdam(n, 0.8, 0.99, 1e-6), () => new Adam(0.8, 0.99, 1e-6)],
  [
    'Clip(Adam, 0.5)',
    (n) => new VecClip(new VecAdam(n), 0.5),
    () => new ClipOptimizer(new Adam(), 0.5),
  ],
]

describe('vector optimizers match scalar optimizers exactly', () => {
  for (const [name, makeVec, makeScalar] of cases) {
    for (const size of [1, 7, 300]) {
      it(`${name}, size ${size}, ${STEPS} steps (Object.is)`, () => {
        compare(makeVec, makeScalar, size, 1000 + size)
      })
    }
  }

  it('accepts Float64Array for params and grads', () => {
    for (const [, makeVec, makeScalar] of cases) compare(makeVec, makeScalar, 7, 5, true)
  })

  it('VecClip does not modify the caller gradients', () => {
    const v = new VecClip(new VecSGD(3), 0.5)
    const g = [5, -5, 0.1]
    v.step([0, 0, 0], g, 1)
    expect(g).toEqual([5, -5, 0.1])
  })

  it('rejects wrong lengths and bad sizes', () => {
    expect(() => new VecSGD(2).step([0], [0, 0], 1)).toThrow(/VecSGD\.step/)
    expect(() => new VecAdam(2).step([0, 0], [0], 1)).toThrow(/VecAdam\.step/)
    expect(() => new VecSGD(-1)).toThrow(/size/)
    expect(() => new VecMomentum(1.5)).toThrow(/size/)
  })
})

describe('vecFromFactory', () => {
  it('returns the right class for each built-in', () => {
    expect(vecFromFactory(() => new SGD(), 4)).toBeInstanceOf(VecSGD)
    expect(vecFromFactory(() => new Momentum(0.5), 4)).toBeInstanceOf(VecMomentum)
    expect(vecFromFactory(() => new Adam(), 4)).toBeInstanceOf(VecAdam)
    const clip = vecFromFactory(ClippedOptimizerFactory(() => new Adam(), 0.5), 4)
    expect(clip).toBeInstanceOf(VecClip)
    expect((clip as VecClip).inner).toBeInstanceOf(VecAdam)
    expect((clip as VecClip).clipValue).toBe(0.5)
    expect(clip.size).toBe(4)
  })

  it('copies hyperparameters', () => {
    const m = vecFromFactory(() => new Momentum(0.5), 3) as VecMomentum
    expect(m.beta).toBe(0.5)
    const a = vecFromFactory(() => new Adam(0.8, 0.99, 1e-6), 3) as VecAdam
    expect([a.beta1, a.beta2, a.epsilon]).toEqual([0.8, 0.99, 1e-6])
    // behaviour identical to scalar with the same non-default hyperparameters
    compare(
      (n) => vecFromFactory(() => new Adam(0.8, 0.99, 1e-6), n),
      () => new Adam(0.8, 0.99, 1e-6),
      7,
      42,
    )
    compare(
      (n) => vecFromFactory(ClippedOptimizerFactory(() => new Momentum(0.5), 0.5), n),
      () => new ClipOptimizer(new Momentum(0.5), 0.5),
      7,
      43,
    )
  })

  it('uses ScalarArrayOptimizer for a custom stateful optimizer, same result as scalar path', () => {
    class Custom implements Optimizer {
      private calls = 0
      step(w: number, g: number, lr: number): number {
        this.calls++
        return w + (lr * g) / this.calls
      }
    }
    const vec = vecFromFactory(() => new Custom(), 5)
    expect(vec).toBeInstanceOf(ScalarArrayOptimizer)
    compare((n) => vecFromFactory(() => new Custom(), n), () => new Custom(), 5, 9)
  })

  it('custom optimizer inside Clip falls back to ScalarArrayOptimizer', () => {
    class Custom implements Optimizer {
      step(w: number, g: number, lr: number): number {
        return w + lr * g * 2
      }
    }
    const f = ClippedOptimizerFactory(() => new Custom(), 0.5)
    expect(vecFromFactory(f, 3)).toBeInstanceOf(ScalarArrayOptimizer)
    compare((n) => vecFromFactory(f, n), () => f(), 3, 11)
  })

  it('ScalarArrayOptimizer calls the factory `size` times, in order', () => {
    let n = 0
    const order: number[] = []
    const factory: OptimizerFactory = () => {
      order.push(n++)
      return new SGD()
    }
    new ScalarArrayOptimizer(factory, 6)
    expect(order).toEqual([0, 1, 2, 3, 4, 5])
  })

  it('vecFromFactory calls a custom factory once more to inspect (size + 1)', () => {
    let n = 0
    class Custom implements Optimizer {
      step(w: number): number {
        return w
      }
    }
    vecFromFactory(() => (n++, new Custom()), 4)
    expect(n).toBe(5)
  })
})

// ─── T028: subclasses of the built-ins take the scalar path ───────────────────
// A subclass may override `step`. The vector classes do not call `step`, so a
// subclass must not be mapped to the vector class of its parent.
class Dbl extends SGD {
  step(w: number, g: number, lr: number): number {
    return w + 2 * lr * g
  }
}

class DblAdam extends Adam {
  step(w: number, g: number, lr: number): number {
    const moved = super.step(w, g, lr)
    return w + 2 * (moved - w)
  }
}

class DblClip extends ClipOptimizer {}

describe('vecFromFactory: subclasses of built-ins use the scalar path (T028)', () => {
  it('a subclass of SGD gives ScalarArrayOptimizer and keeps its own step', () => {
    const vec = vecFromFactory(() => new Dbl(), 4)
    expect(vec).toBeInstanceOf(ScalarArrayOptimizer)
    expect(vec).not.toBeInstanceOf(VecSGD)
    // the overridden step doubles the move: 2 * 0.1 * 1 = 0.2
    const p = [0, 0, 0, 0]
    vec.step(p, [1, 1, 1, 1], 0.1)
    expect(p).toEqual([0.2, 0.2, 0.2, 0.2])
  })

  it('a subclass of Adam gives ScalarArrayOptimizer and moves the weight double', () => {
    const vec = vecFromFactory(() => new DblAdam(), 3)
    expect(vec).toBeInstanceOf(ScalarArrayOptimizer)
    expect(vec).not.toBeInstanceOf(VecAdam)
    const p = [0, 0, 0]
    vec.step(p, [0.5, -0.5, 2], 0.01)
    // first Adam step moves each weight by about lr; doubled, about 2 * lr
    for (const v of p) expect(Math.abs(Math.abs(v) - 0.02)).toBeLessThan(1e-4)
  })

  it('subclasses match the scalar instances exactly (Object.is)', () => {
    compare((n) => vecFromFactory(() => new Dbl(), n), () => new Dbl(), 7, 61)
    compare((n) => vecFromFactory(() => new DblAdam(), n), () => new DblAdam(), 7, 62)
  })

  it('ClipOptimizer around a subclass takes the scalar path', () => {
    const f = ClippedOptimizerFactory(() => new Dbl(), 0.5)
    expect(vecFromFactory(f, 3)).toBeInstanceOf(ScalarArrayOptimizer)
    compare((n) => vecFromFactory(f, n), () => f(), 3, 63)
  })

  it('a subclass of ClipOptimizer takes the scalar path', () => {
    const f = () => new DblClip(new Adam(), 0.5)
    expect(vecFromFactory(f, 3)).toBeInstanceOf(ScalarArrayOptimizer)
    compare((n) => vecFromFactory(f, n), () => f(), 3, 64)
  })

  it('the built-in classes still get their vector classes', () => {
    expect(vecFromFactory(() => new SGD(), 2)).toBeInstanceOf(VecSGD)
    expect(vecFromFactory(() => new Adam(), 2)).toBeInstanceOf(VecAdam)
    expect(vecFromFactory(ClippedOptimizerFactory(() => new Adam(), 0.5), 2)).toBeInstanceOf(VecClip)
  })
})

// ─── T028 (follow-up): an own `step` on the instance is also not ignored ──────
describe('vecFromFactory: an own step on the instance takes the scalar path (T028)', () => {
  const ownStep = () =>
    Object.assign(new SGD(), { step: (w: number, g: number, lr: number) => w + 2 * lr * g })

  it('an instance with own step gives ScalarArrayOptimizer and moves the weight double', () => {
    const vec = vecFromFactory(ownStep, 3)
    expect(vec).toBeInstanceOf(ScalarArrayOptimizer)
    const p = [0, 0, 0]
    vec.step(p, [1, 1, 1], 0.1)
    expect(p).toEqual([0.2, 0.2, 0.2])
  })

  it('ClipOptimizer whose inner has an own step takes the scalar path', () => {
    const f = () => new ClipOptimizer(ownStep(), 0.5)
    expect(vecFromFactory(f, 3)).toBeInstanceOf(ScalarArrayOptimizer)
    compare((n) => vecFromFactory(f, n), () => f(), 3, 65)
  })
})
