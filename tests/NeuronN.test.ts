import { describe, it, expect } from 'vitest'
import { NeuronN } from '../src/NeuronN'
import { sigmoid, tanh } from '../src/activations'
import {
  Adam,
  ClippedOptimizerFactory,
  Momentum,
  SGD,
  defaultOptimizer,
  type Optimizer,
  type OptimizerFactory,
} from '../src/optimizers'
import { seededRandom } from '../src/random'
import { withSeed } from './helpers/seed'

describe('NeuronN', () => {
  it('predict returns a number', () => {
    const n = new NeuronN(3)
    const out = n.predict([1, 2, 3])
    expect(typeof out).toBe('number')
    expect(isFinite(out)).toBe(true)
  })

  it('predict with sigmoid returns between 0 and 1', () => {
    const n = new NeuronN(2, sigmoid)
    const out = n.predict([0.5, -0.5])
    expect(out).toBeGreaterThanOrEqual(0)
    expect(out).toBeLessThanOrEqual(1)
  })

  it('train updates weights', () => {
    const n = new NeuronN(2)
    const origW = [...n.weights]
    n.train([1, 2], 1.0, 0.1)
    // At least one weight should have changed
    const changed = n.weights.some((w, i) => w !== origW[i])
    expect(changed).toBe(true)
  })

  it('validates input length', () => {
    const n = new NeuronN(3)
    expect(() => n.predict([1, 2])).toThrow()
    expect(() => n.predict([1, 2, 3, 4])).toThrow()
  })

  it('validates input values', () => {
    const n = new NeuronN(2)
    expect(() => n.predict([NaN, 1])).toThrow()
    expect(() => n.predict([1, Infinity])).toThrow()
  })

  it('learns XOR-like pattern with single neuron (limited)', () => {
    const n = new NeuronN(2)
    // Train on (1,1)->1 pattern
    for (let i = 0; i < 500; i++) {
      n.train([1, 1], 1.0, 0.5)
    }
    const out = n.predict([1, 1])
    expect(out).toBeGreaterThan(0.5)
  })
})


// ─── T009: memory (vector optimizers) and exactness against NeuronN 0.4 ──────

// NeuronN as it was in 0.4 (one scalar optimizer per weight, `_update` builds a
// new array), pasted here as the reference. Weights are copied from the new
// neuron so the comparison does not depend on the initialization.
class NeuronN04 {
  weights: number[]
  bias = 0
  private _opts: Optimizer[]
  constructor(
    nInputs: number,
    readonly activation = sigmoid,
    optimizerFactory: OptimizerFactory = defaultOptimizer,
  ) {
    this.weights = new Array(nInputs).fill(0)
    this._opts = Array.from({ length: nInputs + 1 }, optimizerFactory)
  }
  predict(inputs: number[]): number {
    const sum = inputs.reduce((acc, e, i) => acc + e * this.weights[i], this.bias)
    return this.activation.fn(sum)
  }
  _update(weightGrads: number[], biasGrad: number, lr: number): void {
    this.weights = this.weights.map((w, i) => this._opts[i].step(w, weightGrads[i], lr))
    this.bias = this._opts[this.weights.length].step(this.bias, biasGrad, lr)
  }
  train(inputs: number[], target: number, lr: number): void {
    const prediction = this.predict(inputs)
    const error = target - prediction
    const grad = error * this.activation.dfn(prediction)
    this._update(inputs.map(inp => grad * inp), grad, lr)
  }
}

// Custom optimizer with state (not a built-in, so it takes the scalar fallback).
class Rms implements Optimizer {
  private s = 0
  step(w: number, g: number, lr: number): number {
    this.s = 0.9 * this.s + 0.1 * g * g
    return w + (lr * g) / (Math.sqrt(this.s) + 1e-8)
  }
}

const factories: Array<[string, OptimizerFactory]> = [
  ['SGD', () => new SGD()],
  ['Momentum', () => new Momentum(0.8)],
  ['Adam', () => new Adam()],
  ['Clip(Adam)', ClippedOptimizerFactory(() => new Adam(0.8, 0.99, 1e-7), 0.05)],
  ['custom Rms', () => new Rms()],
]

function pair(factory: OptimizerFactory, nIn: number) {
  const fresh = withSeed(7, () => new NeuronN(nIn, tanh, factory))
  const ref = new NeuronN04(nIn, tanh, factory)
  ref.weights = [...fresh.weights]
  return { fresh, ref }
}

function same(a: { weights: number[]; bias: number }, b: { weights: number[]; bias: number }) {
  expect(a.weights.length).toBe(b.weights.length)
  for (let i = 0; i < a.weights.length; i++) {
    expect(Object.is(a.weights[i], b.weights[i])).toBe(true)
  }
  expect(Object.is(a.bias, b.bias)).toBe(true)
}

describe('NeuronN 0.5: exactness against the 0.4 class', () => {
  for (const [name, factory] of factories) {
    it(`${name}: 300 train steps give identical weights and bias`, () => {
      const { fresh, ref } = pair(factory, 5)
      const rnd = seededRandom(11)
      for (let k = 0; k < 300; k++) {
        const x = Array.from({ length: 5 }, () => rnd() * 2 - 1)
        const t = rnd() * 2 - 1
        fresh.train(x, t, 0.05)
        ref.train(x, t, 0.05)
      }
      same(fresh, ref)
    })

    it(`${name}: 300 _update steps with random gradients are identical`, () => {
      const { fresh, ref } = pair(factory, 6)
      const rnd = seededRandom(23)
      for (let k = 0; k < 300; k++) {
        const g = Array.from({ length: 6 }, () => (rnd() - 0.5) * 0.4)
        const gb = (rnd() - 0.5) * 0.4
        fresh._update(g, gb, 0.01)
        ref._update(g, gb, 0.01)
      }
      same(fresh, ref)
    })
  }

  it('lr = 0 and NaN-free state: a no-op step leaves the neuron as it was', () => {
    const { fresh, ref } = pair(() => new Adam(), 3)
    fresh._update([0.1, 0.2, 0.3], 0.4, 0)
    ref._update([0.1, 0.2, 0.3], 0.4, 0)
    same(fresh, ref)
  })
})

describe('NeuronN 0.5: in-place update', () => {
  it('keeps the identity of the weights array after train and _update', () => {
    for (const [, factory] of factories) {
      const n = new NeuronN(4, sigmoid, factory)
      const arr = n.weights
      n.train([1, 2, 3, 4], 1, 0.1)
      n._update([0.1, 0.1, 0.1, 0.1], 0.1, 0.1)
      expect(n.weights).toBe(arr)
    }
  })
})

describe('NeuronN 0.5: order of calls to the optimizer factory', () => {
  it('the weights get the first nInputs optimizers and the bias the last one', () => {
    const created: Array<{ calls: number[] }> = []
    const factory: OptimizerFactory = () => {
      const rec = { calls: [] as number[] }
      created.push(rec)
      return {
        step(w: number, g: number, lr: number) {
          rec.calls.push(g)
          return w + lr * g
        },
      }
    }
    const n = new NeuronN(3, sigmoid, factory)
    n._update([10, 20, 30], 40, 0.1)
    // Instances that were never stepped (the one used to inspect the factory) do not count.
    const used = created.filter(c => c.calls.length > 0)
    expect(used.map(c => c.calls)).toEqual([[10], [20], [30], [40]])
    // Same order as the 0.4 neuron: created in order, the last one is the bias.
    expect(created.indexOf(used[3])).toBeGreaterThan(created.indexOf(used[2]))
  })

  it('a custom optimizer with state gives the same result as the reference (200 steps)', () => {
    const p = pair(() => new Rms(), 4)
    const r2 = seededRandom(5)
    for (let k = 0; k < 200; k++) {
      const x = [r2(), r2() - 0.5, r2() * 3, -r2()]
      const t = r2()
      p.fresh.train(x, t, 0.02)
      p.ref.train(x, t, 0.02)
    }
    same(p.fresh, p.ref)
  })
})

// ─── T028: subclasses of the built-ins move exactly as in 0.4 ─────────────────
class DblSGD extends SGD {
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

const subclassFactories: Array<[string, OptimizerFactory]> = [
  ['subclass of SGD', () => new DblSGD()],
  ['subclass of Adam', () => new DblAdam()],
  ['Clip(subclass of SGD)', ClippedOptimizerFactory(() => new DblSGD(), 0.05)],
]

describe('NeuronN 0.5: subclasses of built-ins are not replaced by the parent vector class', () => {
  for (const [name, factory] of subclassFactories) {
    it(`${name}: 300 train steps give identical weights and bias (0.4 reference)`, () => {
      const { fresh, ref } = pair(factory, 5)
      const rnd = seededRandom(31)
      for (let k = 0; k < 300; k++) {
        const x = Array.from({ length: 5 }, () => rnd() * 2 - 1)
        const t = rnd() * 2 - 1
        fresh.train(x, t, 0.05)
        ref.train(x, t, 0.05)
      }
      same(fresh, ref)
    })

    it(`${name}: 300 _update steps with random gradients are identical (0.4 reference)`, () => {
      const { fresh, ref } = pair(factory, 6)
      const rnd = seededRandom(37)
      for (let k = 0; k < 300; k++) {
        const g = Array.from({ length: 6 }, () => (rnd() - 0.5) * 0.4)
        const gb = (rnd() - 0.5) * 0.4
        fresh._update(g, gb, 0.01)
        ref._update(g, gb, 0.01)
      }
      same(fresh, ref)
    })
  }

  it('a subclass of SGD moves each weight twice as far as plain SGD does', () => {
    const plain = new NeuronN(3, sigmoid, () => new SGD())
    const sub = new NeuronN(3, sigmoid, () => new DblSGD())
    sub.weights = [...plain.weights]
    const w0 = [...plain.weights]
    plain._update([0.1, -0.2, 0.3], 0.4, 0.05)
    sub._update([0.1, -0.2, 0.3], 0.4, 0.05)
    for (let i = 0; i < 3; i++) {
      const movedPlain = plain.weights[i] - w0[i]
      const movedSub = sub.weights[i] - w0[i]
      expect(movedSub).toBeCloseTo(2 * movedPlain, 12)
    }
  })
})
