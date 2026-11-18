import { beforeAll, describe, expect, it, vi } from 'vitest'
import { ModelSaver } from '../src/ModelSaver'
import { NetworkN } from '../src/NetworkN'
import { Network } from '../src/Network'
import { Adam, ClippedOptimizerFactory, Momentum, SGD } from '../src/optimizers'
import { elu, leakyRelu, linear, makeElu, makeLeakyRelu, relu, sigmoid, tanh } from '../src/activations'
import type { Activation } from '../src/activations'
import { parseSavedModel } from '../src/core/format2'
import { seedRandom } from './helpers/seed'

beforeAll(() => {
  // src/index.ts does this in the package; the test does not depend on it.
  ModelSaver.register('NetworkN', NetworkN)
  ModelSaver.register('Network', Network)
})

const X = [
  [0.3, -0.7, 1.1],
  [-2, 0.4, 0.9],
  [1.5, 1.5, -1.5],
]

function samePredictions(a: { predict(x: number[]): number[] }, b: { predict(x: number[]): number[] }): void {
  for (const x of X) {
    const pa = a.predict(x)
    const pb = b.predict(x)
    expect(pb.length).toBe(pa.length)
    pa.forEach((v, i) => expect(Object.is(v, pb[i])).toBe(true))
  }
}

function roundTrip(net: NetworkN): NetworkN {
  const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkN
  expect(loaded).toBeInstanceOf(NetworkN)
  expect(loaded.getWeights()).toEqual(net.getWeights())
  samePredictions(net, loaded)
  return loaded
}

describe('ModelSaver format 2: round trip', () => {
  it('save writes format, kind, config, length, weights and timestamp', () => {
    const net = new NetworkN([3, 4, 2])
    const data = JSON.parse(ModelSaver.save(net))
    expect(data.format).toBe(2)
    expect(data.kind).toBe('NetworkN')
    expect(data.length).toBe(net.getWeights().length)
    expect(data.weights).toEqual(net.getWeights())
    expect(typeof data.timestamp).toBe('number')
    expect(NetworkN.kind).toBe('NetworkN')
    expect(net.kind).toBe('NetworkN')
  })

  it.each([
    ['sigmoid', sigmoid], ['relu', relu], ['tanh', tanh], ['linear', linear],
    ['leakyRelu', leakyRelu], ['elu', elu],
  ])('keeps the activation %s', (_n, act) => {
    const net = new NetworkN([3, 5, 2], { activations: [act, act] })
    roundTrip(net)
  })

  it('keeps a different activation per layer', () => {
    const net = new NetworkN([3, 5, 4, 2], { activations: [relu, tanh, linear] })
    const loaded = roundTrip(net)
    expect(loaded.toConfig().activations).toEqual(['relu', 'tanh', 'linear'])
  })

  it('keeps alpha of makeLeakyRelu(0.2) and makeElu(0.7)', () => {
    const net = new NetworkN([3, 5, 2], { activations: [makeLeakyRelu(0.2), makeElu(0.7)] })
    const cfg = JSON.parse(ModelSaver.save(net)).config
    expect(cfg.activations).toEqual([{ name: 'leakyRelu', alpha: 0.2 }, { name: 'elu', alpha: 0.7 }])
    const loaded = roundTrip(net)
    // The negative side of the activation is where alpha matters.
    expect(loaded.layers[0].neurons[0].activation.fn(-2)).toBe(0.2 * -2)
    expect(loaded.layers[1].neurons[0].activation.fn(-1)).toBe(0.7 * (Math.exp(-1) - 1))
  })

  it('saves residual as a function evaluated per layer', () => {
    const net = new NetworkN([3, 3, 3, 2], { residual: i => i < 2, activations: [tanh, tanh, linear] })
    const cfg = JSON.parse(ModelSaver.save(net)).config
    expect(cfg.residual).toEqual([true, true, false])
    const loaded = roundTrip(net)
    expect(loaded.toConfig().residual).toEqual([true, true, false])
  })

  it('saves residual: true as one true per layer and the default as false', () => {
    expect(new NetworkN([3, 3, 3], { residual: true }).toConfig().residual).toEqual([true, true])
    expect(new NetworkN([3, 3, 3]).toConfig().residual).toEqual([false, false])
  })

  it('keeps dropoutRate and the loaded network still has dropout', () => {
    const net = new NetworkN([3, 6, 6, 2], { dropoutRate: 0.3 })
    const loaded = roundTrip(net)
    expect(loaded.toConfig().dropoutRate).toBe(0.3)
    // Training mode applies the mask: same seed, same output as the original.
    seedRandom(7)
    const a = net.predict(X[0], true)
    seedRandom(7)
    const b = loaded.predict(X[0], true)
    expect(b).toEqual(a)
    expect(a).not.toEqual(net.predict(X[0], false))
  })

  it.each([
    ['sgd', () => new SGD(), { name: 'sgd' }],
    ['momentum', () => new Momentum(0.8), { name: 'momentum', beta: 0.8 }],
    ['adam', () => new Adam(0.8, 0.99, 1e-7), { name: 'adam', beta1: 0.8, beta2: 0.99, epsilon: 1e-7 }],
    [
      'clip(adam)',
      ClippedOptimizerFactory(() => new Adam(), 0.5),
      { name: 'clip', clipValue: 0.5, inner: { name: 'adam', beta1: 0.9, beta2: 0.999, epsilon: 1e-8 } },
    ],
  ])('keeps the optimizer %s and the loaded network trains like a fresh one', (_n, factory, spec) => {
    const opts = { activations: [tanh, linear], optimizer: factory }
    const net = new NetworkN([3, 4, 2], opts)
    expect(net.toConfig().optimizer).toEqual(spec)
    // Some training first: its optimizer state is NOT saved (Q4).
    for (let i = 0; i < 5; i++) net.train(X[i % 3], [0.5, -0.5], 0.1)
    const loaded = roundTrip(net)
    expect(loaded.toConfig().optimizer).toEqual(spec)
    const fresh = new NetworkN([3, 4, 2], opts)
    fresh.setWeights(net.getWeights())
    for (let i = 0; i < 10; i++) {
      loaded.train(X[i % 3], [1, 0], 0.05)
      fresh.train(X[i % 3], [1, 0], 0.05)
    }
    expect(loaded.getWeights()).toEqual(fresh.getWeights())
  })

  it('the default optimizer is saved as sgd', () => {
    expect(new NetworkN([2, 2]).toConfig().optimizer).toEqual({ name: 'sgd' })
  })

  it('the loaded network keeps training and its loss goes down', () => {
    const net = new NetworkN([3, 6, 1], { activations: [tanh, linear], optimizer: () => new Adam() })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkN
    const first = loaded.train(X[0], [0.8], 0.05)
    let last = first
    for (let i = 0; i < 100; i++) last = loaded.train(X[0], [0.8], 0.05)
    expect(last).toBeLessThan(first)
  })

  it('Network round trip', () => {
    const net = new Network(3, 4, 2)
    expect(net.kind).toBe('Network')
    expect(Network.kind).toBe('Network')
    const data = JSON.parse(ModelSaver.save(net))
    expect(data.config).toEqual({ nInputs: 3, nHidden: 4, nOutputs: 2 })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as Network
    expect(loaded).toBeInstanceOf(Network)
    expect(loaded.getWeights()).toEqual(net.getWeights())
    samePredictions(net, loaded)
    expect(() => Network.fromConfig({ nInputs: 0, nHidden: 1, nOutputs: 1 })).toThrow(/Network.fromConfig/)
  })
})

describe('ModelSaver format 2: errors', () => {
  const net = new NetworkN([3, 4, 2])
  const good = () => JSON.parse(ModelSaver.save(net))

  function rejects(text: string, re: RegExp): void {
    const spy = vi.spyOn(NetworkN, 'fromConfig')
    expect(() => ModelSaver.load(text)).toThrow(re)
    spy.mockRestore()
  }

  it('invalid JSON and wrong shapes', () => {
    rejects('{nope', /^ModelSaver\.load: the text is not valid JSON/)
    rejects('[1]', /^ModelSaver\.load: invalid model data/)
    rejects(JSON.stringify({ ...good(), format: 1 }), /^ModelSaver\.load: unsupported format 1/)
    rejects(JSON.stringify({ ...good(), config: null }), /^ModelSaver\.load: config must be an object/)
    rejects(JSON.stringify({ ...good(), kind: 5 }), /^ModelSaver\.load: kind must be/)
  })

  it('unknown kind names the registered ones and creates nothing', () => {
    const spy = vi.spyOn(NetworkN, 'fromConfig')
    expect(() => ModelSaver.load(JSON.stringify({ ...good(), kind: 'Nope' }))).toThrow(
      /^ModelSaver\.load: unknown kind "Nope" \(registered: .*NetworkN/,
    )
    expect(spy).not.toHaveBeenCalled()
    spy.mockRestore()
  })

  it('length that does not match weights', () => {
    const d = good()
    d.length = d.weights.length + 1
    rejects(JSON.stringify(d), /^ModelSaver\.load: length \(\d+\) does not match weights/)
    delete d.length
    rejects(JSON.stringify(d), /^ModelSaver\.load: length/)
  })

  it('a different number of weights than the config needs', () => {
    const d = good()
    d.weights.pop()
    d.length = d.weights.length
    rejects(JSON.stringify(d), /^ModelSaver\.load: the NetworkN in the file expects \d+ weights/)
  })

  it('NaN saved as null', () => {
    const bad = new NetworkN([3, 4, 2])
    const w = bad.getWeights()
    w[3] = NaN
    bad.setWeights = () => {}
    bad.getWeights = () => w
    const text = ModelSaver.save(bad)
    rejects(text, /^ModelSaver\.load: weights\[3\] is not a finite number .*saved as null/)
  })

  it('invalid config pieces', () => {
    const mut = (f: (c: Record<string, unknown>) => void) => {
      const d = good()
      f(d.config)
      return JSON.stringify(d)
    }
    rejects(mut(c => { c.activations = ['relu'] }), /^ModelSaver\.load: invalid config for NetworkN: .*activations/)
    rejects(mut(c => { c.activations = ['nope', 'relu'] }), /unknown activation "nope"/)
    rejects(mut(c => { c.optimizer = { name: 'rms' } }), /unknown optimizer "rms"/)
    rejects(mut(c => { c.dropoutRate = 1 }), /dropoutRate/)
    rejects(mut(c => { c.residual = [true] }), /residual/)
    rejects(mut(c => { c.structure = [3] }), /structure/)
  })

  it('an activation without a name cannot be saved', () => {
    const anon: Activation = { fn: x => x, dfn: () => 1 }
    const custom = new NetworkN([3, 4, 2], { activations: [anon, anon] })
    expect(() => ModelSaver.save(custom)).toThrow(/activations\[0\]: the activation has no name/)
  })

  it('a custom optimizer needs allowCustomOptimizer; the load warns and uses SGD', () => {
    const optimizer = () => ({ step: (w: number, g: number, lr: number) => w + lr * g })
    const custom = new NetworkN([3, 4, 2], { optimizer })
    expect(() => ModelSaver.save(custom)).toThrow(/^ModelSaver\.save: .*custom optimizer.*allowCustomOptimizer/)
    const wrapped = new NetworkN([3, 4, 2], { optimizer: ClippedOptimizerFactory(optimizer, 1) })
    expect(() => ModelSaver.save(wrapped)).toThrow(/custom optimizer/)

    const text = ModelSaver.save(custom, { allowCustomOptimizer: true })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const loaded = ModelSaver.load(text) as NetworkN
    expect(warn).toHaveBeenCalledTimes(1)
    expect(warn.mock.calls[0][0]).toMatch(/^ModelSaver\.load: .*custom optimizer/)
    warn.mockRestore()
    expect(loaded.toConfig().optimizer).toEqual({ name: 'sgd' })
  })

  it('save needs kind and toConfig', () => {
    expect(() => ModelSaver.save({ getWeights: () => [] } as never)).toThrow(/^ModelSaver\.save:/)
  })

  it('NetworkN.fromConfig rejects before building anything', () => {
    expect(() => NetworkN.fromConfig(null)).toThrow(/^NetworkN\.fromConfig:/)
  })

  it('parseSavedModel is pure and validates length itself', () => {
    const d = good()
    expect(parseSavedModel(JSON.stringify(d), 'X').weights).toEqual(d.weights)
    d.length = 1
    expect(() => parseSavedModel(JSON.stringify(d), 'X')).toThrow(/^X: length \(1\)/)
  })
})

describe('ModelSaver.register', () => {
  class Toy {
    static readonly kind = 'Toy'
    get kind() { return Toy.kind }
    constructor(public k: number, public w: number[] = [0, 0]) {}
    toConfig() { return { k: this.k } }
    static fromConfig(c: unknown) { return new Toy((c as { k: number }).k) }
    getWeights() { return [...this.w] }
    setWeights(w: number[]) { this.w = [...w] }
  }

  it('saves and loads a class of its own', () => {
    expect(() => ModelSaver.load(ModelSaver.save(new Toy(1)))).toThrow(/unknown kind "Toy"/)
    ModelSaver.register('Toy', Toy)
    const toy = new Toy(5, [1.5, -2])
    const loaded = ModelSaver.load(ModelSaver.save(toy)) as Toy
    expect(loaded).toBeInstanceOf(Toy)
    expect(loaded.k).toBe(5)
    expect(loaded.getWeights()).toEqual([1.5, -2])
  })

  it('validates its arguments', () => {
    expect(() => ModelSaver.register('', Toy)).toThrow(/^ModelSaver\.register:/)
    expect(() => ModelSaver.register('Bad', {} as never)).toThrow(/fromConfig/)
  })
})

describe('fromJSON accepts format 2 and format 1 still loads', () => {
  it('format 2 into a model of the same kind and structure', () => {
    const a = new NetworkN([3, 4, 2])
    const b = new NetworkN([3, 4, 2])
    ModelSaver.fromJSON(b, ModelSaver.save(a))
    expect(b.getWeights()).toEqual(a.getWeights())
  })

  it('rejects another structure, another kind and bad data, leaving the model as it was', () => {
    const a = new NetworkN([3, 4, 2])
    const text = ModelSaver.save(a)
    const other = new NetworkN([3, 5, 2])
    const before = other.getWeights()
    expect(() => ModelSaver.fromJSON(other, text)).toThrow(/^ModelSaver\.fromJSON: file structure \[3,4,2\]/)
    expect(() => ModelSaver.fromJSON(new Network(3, 4, 2), text)).toThrow(/is a "NetworkN"/)
    expect(() => ModelSaver.fromJSON({ getWeights: () => [], setWeights() {} }, text)).toThrow(/class without kind/)
    const d = JSON.parse(text)
    d.length = 1
    expect(() => ModelSaver.fromJSON(new NetworkN([3, 4, 2]), JSON.stringify(d))).toThrow(/^ModelSaver\.fromJSON: length/)
    expect(other.getWeights()).toEqual(before)
  })

  it('format 1 files of the current tests still load', () => {
    const a = new NetworkN([2, 3, 1])
    const b = new NetworkN([2, 3, 1])
    ModelSaver.fromJSON(b, ModelSaver.toJSON(a))
    expect(b.getWeights()).toEqual(a.getWeights())
    const c = new NetworkN([2, 3, 1])
    ModelSaver.fromJSON(c, JSON.stringify({ weights: a.getWeights(), timestamp: 1 }))
    expect(c.getWeights()).toEqual(a.getWeights())
  })
})

describe('fromJSON with format 2 but no kind', () => {
  it('is not a saved model: unsupported format', () => {
    const net = new NetworkN([2, 3, 1])
    const d = JSON.parse(ModelSaver.toJSON(net))
    d.format = 2
    expect(() => ModelSaver.fromJSON(net, JSON.stringify(d))).toThrow(/unsupported format 2/)
  })
})
