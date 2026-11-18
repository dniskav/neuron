import { beforeAll, describe, expect, it, vi } from 'vitest'
import { ModelSaver } from '../src/ModelSaver'
import { NetworkFast } from '../src/NetworkFast'
import { NetworkN } from '../src/NetworkN'
import { Adam, ClippedOptimizerFactory, Momentum, SGD } from '../src/optimizers'
import { elu, leakyRelu, linear, makeElu, makeLeakyRelu, relu, sigmoid, tanh } from '../src/activations'
import type { Activation } from '../src/activations'
import { defaultBackend } from '../src/core/tensor'
import { seedRandom } from './helpers/seed'

beforeAll(() => {
  // src/index.ts does this in the package; the test does not depend on it.
  ModelSaver.register('NetworkFast', NetworkFast)
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

function roundTrip(net: NetworkFast): NetworkFast {
  const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkFast
  expect(loaded).toBeInstanceOf(NetworkFast)
  expect(loaded.getWeights()).toEqual(net.getWeights())
  samePredictions(net, loaded)
  return loaded
}

describe('NetworkFast format 2: kind and toConfig', () => {
  it('kind is NetworkFast on the class and on the instance', () => {
    const net = new NetworkFast([3, 4, 2])
    expect(NetworkFast.kind).toBe('NetworkFast')
    expect(net.kind).toBe('NetworkFast')
    expect(JSON.parse(ModelSaver.save(net)).kind).toBe('NetworkFast')
  })

  it('toConfig has exactly the NetworkN config keys and no backend', () => {
    const backend = { matVec: vi.fn(defaultBackend.matVec) }
    const net = new NetworkFast([3, 4, 2], { backend, dropoutRate: 0.2 })
    expect(Object.keys(net.toConfig()).sort()).toEqual(
      ['activations', 'dropoutRate', 'optimizer', 'residual', 'structure'],
    )
  })

  it('toConfig evaluates residual per layer and names the activations', () => {
    const net = new NetworkFast([3, 3, 3, 2], { residual: i => i < 2, activations: [tanh, tanh, linear] })
    const cfg = net.toConfig()
    expect(cfg.residual).toEqual([true, true, false])
    expect(cfg.activations).toEqual(['tanh', 'tanh', 'linear'])
    expect(cfg.structure).toEqual([3, 3, 3, 2])
  })

  it('toConfig equals NetworkN.toConfig for the same options', () => {
    const opts = {
      activations: [makeLeakyRelu(0.2), makeElu(0.7), linear],
      residual: true,
      dropoutRate: 0.3,
      optimizer: () => new Momentum(0.8),
    }
    expect(new NetworkFast([3, 3, 3, 2], opts).toConfig()).toEqual(new NetworkN([3, 3, 3, 2], opts).toConfig())
  })
})

describe('NetworkFast format 2: round trip', () => {
  it('save writes format, kind, config, length, weights and timestamp', () => {
    const net = new NetworkFast([3, 4, 2])
    const data = JSON.parse(ModelSaver.save(net))
    expect(data.format).toBe(2)
    expect(data.kind).toBe('NetworkFast')
    expect(data.length).toBe(net.getWeights().length)
    expect(data.weights).toEqual(net.getWeights())
    expect(typeof data.timestamp).toBe('number')
  })

  it.each([
    ['sigmoid', sigmoid], ['relu', relu], ['tanh', tanh], ['linear', linear],
    ['leakyRelu', leakyRelu], ['elu', elu],
  ])('keeps the activation %s', (_n, act) => {
    roundTrip(new NetworkFast([3, 5, 2], { activations: [act, act] }))
  })

  it('keeps alpha of makeLeakyRelu(0.2) and makeElu(0.7)', () => {
    const net = new NetworkFast([3, 5, 2], { activations: [makeLeakyRelu(0.2), makeElu(0.7)] })
    const cfg = JSON.parse(ModelSaver.save(net)).config
    expect(cfg.activations).toEqual([{ name: 'leakyRelu', alpha: 0.2 }, { name: 'elu', alpha: 0.7 }])
    const loaded = roundTrip(net)
    expect(loaded.toConfig().activations).toEqual([{ name: 'leakyRelu', alpha: 0.2 }, { name: 'elu', alpha: 0.7 }])
  })

  it('mixed activations, residual as a function and dropout', () => {
    const net = new NetworkFast([3, 6, 6, 2], {
      activations: [makeLeakyRelu(0.2), tanh, linear],
      residual: i => i === 1,
      dropoutRate: 0.3,
    })
    const loaded = roundTrip(net)
    expect(loaded.toConfig()).toEqual(net.toConfig())
    expect(loaded.toConfig().residual).toEqual([false, true, false])
  })

  it('keeps dropoutRate: training-mode output is the same with the same seed', () => {
    const net = new NetworkFast([3, 6, 6, 2], { dropoutRate: 0.3 })
    const loaded = roundTrip(net)
    expect(loaded.toConfig().dropoutRate).toBe(0.3)
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
    const net = new NetworkFast([3, 4, 2], opts)
    expect(net.toConfig().optimizer).toEqual(spec)
    for (let i = 0; i < 5; i++) net.train(X[i % 3], [0.5, -0.5], 0.1)
    const loaded = roundTrip(net)
    expect(loaded.toConfig().optimizer).toEqual(spec)
    const fresh = new NetworkFast([3, 4, 2], opts)
    fresh.setWeights(net.getWeights())
    for (let i = 0; i < 10; i++) {
      loaded.train(X[i % 3], [1, 0], 0.05)
      fresh.train(X[i % 3], [1, 0], 0.05)
    }
    expect(loaded.getWeights()).toEqual(fresh.getWeights())
  })

  it('the default optimizer is saved as sgd', () => {
    expect(new NetworkFast([2, 2]).toConfig().optimizer).toEqual({ name: 'sgd' })
  })

  it('the loaded network keeps training and its loss goes down', () => {
    const net = new NetworkFast([3, 6, 1], { activations: [tanh, linear], optimizer: () => new Adam() })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkFast
    const first = loaded.train(X[0], [0.8], 0.05)
    let last = first
    for (let i = 0; i < 100; i++) last = loaded.train(X[0], [0.8], 0.05)
    expect(last).toBeLessThan(first)
  })

  it('the backend option is not saved: the loaded network uses the global backend', () => {
    const matVec = vi.fn(defaultBackend.matVec)
    const net = new NetworkFast([3, 4, 2], { backend: { matVec } })
    net.predict(X[0])
    expect(matVec).toHaveBeenCalled()
    const json = ModelSaver.save(net)
    expect(JSON.parse(json).config).not.toHaveProperty('backend')
    const loaded = ModelSaver.load(json) as NetworkFast
    const calls = matVec.mock.calls.length
    loaded.predict(X[0])
    expect(matVec.mock.calls.length).toBe(calls)
  })
})

describe('NetworkFast format 2: cross-compatibility with NetworkN', () => {
  const inputs = (n: number) => Array.from({ length: 6 }, (_, i) =>
    Array.from({ length: n }, (_, j) => Math.sin(i + 2 * j + 0.5)))

  it.each([
    ['sigmoid, 3-5-2', [3, 5, 2], { activations: [sigmoid, sigmoid] }],
    ['relu/tanh/linear, residual, 4-8-8-3', [4, 8, 8, 3], {
      activations: [relu, tanh, linear], residual: true,
    }],
    ['leakyRelu(0.2) and elu(0.7), 2-3-1', [2, 3, 1], {
      activations: [makeLeakyRelu(0.2), makeElu(0.7)],
    }],
    ['adam and dropout (not used at predict), 3-6-6-2', [3, 6, 6, 2], {
      activations: [elu, relu, sigmoid], optimizer: () => new Adam(), dropoutRate: 0.25,
    }],
  ])('a NetworkN JSON loads into NetworkFast with the same predictions (%s)', (_n, s, opts) => {
    const netN = new NetworkN(s, opts)
    const data = JSON.parse(ModelSaver.save(netN))
    const fast = NetworkFast.fromConfig(data.config)
    fast.setWeights(data.weights)
    for (const x of inputs(s[0])) {
      const a = netN.predict(x)
      const b = fast.predict(x)
      expect(b.length).toBe(a.length)
      a.forEach((v, i) => expect(Math.abs(b[i] - v)).toBeLessThanOrEqual(1e-12))
    }
  })

  it('a NetworkFast JSON loads into NetworkN with the same predictions', () => {
    const fast = new NetworkFast([4, 8, 3], { activations: [tanh, linear], residual: false })
    const netN = NetworkN.fromConfig(JSON.parse(ModelSaver.save(fast)).config)
    netN.setWeights(fast.getWeights())
    for (const x of inputs(4)) {
      const a = fast.predict(x)
      const b = netN.predict(x)
      a.forEach((v, i) => expect(Math.abs(b[i] - v)).toBeLessThanOrEqual(1e-12))
    }
  })
})

describe('NetworkFast format 2: errors', () => {
  it('fromConfig rejects bad configs with its own name in the message', () => {
    expect(() => NetworkFast.fromConfig(null)).toThrow(/NetworkFast\.fromConfig/)
    expect(() => NetworkFast.fromConfig({
      structure: [2, 2], activations: ['sigmoid', 'sigmoid'], optimizer: { name: 'sgd' },
      residual: [false], dropoutRate: 0,
    })).toThrow(/NetworkFast\.fromConfig/)
    expect(() => NetworkFast.fromConfig({
      structure: [2, 2], activations: ['sigmoid'], optimizer: { name: 'sgd' },
      residual: [false], dropoutRate: 1,
    })).toThrow(/NetworkFast\.fromConfig/)
    expect(() => NetworkFast.fromConfig({
      structure: [2, 2], activations: ['sigmoid'], optimizer: { name: 'rmsprop' },
      residual: [false], dropoutRate: 0,
    })).toThrow(/NetworkFast\.fromConfig/)
  })

  it('toConfig names an activation without a name', () => {
    const anon: Activation = { fn: (x: number) => x, dfn: () => 1 }
    const net = new NetworkFast([2, 2], { activations: [anon] })
    expect(() => net.toConfig()).toThrow(/NetworkFast\.toConfig/)
    expect(() => ModelSaver.save(net)).toThrow(/NetworkFast\.toConfig/)
  })

  it('a custom optimizer needs allowCustomOptimizer; the load warns and uses SGD', () => {
    class MyOpt extends SGD {}
    const net = new NetworkFast([2, 2], { optimizer: () => new MyOpt() })
    expect(() => ModelSaver.save(net)).toThrow(/allowCustomOptimizer/)
    const json = ModelSaver.save(net, { allowCustomOptimizer: true })
    expect(JSON.parse(json).config.optimizer).toEqual({ name: 'custom' })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const loaded = ModelSaver.load(json) as NetworkFast
    expect(warn).toHaveBeenCalled()
    expect(loaded.toConfig().optimizer).toEqual({ name: 'sgd' })
    expect(loaded.getWeights()).toEqual(net.getWeights())
    warn.mockRestore()
  })
})
