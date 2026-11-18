import { beforeAll, describe, expect, it, vi } from 'vitest'
import { ModelSaver } from '../src/ModelSaver'
import { NetworkLSTM } from '../src/NetworkLSTM'
import { Adam, ClippedOptimizerFactory, Momentum } from '../src/optimizers'
import { makeLeakyRelu, tanh } from '../src/activations'

beforeAll(() => {
  // src/index.ts does this in the package; the test does not depend on it.
  ModelSaver.register('NetworkLSTM', NetworkLSTM)
})

const STEPS = [
  [0.1, -0.2, 0.3],
  [0.5, 0.5, -0.4],
  [-1, 0.2, 0.7],
  [0.0, 0.9, -0.9],
  [0.3, -0.3, 0.1],
]

// One episode: resetState, one predict per step. Returns the outputs.
function episode(net: NetworkLSTM): number[][] {
  net.resetState()
  return STEPS.map(x => net.predict(x))
}

function trainOnce(net: NetworkLSTM): void {
  const outs = episode(net)
  net.train(outs.map(() => [0.5, -0.5]), 0.05)
}

function sameEpisodes(a: NetworkLSTM, b: NetworkLSTM): void {
  const pa = episode(a)
  const pb = episode(b)
  expect(pb.length).toBe(pa.length)
  pa.forEach((row, t) => row.forEach((v, i) => expect(Object.is(v, pb[t][i])).toBe(true)))
}

describe('NetworkLSTM format 2: round trip', () => {
  it('save and load give the same episode outputs with Object.is', () => {
    const net = new NetworkLSTM(3, 4, [2], { denseActivation: tanh, optimizer: () => new Adam() })
    trainOnce(net)
    trainOnce(net)
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkLSTM
    expect(loaded).toBeInstanceOf(NetworkLSTM)
    expect(loaded.getWeightsFlat()).toEqual(net.getWeightsFlat())
    sameEpisodes(net, loaded)
  })

  it('keeps the dense activation with its alpha and the optimizer', () => {
    const net = new NetworkLSTM(3, 4, [5, 2], {
      denseActivation: makeLeakyRelu(0.2),
      optimizer: () => new Momentum(0.8),
    })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkLSTM
    expect(loaded.toConfig()).toEqual(net.toConfig())
    expect(loaded.toConfig().denseActivation).toEqual({ name: 'leakyRelu', alpha: 0.2 })
    expect(loaded.toConfig().optimizer).toEqual({ name: 'momentum', beta: 0.8 })
    sameEpisodes(net, loaded)
  })

  it('keeps the clip(adam) optimizer', () => {
    const net = new NetworkLSTM(3, 4, [2], { optimizer: ClippedOptimizerFactory(() => new Adam(), 0.5) })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkLSTM
    expect(loaded.toConfig().optimizer).toEqual({
      name: 'clip', clipValue: 0.5, inner: { name: 'adam', beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
    })
  })

  it('the loaded network keeps training', () => {
    const net = new NetworkLSTM(3, 4, [2], { denseActivation: tanh })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkLSTM
    trainOnce(loaded)
    expect(loaded.getWeightsFlat()).not.toEqual(net.getWeightsFlat())
  })
})

describe('NetworkLSTM format 2: shape of the file', () => {
  it('toConfig has every field (hiddenSize included)', () => {
    const net = new NetworkLSTM(3, 4, [2, 5])
    expect(net.toConfig()).toEqual({
      inputSize: 3,
      hiddenSize: 4,
      denseStructure: [2, 5],
      denseActivation: 'sigmoid',
      optimizer: { name: 'sgd' },
    })
    expect(NetworkLSTM.kind).toBe('NetworkLSTM')
    expect(net.kind).toBe('NetworkLSTM')
  })

  it('save writes weights as a flat array, not the structured object', () => {
    const net = new NetworkLSTM(3, 4, [2])
    const data = JSON.parse(ModelSaver.save(net))
    expect(data.format).toBe(2)
    expect(data.kind).toBe('NetworkLSTM')
    expect(Array.isArray(data.weights)).toBe(true)
    expect(data.weights).toEqual(net.getWeightsFlat())
    expect(data.length).toBe(net.getWeightsFlat().length)
    expect(data.config).toEqual(net.toConfig())
  })
})

describe('NetworkLSTM format 2: errors', () => {
  const good = () => JSON.parse(ModelSaver.save(new NetworkLSTM(3, 4, [2])))

  it('fromConfig rejects bad configs', () => {
    expect(() => NetworkLSTM.fromConfig(null)).toThrow(/^NetworkLSTM\.fromConfig: config must be an object/)
    expect(() => NetworkLSTM.fromConfig({ ...good().config, hiddenSize: 0 })).toThrow(/^NetworkLSTM\.fromConfig: hiddenSize/)
    expect(() => NetworkLSTM.fromConfig({ ...good().config, denseStructure: [] })).toThrow(/denseStructure/)
    expect(() => NetworkLSTM.fromConfig({ ...good().config, denseActivation: 'nope' })).toThrow(/denseActivation/)
    expect(() => NetworkLSTM.fromConfig({ ...good().config, optimizer: { name: 'rms' } })).toThrow(/optimizer/)
  })

  it('load wraps a bad config with the ModelSaver prefix', () => {
    const d = good()
    d.config.hiddenSize = 0
    expect(() => ModelSaver.load(JSON.stringify(d))).toThrow(
      /^ModelSaver\.load: invalid config for NetworkLSTM: NetworkLSTM\.fromConfig: hiddenSize/,
    )
  })

  it('a different number of weights than the config needs', () => {
    const d = good()
    d.weights.pop()
    d.length = d.weights.length
    expect(() => ModelSaver.load(JSON.stringify(d))).toThrow(
      /^ModelSaver\.load: the NetworkLSTM in the file expects \d+ weights/,
    )
  })

  it('a custom optimizer needs allowCustomOptimizer; the load warns and uses SGD', () => {
    const custom = new NetworkLSTM(3, 4, [2], { optimizer: () => ({ step: (w: number, g: number, lr: number) => w + lr * g }) })
    expect(() => ModelSaver.save(custom)).toThrow(/custom optimizer.*allowCustomOptimizer/)
    const text = ModelSaver.save(custom, { allowCustomOptimizer: true })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const loaded = ModelSaver.load(text) as NetworkLSTM
    expect(warn).toHaveBeenCalledTimes(1)
    warn.mockRestore()
    expect(loaded.toConfig().optimizer).toEqual({ name: 'sgd' })
  })
})

describe('ModelSaver with a class that only has getWeights / setWeights', () => {
  class ArrayToy {
    static readonly kind = 'ArrayToy'
    get kind() { return ArrayToy.kind }
    constructor(public k: number, public w: number[] = [0, 0]) {}
    toConfig() { return { k: this.k } }
    static fromConfig(c: unknown) { return new ArrayToy((c as { k: number }).k) }
    getWeights(): number[] { return [...this.w] }
    setWeights(w: number[]): void { this.w = [...w] }
  }

  it('save and load still use getWeights / setWeights', () => {
    ModelSaver.register('ArrayToy', ArrayToy)
    const toy = new ArrayToy(3, [1.25, -4])
    const text = ModelSaver.save(toy)
    expect(JSON.parse(text).weights).toEqual([1.25, -4])
    const loaded = ModelSaver.load(text) as ArrayToy
    expect(loaded).toBeInstanceOf(ArrayToy)
    expect(loaded.k).toBe(3)
    expect(loaded.getWeights()).toEqual([1.25, -4])
  })
})
