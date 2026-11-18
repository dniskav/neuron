import { beforeAll, describe, expect, it } from 'vitest'
import { ModelSaver } from '../src/ModelSaver'
import { NetworkTransformer } from '../src/NetworkTransformer'
import { NetworkTransformerRL } from '../src/NetworkTransformerRL'

// src/index.ts registers the classes in the package; the test does not depend on it.
beforeAll(() => {
  ModelSaver.register('NetworkTransformer', NetworkTransformer)
  ModelSaver.register('NetworkTransformerRL', NetworkTransformerRL)
})

const TOKENS = [0, 1, 2, 3, 4, 1]

const SEQ = [
  [0.1, -0.2, 0.3],
  [0.5, 0.4, -0.6],
  [-0.7, 0.8, 0.9],
  [0.2, 0.1, -0.3],
]

function expectSameOutputs(a: number[], b: number[]): void {
  expect(b.length).toBe(a.length)
  a.forEach((v, i) => expect(Object.is(v, b[i])).toBe(true))
}

describe('NetworkTransformer format 2', () => {
  it('kind is NetworkTransformer on the class and on the instance', () => {
    const net = new NetworkTransformer(6)
    expect(NetworkTransformer.kind).toBe('NetworkTransformer')
    expect(net.kind).toBe('NetworkTransformer')
    expect(JSON.parse(ModelSaver.save(net)).kind).toBe('NetworkTransformer')
  })

  it('toConfig gives the effective defaults, not the options left out', () => {
    expect(new NetworkTransformer(6).toConfig()).toEqual({
      seqLen: 6, vocabSize: 10, d_model: 64, nHeads: 4, d_ff: 128, nBlocks: 4, nClasses: 9,
    })
  })

  it('toConfig keeps every explicit option, nHeads included', () => {
    const net = new NetworkTransformer(6, {
      vocabSize: 5, d_model: 8, nHeads: 2, d_ff: 16, nBlocks: 2, nClasses: 3,
    })
    expect(net.toConfig()).toEqual({
      seqLen: 6, vocabSize: 5, d_model: 8, nHeads: 2, d_ff: 16, nBlocks: 2, nClasses: 3,
    })
  })

  it('round trip with explicit options gives identical logits (Object.is)', () => {
    const net = new NetworkTransformer(6, {
      vocabSize: 5, d_model: 8, nHeads: 2, d_ff: 16, nBlocks: 2, nClasses: 3,
    })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkTransformer
    expect(loaded).toBeInstanceOf(NetworkTransformer)
    expect(loaded.getWeights()).toEqual(net.getWeights())
    expectSameOutputs(net.predict(TOKENS), loaded.predict(TOKENS))
  })

  it('round trip with default options gives identical logits (Object.is)', () => {
    const net = new NetworkTransformer(6)
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkTransformer
    expectSameOutputs(net.predict(TOKENS), loaded.predict(TOKENS))
  })

  it('the loaded model has the same architecture (nHeads, d_ff, nBlocks)', () => {
    const net = new NetworkTransformer(6, { d_model: 8, nHeads: 2, d_ff: 16, nBlocks: 3 })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkTransformer
    expect(loaded.blocks.length).toBe(3)
    expect(loaded.blocks[0].d_ff).toBe(16)
    expect(loaded.blocks[0].attn.heads.length).toBe(2)
    expect(loaded.toConfig()).toEqual(net.toConfig())
  })

  it('fromConfig rejects a config with a missing or invalid field', () => {
    const good = new NetworkTransformer(6, { d_model: 8, nHeads: 2, d_ff: 16, nBlocks: 1 }).toConfig()
    const { nHeads: _omit, ...withoutHeads } = good
    expect(() => NetworkTransformer.fromConfig(withoutHeads)).toThrow('NetworkTransformer.fromConfig')
    expect(() => NetworkTransformer.fromConfig({ ...good, nHeads: 0 })).toThrow('NetworkTransformer.fromConfig')
    expect(() => NetworkTransformer.fromConfig({ ...good, seqLen: 2.5 })).toThrow('NetworkTransformer.fromConfig')
    expect(() => NetworkTransformer.fromConfig(null)).toThrow('NetworkTransformer.fromConfig')
  })
})

describe('NetworkTransformerRL format 2', () => {
  it('kind is NetworkTransformerRL on the class and on the instance', () => {
    const net = new NetworkTransformerRL(4, 3)
    expect(NetworkTransformerRL.kind).toBe('NetworkTransformerRL')
    expect(net.kind).toBe('NetworkTransformerRL')
    expect(JSON.parse(ModelSaver.save(net)).kind).toBe('NetworkTransformerRL')
  })

  it('toConfig gives the effective defaults, pooling included', () => {
    expect(new NetworkTransformerRL(5, 3).toConfig()).toEqual({
      seqLen: 5, inputDim: 3, d_model: 32, nHeads: 2, d_ff: 64, nBlocks: 2, nActions: 2, pooling: 'weighted',
    })
  })

  it.each(['avg', 'max', 'last', 'weighted'] as const)(
    'round trip with pooling %s gives identical Q-values (Object.is)',
    (pooling) => {
      const net = new NetworkTransformerRL(4, 3, {
        d_model: 8, nHeads: 4, d_ff: 12, nBlocks: 1, nActions: 3, pooling,
      })
      expect(net.toConfig()).toEqual({
        seqLen: 4, inputDim: 3, d_model: 8, nHeads: 4, d_ff: 12, nBlocks: 1, nActions: 3, pooling,
      })
      const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkTransformerRL
      expect(loaded).toBeInstanceOf(NetworkTransformerRL)
      expect(loaded.getPoolingType()).toBe(pooling)
      expect(loaded.getWeights()).toEqual(net.getWeights())
      expectSameOutputs(net.predict(SEQ), loaded.predict(SEQ))
    },
  )

  it('round trip with default options gives identical Q-values (Object.is)', () => {
    const net = new NetworkTransformerRL(4, 3)
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkTransformerRL
    expectSameOutputs(net.predict(SEQ), loaded.predict(SEQ))
  })

  it('the loaded model keeps nHeads and d_ff', () => {
    const net = new NetworkTransformerRL(4, 3, { d_model: 8, nHeads: 4, d_ff: 12, nBlocks: 1 })
    const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkTransformerRL
    expect(loaded.blocks[0].attn.heads.length).toBe(4)
    expect(loaded.blocks[0].d_ff).toBe(12)
  })

  it('fromConfig rejects an unknown pooling and an invalid inputDim', () => {
    const good = new NetworkTransformerRL(4, 3).toConfig()
    expect(() => NetworkTransformerRL.fromConfig({ ...good, pooling: 'sum' })).toThrow('NetworkTransformerRL.fromConfig')
    expect(() => NetworkTransformerRL.fromConfig({ ...good, inputDim: 0 })).toThrow('NetworkTransformerRL.fromConfig')
  })
})
