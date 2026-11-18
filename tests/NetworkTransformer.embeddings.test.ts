import { beforeAll, describe, expect, it } from 'vitest'
import { ModelSaver } from '../src/ModelSaver'
import { NetworkTransformer } from '../src/NetworkTransformer'

beforeAll(() => {
  ModelSaver.register('NetworkTransformer', NetworkTransformer)
})

const SEQ = 4
const OPTS = { vocabSize: 5, d_model: 8, nHeads: 2, d_ff: 16, nBlocks: 2, nClasses: 3 }
const TOKENS = [0, 1, 2, 3]
const TARGETS = Array.from({ length: SEQ * OPTS.nClasses }, (_, k) =>
  k % OPTS.nClasses === Math.floor(k / OPTS.nClasses) % OPTS.nClasses ? 1 : 0
)

function expectSame(a: number[], b: number[]): void {
  expect(b.length).toBe(a.length)
  a.forEach((v, i) => expect(Object.is(v, b[i])).toBe(true))
}

function trainSteps(net: NetworkTransformer, steps: number, lr = 0.05): number {
  let loss = 0
  for (let s = 0; s < steps; s++) loss = net.train(TOKENS, TARGETS, lr)
  return loss
}

// Snapshot of the non-embedding weights (blocks + output head).
function otherWeights(net: NetworkTransformer): number[] {
  const out: number[] = []
  for (const block of net.blocks) for (const v of block.getWeights()) out.push(v)
  for (const v of net.outputProj.getWeights()) out.push(v)
  for (const v of net.outputBias.getWeights()) out.push(v)
  return out
}

function changed(before: number[], after: number[]): boolean {
  return before.some((v, i) => !Object.is(v, after[i]))
}

describe('NetworkTransformer freezeEmbeddings', () => {
  it('freezeEmbeddings: true keeps tokenEmb and posEmb fixed over 20 steps, while the rest learns', () => {
    const net = new NetworkTransformer(SEQ, { ...OPTS, freezeEmbeddings: true })
    const tok0 = net.tokenEmb.getWeights()
    const pos0 = net.posEmb.getWeights()
    const other0 = otherWeights(net)
    trainSteps(net, 20)
    expectSame(tok0, net.tokenEmb.getWeights())
    expectSame(pos0, net.posEmb.getWeights())
    expect(changed(other0, otherWeights(net))).toBe(true)
  })

  it('{ token: true } freezes only the token table', () => {
    const net = new NetworkTransformer(SEQ, { ...OPTS, freezeEmbeddings: { token: true } })
    const tok0 = net.tokenEmb.getWeights()
    const pos0 = net.posEmb.getWeights()
    trainSteps(net, 20)
    expectSame(tok0, net.tokenEmb.getWeights())
    expect(changed(pos0, net.posEmb.getWeights())).toBe(true)
  })

  it('{ position: true } freezes only the position table', () => {
    const net = new NetworkTransformer(SEQ, { ...OPTS, freezeEmbeddings: { position: true } })
    const tok0 = net.tokenEmb.getWeights()
    const pos0 = net.posEmb.getWeights()
    trainSteps(net, 20)
    expectSame(pos0, net.posEmb.getWeights())
    expect(changed(tok0, net.tokenEmb.getWeights())).toBe(true)
  })

  it('freezeEmbeddings absent or false gives the same training as before the option existed (Object.is)', () => {
    const a = new NetworkTransformer(SEQ, OPTS)
    const b = new NetworkTransformer(SEQ, { ...OPTS, freezeEmbeddings: false })
    b.setWeights(a.getWeights())
    const lossA = trainSteps(a, 20)
    const lossB = trainSteps(b, 20)
    expect(Object.is(lossA, lossB)).toBe(true)
    expectSame(a.getWeights(), b.getWeights())
    expectSame(a.predict(TOKENS), b.predict(TOKENS))
  })

  it('without freezing, the embeddings do change during training', () => {
    const net = new NetworkTransformer(SEQ, OPTS)
    const tok0 = net.tokenEmb.getWeights()
    const pos0 = net.posEmb.getWeights()
    trainSteps(net, 20)
    expect(changed(tok0, net.tokenEmb.getWeights())).toBe(true)
    expect(changed(pos0, net.posEmb.getWeights())).toBe(true)
  })
})

describe('NetworkTransformer.setTokenEmbeddings', () => {
  const rows = (vocab: number, dim: number, base: number) =>
    Array.from({ length: vocab }, (_, i) =>
      Array.from({ length: dim }, (_, m) => base + i * 0.25 - m * 0.125)
    )

  it('writes the exact rows (Object.is) into tokenEmb', () => {
    const net = new NetworkTransformer(SEQ, OPTS)
    const r = rows(OPTS.vocabSize, OPTS.d_model, 0.3)
    net.setTokenEmbeddings(r)
    expectSame(r.flat(), net.tokenEmb.getWeights())
    expectSame(r[3], net.tokenEmb.get(3))
  })

  it('{ freeze: true } keeps the written token table through later training', () => {
    const net = new NetworkTransformer(SEQ, OPTS)
    const r = rows(OPTS.vocabSize, OPTS.d_model, -0.2)
    net.setTokenEmbeddings(r, { freeze: true })
    trainSteps(net, 20)
    expectSame(r.flat(), net.tokenEmb.getWeights())
  })

  it('rejects a wrong number of rows and leaves the table as it was', () => {
    const net = new NetworkTransformer(SEQ, OPTS)
    const before = net.tokenEmb.getWeights()
    const r = rows(OPTS.vocabSize - 1, OPTS.d_model, 0.1)
    expect(() => net.setTokenEmbeddings(r)).toThrow('NetworkTransformer.setTokenEmbeddings')
    expectSame(before, net.tokenEmb.getWeights())
  })

  it('rejects a row of the wrong width, and the message suggests NetworkTransformerRL and PCA', () => {
    const net = new NetworkTransformer(SEQ, OPTS)
    const r = rows(OPTS.vocabSize, OPTS.d_model + 2, 0.1)
    let message = ''
    try { net.setTokenEmbeddings(r) } catch (e) { message = (e as Error).message }
    expect(message).toContain('NetworkTransformer.setTokenEmbeddings')
    expect(message).toContain('NetworkTransformerRL')
    expect(message).toContain('PCA')
  })

  it('a bad value in row 3 writes nothing: the first rows are not written either', () => {
    const net = new NetworkTransformer(SEQ, OPTS)
    const before = net.tokenEmb.getWeights()
    const r = rows(OPTS.vocabSize, OPTS.d_model, 0.5)
    r[3][2] = Number.NaN
    expect(() => net.setTokenEmbeddings(r)).toThrow('NetworkTransformer.setTokenEmbeddings')
    expectSame(before, net.tokenEmb.getWeights())
  })
})

describe('NetworkTransformer freezeEmbeddings in format 2', () => {
  it('toConfig writes freezeEmbeddings with its effective value', () => {
    expect(new NetworkTransformer(SEQ, OPTS).toConfig().freezeEmbeddings).toBe(false)
    expect(new NetworkTransformer(SEQ, { ...OPTS, freezeEmbeddings: true }).toConfig().freezeEmbeddings).toBe(true)
    expect(new NetworkTransformer(SEQ, { ...OPTS, freezeEmbeddings: { token: true } }).toConfig().freezeEmbeddings)
      .toEqual({ token: true, position: false })
  })

  it.each([true, { token: true }] as const)(
    'round trip with freezeEmbeddings %j: the loaded model keeps the frozen table fixed',
    (fz) => {
      const net = new NetworkTransformer(SEQ, { ...OPTS, freezeEmbeddings: fz })
      const loaded = ModelSaver.load(ModelSaver.save(net)) as NetworkTransformer
      expect(loaded).toBeInstanceOf(NetworkTransformer)
      expect(loaded.toConfig().freezeEmbeddings).toEqual(net.toConfig().freezeEmbeddings)
      const tok0 = loaded.tokenEmb.getWeights()
      const pos0 = loaded.posEmb.getWeights()
      trainSteps(loaded, 20)
      expectSame(tok0, loaded.tokenEmb.getWeights())
      if (fz === true) expectSame(pos0, loaded.posEmb.getWeights())
      else expect(changed(pos0, loaded.posEmb.getWeights())).toBe(true)
    },
  )

  it('fromConfig rejects an invalid freezeEmbeddings', () => {
    const good = new NetworkTransformer(SEQ, OPTS).toConfig()
    expect(() => NetworkTransformer.fromConfig({ ...good, freezeEmbeddings: 'yes' })).toThrow('NetworkTransformer.fromConfig')
    expect(() => NetworkTransformer.fromConfig({ ...good, freezeEmbeddings: { token: 1 } })).toThrow('NetworkTransformer.fromConfig')
  })
})
