import { describe, it, expect, afterEach } from 'vitest'
import { loadForInference } from '../src/infer'
import { NetworkN } from '../src/NetworkN'
import { NetworkFast } from '../src/NetworkFast'
import { Network } from '../src/Network'
import { ModelSaver } from '../src/ModelSaver'
import { NetworkLSTM } from '../src/NetworkLSTM'
import { setBackend } from '../src/core/backend'
import { defaultBackend } from '../src/core/tensor'
import { sigmoid, relu, tanh, linear, leakyRelu, elu, makeLeakyRelu, makeElu } from '../src/activations'

afterEach(() => setBackend(null))

function close(a: number[], b: number[]) {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(b[i])))
  }
}

// Fresh nets start with bias 0; give every weight and bias a value so a dropped bias shows.
function withBias<T extends { getWeights(): number[]; setWeights(w: number[]): void }>(m: T): T {
  m.setWeights(m.getWeights().map((_, i) => Math.sin(i * 1.7 + 0.3) * 0.8))
  return m
}

const INPUTS = [[0.3, -0.7, 1.2], [-2, 0.5, 0.1], [0, 0, 0], [5, -5, 2]]

describe('infer: predictions equal model.predict', () => {
  const acts = [
    ['sigmoid', sigmoid], ['relu', relu], ['tanh', tanh], ['linear', linear],
    ['leakyRelu', leakyRelu], ['elu', elu], ['makeLeakyRelu', makeLeakyRelu(0.2)], ['makeElu', makeElu(0.7)],
  ] as const

  for (const [name, act] of acts) {
    it(`NetworkN with ${name}`, () => {
      const net = withBias(new NetworkN([3, 5, 4, 2], { activations: [act, act, linear] }))
      const m = loadForInference(ModelSaver.save(net))
      expect(m.kind).toBe('NetworkN')
      expect(m.structure).toEqual([3, 5, 4, 2])
      for (const x of INPUTS) close(m.predict(x), net.predict(x))
    })

    it(`NetworkFast with ${name} (saved as NetworkFast)`, () => {
      const fast = withBias(new NetworkFast([3, 5, 4, 2], { activations: [act, act, linear] }))
      // NetworkFast has the weight order of NetworkN; save through a twin NetworkN and relabel the kind.
      const twin = withBias(new NetworkN([3, 5, 4, 2], { activations: [act, act, linear] }))
      twin.setWeights(fast.getWeights())
      const json = JSON.parse(ModelSaver.save(twin))
      json.kind = 'NetworkFast'
      const m = loadForInference(JSON.stringify(json))
      expect(m.kind).toBe('NetworkFast')
      for (const x of INPUTS) close(m.predict(x), fast.predict(x))
    })
  }

  it('Network (fixed sigmoid, one hidden layer)', () => {
    const net = withBias(new Network(3, 4, 2))
    const m = loadForInference(ModelSaver.save(net))
    expect(m.kind).toBe('Network')
    expect(m.structure).toEqual([3, 4, 2])
    for (const x of INPUTS) close(m.predict(x), net.predict(x))
  })

  it('residual applies only where the layer keeps its width, as in NetworkN', () => {
    const net = withBias(new NetworkN([3, 3, 3, 2], { activations: [tanh, tanh, linear], residual: true }))
    const m = loadForInference(ModelSaver.save(net))
    for (const x of INPUTS) close(m.predict(x), net.predict(x))
    // without residual the output differs, so the flag matters
    const plain = new NetworkN([3, 3, 3, 2], { activations: [tanh, tanh, linear] })
    plain.setWeights(net.getWeights())
    expect(Math.abs(plain.predict(INPUTS[0])[0] - net.predict(INPUTS[0])[0])).toBeGreaterThan(1e-6)
  })

  it('dropoutRate does not change inference', () => {
    const net = withBias(new NetworkN([3, 6, 2], { dropoutRate: 0.5 }))
    const m = loadForInference(ModelSaver.save(net))
    for (const x of INPUTS) close(m.predict(x), net.predict(x))
  })
})

describe('infer: predictBatch', () => {
  it('equals predict row by row and model.predict', () => {
    const net = withBias(new NetworkN([3, 5, 2], { activations: [relu, sigmoid], residual: false }))
    const m = loadForInference(ModelSaver.save(net))
    const out = m.predictBatch(INPUTS)
    expect(out.length).toBe(INPUTS.length)
    out.forEach((row, i) => { close(row, net.predict(INPUTS[i])); close(row, m.predict(INPUTS[i])) })
    expect(m.predictBatch([])).toEqual([])
  })

  it('rejects a bad row before returning anything', () => {
    const m = loadForInference(ModelSaver.save(new NetworkN([3, 2])))
    expect(() => m.predictBatch([[1, 2, 3], [1, 2]])).toThrow(/row 1.*expected array of length 3/)
    expect(() => m.predictBatch([[1, 2, NaN]])).toThrow(/invalid value/)
    expect(() => m.predict([1, 2])).toThrow(/infer\.predict: expected array of length 3/)
  })
})

describe('infer: errors', () => {
  it('a non dense kind points to ModelSaver.load', () => {
    const lstm = new NetworkLSTM(2, 3, [1])
    expect(() => loadForInference(ModelSaver.save(lstm))).toThrow(/NetworkLSTM.*not supported in infer.*ModelSaver\.load/)
  })

  it('rejects bad JSON, a format 1 file and a wrong weight count', () => {
    expect(() => loadForInference('{nope')).toThrow(/loadForInference: the text is not valid JSON/)
    expect(() => loadForInference('{"format":1}')).toThrow(/unsupported format/)
    const j = JSON.parse(ModelSaver.save(new NetworkN([2, 2])))
    j.weights.pop()
    j.length = j.weights.length
    expect(() => loadForInference(JSON.stringify(j))).toThrow(/needs 6 weights, got 5/)
  })
})

describe('infer: backend', () => {
  function counting() {
    const calls = { matVec: 0, matMulBatch: 0 }
    return {
      calls,
      backend: {
        matVec: (...a: Parameters<typeof defaultBackend.matVec>) => { calls.matVec++; return defaultBackend.matVec(...a) },
        matMulBatch: (...a: Parameters<typeof defaultBackend.matMulBatch>) => { calls.matMulBatch++; return defaultBackend.matMulBatch(...a) },
      },
    }
  }

  it('uses the backend option', () => {
    const net = withBias(new NetworkN([3, 4, 2]))
    const c = counting()
    const m = loadForInference(ModelSaver.save(net), { backend: c.backend })
    m.predict(INPUTS[0])
    m.predictBatch(INPUTS)
    expect(c.calls).toEqual({ matVec: 2, matMulBatch: 2 })
  })

  it('uses the global backend, read at each call', () => {
    const net = withBias(new NetworkN([3, 4, 2]))
    const m = loadForInference(ModelSaver.save(net))
    const c = counting()
    setBackend(c.backend)
    m.predict(INPUTS[0])
    m.predictBatch(INPUTS)
    expect(c.calls).toEqual({ matVec: 2, matMulBatch: 2 })
  })

  it('the option wins over the global backend', () => {
    const net = withBias(new NetworkN([3, 2]))
    const g = counting()
    const o = counting()
    setBackend(g.backend)
    loadForInference(ModelSaver.save(net), { backend: o.backend }).predict(INPUTS[0])
    expect(o.calls.matVec).toBe(1)
    expect(g.calls.matVec).toBe(0)
  })

  it('rejects a backend operation that is not a function', () => {
    const json = ModelSaver.save(new NetworkN([2, 2]))
    expect(() => loadForInference(json, { backend: { matVec: 5 as never } })).toThrow(/matVec must be a function/)
  })
})

// ─── static check: the chain of imports of infer.ts ──────────────────────────
interface FsLike { readFileSync(p: URL, enc: string): string }
const FORBIDDEN = ['optimizers', 'random', 'NetworkN', 'NetworkFast', 'Network', 'ModelSaver', 'Trainer']

describe('infer: imports', () => {
  it('no module reachable from infer.ts is an optimizer, random, a model, ModelSaver or Trainer', async () => {
    const mod = 'node:fs'
    const fs = (await import(/* @vite-ignore */ mod)) as FsLike
    const seen = new Set<string>()
    const bad: string[] = []
    const visit = (file: URL, rel: string) => {
      if (seen.has(rel)) return
      seen.add(rel)
      const src = fs.readFileSync(file, 'utf8')
      for (const m of src.matchAll(/^\s*(?:import|export)\s[^'"]*?from\s+['"](\.[^'"]+)['"]/gm)) {
        const target = new URL(m[1] + '.ts', file)
        const name = m[1].slice(m[1].lastIndexOf('/') + 1)
        if (FORBIDDEN.includes(name)) bad.push(`${rel} -> ${m[1]}`)
        else visit(target, target.pathname.slice(target.pathname.indexOf('/src/') + 1))
      }
    }
    visit(new URL('../src/infer.ts', import.meta.url), 'src/infer.ts')
    expect(bad).toEqual([])
    // the walk reached the expected modules, so an empty result is not a broken regex
    for (const f of ['src/core/format2.ts', 'src/core/tensor.ts', 'src/core/backend.ts', 'src/activations.ts', 'src/Validation.ts']) {
      expect(seen.has(f)).toBe(true)
    }
  })
})
