import { describe, it, expect, vi, afterEach } from 'vitest'
import { NetworkFast } from '../src/NetworkFast'
import { NetworkN, NetworkNOptions } from '../src/NetworkN'
import { sigmoid, relu, tanh, linear, leakyRelu, elu, Activation } from '../src/activations'
import { SGD, Momentum, Adam, ClipOptimizer, OptimizerFactory } from '../src/optimizers'
import { Dropout } from '../src/Dropout'
import { setBackend, getBackend } from '../src/core/backend'
import { defaultBackend, Backend } from '../src/core/tensor'
import { expectSameWeights, runPair } from './helpers/equivalence'
import { expectLearns } from './helpers/learns'
import { mulberry32 } from './helpers/seed'

afterEach(() => {
  setBackend(null)
  vi.restoreAllMocks()
})

type Net = NetworkN | NetworkFast

const ACTS: [string, Activation][] = [
  ['sigmoid', sigmoid], ['relu', relu], ['tanh', tanh],
  ['linear', linear], ['leakyRelu', leakyRelu], ['elu', elu],
]
const OPTS: [string, OptimizerFactory][] = [
  ['SGD', () => new SGD()],
  ['Momentum', () => new Momentum(0.9)],
  ['Adam', () => new Adam()],
  ['Clip(Adam)', () => new ClipOptimizer(new Adam(), 0.5)],
]
const STRUCTS = [[2, 3, 1], [4, 8, 8, 3], [5, 5, 5, 5]]

function opts(structure: number[], act: Activation, optimizer: OptimizerFactory,
  extra: Partial<NetworkNOptions> = {}): NetworkNOptions {
  return { activations: Array.from({ length: structure.length - 1 }, () => act), optimizer, ...extra }
}

describe('NetworkFast: initialisation identical to NetworkN (Object.is)', () => {
  const cases: [string, number[], NetworkNOptions][] = [
    ['[2,3,1] default', [2, 3, 1], {}],
    ['[4,8,8,3] relu', [4, 8, 8, 3], { activations: [relu, relu, sigmoid] }],
    ['[5,5,5,5] residual + dropout', [5, 5, 5, 5], { residual: true, dropoutRate: 0.3 }],
    ['[7,1] one layer', [7, 1], {}],
  ]
  for (const [name, s, o] of cases) {
    it(name, () => {
      const { a, b } = runPair(() => new NetworkN(s, o), () => new NetworkFast(s, o), () => {}, 123)
      expectSameWeights(a, b, 0)
      // biases are 0
      expect(b.length).toBe(a.length)
    })
  }

  it('consumes the same number of random() draws', () => {
    const next = (mk: () => Net) => {
      const r = mulberry32(5)
      let n = 0
      const spy = vi.spyOn(Math, 'random').mockImplementation(() => { n++; return r() })
      mk()
      spy.mockRestore()
      return n
    }
    expect(next(() => new NetworkFast([4, 8, 8, 3]))).toBe(next(() => new NetworkN([4, 8, 8, 3])))
  })
})

// ─── equivalence after 50 steps ────────────────────────────────────────────────
// One log per run: losses, outputs and the next random() value, compared too.
function data(nIn: number, nOut: number) {
  const r = mulberry32(99)
  const vec = (n: number) => Array.from({ length: n }, () => r() * 2 - 1)
  return { x: () => vec(nIn), t: () => vec(nOut).map(v => (v + 1) / 2), d: () => vec(nOut).map(v => v * 0.5) }
}

function pair(
  s: number[], o: NetworkNOptions, steps: (net: Net, log: number[]) => void, relTol = 1e-12, seed = 17,
) {
  const logs: number[][] = []
  const { a, b } = runPair(
    () => new NetworkN(s, o), () => new NetworkFast(s, o),
    net => { const log: number[] = []; steps(net, log); log.push(Math.random()); logs.push(log) },
    seed,
  )
  expectSameWeights(a, b, relTol)
  expectSameWeights(logs[0], logs[1], relTol)
  // the last logged value is the next random() draw: it must be identical
  expect(logs[1][logs[1].length - 1]).toBe(logs[0][logs[0].length - 1])
}

const trainSteps = (s: number[]) => (net: Net, log: number[]) => {
  const d = data(s[0], s[s.length - 1])
  for (let i = 0; i < 50; i++) log.push(net.train(d.x(), d.t(), 0.05))
}

describe('NetworkFast equals NetworkN after 50 train steps (<= 1e-12)', () => {
  for (const s of STRUCTS) {
    for (const [an, act] of ACTS) {
      for (const [on, opt] of OPTS) {
        it(`[${s}] ${an} ${on}`, () => pair(s, opts(s, act, opt), trainSteps(s)))
      }
    }
  }
})

describe('NetworkFast equals NetworkN: residual and dropout (train)', () => {
  const resid: [string, NetworkNOptions['residual']][] = [
    ['false', false], ['true', true], ['i===1', (i: number) => i === 1],
  ]
  for (const s of [[4, 8, 8, 3], [5, 5, 5, 5], [2, 3, 1]]) {
    for (const [rn, residual] of resid) {
      for (const dropoutRate of [0, 0.3]) {
        for (const [on, opt] of [OPTS[0], OPTS[2]]) {
          it(`[${s}] residual ${rn} dropout ${dropoutRate} ${on}`, () =>
            pair(s, opts(s, relu, opt, { residual, dropoutRate }), trainSteps(s)))
        }
      }
    }
  }
})

describe('NetworkFast equals NetworkN: trainWithDeltas and inputGradient', () => {
  const resid: [string, NetworkNOptions['residual']][] = [
    ['false', false], ['hidden only', (i: number) => i < 2],
  ]
  for (const s of [[4, 8, 8, 3], [5, 5, 5, 5]]) {
    for (const [rn, residual] of resid) {
      for (const dropoutRate of [0, 0.3]) {
        for (const [on, opt] of [OPTS[0], OPTS[2], OPTS[3]]) {
          const o = () => opts(s, tanh, opt, { residual, dropoutRate })

          it(`trainWithDeltas [${s}] residual ${rn} dropout ${dropoutRate} ${on}`, () =>
            pair(s, o(), (net, log) => {
              const d = data(s[0], s[s.length - 1])
              for (let i = 0; i < 50; i++) {
                net.trainWithDeltas(d.x(), d.d(), 0.05)
                log.push(...net.predict(d.x()))
              }
            }))

          it(`inputGradient [${s}] residual ${rn} dropout ${dropoutRate} ${on}`, () =>
            pair(s, o(), (net, log) => {
              const d = data(s[0], s[s.length - 1])
              for (let i = 0; i < 50; i++) {
                const x = d.x()
                const gx = net.inputGradient(x, d.d())
                log.push(...gx)
                net.trainWithDeltas(x.map((v, j) => v + 0.1 * gx[j]), d.d(), 0.05)
              }
            }))

          if (dropoutRate > 0) {
            it(`reuseMasks [${s}] residual ${rn} ${on}`, () =>
              pair(s, o(), (net, log) => {
                const d = data(s[0], s[s.length - 1])
                for (let i = 0; i < 50; i++) {
                  const x = d.x(), dl = d.d()
                  log.push(...net.predict(x, true))
                  log.push(...net.inputGradient(x, dl, true))
                  net.trainWithDeltas(x, dl, 0.05, true)
                  // the masks stay saved: a second reuse of the same mask must work
                  log.push(...net.inputGradient(x, dl, true))
                }
              }))
          }
        }
      }
    }
  }

  it('lr = 0 changes no weight', () => {
    const net = new NetworkFast([3, 4, 2], { optimizer: () => new Adam() })
    const w = net.getWeights()
    net.train([0.1, 0.2, 0.3], [1, 0], 0)
    net.trainWithDeltas([0.1, 0.2, 0.3], [0.1, 0.1], 0)
    expect(net.getWeights()).toEqual(w)
  })

  it('inputGradient does not change weights and is repeatable', () => {
    const net = new NetworkFast([3, 4, 2])
    const w = net.getWeights()
    const a = net.inputGradient([0.1, 0.2, 0.3], [0.4, -0.2])
    expect(net.inputGradient([0.1, 0.2, 0.3], [0.4, -0.2])).toEqual(a)
    expect(net.getWeights()).toEqual(w)
  })
})

describe('NetworkFast: get/setWeights and predict', () => {
  it('weights set in NetworkN predict identically in NetworkFast (and back)', () => {
    const s = [4, 8, 8, 3]
    const n = new NetworkN(s, { activations: [relu, tanh, sigmoid] })
    const f = new NetworkFast(s, { activations: [relu, tanh, sigmoid] })
    f.setWeights(n.getWeights())
    expect(f.getWeights()).toEqual(n.getWeights())
    expect(f.predict([0.1, -0.2, 0.3, 0.9])).toEqual(n.predict([0.1, -0.2, 0.3, 0.9]))
    const g = new NetworkN(s, { activations: [relu, tanh, sigmoid] })
    g.setWeights(f.getWeights())
    expect(g.getWeights()).toEqual(f.getWeights())
  })

  it('predict returns a fresh array each call', () => {
    const f = new NetworkFast([2, 3, 2])
    const a = f.predict([1, 2])
    const b = f.predict([3, 4])
    expect(a).not.toBe(b)
    expect(a).not.toEqual(b)
  })

  it('structure is exposed', () => {
    expect(new NetworkFast([2, 3, 1]).structure).toEqual([2, 3, 1])
  })
})

// ─── learns ────────────────────────────────────────────────────────────────────
describe('NetworkFast learns', () => {
  it('XOR with Adam', () => {
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    const T = [[0], [1], [1], [0]]
    const net = new NetworkFast([2, 6, 1], { activations: [tanh, sigmoid], optimizer: () => new Adam() })
    expectLearns({
      loss: () => X.reduce((s, x, i) => s + (net.predict(x)[0] - T[i][0]) ** 2, 0) / 4,
      step: () => { for (let i = 0; i < 4; i++) net.train(X[i], T[i], 0.05) },
      steps: 300,
    })
  })
})

// ─── validation (same cases and messages as NetworkN, with NetworkFast.) ───────
describe('NetworkFast validation', () => {
  const pattern = (n: number) => Array.from({ length: n }, (_, i) => 0.05 * ((i % 7) - 3))
  const patterned = (s: number[], o?: NetworkNOptions) => {
    const net = new NetworkFast(s, o)
    net.setWeights(pattern(net.getWeights().length))
    return net
  }

  it('constructor: activations count and dropout rate (same messages as NetworkN)', () => {
    expect(() => new NetworkFast([3, 8, 2], { activations: [] })).toThrow('Expected 2 activations, got 0')
    expect(() => new NetworkFast([4, 8, 2], { dropoutRate: -0.1 })).toThrow('Dropout rate must be in [0, 1), got -0.1')
    expect(() => new NetworkFast([4, 8, 2], { dropoutRate: 1 })).toThrow('Dropout rate must be in [0, 1), got 1')
  })

  it('constructor: structure must be at least two integers >= 1', () => {
    expect(() => new NetworkFast([3])).toThrow('NetworkFast: structure')
    expect(() => new NetworkFast([3, 0, 1])).toThrow('NetworkFast: structure')
    expect(() => new NetworkFast([3, 2.5, 1])).toThrow('NetworkFast: structure')
  })

  it('constructor: backend option must hold functions', () => {
    expect(() => new NetworkFast([2, 2], { backend: { matVec: 3 as unknown as Backend['matVec'] } }))
      .toThrow('NetworkFast: backend.matVec must be a function')
  })

  it('predict: wrong length, NaN, undefined', () => {
    const net = new NetworkFast([3, 4, 2])
    expect(() => net.predict([1, 2])).toThrow('NetworkFast.predict: expected array of length 3, got 2')
    expect(() => net.predict([1, NaN, 2])).toThrow('NetworkFast.predict: invalid value at index 1: NaN')
    expect(() => net.predict(undefined as unknown as number[])).toThrow('NetworkFast.predict: expected array, got undefined')
  })

  it('train: lengths and lr, weights intact', () => {
    const net = patterned([3, 4, 2])
    const before = net.getWeights()
    expect(() => net.train([1, 2], [1, 0], 0.1)).toThrow('NetworkFast.train: expected array of length 3, got 2')
    expect(() => net.train([1, 2, 3], [1], 0.1)).toThrow('NetworkFast.train: expected array of length 2, got 1')
    expect(() => net.train([1, 2, 3], [1, NaN], 0.1)).toThrow('NetworkFast.train: invalid value at index 1: NaN')
    for (const lr of [NaN, Infinity, -1]) {
      expect(() => net.train([1, 2, 3], [1, 0], lr)).toThrow('NetworkFast.train: lr must be a finite number >= 0')
      expect(() => net.trainWithDeltas([1, 2, 3], [1, 0], lr)).toThrow('NetworkFast.trainWithDeltas: lr must be a finite number >= 0')
    }
    expect(net.getWeights()).toEqual(before)
  })

  it('trainWithDeltas: inputs and deltas, weights intact, no random() drawn', () => {
    const method = 'NetworkFast.trainWithDeltas'
    const net = patterned([4, 8, 2], { dropoutRate: 0.5 })
    const before = net.getWeights()
    const spy = vi.spyOn(Math, 'random')
    expect(() => net.trainWithDeltas([1, 2, 3], [0.1, 0.2], 0.5)).toThrow(`${method}: expected array of length 4, got 3`)
    expect(() => net.trainWithDeltas([1, 2, 3, 4], [0.1], 0.5)).toThrow(`${method}: expected array of length 2, got 1`)
    expect(() => net.trainWithDeltas([1, 2, 3, 4], [0.1, NaN], 0.5)).toThrow(`${method}: invalid value at index 1: NaN`)
    expect(() => net.trainWithDeltas([1, 2, 3, 4], undefined as unknown as number[], 0.5)).toThrow(`${method}: expected array, got undefined`)
    expect(spy).not.toHaveBeenCalled()
    expect(net.getWeights()).toEqual(before)
    net.trainWithDeltas([1, 2, 3, 4], [0.1, 0.2], 0.5)
    expect(spy).toHaveBeenCalled()
  })

  it('inputGradient: lengths', () => {
    const net = patterned([4, 8, 2], { dropoutRate: 0.5 })
    const spy = vi.spyOn(Math, 'random')
    expect(() => net.inputGradient([1, 2, 3], [0.1, 0.2])).toThrow('NetworkFast.inputGradient: expected array of length 4, got 3')
    expect(() => net.inputGradient([1, 2, 3, 4], [0.1])).toThrow('NetworkFast.inputGradient: expected array of length 2, got 1')
    expect(spy).not.toHaveBeenCalled()
  })

  it('residual on the last layer: trainWithDeltas and inputGradient throw, weights intact; train works', () => {
    const net = new NetworkFast([3, 3, 3, 3], { residual: true })
    const before = net.getWeights()
    expect(() => net.trainWithDeltas([1, 2, 3], [0.1, 0.1, 0.1], 0.1)).toThrow(/NetworkFast\.trainWithDeltas: residual/)
    expect(() => net.inputGradient([1, 2, 3], [0.1, 0.1, 0.1])).toThrow(/NetworkFast\.inputGradient: residual/)
    expect(net.getWeights()).toEqual(before)
    expect(Number.isFinite(net.train([1, 2, 3], [0, 0, 1], 0.1))).toBe(true)
  })

  it('reuseMasks without saved masks throws; with dropout 0 it does not', () => {
    const net = new NetworkFast([4, 8, 2], { dropoutRate: 0.5 })
    expect(() => net.trainWithDeltas([1, 2, 3, 4], [0.1, 0.1], 0.1, true)).toThrow('NetworkFast.trainWithDeltas: reuseMasks = true needs the dropout masks')
    expect(() => net.inputGradient([1, 2, 3, 4], [0.1, 0.1], true)).toThrow('NetworkFast.inputGradient: reuseMasks = true needs the dropout masks')
    net.predict([1, 2, 3, 4], true)
    expect(() => net.inputGradient([1, 2, 3, 4], [0.1, 0.1], true)).not.toThrow()
    net.predict([1, 2, 3, 4], false) // inference clears the masks
    expect(() => net.inputGradient([1, 2, 3, 4], [0.1, 0.1], true)).toThrow('needs the dropout masks')
    net.predict([1, 2, 3, 4], true)
    net.getWeights() // clears them too
    expect(() => net.inputGradient([1, 2, 3, 4], [0.1, 0.1], true)).toThrow('needs the dropout masks')
    net.predict([1, 2, 3, 4], true)
    net.setWeights(net.getWeights())
    expect(() => net.inputGradient([1, 2, 3, 4], [0.1, 0.1], true)).toThrow('needs the dropout masks')

    const plain = new NetworkFast([4, 8, 2])
    expect(() => plain.trainWithDeltas([1, 2, 3, 4], [0.1, 0.1], 0.1, true)).not.toThrow()
  })

  describe.each([[[3, 4, 2], undefined], [[3, 4, 4, 2], { residual: true }], [[4, 8, 2], { dropoutRate: 0.5 }]] as
    [number[], NetworkNOptions | undefined][])('setWeights on [%j]', (s, o) => {
    const expected = s.slice(0, -1).reduce((t, n, i) => t + (n + 1) * s[i + 1], 0)
    const method = 'NetworkFast.setWeights'
    const rejected = (bad: unknown, msg: string) => {
      const net = patterned(s, o)
      const before = net.getWeights()
      expect(() => net.setWeights(bad as number[])).toThrow(msg)
      expect(net.getWeights()).toEqual(before)
    }
    it('length of getWeights', () => expect(new NetworkFast(s, o).getWeights().length).toBe(expected))
    it('round trip and shifted values', () => {
      const net = patterned(s, o)
      const w = net.getWeights()
      net.setWeights(net.getWeights())
      expect(net.getWeights()).toEqual(w)
      const sh = w.map(v => v + 1)
      net.setWeights(sh)
      expect(net.getWeights()).toEqual(sh)
    })
    it('short, long, NaN, Infinity, undefined, non-array', () => {
      const w = patterned(s, o).getWeights()
      rejected(w.slice(0, expected - 1), `${method}: expected array of length ${expected}, got ${expected - 1}`)
      rejected([...w, 0.5], `${method}: expected array of length ${expected}, got ${expected + 1}`)
      const nan = [...w]; nan[expected - 1] = NaN
      rejected(nan, `${method}: invalid value at index ${expected - 1}`)
      const inf = [...w]; inf[0] = Infinity
      rejected(inf, `${method}: invalid value at index 0`)
      rejected(undefined, `${method}: expected array, got undefined`)
      rejected({ length: expected }, `${method}: expected array, got object`)
    })
  })

  it('an invalid setWeights does not reset dropout masks', () => {
    const net = patterned([4, 8, 2], { dropoutRate: 0.5 })
    const w = net.getWeights()
    const spy = vi.spyOn(Dropout.prototype, 'resetMask')
    expect(() => net.setWeights([1, 2, 3])).toThrow('NetworkFast.setWeights: expected array of length 58, got 3')
    expect(spy).not.toHaveBeenCalled()
    net.setWeights(w)
    expect(spy).toHaveBeenCalledTimes(1)
  })
})

// ─── backend ───────────────────────────────────────────────────────────────────
describe('NetworkFast backend', () => {
  const spyBackend = () => {
    const calls: string[] = []
    const wrap = <K extends keyof Backend>(op: K): Backend[K] =>
      ((...args: unknown[]) => { calls.push(op); return (defaultBackend[op] as (...a: unknown[]) => unknown)(...args) }) as Backend[K]
    return { calls, be: { matVec: wrap('matVec'), matTVec: wrap('matTVec'), addOuter: wrap('addOuter') } as Partial<Backend> }
  }
  const run = (net: NetworkFast) => {
    net.predict([0.1, 0.2, 0.3])
    net.train([0.1, 0.2, 0.3], [1, 0], 0.1)
    net.inputGradient([0.1, 0.2, 0.3], [0.1, 0.1])
  }

  it('the backend option receives the calls and gives the same result', () => {
    const s = spyBackend()
    const a = new NetworkFast([3, 4, 2], { backend: s.be })
    const b = new NetworkFast([3, 4, 2])
    b.setWeights(a.getWeights())
    run(a); run(b)
    expect(new Set(s.calls)).toEqual(new Set(['matVec', 'matTVec', 'addOuter']))
    expect(a.getWeights()).toEqual(b.getWeights())
  })

  it('a global backend receives the calls (read at every public call)', () => {
    const net = new NetworkFast([3, 4, 2])
    const s = spyBackend()
    run(net)
    expect(s.calls).toEqual([])
    setBackend(s.be)
    run(net)
    expect(new Set(s.calls)).toEqual(new Set(['matVec', 'matTVec', 'addOuter']))
    const n = s.calls.length
    setBackend(null)
    run(net)
    expect(s.calls.length).toBe(n)
  })

  it('the option has priority over the global backend', () => {
    const g = spyBackend()
    const o = spyBackend()
    setBackend(g.be)
    run(new NetworkFast([3, 4, 2], { backend: o.be }))
    expect(g.calls).toEqual([])
    expect(o.calls.length).toBeGreaterThan(0)
  })

  it('operations missing in the option come from the global backend', () => {
    const g = spyBackend()
    const o = spyBackend()
    setBackend(g.be)
    run(new NetworkFast([3, 4, 2], { backend: { matVec: o.be.matVec } }))
    expect(new Set(o.calls)).toEqual(new Set(['matVec']))
    expect(new Set(g.calls)).toEqual(new Set(['matTVec', 'addOuter']))
  })

  it('calls matVec once per layer in predict', () => {
    const net = new NetworkFast([3, 4, 4, 2])
    let n = 0
    setBackend({ matVec: (...a) => { n++; return defaultBackend.matVec(...a) } })
    net.predict([1, 2, 3])
    expect(n).toBe(3)
  })

  it('a backend that adds 1 to every matVec output changes the result (backend really used)', () => {
    const bad: Partial<Backend> = { matVec: (W, r, c, x, b, out) => { defaultBackend.matVec(W, r, c, x, b, out); out[0] += 1; return out } }
    const a = new NetworkFast([3, 4, 2], { backend: bad })
    const b = new NetworkFast([3, 4, 2])
    expect(a.predict([1, 2, 3])).not.toEqual(b.predict([1, 2, 3]))
  })
})
