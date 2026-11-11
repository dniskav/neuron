// T018: `lr` must be a finite number >= 0 in the core networks.
//
// Rule: NaN, +-Infinity, negative and non-numeric `lr` throw
// `Class.method: lr must be a finite number >= 0, got X` BEFORE any weight, state,
// trajectory or cache is touched. `lr = 0` stays valid (a no-op for SGD).
//
// Perceptron.train is not in this table on purpose: it already rejects NaN,
// Infinity, negatives AND 0 (`learning rate must be positive`), which is stricter
// than the rule; its behavior is covered in tests/Perceptron.test.ts.
//
// Golden: with a valid `lr` the losses and weights are the ones captured on the
// base commit (6f20f36), tolerance 1e-12.

import { describe, it, expect } from 'vitest'
import { NetworkN } from '../../src/NetworkN'
import { Network } from '../../src/Network'
import { Neuron } from '../../src/Neuron'
import { NeuronN } from '../../src/NeuronN'
import { NetworkLSTM } from '../../src/NetworkLSTM'
import { LSTMLayer } from '../../src/LSTMLayer'
import { GRULayer } from '../../src/GRU'
import { NetworkTransformer } from '../../src/NetworkTransformer'
import { NetworkTransformerRL } from '../../src/NetworkTransformerRL'

interface Case {
  method: string
  make: () => any
  call: (s: any, lr: any) => unknown
  // Everything the call could touch: weights plus stored trajectory length.
  state: (s: any) => number[]
  // Weights only (what `lr = 0` must leave identical).
  weights: (s: any) => number[]
}

const seq = [[0.1, 0.2], [0.3, -0.1], [0.2, 0.4]]
const onehot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
const dh = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]

const makeTransformer = () =>
  new NetworkTransformer(3, { vocabSize: 4, d_model: 4, nHeads: 2, d_ff: 4, nBlocks: 1, nClasses: 3 })
const makeTransformerRL = () =>
  new NetworkTransformerRL(3, 2, { d_model: 4, nHeads: 2, d_ff: 4, nBlocks: 1, nActions: 2 })
const makeLSTMNet = () => {
  const n = new NetworkLSTM(2, 3, [1])
  n.resetState()
  n.predict([0.1, 0.2])
  n.predict([0.3, 0.1])
  return n
}
const makeLSTMLayer = () => {
  const l = new LSTMLayer(2, 3)
  l.predict([0.1, 0.2])
  l.predict([0.3, 0.1])
  return l
}
const makeGRU = () => {
  const g = new GRULayer(2, 3)
  g.predict([0.1, 0.2])
  g.predict([0.3, 0.1])
  return g
}

const cases: Case[] = [
  {
    method: 'NetworkN.train',
    make: () => new NetworkN([2, 3, 1]),
    call: (n, lr) => n.train([0, 1], [1], lr),
    state: n => n.getWeights(),
    weights: n => n.getWeights(),
  },
  {
    method: 'NetworkN.trainWithDeltas',
    make: () => new NetworkN([2, 3, 1]),
    call: (n, lr) => n.trainWithDeltas([0, 1], [0.5], lr),
    state: n => n.getWeights(),
    weights: n => n.getWeights(),
  },
  {
    method: 'Network.train',
    make: () => new Network(2, 3, 1),
    call: (n, lr) => n.train([0, 1], 1, lr),
    state: n => n.getWeights(),
    weights: n => n.getWeights(),
  },
  {
    method: 'Neuron.train',
    make: () => new Neuron(),
    call: (n, lr) => n.train(0.5, 1, lr),
    state: n => [n.weight, n.bias],
    weights: n => [n.weight, n.bias],
  },
  {
    method: 'NeuronN.train',
    make: () => new NeuronN(2),
    call: (n, lr) => n.train([0, 1], 1, lr),
    state: n => [...n.weights, n.bias],
    weights: n => [...n.weights, n.bias],
  },
  {
    method: 'NetworkLSTM.train',
    make: makeLSTMNet,
    call: (n, lr) => n.train([[1], [0]], lr),
    state: n => [...n.getWeightsFlat(), n.lstm._traj.length, n._acts.length],
    weights: n => n.getWeightsFlat(),
  },
  {
    method: 'LSTMLayer.backprop',
    make: makeLSTMLayer,
    call: (l, lr) => l.backprop(dh, lr),
    state: l => [...l.getWeightsFlat(), l._traj.length],
    weights: l => l.getWeightsFlat(),
  },
  {
    method: 'GRULayer.backprop',
    make: makeGRU,
    call: (g, lr) => g.backprop(dh, lr),
    state: g => [...g.getWeightsFlat(), g._traj.length],
    weights: g => g.getWeightsFlat(),
  },
  {
    method: 'NetworkTransformer.train',
    make: makeTransformer,
    call: (n, lr) => n.train([0, 1, 2], onehot, lr),
    state: n => n.getWeights(),
    weights: n => n.getWeights(),
  },
  {
    method: 'NetworkTransformerRL.train',
    make: makeTransformerRL,
    call: (n, lr) => n.train(seq, [1, 0], lr),
    state: n => n.getWeights(),
    weights: n => n.getWeights(),
  },
]

const BAD: [string, number][] = [
  ['NaN', NaN],
  ['Infinity', Infinity],
  ['-Infinity', -Infinity],
  ['-1', -1],
]

describe('T018: lr must be a finite number >= 0', () => {
  for (const c of cases) {
    describe(c.method, () => {
      for (const [label, lr] of BAD) {
        it(`lr = ${label} throws with the method prefix and leaves the state intact`, () => {
          const s = c.make()
          const before = c.state(s)
          expect(() => c.call(s, lr)).toThrow(
            new RegExp(`^${c.method.replace('.', '\\.')}: lr must be a finite number >= 0, got `),
          )
          expect(c.state(s)).toEqual(before)
        })
      }

      it('non-numeric lr (string, null, undefined) throws and leaves the state intact', () => {
        for (const lr of ['0.1', null, undefined]) {
          const s = c.make()
          const before = c.state(s)
          expect(() => c.call(s, lr)).toThrow(`${c.method}: lr must be a finite number >= 0`)
          expect(c.state(s)).toEqual(before)
        }
      })

      it('lr = 0 does not throw and does not change the weights', () => {
        const s = c.make()
        const before = c.weights(s)
        expect(() => c.call(s, 0)).not.toThrow()
        expect(c.weights(s)).toEqual(before)
      })
    })
  }

  it('NetworkLSTM.train rejected: the episode is still trainable afterwards', () => {
    const n = makeLSTMNet()
    expect(() => n.train([[1], [0]], -1)).toThrow('NetworkLSTM.train: lr must be')
    expect((n as any).lstm._traj.length).toBe(2)
    expect(() => n.train([[1], [0]], 0.1)).not.toThrow()
    expect((n as any).lstm._traj.length).toBe(0)
  })

  it('NetworkN.train NaN no longer poisons the weights (the original defect)', () => {
    const n = new NetworkN([2, 3, 1])
    expect(() => n.train([0, 1], [1], NaN)).toThrow('NetworkN.train: lr must be')
    expect(n.getWeights().every(Number.isFinite)).toBe(true)
  })
})

// ── Golden: valid lr behaves exactly as on the base commit ────────────────────
// Values are [loss_1 .. loss_k, sum(w), sum((i + 1) * w_i)] captured on 6f20f36.

function summarize(losses: number[], w: number[]): number[] {
  return [
    ...losses,
    w.reduce((s, v) => s + v, 0),
    w.reduce((s, v, i) => s + (i + 1) * v, 0),
  ]
}

const scenarios: Record<string, () => number[]> = {
  'NetworkN.train + trainWithDeltas': () => {
    const n = new NetworkN([2, 3, 1])
    const losses: number[] = []
    for (let i = 0; i < 5; i++) losses.push(n.train([i % 2, 1], [(i + 1) % 2], 0.3))
    for (let i = 0; i < 2; i++) n.trainWithDeltas([1, i % 2], [0.2], 0.1)
    return summarize(losses, n.getWeights())
  },
  'Network.train': () => {
    const n = new Network(2, 3, 1)
    const losses: number[] = []
    for (let i = 0; i < 5; i++) losses.push(n.train([i % 2, 1], (i + 1) % 2, 0.3))
    return summarize(losses, n.getWeights())
  },
  'Neuron.train': () => {
    const n = new Neuron()
    for (let i = 0; i < 5; i++) n.train(0.5 + i * 0.1, i % 2, 0.5)
    return summarize([], [n.weight, n.bias])
  },
  'NeuronN.train': () => {
    const n = new NeuronN(2)
    for (let i = 0; i < 5; i++) n.train([i % 2, 1], (i + 1) % 2, 0.5)
    return summarize([], [...n.weights, n.bias])
  },
  'NetworkLSTM.train': () => {
    const n = new NetworkLSTM(2, 3, [1])
    for (let e = 0; e < 3; e++) {
      n.resetState()
      n.predict([0.1, 0.2])
      n.predict([0.3, 0.1])
      n.predict([0.2, 0.4])
      n.train([[1], [0], [1]], 0.1)
    }
    return summarize([], n.getWeightsFlat())
  },
  'LSTMLayer.backprop': () => {
    const l = new LSTMLayer(2, 3)
    for (let e = 0; e < 2; e++) {
      l.reset()
      l.predict([0.1, 0.2])
      l.predict([0.3, 0.1])
      l.backprop(dh, 0.1)
    }
    return summarize([], l.getWeightsFlat())
  },
  'GRULayer.backprop': () => {
    const g = new GRULayer(2, 3)
    for (let e = 0; e < 2; e++) {
      g.reset()
      g.predict([0.1, 0.2])
      g.predict([0.3, 0.1])
      g.backprop(dh, 0.1)
    }
    return summarize([], g.getWeightsFlat())
  },
  'NetworkTransformer.train': () => {
    const n = makeTransformer()
    const losses: number[] = []
    for (let i = 0; i < 3; i++) losses.push(n.train([0, 1, 2], onehot, 0.01))
    return summarize(losses, n.getWeights())
  },
  'NetworkTransformerRL.train': () => {
    const n = makeTransformerRL()
    const losses: number[] = []
    for (let i = 0; i < 3; i++) losses.push(n.train(seq, [1, 0], 0.01))
    return summarize(losses, n.getWeights())
  },
}

const GOLDEN: Record<string, number[]> = {
  'NetworkN.train + trainWithDeltas': [0.2842555878951204, 0.23732766959869867, 0.28261064916907375, 0.23820721847870446, 0.2811261094603293, 0.024290306231568193, -0.3480598211526784],
  'Network.train': [0.2842555878951204, 0.23732766959869867, 0.28261064916907375, 0.23820721847870446, 0.2811261094603293, -0.07391144046858532, -1.5654591660099382],
  'Neuron.train': [-0.0178076960789122, -0.05814514733429756],
  'NeuronN.train': [0.021748832044788072, -0.043276847343189706],
  'NetworkLSTM.train': [2.1637553576116173, 52.279165847281206],
  'LSTMLayer.backprop': [1.5865370991229073, 8.093917769799647],
  'GRULayer.backprop': [-1.242346715177187, -29.687984488484904],
  'NetworkTransformer.train': [1.6879877631544764, 1.5384237563438348, 1.3876618502493177, 10.032807625978265, 1336.343085113453],
  'NetworkTransformerRL.train': [0.0784881334077541, 0.031815418187991586, 0.0068126663516303665, 6.121019557622704, 704.310869192326],
}

describe('T018 golden: valid lr is unchanged', () => {
  for (const [name, run] of Object.entries(scenarios)) {
    it(name, () => {
      const got = run()
      const want = GOLDEN[name]
      expect(want).toBeDefined()
      expect(got.length).toBe(want.length)
      got.forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThan(1e-12))
    })
  }
})
