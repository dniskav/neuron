import { describe, it, expect } from 'vitest'
import { RNN } from '../src/RNN'
import { Seq2Seq } from '../src/Seq2Seq'
import { LSTMLayer } from '../src/LSTMLayer'
import { Adam, Momentum, SGD, type Optimizer, type OptimizerFactory } from '../src/optimizers'
import { random, seededRandom } from '../src/random'
import { validateArray } from '../src/Validation'
import { withSeed } from './helpers/seed'

// ─── T022: memory (vector optimizers) and exactness against RNN and Seq2Seq 0.4 ──
//
// RNN04 and Seq2Seq04 are the classes as they were before T022 (one scalar
// Optimizer per weight), pasted here as the reference. Their only difference from
// the new classes is the storage of the optimizer state.

// ── RNN 0.4 (copied from `git show feat/0.5-core:src/RNN.ts`) ─────────────────────

function tanh(x: number): number {
  return Math.tanh(x)
}

function checkLr(where: string, lr: number): void {
  if (!Number.isFinite(lr) || lr < 0) {
    throw new Error(`${where}: lr must be a finite number >= 0`)
  }
}

function checkRows(where: string, name: string, rows: unknown, width: number): void {
  if (!Array.isArray(rows)) {
    throw new Error(`${where}: ${name} must be an array of rows`)
  }
  rows.forEach((row, t) => {
    if (!Array.isArray(row) || row.length !== width) {
      throw new Error(`${where}: ${name}[${t}] must be an array of length ${width}`)
    }
    validateArray(row, width, `${where}: ${name}[${t}]`)
  })
}

function isSize(n: unknown): boolean {
  return typeof n === 'number' && Number.isInteger(n) && n > 0
}

interface RNNStep {
  x: number[]
  h: number[]
  hRaw: number[]
  hPrev: number[]
}

class RNN04 {
  readonly inputSize: number
  readonly hiddenSize: number
  readonly outputSize: number

  Wxh: number[][]
  Whh: number[][]
  Why: number[][]
  bh: number[]
  by: number[]

  private _h: number[]
  private _traj: RNNStep[] = []
  private _outputs: number[][] = []

  private _opts: {
    Wxh: Optimizer[][]
    Whh: Optimizer[][]
    Why: Optimizer[][]
    bh: Optimizer[]
    by: Optimizer[]
  }

  constructor(
    inputSize: number,
    hiddenSize: number,
    outputSize: number,
    optimizerFactory: OptimizerFactory = () => new SGD(),
  ) {
    if (!isSize(inputSize) || !isSize(hiddenSize) || !isSize(outputSize)) {
      throw new Error('RNN: all sizes must be positive integers')
    }
    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.outputSize = outputSize

    const limXH = Math.sqrt(2 / inputSize)
    const limHH = Math.sqrt(2 / hiddenSize)
    const limHY = Math.sqrt(2 / hiddenSize)

    this.Wxh = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => (random() * 2 - 1) * limXH),
    )
    this.Whh = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: hiddenSize }, () => (random() * 2 - 1) * limHH),
    )
    this.Why = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => (random() * 2 - 1) * limHY),
    )
    this.bh = new Array(hiddenSize).fill(0)
    this.by = new Array(outputSize).fill(0)

    this._h = new Array(hiddenSize).fill(0)

    this._opts = {
      Wxh: Array.from({ length: hiddenSize }, () =>
        Array.from({ length: inputSize }, () => optimizerFactory()),
      ),
      Whh: Array.from({ length: hiddenSize }, () =>
        Array.from({ length: hiddenSize }, () => optimizerFactory()),
      ),
      Why: Array.from({ length: outputSize }, () =>
        Array.from({ length: hiddenSize }, () => optimizerFactory()),
      ),
      bh: Array.from({ length: hiddenSize }, () => optimizerFactory()),
      by: Array.from({ length: outputSize }, () => optimizerFactory()),
    }
  }

  reset(): void {
    this._h = new Array(this.hiddenSize).fill(0)
    this._traj = []
    this._outputs = []
  }

  forward(sequence: number[][]): { outputs: number[][]; hiddens: number[][] } {
    if (sequence.length === 0) {
      throw new Error('RNN.forward: sequence must not be empty')
    }
    checkRows('RNN.forward', 'sequence', sequence, this.inputSize)

    this._traj = []
    this._outputs = []

    const outputs: number[][] = []
    const hiddens: number[][] = []

    let hPrev = [...this._h]

    for (const x of sequence) {
      const hRaw = this.bh.map(
        (b, i) =>
          b +
          this.Wxh[i].reduce((s, w, j) => s + w * x[j], 0) +
          this.Whh[i].reduce((s, w, j) => s + w * hPrev[j], 0),
      )
      const h = hRaw.map(tanh)

      const o = this.by.map((b, i) => b + this.Why[i].reduce((s, w, j) => s + w * h[j], 0))

      this._traj.push({ x: [...x], h: [...h], hRaw: [...hRaw], hPrev: [...hPrev] })
      this._outputs.push(o)
      outputs.push(o)
      hiddens.push(h)

      hPrev = h
    }

    this._h = hPrev
    return { outputs, hiddens }
  }

  backward(sequence: number[][], targets: number[][], lr: number): number {
    checkLr('RNN.backward', lr)
    if (Array.isArray(sequence) && sequence.length === 0) {
      throw new Error('RNN.backward: sequence must not be empty')
    }
    checkRows('RNN.backward', 'sequence', sequence, this.inputSize)
    checkRows('RNN.backward', 'targets', targets, this.outputSize)
    if (targets.length !== sequence.length) {
      throw new Error(`RNN.backward: targets must have one row per time step (${sequence.length})`)
    }

    this.reset()
    const { outputs } = this.forward(sequence)

    const T = this._traj.length

    let loss = 0
    const dOutputs: number[][] = outputs.map((o, t) =>
      o.map((v, k) => {
        const diff = v - targets[t][k]
        loss += diff * diff
        return (-2 * diff) / this.outputSize
      }),
    )
    loss /= T * this.outputSize

    const dWxh = Array.from({ length: this.hiddenSize }, () => new Array(this.inputSize).fill(0))
    const dWhh = Array.from({ length: this.hiddenSize }, () => new Array(this.hiddenSize).fill(0))
    const dWhy = Array.from({ length: this.outputSize }, () => new Array(this.hiddenSize).fill(0))
    const dbh = new Array(this.hiddenSize).fill(0)
    const dby = new Array(this.outputSize).fill(0)

    let dhNext: number[] = new Array(this.hiddenSize).fill(0)

    for (let t = T - 1; t >= 0; t--) {
      const s = this._traj[t]
      const do_ = dOutputs[t]

      for (let i = 0; i < this.outputSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dWhy[i][j] += do_[i] * s.h[j]
        }
        dby[i] += do_[i]
      }

      const dh: number[] =
        this.hiddenSize > 0
          ? Array.from(
              { length: this.hiddenSize },
              (_, j) => this.Why.reduce((sum, row, i) => sum + row[j] * do_[i], 0) + dhNext[j],
            )
          : []

      const dhRaw: number[] = dh.map((d, k) => d * (1 - s.h[k] ** 2))

      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.inputSize; j++) {
          dWxh[i][j] += dhRaw[i] * s.x[j]
        }
        dbh[i] += dhRaw[i]
      }

      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dWhh[i][j] += dhRaw[i] * s.hPrev[j]
        }
      }

      dhNext = Array.from({ length: this.hiddenSize }, (_, j) =>
        this.Whh.reduce((sum, row, i) => sum + row[j] * dhRaw[i], 0),
      )
    }

    const scale = lr / T
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.Wxh[i][j] = this._opts.Wxh[i][j].step(this.Wxh[i][j], dWxh[i][j], scale)
      }
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Whh[i][j] = this._opts.Whh[i][j].step(this.Whh[i][j], dWhh[i][j], scale)
      }
      this.bh[i] = this._opts.bh[i].step(this.bh[i], dbh[i], scale)
    }
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.Why[i][j] = this._opts.Why[i][j].step(this.Why[i][j], dWhy[i][j], scale)
      }
      this.by[i] = this._opts.by[i].step(this.by[i], dby[i], scale)
    }

    this._traj = []
    this._outputs = []
    return loss
  }
}

// ── Seq2Seq 0.4 (copied from `git show feat/0.5-core:src/Seq2Seq.ts`) ─────────────

function checkVector(where: string, name: string, v: unknown, width: number): void {
  if (!Array.isArray(v) || v.length !== width) {
    throw new Error(`${where}: ${name} must be an array of length ${width}`)
  }
  validateArray(v, width, `${where}: ${name}`)
}

class Seq2Seq04 {
  readonly encoder: LSTMLayer
  readonly decoder: LSTMLayer

  W_out: number[][]
  b_out: number[]

  readonly inputSize: number
  readonly hiddenSize: number
  readonly outputSize: number

  private _wOutOpts: Optimizer[][]
  private _bOutOpts: Optimizer[]

  constructor(
    inputSize: number,
    hiddenSize: number,
    outputSize: number,
    options?: { optimizerFactory?: OptimizerFactory },
  ) {
    const isSize = (n: number): boolean => Number.isInteger(n) && n > 0
    if (!isSize(inputSize) || !isSize(hiddenSize) || !isSize(outputSize)) {
      throw new Error('Seq2Seq: all sizes must be positive integers')
    }

    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.outputSize = outputSize

    const factory: OptimizerFactory = options?.optimizerFactory ?? (() => new SGD())

    this.encoder = new LSTMLayer(inputSize, hiddenSize, factory)
    this.decoder = new LSTMLayer(outputSize, hiddenSize, factory)

    const limit = Math.sqrt(2 / hiddenSize)
    this.W_out = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => (random() * 2 - 1) * limit),
    )
    this.b_out = new Array(outputSize).fill(0)

    this._wOutOpts = Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => factory()),
    )
    this._bOutOpts = Array.from({ length: outputSize }, () => factory())
  }

  private _project(h: number[]): number[] {
    return this.b_out.map((b, i) => b + this.W_out[i].reduce((s, w, j) => s + w * h[j], 0))
  }

  trainStep(inputSeq: number[][], targetSeq: number[][], lr: number): number {
    checkLr('Seq2Seq.trainStep', lr)
    if (inputSeq.length === 0) {
      throw new Error('Seq2Seq.trainStep: inputSeq must not be empty')
    }
    checkRows('Seq2Seq.trainStep', 'inputSeq', inputSeq, this.inputSize)
    if (targetSeq.length === 0) {
      throw new Error('Seq2Seq.trainStep: targetSeq must not be empty')
    }
    checkRows('Seq2Seq.trainStep', 'targetSeq', targetSeq, this.outputSize)
    const T = targetSeq.length

    this.encoder.reset()
    for (const x of inputSeq) {
      this.encoder.predict(x)
    }
    const contextH = [...this.encoder.h]
    const contextC = [...this.encoder.c]

    this.decoder.reset()
    this.decoder.h = [...contextH]
    this.decoder.c = [...contextC]

    const hiddens: number[][] = []
    const projOuts: number[][] = []

    let prevTeacher: number[] = new Array(this.outputSize).fill(0)
    for (let t = 0; t < T; t++) {
      const h = this.decoder.predict(prevTeacher)
      const out = this._project(h)
      hiddens.push(h)
      projOuts.push(out)
      prevTeacher = targetSeq[t]
    }

    let loss = 0
    const dProjOut: number[][] = projOuts.map((o, t) =>
      o.map((v, k) => {
        const diff = v - targetSeq[t][k]
        loss += diff * diff
        return (-2 * diff) / this.outputSize
      }),
    )
    loss /= T * this.outputSize

    const dhSeq: number[][] = Array.from({ length: T }, () => new Array(this.hiddenSize).fill(0))

    const dWout = Array.from({ length: this.outputSize }, () => new Array(this.hiddenSize).fill(0))
    const dbOut = new Array(this.outputSize).fill(0)

    for (let t = 0; t < T; t++) {
      for (let i = 0; i < this.outputSize; i++) {
        const dv = dProjOut[t][i]
        dbOut[i] += dv
        for (let j = 0; j < this.hiddenSize; j++) {
          dWout[i][j] += dv * hiddens[t][j]
          dhSeq[t][j] += dv * this.W_out[i][j]
        }
      }
    }

    const scale = lr / T
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.W_out[i][j] = this._wOutOpts[i][j].step(this.W_out[i][j], dWout[i][j], scale)
      }
      this.b_out[i] = this._bOutOpts[i].step(this.b_out[i], dbOut[i], scale)
    }

    const { dh0, dc0 } = this.decoder.backprop(dhSeq, lr)

    const Tin = inputSeq.length
    const encoderDhSeq: number[][] = inputSeq.map((_, t) =>
      t === Tin - 1 ? [...dh0] : new Array(this.hiddenSize).fill(0),
    )
    this.encoder.backprop(encoderDhSeq, (lr * Tin) / T, dc0)

    return loss
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const factories: Array<[string, OptimizerFactory]> = [
  ['SGD', () => new SGD()],
  ['Momentum', () => new Momentum(0.8)],
  ['Adam', () => new Adam(0.8, 0.99, 1e-7)],
]

/** Every number inside a nested structure, in a fixed order. */
function flat(x: unknown, out: number[] = []): number[] {
  if (typeof x === 'number') out.push(x)
  else if (Array.isArray(x)) x.forEach(v => flat(v, out))
  else if (x && typeof x === 'object') Object.values(x).forEach(v => flat(v, out))
  return out
}

function expectSame(a: unknown, b: unknown): void {
  const fa = flat(a)
  const fb = flat(b)
  expect(fa.length).toBe(fb.length)
  for (let i = 0; i < fa.length; i++) {
    expect(Object.is(fa[i], fb[i])).toBe(true)
  }
}

function countingFactory(make: () => Optimizer): { factory: OptimizerFactory; count: () => number } {
  let n = 0
  return { factory: () => { n++; return make() }, count: () => n }
}

// ─── RNN ──────────────────────────────────────────────────────────────────────

describe('RNN 0.5: exactness against the 0.4 class (100 backward steps)', () => {
  for (const [name, factory] of factories) {
    it(`${name}: loss and every weight are identical`, () => {
      const fresh = withSeed(7, () => new RNN(3, 4, 2, factory))
      const ref = withSeed(7, () => new RNN04(3, 4, 2, factory))
      ref.Wxh = fresh.Wxh.map(r => [...r])
      ref.Whh = fresh.Whh.map(r => [...r])
      ref.Why = fresh.Why.map(r => [...r])
      ref.bh = [...fresh.bh]
      ref.by = [...fresh.by]

      const rnd = seededRandom(11)
      for (let k = 0; k < 100; k++) {
        const seq = Array.from({ length: 5 }, () => [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1])
        const tgt = Array.from({ length: 5 }, () => [rnd() * 2 - 1, rnd() * 2 - 1])
        const la = fresh.backward(seq, tgt, 0.05)
        const lb = ref.backward(seq, tgt, 0.05)
        expect(Object.is(la, lb)).toBe(true)
      }
      expectSame(fresh.Wxh, ref.Wxh)
      expectSame(fresh.Whh, ref.Whh)
      expectSame(fresh.Why, ref.Why)
      expectSame(fresh.bh, ref.bh)
      expectSame(fresh.by, ref.by)
    })
  }
})

describe('RNN 0.5: optimizer memory', () => {
  it('asks the factory once per row and once per bias vector, not once per weight', () => {
    // RNN(3, 4, 2): rows Wxh 4, rows Whh 4, rows Why 2, bh 1, by 1 => 12 vectors.
    // The 0.4 class created 4*3 + 4*4 + 2*4 + 4 + 2 = 42 scalar optimizers.
    const c = countingFactory(() => new SGD())
    new RNN(3, 4, 2, c.factory)
    expect(c.count()).toBe(4 + 4 + 2 + 1 + 1)
  })

  it('keeps the identity of every weight row after backward', () => {
    const net = new RNN(3, 4, 2, () => new Adam())
    const rows = net.Wxh.slice()
    const wh = net.Whh.slice()
    const why = net.Why.slice()
    const bh = net.bh
    const by = net.by
    net.backward([[1, 0, -1], [0.5, 0.5, 0.5]], [[0.1, 0.2], [0.3, -0.1]], 0.1)
    rows.forEach((r, i) => expect(net.Wxh[i]).toBe(r))
    wh.forEach((r, i) => expect(net.Whh[i]).toBe(r))
    why.forEach((r, i) => expect(net.Why[i]).toBe(r))
    expect(net.bh).toBe(bh)
    expect(net.by).toBe(by)
  })
})

// ─── Seq2Seq ──────────────────────────────────────────────────────────────────

describe('Seq2Seq 0.5: exactness against the 0.4 class (100 trainStep calls)', () => {
  for (const [name, factory] of factories) {
    it(`${name}: loss, output projection and both LSTMs are identical`, () => {
      const fresh = withSeed(7, () => new Seq2Seq(2, 3, 2, { optimizerFactory: factory }))
      const ref = withSeed(7, () => new Seq2Seq04(2, 3, 2, { optimizerFactory: factory }))
      ref.W_out = fresh.W_out.map(r => [...r])
      ref.b_out = [...fresh.b_out]
      ref.encoder.setWeights(fresh.encoder.getWeights())
      ref.decoder.setWeights(fresh.decoder.getWeights())

      const rnd = seededRandom(19)
      for (let k = 0; k < 100; k++) {
        const inSeq = Array.from({ length: 4 }, () => [rnd() * 2 - 1, rnd() * 2 - 1])
        const tgt = Array.from({ length: 3 }, () => [rnd() * 2 - 1, rnd() * 2 - 1])
        const la = fresh.trainStep(inSeq, tgt, 0.05)
        const lb = ref.trainStep(inSeq, tgt, 0.05)
        expect(Object.is(la, lb)).toBe(true)
      }
      expectSame(fresh.W_out, ref.W_out)
      expectSame(fresh.b_out, ref.b_out)
      expectSame(fresh.encoder.getWeights(), ref.encoder.getWeights())
      expectSame(fresh.decoder.getWeights(), ref.decoder.getWeights())
    })
  }
})

describe('Seq2Seq 0.5: optimizer memory', () => {
  it('the output projection asks the factory once per row and once for the bias', () => {
    // Seq2Seq(2, 3, 2): the LSTMs ask the factory for their own optimizers. Subtract
    // them to count only the projection. 0.4 made outputSize*hiddenSize + outputSize = 8;
    // now it is one per row (2) plus one for the bias vector (1) = 3.
    const plain = countingFactory(() => new SGD())
    new LSTMLayer(2, 3, plain.factory)
    new LSTMLayer(2, 3, plain.factory)
    const lstmOnly = plain.count()

    const full = countingFactory(() => new SGD())
    new Seq2Seq(2, 3, 2, { optimizerFactory: full.factory })
    expect(full.count() - lstmOnly).toBe(2 + 1)
  })

  it('keeps the identity of W_out rows after trainStep', () => {
    const net = new Seq2Seq(2, 3, 2, { optimizerFactory: () => new Momentum(0.9) })
    const rows = net.W_out.slice()
    const bias = net.b_out
    net.trainStep([[0.2, -0.4], [0.9, 0.1]], [[0.3, -0.2], [0.1, 0.4], [0.0, 0.2]], 0.1)
    rows.forEach((r, i) => expect(net.W_out[i]).toBe(r))
    expect(net.b_out).toBe(bias)
  })
})
