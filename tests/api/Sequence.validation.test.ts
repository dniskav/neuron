// Validación de entradas de RNN, TCN (incluido CausalConv1D.backward) y Seq2Seq (T040, feature 006).
//
// Cada error debe:
//   - lanzar con el formato `Clase.método: motivo`;
//   - dejar intactos los pesos (se comparan antes y después);
//   - dejar intacto el estado (RNN: h; Seq2Seq: h del decoder), porque las comprobaciones
//     van ANTES de cualquier reset o escritura.
//
// Excepciones documentadas (fijadas por tests existentes que no se pueden tocar en T040):
//   - lr = 0 se acepta (RNN.test.ts y Seq2Seq.test.ts lo usan como "no cambia los pesos").
//   - Seq2Seq.encode([]) devuelve contexto cero y RNN.backward([], [], lr) devuelve 0.
//
// Valores a mano: no hay cifras numéricas esperadas; todas las comprobaciones son de
// mensaje, de igualdad de pesos/estado y de continuación frente a una red de control.
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { RNN } from '../../src/RNN'
import { CausalConv1D, TCN } from '../../src/TCN'
import { Seq2Seq } from '../../src/Seq2Seq'
import { restoreRandom, seedRandom } from '../helpers/seed'

beforeEach(() => seedRandom(7))
afterEach(() => restoreRandom())

const BAD_LR: number[] = [Number.NaN, Infinity, -Infinity, -0.1]
const LR_MSG = 'lr must be a finite number >= 0'

// ── helpers ──────────────────────────────────────────────────────────────────

function copyRnnWeights(from: RNN, to: RNN): void {
  to.Wxh = structuredClone(from.Wxh)
  to.Whh = structuredClone(from.Whh)
  to.Why = structuredClone(from.Why)
  to.bh = structuredClone(from.bh)
  to.by = structuredClone(from.by)
}

const rnnWeights = (n: RNN): string => JSON.stringify([n.Wxh, n.Whh, n.Why, n.bh, n.by])

const SEQ_TCN: number[][] = [[0.5, -0.4], [-0.3, 0.7], [0.8, 0.1], [-0.6, 0.9], [0.4, -0.5], [0.1, 0.2]]
const TGT_TCN: number[][] = [[0.3], [-0.4], [0.5], [0.1], [0.4], [-0.1]]

// Pesos del TCN observables sin acceso privado: kernels, sesgos y la salida de un forward.
const tcnState = (n: TCN): string =>
  JSON.stringify([n.layers.map((l) => [l.kernels, l.biases]), n.forward(SEQ_TCN)])

const s2sState = (m: Seq2Seq): string =>
  JSON.stringify([m.encoder.getWeightsFlat(), m.decoder.getWeightsFlat(), m.W_out, m.b_out])

// ── RNN ──────────────────────────────────────────────────────────────────────

describe('RNN.forward validation (T040)', () => {
  it('rejects an empty sequence', () => {
    expect(() => new RNN(2, 3, 1).forward([])).toThrow('RNN.forward: sequence must not be empty')
  })

  it('rejects a row of the wrong width', () => {
    const net = new RNN(2, 3, 1)
    expect(() => net.forward([[0.1, 0.2], [0.3]])).toThrow(
      'RNN.forward: sequence[1] must be an array of length 2'
    )
  })

  it('a rejected call keeps the hidden state: the next forward equals a control net that never failed', () => {
    // Con h ≠ 0 de un forward válido, un error que reiniciara h cambiaría la salida siguiente.
    const failing = new RNN(2, 3, 1)
    const control = new RNN(2, 3, 1)
    copyRnnWeights(failing, control)
    failing.forward([[0.5, -0.8]])
    control.forward([[0.5, -0.8]])
    expect(() => failing.forward([[0.1, 0.2], [0.3]])).toThrow('RNN.forward')
    expect(failing.forward([[0, 0]]).outputs).toEqual(control.forward([[0, 0]]).outputs)
  })
})

describe('RNN.backward validation (T040)', () => {
  it('rejects targets with a different number of rows and leaves the weights unchanged', () => {
    const net = new RNN(1, 2, 1)
    const before = rnnWeights(net)
    expect(() => net.backward([[1], [0]], [[1]], 0.1)).toThrow(
      'RNN.backward: targets must have one row per time step (2)'
    )
    expect(rnnWeights(net)).toBe(before)
  })

  it('rejects a target row of the wrong width', () => {
    const net = new RNN(1, 2, 1)
    expect(() => net.backward([[1]], [[1, 2]], 0.1)).toThrow(
      'RNN.backward: targets[0] must be an array of length 1'
    )
  })

  it.each(BAD_LR)('rejects lr = %s and leaves the weights unchanged', (lr) => {
    const net = new RNN(1, 2, 1)
    const before = rnnWeights(net)
    expect(() => net.backward([[1]], [[0]], lr)).toThrow(`RNN.backward: ${LR_MSG}`)
    expect(rnnWeights(net)).toBe(before)
  })

  it('a rejected call keeps the hidden state: the next forward equals a control net that never failed', () => {
    const failing = new RNN(1, 2, 1)
    const control = new RNN(1, 2, 1)
    copyRnnWeights(failing, control)
    failing.forward([[0.7]])
    control.forward([[0.7]])
    expect(() => failing.backward([[1]], [[0]], Number.NaN)).toThrow('RNN.backward')
    expect(failing.forward([[0]]).outputs).toEqual(control.forward([[0]]).outputs)
  })

  it('lr = 0 is accepted and leaves the weights unchanged (pinned by RNN.test.ts)', () => {
    const net = new RNN(1, 2, 1)
    const before = rnnWeights(net)
    const loss = net.backward([[1], [0]], [[0], [1]], 0)
    expect(Number.isFinite(loss)).toBe(true)
    expect(rnnWeights(net)).toBe(before)
  })
})

// ── TCN ──────────────────────────────────────────────────────────────────────

describe('TCN validation (T040)', () => {
  it('forward rejects an empty sequence', () => {
    expect(() => new TCN(2, 3, 2, 2, 1).forward([])).toThrow('TCN.forward: sequence must not be empty')
  })

  it('forward rejects a row whose width is not inputChannels', () => {
    expect(() => new TCN(2, 3, 2, 2, 1).forward([[0.1, 0.2], [0.3]])).toThrow(
      'TCN.forward: sequence[1] must be an array of length 2'
    )
  })

  it('train rejects an empty sequence', () => {
    expect(() => new TCN(2, 3, 2, 2, 1).train([], [], 0.1)).toThrow('TCN.train: sequence must not be empty')
  })

  it('train rejects targets with a different number of rows and leaves the weights unchanged', () => {
    const net = new TCN(2, 3, 2, 2, 1)
    const before = tcnState(net)
    expect(() => net.train(SEQ_TCN, TGT_TCN.slice(1), 0.1)).toThrow(
      'TCN.train: targets must have one row per time step (6)'
    )
    expect(tcnState(net)).toBe(before)
  })

  it('train rejects a target row of the wrong width', () => {
    const net = new TCN(2, 3, 2, 2, 1)
    const bad = TGT_TCN.map((r, t) => (t === 2 ? [0.5, 0.5] : r))
    expect(() => net.train(SEQ_TCN, bad, 0.1)).toThrow('TCN.train: targets[2] must be an array of length 1')
  })

  it.each(BAD_LR)('train rejects lr = %s and leaves the weights unchanged', (lr) => {
    const net = new TCN(2, 3, 2, 2, 1)
    const before = tcnState(net)
    expect(() => net.train(SEQ_TCN, TGT_TCN, lr)).toThrow(`TCN.train: ${LR_MSG}`)
    expect(tcnState(net)).toBe(before)
  })
})

describe('CausalConv1D.backward validation (T040)', () => {
  const INPUT3: number[][] = [[0.5, -0.4], [-0.3, 0.7], [0.8, 0.1]]

  it('rejects dOutput with a different number of rows than the forward input', () => {
    const conv = new CausalConv1D(2, 2, 2, 1)
    conv.forward(INPUT3)
    const before = JSON.stringify([conv.kernels, conv.biases])
    expect(() => conv.backward([[0, 0], [0, 0]], 0.1)).toThrow(
      'CausalConv1D.backward: dOutput must have 3 rows (one per time step)'
    )
    expect(JSON.stringify([conv.kernels, conv.biases])).toBe(before)
  })

  it('rejects a dOutput row of the wrong width', () => {
    const conv = new CausalConv1D(2, 2, 2, 1)
    conv.forward(INPUT3)
    expect(() => conv.backward([[0, 0], [0], [0, 0]], 0.1)).toThrow(
      'CausalConv1D.backward: dOutput[1] must be an array of length 2'
    )
  })

  it.each(BAD_LR)('rejects lr = %s and leaves the kernels unchanged', (lr) => {
    const conv = new CausalConv1D(2, 2, 2, 1)
    conv.forward(INPUT3)
    const before = JSON.stringify([conv.kernels, conv.biases])
    expect(() => conv.backward([[0, 0], [0, 0], [0, 0]], lr)).toThrow(`CausalConv1D.backward: ${LR_MSG}`)
    expect(JSON.stringify([conv.kernels, conv.biases])).toBe(before)
  })
})

// ── Seq2Seq ──────────────────────────────────────────────────────────────────

describe('Seq2Seq constructor validation (T040)', () => {
  it.each([
    [Number.NaN, 2, 1],
    [1, Number.NaN, 1],
    [1, 2, Number.NaN],
    [1.5, 2, 1],
    [1, Infinity, 1],
    [1, 2, -Infinity],
  ])('rejects non-integer or non-finite sizes (%s, %s, %s)', (i, h, o) => {
    expect(() => new Seq2Seq(i, h, o)).toThrow('Seq2Seq: all sizes must be positive integers')
  })
})

describe('Seq2Seq.encode / decode validation (T040)', () => {
  it('encode rejects a row of the wrong width', () => {
    const m = new Seq2Seq(1, 2, 1)
    expect(() => m.encode([[0.5], [0.1, 0.2]])).toThrow(
      'Seq2Seq.encode: inputSequence[1] must be an array of length 1'
    )
  })

  it('decode rejects a context whose h has the wrong length and keeps the decoder state', () => {
    const m = new Seq2Seq(1, 2, 1)
    m.decode({ h: [0.3, -0.2], c: [0.1, 0.4] }, 2) // deja un estado del decoder no nulo
    const hBefore = [...m.decoder.h]
    expect(() => m.decode({ h: [0.3], c: [0.1, 0.4] }, 2)).toThrow(
      'Seq2Seq.decode: context.h must be an array of length 2'
    )
    expect(m.decoder.h).toEqual(hBefore)
  })

  it('decode rejects a context whose c has the wrong length', () => {
    const m = new Seq2Seq(1, 2, 1)
    expect(() => m.decode({ h: [0.3, -0.2], c: [0.1, 0.4, 0.5] }, 2)).toThrow(
      'Seq2Seq.decode: context.c must be an array of length 2'
    )
  })
})

describe('Seq2Seq.trainStep validation (T040)', () => {
  const IN: number[][] = [[0.1], [0.5], [0.9]]
  const OUT: number[][] = [[0.9], [0.5], [0.1]]

  it('rejects an empty targetSeq (T = 0) and does not train: no silent 0', () => {
    const m = new Seq2Seq(1, 3, 1)
    const before = s2sState(m)
    expect(() => m.trainStep(IN, [], 0.1)).toThrow('Seq2Seq.trainStep: targetSeq must not be empty')
    expect(s2sState(m)).toBe(before)
  })

  it('rejects an empty inputSeq and does not train', () => {
    const m = new Seq2Seq(1, 3, 1)
    const before = s2sState(m)
    expect(() => m.trainStep([], OUT, 0.1)).toThrow('Seq2Seq.trainStep: inputSeq must not be empty')
    expect(s2sState(m)).toBe(before)
  })

  it('rejects an inputSeq row of the wrong width and leaves the weights unchanged', () => {
    const m = new Seq2Seq(1, 3, 1)
    const before = s2sState(m)
    expect(() => m.trainStep([[0.1], [0.5, 0.2]], OUT.slice(0, 2), 0.1)).toThrow(
      'Seq2Seq.trainStep: inputSeq[1] must be an array of length 1'
    )
    expect(s2sState(m)).toBe(before)
  })

  it('rejects a targetSeq row of the wrong width and leaves the weights unchanged', () => {
    const m = new Seq2Seq(1, 3, 1)
    const before = s2sState(m)
    expect(() => m.trainStep(IN, [[0.9], [0.5, 0.5], [0.1]], 0.1)).toThrow(
      'Seq2Seq.trainStep: targetSeq[1] must be an array of length 1'
    )
    expect(s2sState(m)).toBe(before)
  })

  it.each(BAD_LR)('rejects lr = %s and leaves every weight unchanged', (lr) => {
    const m = new Seq2Seq(1, 3, 1)
    const before = s2sState(m)
    expect(() => m.trainStep(IN, OUT, lr)).toThrow(`Seq2Seq.trainStep: ${LR_MSG}`)
    expect(s2sState(m)).toBe(before)
  })

  it('lr = 0 is accepted and leaves the weights unchanged (pinned by Seq2Seq.test.ts)', () => {
    const m = new Seq2Seq(1, 3, 1)
    const before = s2sState(m)
    expect(Number.isFinite(m.trainStep(IN, OUT, 0))).toBe(true)
    expect(s2sState(m)).toBe(before)
  })
})
