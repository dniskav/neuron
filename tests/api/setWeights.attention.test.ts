// T002 (I2): setWeights de AttentionHead, MultiHeadAttention y TransformerBlock
// valida la longitud TOTAL esperada ANTES de escribir.
//
// Batería común (la misma que tests/api/setWeights.layers.test.ts en T001):
//   - set(get()) es identidad y set(get() + 1) escribe exactamente lo pedido;
//   - array corto (-1), largo (+1), NaN, Infinity, undefined y no-array lanzan;
//   - el mensaje contiene `Clase.setWeights` y, para longitudes, la esperada y la recibida;
//   - getWeights() es idéntico antes y después del error (atomicidad).
// TransformerBlock valida en el nivel exterior: un array largo no se recorta en silencio.

import { describe, it, expect } from 'vitest'
import { AttentionHead } from '../../src/AttentionHead'
import { MultiHeadAttention } from '../../src/MultiHeadAttention'
import { TransformerBlock } from '../../src/TransformerBlock'

interface Settable {
  getWeights(): number[]
  setWeights(w: number[]): void
}

// Estado no trivial y determinista: así un error que escribiera antes de
// validar cambiaría los pesos y el test lo detectaría.
function setPatternState(m: Settable): void {
  const n = m.getWeights().length
  const pattern = Array.from({ length: n }, (_, i) => 0.05 * ((i % 7) - 3))
  m.setWeights(pattern)
  expect(m.getWeights()).toEqual(pattern)
}

function battery(cls: string, make: () => Settable, expected: number): void {
  const method = `${cls}.setWeights`

  function fresh(): Settable {
    const m = make()
    setPatternState(m)
    return m
  }

  function expectRejectedAndIntact(m: Settable, bad: unknown, msg: string): void {
    const before = m.getWeights()
    expect(() => m.setWeights(bad as number[])).toThrow(msg)
    expect(m.getWeights()).toEqual(before)
  }

  describe(`${method} (${expected} valores)`, () => {
    it('reports the expected total length from the structure', () => {
      expect(make().getWeights().length).toBe(expected)
    })

    it('set(get()) keeps the weights identical', () => {
      const m = fresh()
      const w = m.getWeights()
      m.setWeights(m.getWeights())
      expect(m.getWeights()).toEqual(w)
    })

    it('set(get() + 1) writes exactly the given values', () => {
      const m = fresh()
      const shifted = m.getWeights().map(v => v + 1)
      m.setWeights(shifted)
      expect(m.getWeights()).toEqual(shifted)
    })

    it('rejects a short array (one value less) with name and both lengths', () => {
      const m = fresh()
      const short = m.getWeights().slice(0, expected - 1)
      expectRejectedAndIntact(
        m, short,
        `${method}: expected array of length ${expected}, got ${expected - 1}`,
      )
    })

    it('rejects a long array (one value more) with name and both lengths', () => {
      const m = fresh()
      const long = [...m.getWeights(), 0.5]
      expectRejectedAndIntact(
        m, long,
        `${method}: expected array of length ${expected}, got ${expected + 1}`,
      )
    })

    it('rejects NaN inside a correctly sized array', () => {
      const m = fresh()
      const bad = m.getWeights()
      bad[bad.length - 1] = NaN
      expectRejectedAndIntact(m, bad, `${method}: invalid value at index ${expected - 1}`)
    })

    it('rejects Infinity inside a correctly sized array', () => {
      const m = fresh()
      const bad = m.getWeights()
      bad[0] = Infinity
      expectRejectedAndIntact(m, bad, `${method}: invalid value at index 0`)
    })

    it('rejects undefined', () => {
      const m = fresh()
      expectRejectedAndIntact(m, undefined, `${method}: expected array, got undefined`)
    })

    it('rejects a value that is not an array', () => {
      const m = fresh()
      expectRejectedAndIntact(m, { length: expected }, `${method}: expected array`)
    })
  })
}

// AttentionHead: Wq, Wk, Wv → (2d_k + d_v) · d_model
battery('AttentionHead', () => new AttentionHead(4, 2, 2), (2 + 2 + 2) * 4)
battery('AttentionHead', () => new AttentionHead(4, 3, 1), (3 + 3 + 1) * 4)

// MultiHeadAttention con 1 y 2 cabezas (d_model 4).
// Total = nHeads · 3 · d_k · d_model + d_model · nHeads · d_k = 4 · d_model² = 64.
battery('MultiHeadAttention', () => new MultiHeadAttention(4, 1), 64)
battery('MultiHeadAttention', () => new MultiHeadAttention(4, 2), 64)

// TransformerBlock con 1 y 2 cabezas (d_model 4, d_ff 6).
// attn 64 + norm1 8 + ff1 24 + b1 6 + ff2 24 + b2 4 + norm2 8 = 138.
battery('TransformerBlock', () => new TransformerBlock({ d_model: 4, nHeads: 1, d_ff: 6 }), 138)
battery('TransformerBlock', () => new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 6 }), 138)

describe('TransformerBlock.setWeights, validación en el nivel exterior', () => {
  it('a long array is rejected even though its attn slice would be exactly right', () => {
    const block = new TransformerBlock({ d_model: 4, nHeads: 2, d_ff: 6 })
    setPatternState(block)
    const before = block.getWeights()
    const long = [...before, 0.1, 0.2, 0.3]
    expect(() => block.setWeights(long)).toThrow(
      'TransformerBlock.setWeights: expected array of length 138, got 141',
    )
    expect(block.getWeights()).toEqual(before)
  })
})
