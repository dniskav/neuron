// T008 (I4, Q4): Conv1D con padding 'same' y kernel par, validacion de dOut y setWeights.
//
// Relleno 'same': total = k - 1, left = floor(total / 2), right = total - left.
// Con k impar left = right (comportamiento anterior, bit a bit); con k par el
// relleno es asimetrico (estilo TensorFlow) y la salida mide ceil(L / stride).
//
// Los valores de pesos y entradas son formulas deterministas (sin semilla ni
// Math.random), asi que los tests no dependen de la plataforma. Las comparaciones
// golden usan toEqual (exactas): la replica antigua hace las mismas operaciones
// en el mismo orden.
import { describe, it, expect } from 'vitest'
import { Conv1D } from '../../src/Conv1D'
import { SGD } from '../../src/optimizers'

const val = (i: number, salt: number) => Math.sin(1.7 * i + salt) * 0.9 + 0.05

function makeConv(L: number, k: number, filters: number, stride: number, padding: 'valid' | 'same', ch: number) {
  const conv = new Conv1D(L, k, filters, stride, padding, () => new SGD(), ch)
  const n = filters * k * ch + filters
  conv.setWeights(Array.from({ length: n }, (_, i) => val(i, 0.3)))
  return conv
}

const makeInput = (L: number, ch: number) =>
  Array.from({ length: L }, (_, p) => Array.from({ length: ch }, (_, c) => val(p * ch + c, 1.1)))

// ── Replica del codigo ANTIGUO (relleno floor((k-1)/2) por cada lado) ───────
function oldRun(
  kernels: number[][][], biases: number[], input: number[][], k: number, stride: number,
  padding: 'valid' | 'same', ch: number, dOut: (y: number[][]) => number[][], lr: number,
) {
  const filters = kernels.length
  const L = input.length
  let padded = input
  const padSize = Math.floor((k - 1) / 2)
  if (padding === 'same') {
    const row = () => new Array(ch).fill(0)
    padded = new Array(padSize).fill(null).map(row).concat(input).concat(new Array(padSize).fill(null).map(row))
  }
  const outLen = Math.floor((padded.length - k) / stride) + 1
  const y: number[][] = Array.from({ length: filters }, () => new Array(outLen).fill(0))
  for (let f = 0; f < filters; f++)
    for (let pos = 0; pos < outLen; pos++) {
      let sum = biases[f]
      for (let kk = 0; kk < k; kk++) for (let c = 0; c < ch; c++) sum += kernels[f][kk][c] * padded[pos * stride + kk][c]
      y[f][pos] = sum
    }
  const d = dOut(y)
  const dK = kernels.map(kr => kr.map(() => new Array(ch).fill(0)))
  const dB = new Array(filters).fill(0)
  const dP = padded.map(() => new Array(ch).fill(0))
  for (let f = 0; f < filters; f++)
    for (let pos = 0; pos < outLen; pos++) {
      dB[f] += d[f][pos]
      for (let kk = 0; kk < k; kk++)
        for (let c = 0; c < ch; c++) {
          dK[f][kk][c] += d[f][pos] * padded[pos * stride + kk][c]
          dP[pos * stride + kk][c] += d[f][pos] * kernels[f][kk][c]
        }
    }
  // SGD: w + lr * g
  const newK = kernels.map((kr, f) => kr.map((r, kk) => r.map((w, c) => w + lr * dK[f][kk][c])))
  const newB = biases.map((b, f) => b + lr * dB[f])
  const dIn = padding === 'same' ? dP.slice(padSize, padSize + L) : dP.slice(0, L)
  return { y, dIn, newK, newB }
}

describe('Conv1D padding: propiedad de forma', () => {
  it('same: forward mide getOutputLength() === ceil(L/stride), k 1..6, stride 1..3, L 5..12, 1 y 2 canales', () => {
    for (let k = 1; k <= 6; k++)
      for (let stride = 1; stride <= 3; stride++)
        for (let L = 5; L <= 12; L++)
          for (const ch of [1, 2]) {
            const conv = makeConv(L, k, 2, stride, 'same', ch)
            const y = conv.forward(makeInput(L, ch))
            const expected = Math.ceil(L / stride)
            const tag = `k=${k} s=${stride} L=${L} ch=${ch}`
            expect(y, tag).toHaveLength(2)
            for (const row of y) expect(row.length, tag).toBe(expected)
            expect(conv.getOutputLength(), tag).toBe(expected)
            // backward con la forma anunciada devuelve [L][ch]
            const dIn = conv.backward(y.map(r => r.map(() => 0.1)), 0.01)
            expect(dIn, tag).toHaveLength(L)
            for (const r of dIn) expect(r, tag).toHaveLength(ch)
          }
  })

  it('valid: forward mide getOutputLength() === floor((L-k)/stride)+1', () => {
    for (let k = 1; k <= 5; k++)
      for (let stride = 1; stride <= 3; stride++)
        for (let L = 5; L <= 12; L++)
          for (const ch of [1, 2]) {
            const conv = makeConv(L, k, 2, stride, 'valid', ch)
            const y = conv.forward(makeInput(L, ch))
            const expected = Math.floor((L - k) / stride) + 1
            for (const row of y) expect(row.length).toBe(expected)
            expect(conv.getOutputLength()).toBe(expected)
          }
  })
})

describe('Conv1D padding: golden con k impar (identico a la replica antigua)', () => {
  const cases: Array<['valid' | 'same', number, number, number, number]> = []
  for (const padding of ['valid', 'same'] as const)
    for (const k of [1, 3, 5])
      for (const stride of [1, 2])
        for (const ch of [1, 2]) cases.push([padding, k, stride, ch, 9])

  it.each(cases)('%s k=%i stride=%i ch=%i L=%i: salida, dInput y pesos tras backward identicos', (padding, k, stride, ch, L) => {
    const conv = makeConv(L, k, 2, stride, padding, ch)
    const input = makeInput(L, ch)
    const kernels = conv.kernels.map(kr => kr.map(r => [...r]))
    const biases = [...conv.biases]
    const dOutFn = (y: number[][]) => y.map((r, f) => r.map((v, p) => Math.cos(v + f + 0.37 * p)))
    const old = oldRun(kernels, biases, input, k, stride, padding, ch, dOutFn, 0.05)

    const y = conv.forward(input)
    expect(y).toEqual(old.y)
    const dIn = conv.backward(dOutFn(y), 0.05)
    expect(dIn).toEqual(old.dIn)
    expect(conv.kernels).toEqual(old.newK)
    expect(conv.biases).toEqual(old.newB)
  })
})

describe('Conv1D padding: k par calculado a mano', () => {
  // k = 4, L = 6, same => total = 3, left = 1, right = 2.
  // entrada [1..6], kernel [1,2,3,4], sesgo 0.5; padded = [0,1,2,3,4,5,6,0,0] (9)
  //   pos0: [0,1,2,3] -> 0+2+6+12   = 20 + 0.5 = 20.5
  //   pos1: [1,2,3,4] -> 1+4+9+16   = 30 + 0.5 = 30.5
  //   pos2: [2,3,4,5] -> 2+6+12+20  = 40 + 0.5 = 40.5
  //   pos3: [3,4,5,6] -> 3+8+15+24  = 50 + 0.5 = 50.5
  //   pos4: [4,5,6,0] -> 4+10+18+0  = 32 + 0.5 = 32.5
  //   pos5: [5,6,0,0] -> 5+12+0+0   = 17 + 0.5 = 17.5
  // Con dOut = 1 en todas las posiciones, dInput[i] suma los pesos w[j] con
  // i + left = pos + j para algun pos en 0..5:
  //   x0 (padded 1): pos 0,1 -> w1 + w0 = 3
  //   x1 (padded 2): pos 0,1,2 -> w2 + w1 + w0 = 6
  //   x2, x3, x4 (padded 3, 4, 5): cuatro posiciones -> 1+2+3+4 = 10
  //   x5 (padded 6): pos 3,4,5 -> w3 + w2 + w1 = 9
  it('salida y dInput', () => {
    const conv = new Conv1D(6, 4, 1, 1, 'same', () => new SGD(), 1)
    conv.setWeights([1, 2, 3, 4, 0.5])
    const y = conv.forward([1, 2, 3, 4, 5, 6])
    expect(y).toHaveLength(1)
    expect(y[0]).toHaveLength(6)
    const expected = [20.5, 30.5, 40.5, 50.5, 32.5, 17.5]
    expected.forEach((v, i) => expect(y[0][i]).toBeCloseTo(v, 12))
    const dIn = conv.backward([[1, 1, 1, 1, 1, 1]], 0)
    const exp = [3, 6, 10, 10, 10, 9]
    exp.forEach((v, i) => expect(dIn[i][0]).toBeCloseTo(v, 12))
    expect(conv.getOutputLength()).toBe(6)
  })
})

describe('Conv1D validacion', () => {
  it('backward con dOut de forma erronea lanza y no modifica pesos', () => {
    const conv = makeConv(8, 3, 2, 1, 'same', 2)
    const y = conv.forward(makeInput(8, 2))
    const before = conv.getWeights()
    const good = y.map(r => r.map(() => 0.1))
    const bad: Array<[string, unknown]> = [
      ['menos filtros', good.slice(0, 1)],
      ['mas filtros', [...good, good[0]]],
      ['fila corta', [good[0].slice(0, -1), good[1]]],
      ['fila larga', [[...good[0], 0.1], good[1]]],
      ['NaN', [[...good[0].slice(1), NaN], good[1]]],
      ['no array', 5],
      ['vacio', []],
    ]
    for (const [label, d] of bad) {
      expect(() => conv.backward(d as number[][], 0.1), label).toThrow(/Conv1D\.backward/)
      expect(conv.getWeights(), label).toEqual(before)
    }
    // sigue funcionando tras un error
    expect(() => conv.backward(good, 0.1)).not.toThrow()
  })

  it('stride 2: la forma de dOut es la de stride 2, no la de stride 1', () => {
    const conv = makeConv(9, 3, 1, 2, 'same', 1)
    conv.forward(makeInput(9, 1))
    expect(() => conv.backward([new Array(9).fill(0.1)], 0.1)).toThrow(/Conv1D\.backward/)
    expect(() => conv.backward([new Array(5).fill(0.1)], 0.1)).not.toThrow()
  })

  it('setWeights invalido lanza con Conv1D.setWeights y deja el estado intacto', () => {
    const conv = makeConv(8, 3, 2, 1, 'valid', 2)
    const w = conv.getWeights()
    expect(w).toHaveLength(2 * 3 * 2 + 2)
    const shifted = (a: number[]) => a.map(v => v + 0.5)
    const bad: unknown[] = [
      shifted(w.slice(0, -1)),
      [...shifted(w), 1],
      shifted(w).map((v, i) => (i === 0 ? NaN : v)),
      undefined,
      'abc',
    ]
    for (const b of bad) {
      expect(() => conv.setWeights(b as number[])).toThrow(/Conv1D\.setWeights/)
      expect(conv.getWeights()).toEqual(w)
    }
    conv.setWeights(w)
    expect(conv.getWeights()).toEqual(w)
    conv.setWeights(shifted(w))
    expect(conv.getWeights()).toEqual(shifted(w))
  })
})
