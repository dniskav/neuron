// T009 (feature 005, I4 + N5, Q4): relleno 'same' asimetrico por eje en Conv2D.
//   left/top = floor((k-1)/2), right/bottom = k-1-left  (estilo TensorFlow).
// Con kernels impares en ambos ejes el comportamiento es identico al anterior
// (golden contra una replica del codigo antiguo, mas abajo).
import { describe, it, expect } from 'vitest'
import { Conv2D } from '../../src/Conv2D'
import { SGD } from '../../src/optimizers'

type T3 = number[][][]

function makeInput(H: number, W: number, C: number): T3 {
  return Array.from({ length: H }, (_, h) =>
    Array.from({ length: W }, (_, w) =>
      Array.from({ length: C }, (_, c) => Math.sin(1 + h * 1.7 + w * 0.9 + c * 2.3)),
    ),
  )
}

function makeDOut(shape: [number, number, number]): T3 {
  const [oh, ow, f] = shape
  return Array.from({ length: oh }, (_, h) =>
    Array.from({ length: ow }, (_, w) =>
      Array.from({ length: f }, (_, k) => Math.cos(0.5 + h * 1.3 + w * 0.7 + k * 1.9)),
    ),
  )
}

// ── Replica del codigo ANTIGUO (padding simetrico floor(k/2), SGD escalar) ──
function oldConv(
  H: number, W: number, C: number, kH: number, kW: number, F: number,
  stride: number, padding: 'valid' | 'same',
  weights: number[], input: T3, dOutFn: (shape: [number, number, number]) => T3, lr: number,
) {
  const kernels: number[][][][] = []
  let idx = 0
  for (let f = 0; f < F; f++) {
    const a: number[][][] = []
    for (let i = 0; i < kH; i++) {
      const b: number[][] = []
      for (let j = 0; j < kW; j++) {
        const c: number[] = []
        for (let k = 0; k < C; k++) c.push(weights[idx++])
        b.push(c)
      }
      a.push(b)
    }
    kernels.push(a)
  }
  const biases: number[] = []
  for (let f = 0; f < F; f++) biases.push(weights[idx++])

  const padH = padding === 'same' ? Math.floor(kH / 2) : 0
  const padW = padding === 'same' ? Math.floor(kW / 2) : 0
  const pH = H + 2 * padH
  const pW = W + 2 * padW
  const padded: T3 = Array.from({ length: pH }, () =>
    Array.from({ length: pW }, () => new Array(C).fill(0)),
  )
  for (let h = 0; h < H; h++)
    for (let w = 0; w < W; w++)
      for (let c = 0; c < C; c++) padded[h + padH][w + padW][c] = input[h][w][c]
  const outH = Math.floor((H - kH + 2 * padH) / stride) + 1
  const outW = Math.floor((W - kW + 2 * padW) / stride) + 1
  const output: T3 = Array.from({ length: outH }, () =>
    Array.from({ length: outW }, () => new Array(F).fill(0)),
  )
  for (let f = 0; f < F; f++)
    for (let h = 0; h < outH; h++)
      for (let w = 0; w < outW; w++) {
        let sum = biases[f]
        for (let kh = 0; kh < kH; kh++)
          for (let kw = 0; kw < kW; kw++)
            for (let c = 0; c < C; c++)
              sum += kernels[f][kh][kw][c] * padded[h * stride + kh][w * stride + kw][c]
        output[h][w][f] = sum
      }

  const dOutput = dOutFn([outH, outW, F])
  const dK = kernels.map((a) => a.map((b) => b.map((c) => c.map(() => 0))))
  const dB = new Array(F).fill(0)
  const dPadded: T3 = Array.from({ length: pH }, () =>
    Array.from({ length: pW }, () => new Array(C).fill(0)),
  )
  for (let f = 0; f < F; f++)
    for (let h = 0; h < outH; h++)
      for (let w = 0; w < outW; w++) {
        const dv = dOutput[h][w][f]
        dB[f] += dv
        for (let kh = 0; kh < kH; kh++)
          for (let kw = 0; kw < kW; kw++)
            for (let c = 0; c < C; c++) {
              const ph = h * stride + kh
              const pw = w * stride + kw
              dK[f][kh][kw][c] += dv * padded[ph][pw][c]
              dPadded[ph][pw][c] += dv * kernels[f][kh][kw][c]
            }
      }
  const sgd = new SGD()
  const flat: number[] = []
  for (let f = 0; f < F; f++)
    for (let kh = 0; kh < kH; kh++)
      for (let kw = 0; kw < kW; kw++)
        for (let c = 0; c < C; c++) flat.push(sgd.step(kernels[f][kh][kw][c], dK[f][kh][kw][c], lr))
  for (let f = 0; f < F; f++) flat.push(sgd.step(biases[f], dB[f], lr))
  const dInput = dPadded.slice(padH, padH + H).map((row) => row.slice(padW, padW + W))
  return { output, dInput, weightsAfter: flat }
}

describe('Conv2D same padding (I4, N5)', () => {
  it('propiedad: forward mide outputShape() y ceil(H/stride) x ceil(W/stride)', () => {
    for (const [H, W] of [[6, 7], [4, 4], [5, 3]]) {
      for (let kH = 1; kH <= 5; kH++)
        for (let kW = 1; kW <= 5; kW++)
          for (const stride of [1, 2]) {
            const conv = new Conv2D(H, W, 2, [kH, kW], 3, { padding: 'same', stride })
            const y = conv.forward(makeInput(H, W, 2))
            const eh = Math.ceil(H / stride)
            const ew = Math.ceil(W / stride)
            const label = `${H}x${W} k=[${kH},${kW}] s=${stride}`
            expect(conv.outputShape(), label).toEqual([eh, ew, 3])
            expect(y.length, label).toBe(eh)
            expect(y.every((r) => r.length === ew && r.every((c) => c.length === 3)), label).toBe(true)
            const dIn = conv.backward(makeDOut([eh, ew, 3]), 0.01)
            expect(dIn.length, label).toBe(H)
            expect(dIn.every((r) => r.length === W), label).toBe(true)
          }
    }
  })

  it('4x4 con k=4 da 4x4 (antes 5x5)', () => {
    const conv = new Conv2D(4, 4, 1, 4, 1, { padding: 'same' })
    expect(conv.outputShape()).toEqual([4, 4, 1])
    const y = conv.forward(makeInput(4, 4, 1))
    expect(y.length).toBe(4)
    expect(y[0].length).toBe(4)
  })

  it("'valid' no cambia (cualquier k)", () => {
    const conv = new Conv2D(6, 7, 1, [4, 3], 2, { padding: 'valid', stride: 2 })
    expect(conv.outputShape()).toEqual([2, 3, 2])
    const y = conv.forward(makeInput(6, 7, 1))
    expect([y.length, y[0].length]).toEqual([2, 3])
  })

  it('golden: kernels impares en ambos ejes == replica del codigo antiguo (bit a bit)', () => {
    // H, W, kH, kW, stride, C, padding
    const cases: Array<[number, number, number, number, number, number, 'valid' | 'same']> = [
      [5, 5, 3, 3, 1, 1, 'same'],
      [6, 7, 3, 5, 1, 2, 'same'],
      [7, 6, 5, 1, 2, 2, 'same'],
      [8, 8, 3, 3, 2, 3, 'same'],
      [5, 4, 1, 1, 1, 2, 'same'],
      [7, 7, 5, 5, 1, 1, 'same'],
      [6, 6, 3, 3, 1, 2, 'valid'],
      [7, 6, 4, 2, 2, 1, 'valid'],
    ]
    for (const [H, W, kH, kW, stride, C, padding] of cases) {
      const F = 2
      const conv = new Conv2D(H, W, C, [kH, kW], F, {
        padding,
        stride,
        optimizerFactory: () => new SGD(),
      })
      const w0 = conv.getWeights().map((v, i) => v + 0.01 * i)
      conv.setWeights(w0)
      const input = makeInput(H, W, C)
      const old = oldConv(H, W, C, kH, kW, F, stride, padding, w0, input, makeDOut, 0.05)
      const y = conv.forward(input)
      const dIn = conv.backward(makeDOut(conv.outputShape()), 0.05)
      const label = `${H}x${W} k=[${kH},${kW}] s=${stride} C=${C} ${padding}`
      expect(y, label).toEqual(old.output)
      expect(dIn, label).toEqual(old.dInput)
      expect(conv.getWeights(), label).toEqual(old.weightsAfter)
    }
  })

  it('caso a mano k=2 en 3x3 de un canal: pad solo abajo/derecha', () => {
    const conv = new Conv2D(3, 3, 1, 2, 1, { padding: 'same', optimizerFactory: () => new SGD() })
    conv.setWeights([1, 2, 3, 4, 0.5])
    const x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].map((r) => r.map((v) => [v]))
    const y = conv.forward(x)
    const expected = [
      [37.5, 47.5, 21.5],
      [67.5, 77.5, 33.5],
      [23.5, 26.5, 9.5],
    ]
    expect(conv.outputShape()).toEqual([3, 3, 1])
    for (let h = 0; h < 3; h++)
      for (let w = 0; w < 3; w++) expect(y[h][w][0]).toBeCloseTo(expected[h][w], 12)
    // dOutput = 1: dx[i][j] = a + b[j>=1] + c[i>=1] + d[i>=1 && j>=1]
    const ones = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => [1]))
    const dIn = conv.backward(ones, 0)
    const dx = [
      [1, 3, 3],
      [4, 10, 10],
      [4, 10, 10],
    ]
    for (let h = 0; h < 3; h++)
      for (let w = 0; w < 3; w++) expect(dIn[h][w][0]).toBeCloseTo(dx[h][w], 12)
  })

  it('errores de forma lanzan', () => {
    const conv = new Conv2D(4, 4, 1, 3, 2, { padding: 'same', stride: 2 })
    conv.forward(makeInput(4, 4, 1))
    // outShape = [2, 2, 2]
    expect(() => conv.backward(makeDOut([2, 2, 2]).slice(0, 1), 0.1)).toThrow(/Conv2D\.backward/)
    expect(() =>
      conv.backward(makeDOut([2, 2, 2]).map((r) => r.slice(0, 1)), 0.1),
    ).toThrow(/Conv2D\.backward/)
    expect(() => conv.backward(makeDOut([2, 2, 1]), 0.1)).toThrow(/Conv2D\.backward/)
    expect(() => conv.backward(makeDOut([4, 4, 2]), 0.1)).toThrow(/Conv2D\.backward/)
    expect(() => conv.backward([], 0.1)).toThrow(/Conv2D\.backward/)
    // demasiadas filas (anchura y filtros correctos)
    expect(() => conv.backward(makeDOut([3, 2, 2]), 0.1)).toThrow(/Conv2D\.backward/)
    // valor no finito
    const nan = makeDOut([2, 2, 2])
    nan[1][1][0] = NaN
    expect(() => conv.backward(nan, 0.1)).toThrow(/Conv2D\.backward/)
    // un dOutput valido sigue funcionando tras los errores
    expect(() => conv.backward(makeDOut([2, 2, 2]), 0.1)).not.toThrow()
  })

  it('backward con dOutput invalido no modifica los pesos', () => {
    const conv = new Conv2D(4, 4, 1, 3, 2, { padding: 'same' })
    conv.forward(makeInput(4, 4, 1))
    const before = conv.getWeights()
    expect(() => conv.backward(makeDOut([3, 4, 2]), 0.1)).toThrow()
    expect(conv.getWeights()).toEqual(before)
  })

  it('setWeights valida filters*kH*kW*channels + filters', () => {
    const conv = new Conv2D(4, 4, 2, [2, 3], 3)
    const n = 3 * 2 * 3 * 2 + 3
    const w = conv.getWeights()
    expect(w).toHaveLength(n)
    expect(() => conv.setWeights(w.slice(1))).toThrow(/Conv2D\.setWeights/)
    expect(() => conv.setWeights([...w, 0])).toThrow(/Conv2D\.setWeights/)
    const bad = w.slice()
    bad[2] = NaN
    expect(() => conv.setWeights(bad)).toThrow(/Conv2D\.setWeights/)
    expect(() => conv.setWeights(5 as unknown as number[])).toThrow(/Conv2D\.setWeights/)
    expect(conv.getWeights()).toEqual(w)
    conv.setWeights(w.map((v) => v + 1))
    expect(conv.getWeights()).toEqual(w.map((v) => v + 1))
    conv.setWeights(w)
    expect(conv.getWeights()).toEqual(w)
  })
})

// T022: validacion de stride y valid con kernel mayor que la entrada
describe('Conv2D stride and valid-kernel validation (T022)', () => {
  for (const bad of [0, -1, 1.5, NaN, Infinity]) {
    it(`stride ${bad} lanza Error con texto`, () => {
      expect(() => new Conv2D(4, 4, 1, 2, 1, { stride: bad })).toThrow(/Conv2D: stride must be a finite integer >= 1/)
    })
  }
  it('valid con k > H lanza Error claro', () => {
    expect(() => new Conv2D(2, 4, 1, [3, 2], 1)).toThrow(/cannot exceed input.*valid padding/)
  })
  it('valid con k > W lanza Error claro', () => {
    expect(() => new Conv2D(4, 2, 1, [2, 3], 1, { padding: 'valid' })).toThrow(/cannot exceed input.*valid padding/)
  })
  it('valid con k == H y same con k > H siguen permitidos', () => {
    expect(new Conv2D(3, 3, 1, 3, 1).outputShape()).toEqual([1, 1, 1])
    expect(new Conv2D(2, 2, 1, 3, 1, { padding: 'same' }).outputShape()).toEqual([2, 2, 1])
  })
})
