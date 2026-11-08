// Gradient-oracle test for Conv2D.
//
// Same template as tests/oracle/Conv1D.oracle.test.ts:
//   1. smoke   - forma de la salida, numero de pesos, forward/backward finitos.
//   2. oracle  - expectGradientMatches: el paso REAL de backward(dOut, lr) sigue
//                -dL/dw (coseno > 0.999). L = 0.5 * sum (t - y)^2 y
//                dOut = -dL/dy = t - y (convencion de Conv2D.backward).
//   3. learns  - expectLearns: la perdida sobre datos fijos baja a menos de la mitad.
// Extra: forward valid comparado con una convolucion calculada a mano (valores
// esperados independientes del codigo). En el oraculo de pesos solo se asserta el coseno:
// la escala del gradiente no se compara.
// Conv2D es lineal en sus pesos (sin activacion). Los pesos iniciales los fija la
// semilla global (tests/setup.ts). Conv2D usa SGD escalar por defecto: no hace falta
// withPlainSGD (ese wrapper es solo para WeightMatrix/Adam).
import { describe, it, expect } from 'vitest'
import { Conv2D } from '../../src/Conv2D'
import { SGD } from '../../src/optimizers'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const H = 4
const W = 4
const CHANNELS = 1
const KERNEL = 2
const FILTERS = 2
const OUT = 3 // valid: 4 - 2 + 1

// [H][W][C], todos |x| >= 0.1
const INPUT: number[][][] = [
  [[0.5], [-0.3], [0.8], [-0.6]],
  [[0.4], [0.7], [-0.5], [0.2]],
  [[-0.9], [0.1], [0.6], [-0.4]],
  [[0.3], [-0.7], [0.25], [0.85]],
]

// [outH][outW][filters]
const TARGET: number[][][] = [
  [
    [0.3, -0.2],
    [0.5, 0.1],
    [-0.4, 0.6],
  ],
  [
    [0.2, 0.7],
    [-0.1, 0.4],
    [0.9, -0.3],
  ],
  [
    [0.0, 0.8],
    [0.25, -0.5],
    [0.6, 0.1],
  ],
]

function setup() {
  const conv = new Conv2D(H, W, CHANNELS, KERNEL, FILTERS, {
    padding: 'valid',
    optimizerFactory: () => new SGD(),
  })
  const loss = () => {
    const y = conv.forward(INPUT)
    let s = 0
    for (let h = 0; h < OUT; h++)
      for (let w = 0; w < OUT; w++)
        for (let f = 0; f < FILTERS; f++) s += 0.5 * (TARGET[h][w][f] - y[h][w][f]) ** 2
    return s
  }
  return {
    conv,
    loss,
    // ONE real update: forward refresca la cache, backward aplica w + lr*g.
    step: (lr: number) => {
      const y = conv.forward(INPUT)
      const dOut = y.map((row, h) => row.map((cell, w) => cell.map((v, f) => TARGET[h][w][f] - v)))
      conv.backward(dOut, lr)
    },
    getParams: () => conv.getWeights(),
    setParams: (p: number[]) => conv.setWeights(p),
  }
}

describe('Conv2D oraculo', () => {
  it('humo: forward con forma [outH][outW][filters], backward finito', () => {
    const m = setup()
    // valid: 3x3 posiciones; pesos = 2 filtros * 2*2*1 kernels + 2 sesgos = 10
    expect(m.conv.outputShape()).toEqual([3, 3, 2])
    expect(m.getParams()).toHaveLength(10)
    const y = m.conv.forward(INPUT)
    expect(y).toHaveLength(OUT)
    expect(y.every((row) => row.length === OUT && row.every((c) => c.length === FILTERS))).toBe(
      true,
    )
    expect(y.flat(2).every(Number.isFinite)).toBe(true)
    const dIn = m.conv.backward(
      y.map((row) => row.map((c) => c.map(() => 0.1))),
      0.01,
    )
    expect(dIn).toHaveLength(H)
    expect(dIn.every((r) => r.length === W && r.every((c) => c.length === CHANNELS && Number.isFinite(c[0])))).toBe(true)
  })

  it('forward valid coincide con la convolucion calculada a mano', () => {
    const conv = new Conv2D(4, 4, 1, 2, 2, { padding: 'valid', optimizerFactory: () => new SGD() })
    // getWeights: filtro0 [[1,2],[0,-1]], filtro1 [[0,1],[0,0]], sesgos [0.5, 0]
    conv.setWeights([1, 2, 0, -1, 0, 1, 0, 0, 0.5, 0])
    const x = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ].map((row) => row.map((v) => [v]))
    const y = conv.forward(x)
    // filtro0(h,w) = x[h][w] + 2 x[h][w+1] - x[h+1][w+1] + 0.5
    const f0 = [
      [-0.5, 1.5, 3.5],
      [7.5, 9.5, 11.5],
      [15.5, 17.5, 19.5],
    ]
    // filtro1(h,w) = x[h][w+1]
    const f1 = [
      [2, 3, 4],
      [6, 7, 8],
      [10, 11, 12],
    ]
    for (let h = 0; h < OUT; h++)
      for (let w = 0; w < OUT; w++) {
        expect(y[h][w][0]).toBeCloseTo(f0[h][w], 12)
        expect(y[h][w][1]).toBeCloseTo(f1[h][w], 12)
      }
  })

  it('oraculo valid: el paso sigue -dL/dw', () => {
    const r = expectGradientMatches(setup())
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('expectLearns valid: la perdida baja a la mitad', () => {
    const m = setup()
    expectLearns({ loss: m.loss, step: () => m.step(0.05), steps: 200 })
  })
})

// ── Oraculo de dInput por componente ──────────────────────────────────────────
// L = sum_{h,w,f} c[h][w][f] * y[h][w][f]  (lineal en la entrada), asi dL/dx es
// exactamente dInput de backward(c, 0). Se compara con diferencias finitas
// centrales sobre CADA componente de la entrada, con toBeCloseTo de 6 decimales
// (|diferencia| < 5e-7).
// Limite: el oraculo compara backward con forward del mismo codigo, asi que un
// intercambio de top/bottom o left/right aplicado a los dos no se detecta aqui; lo
// cubre el caso calculado a mano (k = 2 en 3x3, relleno solo abajo/derecha) de
// tests/api/Conv2D.padding.test.ts.
describe('Conv2D oraculo de dInput (por componente)', () => {
  type Case = [string, number, number, number | [number, number], 'valid' | 'same', number]
  const cases: Case[] = [
    ['valid', 5, 6, 3, 'valid', 1],
    ['same impar', 5, 6, 3, 'same', 1],
    ['same par', 4, 4, 4, 'same', 1],
    ['same rectangular par/impar [4,3]', 5, 6, [4, 3], 'same', 1],
    ['same rectangular impar/par [2,5]', 6, 5, [2, 5], 'same', 1],
    ['valid rectangular [2,3]', 5, 6, [2, 3], 'valid', 1],
    ['same stride 2 impar', 7, 6, 3, 'same', 2],
    ['same stride 2 par [4,2]', 7, 6, [4, 2], 'same', 2],
    ['valid stride 2', 7, 8, [3, 2], 'valid', 2],
  ]
  for (const [name, h, w, k, padding, stride] of cases) {
    it(`dInput == dL/dx por diferencias finitas: ${name}`, () => {
      const C = 2
      const F = 2
      const conv = new Conv2D(h, w, C, k, F, { padding, stride, optimizerFactory: () => new SGD() })
      const x: number[][][] = Array.from({ length: h }, (_, i) =>
        Array.from({ length: w }, (_, j) =>
          Array.from({ length: C }, (_, c) => Math.sin(1 + i * 1.7 + j * 0.9 + c * 2.3)),
        ),
      )
      const [oh, ow] = conv.outputShape()
      const coef: number[][][] = Array.from({ length: oh }, (_, i) =>
        Array.from({ length: ow }, (_, j) =>
          Array.from({ length: F }, (_, f) => Math.cos(0.5 + i * 1.3 + j * 0.7 + f * 1.9)),
        ),
      )
      const loss = () => {
        const y = conv.forward(x)
        let s = 0
        for (let i = 0; i < oh; i++)
          for (let j = 0; j < ow; j++)
            for (let f = 0; f < F; f++) s += coef[i][j][f] * y[i][j][f]
        return s
      }
      loss()
      const dIn = conv.backward(coef, 0)
      expect(dIn).toHaveLength(h)
      const eps = 1e-5
      for (let i = 0; i < h; i++)
        for (let j = 0; j < w; j++)
          for (let c = 0; c < C; c++) {
            const orig = x[i][j][c]
            x[i][j][c] = orig + eps
            const lp = loss()
            x[i][j][c] = orig - eps
            const lm = loss()
            x[i][j][c] = orig
            expect(dIn[i][j][c]).toBeCloseTo((lp - lm) / (2 * eps), 6)
          }
    })
  }
})
