// Gradient-oracle test for Conv1D.
//
// Same template as tests/oracle/NetworkN.oracle.test.ts:
//   1. smoke   - forward/backward finitos, forma de la salida y numero de pesos.
//   2. oracle  - expectGradientMatches: el paso REAL de backward(dOut, lr) sigue
//                -dL/dw (coseno > 0.999). Perdida L = 0.5 * sum (t - y)^2 y
//                dOut = -dL/dy = t - y (convencion de Conv1D.backward).
//   3. learns  - expectLearns: la perdida sobre datos fijos baja a menos de la mitad.
// Extra: forward comparado con convoluciones calculadas a mano (valores esperados
// independientes del codigo) para valid y same.
// Entrada sin valores cercanos a cero y sin activacion: Conv1D es lineal en sus pesos.
// Los pesos iniciales los fija la semilla global (tests/setup.ts).
import { describe, it, expect } from 'vitest'
import { Conv1D } from '../../src/Conv1D'
import { SGD } from '../../src/optimizers'
import { expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const INPUT_LEN = 6
const KERNEL = 3
const FILTERS = 2
const CHANNELS = 2

// [inputLength][inputChannels], todos |x| >= 0.1
const INPUT = [
  [0.5, -0.4],
  [-0.3, 0.7],
  [0.8, 0.2],
  [-0.6, 0.9],
  [0.4, -0.5],
  [0.1, 0.6],
]

// [filters][outputLength]; valid => 4 posiciones, same => 6 posiciones.
const TARGET_FULL = [
  [0.3, -0.2, 0.5, 0.1, 0.4, -0.1],
  [-0.4, 0.2, 0.0, 0.6, -0.3, 0.25],
]

function setup(padding: 'valid' | 'same' = 'valid') {
  const conv = new Conv1D(INPUT_LEN, KERNEL, FILTERS, 1, padding, () => new SGD(), CHANNELS)
  const outLen = conv.getOutputLength()
  const target = TARGET_FULL.map((row) => row.slice(0, outLen))
  const loss = () => {
    const y = conv.forward(INPUT)
    let s = 0
    for (let f = 0; f < FILTERS; f++)
      for (let p = 0; p < outLen; p++) s += 0.5 * (target[f][p] - y[f][p]) ** 2
    return s
  }
  return {
    conv,
    loss,
    // ONE real update: forward refresca la cache, backward aplica w + lr*g.
    step: (lr: number) => {
      const y = conv.forward(INPUT)
      const dOut = y.map((row, f) => row.map((v, p) => target[f][p] - v))
      conv.backward(dOut, lr)
    },
    getParams: () => conv.getWeights(),
    setParams: (p: number[]) => conv.setWeights(p),
  }
}

describe('Conv1D oraculo', () => {
  it('humo: forward con forma [filters][outputLength], backward finito', () => {
    const m = setup('valid')
    // valid: 6 - 3 + 1 = 4 posiciones; pesos = 2*3*2 kernels + 2 sesgos = 14
    expect(m.conv.getOutputLength()).toBe(4)
    expect(m.getParams()).toHaveLength(14)
    const y = m.conv.forward(INPUT)
    expect(y).toHaveLength(FILTERS)
    expect(y.every((row) => row.length === 4)).toBe(true)
    expect(y.flat().every(Number.isFinite)).toBe(true)
    const dIn = m.conv.backward(y.map((row) => row.map(() => 0.1)), 0.01)
    expect(dIn).toHaveLength(INPUT_LEN)
    expect(dIn.every((r) => r.length === CHANNELS && r.every(Number.isFinite))).toBe(true)
  })

  it('forward valid coincide con la convolucion calculada a mano', () => {
    // kernel [2, -1, 0.5], sesgo 0.1, entrada [1, 2, 3, 4] (1 canal)
    // pos0: 2*1 - 1*2 + 0.5*3 + 0.1 = 1.6 ; pos1: 2*2 - 1*3 + 0.5*4 + 0.1 = 3.1
    const conv = new Conv1D(4, 3, 1, 1, 'valid', () => new SGD(), 1)
    conv.setWeights([2, -1, 0.5, 0.1])
    const y = conv.forward([1, 2, 3, 4])
    expect(y[0][0]).toBeCloseTo(1.6, 12)
    expect(y[0][1]).toBeCloseTo(3.1, 12)
  })

  it('forward same (relleno 1 a cada lado) coincide con la convolucion calculada a mano', () => {
    // kernel [1, 1, 1], sesgo 0, entrada [1, 2, 3] => padded [0,1,2,3,0]
    // salidas: 0+1+2 = 3 ; 1+2+3 = 6 ; 2+3+0 = 5
    const conv = new Conv1D(3, 3, 1, 1, 'same', () => new SGD(), 1)
    conv.setWeights([1, 1, 1, 0])
    const y = conv.forward([1, 2, 3])
    expect(y[0]).toHaveLength(3)
    expect(y[0][0]).toBeCloseTo(3, 12)
    expect(y[0][1]).toBeCloseTo(6, 12)
    expect(y[0][2]).toBeCloseTo(5, 12)
  })

  // ratio = |delta| / (lr |g|): con SGD y gradiente correcto vale 1 (la escala tambien cuenta).
  it('oraculo valid: el paso sigue -dL/dw', () => {
    const r = expectGradientMatches(setup('valid'))
    expect(r.cosine).toBeGreaterThan(0.999)
    expect(r.ratio).toBeCloseTo(1, 6)
  })

  it('oraculo same: el paso sigue -dL/dw', () => {
    const r = expectGradientMatches(setup('same'))
    expect(r.cosine).toBeGreaterThan(0.999)
    expect(r.ratio).toBeCloseTo(1, 6)
  })

  it('expectLearns valid: la perdida baja a la mitad', () => {
    const m = setup('valid')
    expectLearns({ loss: m.loss, step: () => m.step(0.05), steps: 200 })
  })
})

// ── Oraculo de dInput por componente ────────────────────────────────────────
// L = 1/2 * sum (y - t)^2 sobre la ENTRADA. Conv1D es lineal en la entrada, asi que
// la diferencia central con h = 1e-4 es exacta salvo redondeo (~1e-11): el umbral
// 1e-6 deja cinco ordenes de margen. Pesos, entrada y objetivo salen de formulas
// deterministas (sin semilla). backward(dOut = y - t) devuelve dL/dx; lr = 0 no
// modifica los pesos.
// Limite: el oraculo compara backward con forward del mismo codigo, asi que un
// intercambio de left/right aplicado a los dos no se detecta aqui; lo cubre el caso
// calculado a mano (k = 4, L = 6, same) de tests/api/Conv1D.padding.test.ts.
describe('Conv1D oraculo de dInput (diferencias finitas)', () => {
  const val = (i: number, salt: number) => Math.sin(1.9 * i + salt) * 0.8 + 0.1

  const configs: Array<[string, number, 'valid' | 'same', number]> = [
    ['valid k=3 stride 1', 3, 'valid', 1],
    ['valid k=4 stride 1', 4, 'valid', 1],
    ['same k=3 (impar)', 3, 'same', 1],
    ['same k=5 (impar)', 5, 'same', 1],
    ['same k=2 (par)', 2, 'same', 1],
    ['same k=4 (par)', 4, 'same', 1],
    ['same k=6 (par)', 6, 'same', 1],
    ['valid k=3 stride 2', 3, 'valid', 2],
    ['same k=3 stride 2', 3, 'same', 2],
    ['same k=4 stride 2', 4, 'same', 2],
    ['same k=4 stride 3', 4, 'same', 3],
  ]

  for (const ch of [1, 2]) {
    for (const [label, k, padding, stride] of configs) {
      it(`${label}, ${ch} canal(es)`, () => {
        const L = 9
        const filters = 2
        const conv = new Conv1D(L, k, filters, stride, padding, () => new SGD(), ch)
        conv.setWeights(Array.from({ length: filters * k * ch + filters }, (_, i) => val(i, 0.2)))
        const x0 = Array.from({ length: L }, (_, p) => Array.from({ length: ch }, (_, c) => val(p * ch + c, 1.3)))
        const outLen = conv.getOutputLength()
        const t = Array.from({ length: filters }, (_, f) =>
          Array.from({ length: outLen }, (_, p) => val(f * outLen + p, 2.1)))
        const loss = (x: number[][]) => {
          const y = conv.forward(x)
          let s = 0
          for (let f = 0; f < filters; f++)
            for (let p = 0; p < outLen; p++) s += 0.5 * (y[f][p] - t[f][p]) ** 2
          return s
        }
        const y = conv.forward(x0)
        const analytic = conv.backward(y.map((row, f) => row.map((v, p) => v - t[f][p])), 0)
        const h = 1e-4
        for (let p = 0; p < L; p++)
          for (let c = 0; c < ch; c++) {
            const xp = x0.map((r) => [...r]); xp[p][c] += h
            const xm = x0.map((r) => [...r]); xm[p][c] -= h
            const numeric = (loss(xp) - loss(xm)) / (2 * h)
            expect(Math.abs(analytic[p][c] - numeric), `x[${p}][${c}]`).toBeLessThan(1e-6)
          }
      })
    }
  }
})
