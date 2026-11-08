// Gradient-oracle test for CausalConv1D and TCN (feature 002, T022).
//
//   1. smoke   - forward/backward finitos, formas, causalidad. 
//   2. oracle  - CausalConv1D aislada: backward(dOut, lr) sigue -dL/dw (coseno > 0.999).
//                TCN: train() sigue -dL/dw, coseno > 0.999 y ratio ~ 1.
//   3. learns  - TCN: la perdida sobre datos fijos baja a menos de la mitad.
// Solo se asierta el coseno (Q9). Los campos private de TCN (_outputW/_outputB) se leen
// con `as any` (Q5). El optimizador por defecto es SGD (`w + lr*g`), sin withPlainSGD.
// Perdida de TCN = MSE medio sobre T*outputSize, la que devuelve train().
//
// TCN.train entrega dOut = -2*diff/(T*outputSize) (= -dL/dy de la perdida media) y usa lr
// sin dividir en proyeccion y capas conv, de modo que el paso tiene la escala de la perdida.
import { describe, it, expect } from 'vitest'
import { CausalConv1D, TCN } from '../../src/TCN'
import { bindParams, expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const LR = 1e-3

// [T=6][2 canales], todos |x| >= 0.1
const SEQ = [
  [0.5, -0.4],
  [-0.3, 0.7],
  [0.8, 0.2],
  [-0.6, 0.9],
  [0.4, -0.5],
  [0.1, 0.6],
]
const TARGET_CONV = [
  [0.3, -0.2],
  [-0.4, 0.2],
  [0.5, 0.0],
  [0.1, 0.6],
  [0.4, -0.3],
  [-0.1, 0.25],
]
const TARGET_TCN = [[0.3], [-0.4], [0.5], [0.1], [0.4], [-0.1]]

// ── CausalConv1D aislada ─────────────────────────────────────────────────────
function setupConv(dilation = 2) {
  const conv = new CausalConv1D(2, 2, 3, dilation)
  const loss = () => {
    const y = conv.forward(SEQ)
    let s = 0
    y.forEach((row, t) => row.forEach((v, f) => (s += 0.5 * (TARGET_CONV[t][f] - v) ** 2)))
    return s
  }
  return {
    conv,
    loss,
    // dOut = -dL/dy = target - y (convencion de backward)
    step: (lr: number) => {
      const y = conv.forward(SEQ)
      conv.backward(
        y.map((row, t) => row.map((v, f) => TARGET_CONV[t][f] - v)),
        lr,
      )
    },
    ...bindParams(conv.kernels, conv.biases),
  }
}

// ── TCN ──────────────────────────────────────────────────────────────────────
// inCh = channels = 2 => hay conexion residual en todos los niveles.
function buildTcn(): TCN {
  return new TCN(2, 2, 2, 2, 1)
}

function tcnLoss(net: TCN): number {
  const y = net.forward(SEQ)
  let s = 0
  y.forEach((row, t) => row.forEach((v, k) => (s += (v - TARGET_TCN[t][k]) ** 2)))
  return s / (y.length * y[0].length)
}

function tcnParams(net: TCN) {
  const n = net as any
  return bindParams(
    ...net.layers.flatMap((l) => [l.kernels, l.biases]),
    n._outputW,
    n._outputB,
  )
}

function tcnStep(net: TCN, lr: number): void {
  net.train(SEQ, TARGET_TCN, lr)
}

describe('CausalConv1D oraculo', () => {
  it('humo: forma [T][outputChannels], backward finito y causal', () => {
    const m = setupConv()
    const y = m.conv.forward(SEQ)
    expect(y).toHaveLength(SEQ.length)
    expect(y.every((r) => r.length === 2 && r.every(Number.isFinite))).toBe(true)
    // Causalidad: cambiar la ultima entrada no altera salidas anteriores.
    const alt = SEQ.map((r) => [...r])
    alt[5] = [9, -9]
    const y2 = m.conv.forward(alt)
    for (let t = 0; t < 5; t++) expect(y2[t]).toEqual(y[t])
    m.conv.forward(SEQ)
    const dIn = m.conv.backward(
      y.map((r) => r.map(() => 0.1)),
      0.01,
    )
    expect(dIn).toHaveLength(SEQ.length)
    expect(dIn.every((r) => r.length === 2 && r.every(Number.isFinite))).toBe(true)
  })

  it('oraculo (dilatacion 2): el paso sigue -dL/dw', () => {
    const r = expectGradientMatches(setupConv(2))
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('oraculo (dilatacion 1): el paso sigue -dL/dw', () => {
    const r = expectGradientMatches(setupConv(1))
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('expectLearns: la perdida baja a la mitad', () => {
    const m = setupConv()
    expectLearns({ loss: m.loss, step: () => m.step(0.02), steps: 200 })
  })
})

describe('TCN oraculo', () => {
  it('humo: forward [T][outputSize] finito, train devuelve el MSE y mueve los pesos', () => {
    const net = buildTcn()
    const y = net.forward(SEQ)
    expect(y).toHaveLength(SEQ.length)
    expect(y.every((r) => r.length === 1 && r.every(Number.isFinite))).toBe(true)
    const p = tcnParams(net)
    const before = p.getParams()
    const expected = tcnLoss(net)
    const reported = net.train(SEQ, TARGET_TCN, LR)
    expect(reported).toBeCloseTo(expected, 12)
    const after = p.getParams()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oraculo: train() sigue -dL/dw (coseno > 0.999)', () => {
    const net = buildTcn()
    const r = expectGradientMatches({
      ...tcnParams(net),
      loss: () => tcnLoss(net),
      step: (lr) => tcnStep(net, lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
    // Escala: el paso tiene la escala de la perdida media (ratio en (0.9, 1.1)).
    expect(r.ratio).toBeGreaterThan(0.9)
    expect(r.ratio).toBeLessThan(1.1)
  })

  it('learns: la perdida baja a la mitad', () => {
    const net = buildTcn()
    expectLearns({ loss: () => tcnLoss(net), step: () => tcnStep(net, 0.01), steps: 200 })
  })
})
