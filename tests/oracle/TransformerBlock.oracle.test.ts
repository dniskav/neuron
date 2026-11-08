// Gradient-oracle test for TransformerBlock (feature 002, T016).
//
// Template: tests/oracle/AttentionHead.oracle.test.ts (same family).
//   1. smoke   - predicts a 3-token sequence with finite outputs and backward()
//                returns finite dX and moves the weights.
//   2. oracle  - expectGradientMatches: the update backward() really performs
//                (attention heads, Wo, ff1/ff2, b1/b2 through the plain `w + lr*g`
//                step, and norm1/norm2 gamma/beta through their SGD) follows
//                -dL/dw from central finite differences (cosine ~ 1).
//                Loss is L = 0.5 * sum (out - T)^2 with a fixed target T.
//                backward receives dOut = -dL/dout = -(out - T).
//   3. learns  - expectLearns: the same loss, measured outside backward on fixed
//                data, drops below half after a few hundred steps.
// TransformerBlock keeps its Wq/Wk/Wv/Wo/ff1/ff2 as WeightMatrix and b1/b2 as
// BiasVector (Adam), so every real update runs inside withPlainSGD (Q1).
// The cosine is the only criterion (Q9: the ratio/scale is not asserted).
// A fixed seed (tests/setup.ts) fixes the initial weights.
//
// Why a fixed target instead of L = 0.5 * sum out^2: the output of a post-norm
// block is LayerNorm(...), whose row norm is almost constant, so sum out^2 is
// nearly flat and would be a weak oracle. The target makes dL/dout vary per
// element.
//
// ReLU: the inputs have magnitudes ~1-3, so the FFN pre-activations are not
// near zero and central differences (eps 1e-5) do not cross the ReLU kink.
import { describe, it, expect } from 'vitest'
import { TransformerBlock } from '../../src/TransformerBlock'
import { expectGradientMatches, withPlainSGD } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const D_MODEL = 4
const N_HEADS = 2
const D_FF = 8
// Magnitudes ~1-3 (see AttentionHead.oracle.test.ts for the reason).
const X: number[][] = [
  [1.5, -0.9, 2.4, 0.6],
  [-1.2, 1.8, 0.3, -2.1],
  [2.7, 0.6, -1.5, 0.9],
]
// Fixed target, same shape as the output, magnitude ~1.
const T: number[][] = [
  [0.8, -0.4, 1.1, -0.7],
  [-0.9, 0.5, -0.3, 1.0],
  [0.2, 1.2, -0.8, -0.5],
]
const LR = 1e-3

/** L = 0.5 * sum_t sum_i (out[t][i] - T[t][i])^2, measured with predict() only. */
function loss(block: TransformerBlock): number {
  const out = block.predict(X)
  let s = 0
  for (let t = 0; t < out.length; t++)
    for (let i = 0; i < out[t].length; i++) {
      const d = out[t][i] - T[t][i]
      s += 0.5 * d * d
    }
  return s
}

/** One real update: forward, then backward with dOut = -dL/dout = -(out - T). */
function step(block: TransformerBlock, lr: number): void {
  withPlainSGD(() => {
    const out = block.predict(X)
    block.backward(
      out.map((row, t) => row.map((v, i) => -(v - T[t][i]))),
      lr,
    )
  })
}

function build(): TransformerBlock {
  return new TransformerBlock({ d_model: D_MODEL, nHeads: N_HEADS, d_ff: D_FF, causal: false })
}

describe('TransformerBlock oracle', () => {
  it('smoke: predicts finite outputs and backward moves the weights', () => {
    const block = build()
    const out = block.predict(X)
    expect(out).toHaveLength(X.length)
    for (const row of out) {
      expect(row).toHaveLength(D_MODEL)
      expect(row.every(Number.isFinite)).toBe(true)
    }
    const before = block.getWeights()
    const dOut = out.map((row, t) => row.map((v, i) => -(v - T[t][i])))
    const dX = withPlainSGD(() => block.backward(dOut, LR))
    expect(dX).toHaveLength(X.length)
    for (const row of dX) {
      expect(row).toHaveLength(D_MODEL)
      expect(row.every(Number.isFinite)).toBe(true)
    }
    const after = block.getWeights()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  it('oracle: backward() follows -dL/dw from finite differences', () => {
    const block = build()
    const r = expectGradientMatches({
      getParams: () => block.getWeights(),
      setParams: (p) => block.setWeights(p),
      loss: () => loss(block),
      step: (lr) => step(block, lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('learns: the output loss drops below half after a few hundred steps', () => {
    const block = build()
    expectLearns({
      loss: () => loss(block),
      step: () => step(block, 0.05),
      steps: 300,
    })
  })
})
