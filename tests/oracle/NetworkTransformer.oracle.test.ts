// Gradient-oracle test for NetworkTransformer (feature 002, T019).
//
// Template: tests/oracle/TransformerBlock.oracle.test.ts (same family).
//   1. smoke    - predict gives seqLen*nClasses finite logits, train returns the
//                 cross-entropy it reports (checked against an independent
//                 computation from predict()) and moves the weights. No it.fails.
//   2. oracle   - it.fails (AUDIT C1): the update train() really performs follows
//                 +dL/dw instead of -dL/dw (cosine ~ -1).
//   3. learns   - it.fails (AUDIT C1): the loss does not drop; it grows.
//   4. control  - NOT it.fails: the same step with its delta negated by hand gives
//                 cosine ~ +1, proving the defect is the sign only.
// Loss = mean over positions of -sum_c t_c * log softmax(logits)_c, which is what
// train() implements (cross-entropy + softmax), NOT the "MSE" its doc comment says.
// Every weight matrix is Adam (Q1), so every real update runs inside withPlainSGD.
// Only the cosine is asserted (Q9). A fixed seed (tests/setup.ts) fixes the init.
import { describe, it, expect } from 'vitest'
import { NetworkTransformer } from '../../src/NetworkTransformer'
import { softmax } from '../../src/MatMul'
import { gradientOracle, withPlainSGD } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'

const SEQ = 3
const N_CLASSES = 3
const TOKENS = [1, 3, 0]
// One-hot target class per position, flattened (seqLen * nClasses).
const CLASSES = [2, 0, 1]
const TARGETS: number[] = CLASSES.flatMap((k) =>
  Array.from({ length: N_CLASSES }, (_, c) => (c === k ? 1 : 0)),
)
const LR = 1e-3

function build(): NetworkTransformer {
  return new NetworkTransformer(SEQ, {
    vocabSize: 4,
    d_model: 4,
    nHeads: 2,
    d_ff: 8,
    nBlocks: 1,
    nClasses: N_CLASSES,
  })
}

/** Mean over positions of cross-entropy of softmax(logits), measured with predict() only. */
function loss(net: NetworkTransformer): number {
  const logits = net.predict(TOKENS)
  let s = 0
  for (let i = 0; i < SEQ; i++) {
    const p = softmax(logits.slice(i * N_CLASSES, (i + 1) * N_CLASSES))
    for (let c = 0; c < N_CLASSES; c++) {
      if (TARGETS[i * N_CLASSES + c] > 0) s -= Math.log(Math.max(p[c], 1e-7))
    }
  }
  return s / SEQ
}

/** One real update of the model, with plain `w + lr*g` instead of Adam. */
function step(net: NetworkTransformer, lr: number): number {
  return withPlainSGD(() => net.train(TOKENS, TARGETS, lr))
}

/** Same step but with the delta negated by hand (control for the sign defect). */
function negatedStep(net: NetworkTransformer, lr: number): void {
  const before = [...net.getWeights()]
  step(net, lr)
  const after = net.getWeights()
  net.setWeights(before.map((w, i) => w - (after[i] - w)))
}

describe('NetworkTransformer oracle', () => {
  it('smoke: finite logits, train returns the cross-entropy and moves the weights', () => {
    const net = build()
    const logits = net.predict(TOKENS)
    expect(logits).toHaveLength(SEQ * N_CLASSES)
    expect(logits.every(Number.isFinite)).toBe(true)

    const before = [...net.getWeights()]
    const expected = loss(net)
    const reported = step(net, LR)
    expect(Number.isFinite(reported)).toBe(true)
    // train() reports the cross-entropy of the forward pass it started from.
    expect(reported).toBeCloseTo(expected, 10)
    const after = net.getWeights()
    expect(after.every(Number.isFinite)).toBe(true)
    expect(after.some((v, i) => v !== before[i])).toBe(true)
  })

  // AUDIT C1: train() backpropagates dLogits = prob - target = +dL/dz, but the
  // optimizers do `w + lr*g` and expect g = -dL/dz, so every weight moves uphill
  // (cosine -1.0000 measured in the audit). Remove `.fails` when fixed (feature 003).
  it.fails('oracle: train() follows -dL/dw from finite differences (cosine > 0.999)', () => {
    const net = build()
    const r = gradientOracle({
      getParams: () => net.getWeights(),
      setParams: (p) => net.setWeights(p),
      loss: () => loss(net),
      step: (lr) => {
        step(net, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  // AUDIT C1: same sign defect; the cross-entropy rises instead of dropping.
  it.fails('learns: the cross-entropy drops below half after a few hundred steps', () => {
    const net = build()
    expectLearns({
      loss: () => loss(net),
      step: () => {
        step(net, 0.05)
      },
      steps: 300,
    })
  })

  it('control: with the delta negated by hand the oracle gives cosine ~ +1 (sign-only defect)', () => {
    const net = build()
    const r = gradientOracle({
      getParams: () => net.getWeights(),
      setParams: (p) => net.setWeights(p),
      loss: () => loss(net),
      step: (lr) => negatedStep(net, lr),
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  // Guard migrated from GradientCheck.test.ts (T035): not a learning test (C1
  // makes the model train backwards), only that a short run at a small lr does
  // not blow up. Keep it as a plain `it` after C1 is fixed.
  it('stability: 15 steps at lr 1e-3 keep the loss finite and below 1.5x the initial', () => {
    const net = build()
    const initial = loss(net)
    for (let i = 0; i < 15; i++) step(net, LR)
    const final = loss(net)
    expect(Number.isFinite(final)).toBe(true)
    expect(final).toBeLessThan(1.5 * initial)
  })
})
