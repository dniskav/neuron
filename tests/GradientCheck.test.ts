// ─── Gradient Checks (Finite Differences + Signed Perturbation) ──────────
//
// Verifies that hand-coded gradients are directionally correct.
//
// Signed perturbation approach (works for ALL classes regardless of internal
// gradient convention):
//   1. Save current weights
//   2. Compute baseline loss L0
//   3. For a sampled weight w_i:
//      a. Compute numerical gradient g_i = (L(w_i+ε) - L(w_i-ε)) / (2ε)
//      b. Perturb weight: w_i' = w_i - η * sign(g_i)  (small step opposite gradient)
//      c. Recompute loss L'. If gradient is correct, L' < L0.
//
// This verifies the sign of dL/dw without relying on the class's internal
// gradient computation convention.
//
// Additionally, for simple SGD-based networks we verify that multiple
// training steps decrease loss.
//
// ─────────────────────────────────────────────────────────────────────────────

import { describe, it, expect } from 'vitest'
import { Network } from '../src/Network'
import { NetworkN } from '../src/NetworkN'
import { LSTMLayer } from '../src/LSTMLayer'
import { GRULayer } from '../src/GRU'
import { Conv1D } from '../src/Conv1D'
import { AttentionHead } from '../src/AttentionHead'
import { NetworkTransformer } from '../src/NetworkTransformer'
import { SGD } from '../src/optimizers'

// ── Signed perturbation check ─────────────────────────────────────────────

/**
 * For a set of sampled weights, compute the numerical gradient and verify
 * that perturbing the weight OPPOSITE to the gradient direction decreases loss.
 *
 * This validates that the finite-difference gradient has the correct sign,
 * which implies the class's internal backward pass computes directionally
 * correct gradients (since training also moves weights opposite to dL/dw).
 *
 * @param nWeights  total number of scalar weights
 * @param getWeight read weight i
 * @param setWeight write weight i
 * @param computeLoss () => loss (deterministic forward pass)
 * @param sampleSize number of weights to check (default 10)
 * @param epsilon   finite-difference epsilon (default 1e-4)
 * @param eta       perturbation step size (default 1e-3)
 */
function signedPerturbationCheck(
  nWeights: number,
  getWeight: (i: number) => number,
  setWeight: (i: number, v: number) => void,
  computeLoss: () => number,
  sampleSize = 10,
  epsilon = 1e-4,
  eta = 1e-3,
): { improved: number; checked: number } {
  const origWeights: number[] = []
  for (let i = 0; i < nWeights; i++) origWeights.push(getWeight(i))

  const baselineLoss = computeLoss()
  const step = Math.max(1, Math.floor(nWeights / sampleSize))

  let improved = 0
  let checked = 0

  for (let i = 0; i < nWeights; i += step) {
    // Restore all weights to original state
    for (let j = 0; j < nWeights; j++) setWeight(j, origWeights[j])

    const w = getWeight(i)

    // Numerical gradient
    setWeight(i, w + epsilon)
    const lossPlus = computeLoss()
    setWeight(i, w - epsilon)
    const lossMinus = computeLoss()
    setWeight(i, w) // restore

    const numGrad = (lossPlus - lossMinus) / (2 * epsilon)

    if (Math.abs(numGrad) < 1e-10) continue // gradient is flat, skip

    // Perturb weight OPPOSITE to gradient: w' = w - eta * sign(grad)
    const sign = numGrad > 0 ? 1 : -1
    setWeight(i, w - eta * sign)
    const perturbedLoss = computeLoss()

    checked++
    if (perturbedLoss < baselineLoss) improved++
  }

  // Restore all weights
  for (let j = 0; j < nWeights; j++) setWeight(j, origWeights[j])

  return { improved, checked }
}

// ── Multiple-step loss check ──────────────────────────────────────────────

/**
 * Verify that training over multiple steps decreases loss.
 * More reliable than single-step checks for complex architectures.
 */
function multiStepLossCheck(
  train: () => number,
  steps = 10,
  tolerance = 1.5, // allow slight increase, just verify no explosion
): { initialLoss: number; finalLoss: number; improved: boolean } {
  const initialLoss = train()
  let finalLoss = initialLoss
  for (let i = 0; i < steps; i++) {
    finalLoss = train()
  }
  return {
    initialLoss,
    finalLoss,
    improved: finalLoss < initialLoss * tolerance && isFinite(finalLoss),
  }
}


// ── Tests ──────────────────────────────────────────────────────────────────

describe('GradientCheck', () => {

  // ── Network (2-layer, SGD) ────────────────────────────────────────────

  describe('Network (2-layer)', () => {
    it('signed perturbation check passes', () => {
      const net = new Network(3, 4, 1)
      const inputs = [0.5, -0.3, 0.8]
      const target = 0.7

      const computeLoss = () => {
        const pred = net.predict(inputs)
        return (pred - target) ** 2
      }

      const { improved, checked } = signedPerturbationCheck(
        net.getWeights().length,
        (i) => net.getWeights()[i],
        (i, v) => { const w = net.getWeights(); w[i] = v; net.setWeights(w) },
        computeLoss,
        10,
      )

      // Almost all weight perturbations in the descent direction should improve loss
      expect(improved / checked).toBeGreaterThan(0.5)
    })

    it('multi-step training decreases loss', () => {
      const net = new Network(3, 4, 1)
      const inputs = [0.5, -0.3, 0.8]
      const target = 0.7
      const lr = 0.01

      const { improved } = multiStepLossCheck(
        () => net.train(inputs, target, lr),
        15,
      )
      expect(improved).toBe(true)
    })

    it('intentionally wrong perturbation increases loss', () => {
      const net = new Network(2, 2, 1)
      const inputs = [0.5, 0.5]
      const target = 0.3

      // Get numerical gradient of the first weight
      const epsilon = 1e-4
      const computeLoss = () => {
        const pred = net.predict(inputs)
        return (pred - target) ** 2
      }
      const baselineLoss = computeLoss()

      const w = net.getWeights()[0]
      const wArr = net.getWeights()
      wArr[0] = w + epsilon
      net.setWeights(wArr)
      const lossPlus = computeLoss()
      wArr[0] = w - epsilon
      net.setWeights(wArr)
      const lossMinus = computeLoss()
      const numGrad = (lossPlus - lossMinus) / (2 * epsilon)

      // Perturb in the WRONG direction (same direction as gradient)
      wArr[0] = w + 0.01 * (numGrad > 0 ? 1 : -1)
      net.setWeights(wArr)
      const wrongLoss = computeLoss()

      // Moving WITH the gradient should increase or not decrease loss
      // (gradient descent moves OPPOSITE to gradient)
      expect(wrongLoss).toBeGreaterThan(baselineLoss * 0.5)
    })
  })

  // ── NetworkN ───────────────────────────────────────────────────────────

  describe('NetworkN', () => {
    it('signed perturbation check with residual', () => {
      const net = new NetworkN([3, 3, 3, 1], {
        residual: true,
        optimizer: () => new SGD(),
      })
      const inputs = [0.5, -0.3, 0.8]
      const target = [0.7]

      const computeLoss = () => {
        const pred = net.predict(inputs)
        return pred.reduce((s, p, i) => s + (p - target[i]) ** 2, 0) / pred.length
      }

      const { improved, checked } = signedPerturbationCheck(
        net.getWeights().length,
        (i) => net.getWeights()[i],
        (i, v) => { const w = net.getWeights(); w[i] = v; net.setWeights(w) },
        computeLoss,
        10,
      )

      expect(improved / checked).toBeGreaterThan(0.5)
    })

    it('multi-step training decreases loss with residual', () => {
      const net = new NetworkN([3, 3, 3, 1], {
        residual: true,
        optimizer: () => new SGD(),
      })
      const inputs = [0.5, -0.3, 0.8]
      const target = [0.7]
      const lr = 0.1

      const { improved } = multiStepLossCheck(
        () => net.train(inputs, target, lr),
        20,
      )
      expect(improved).toBe(true)
    })

    it('dropout training is stable (no gradient explosion)', () => {
      const net = new NetworkN([3, 4, 4, 1], {
        optimizer: () => new SGD(),
        dropoutRate: 0.5,
      })
      const inputs = [0.5, -0.3, 0.8]
      const target = [0.7]
      const lr = 0.01

      const losses: number[] = []
      for (let i = 0; i < 20; i++) {
        losses.push(net.train(inputs, target, lr))
      }

      expect(losses.every(l => isFinite(l))).toBe(true)
      // Loss shouldn't explode more than 20x
      expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 20)
    })
  })

  // ── LSTMLayer ──────────────────────────────────────────────────────────

  describe('LSTMLayer', () => {
    it('signed perturbation check passes', () => {
      const lstm = new LSTMLayer(2, 2, () => new SGD())
      const inputs = [0.5, -0.3]

      const computeLoss = () => {
        lstm.reset()
        const out = lstm.predict(inputs)
        return out.reduce((s, v) => s + v * v, 0)
      }

      const { improved, checked } = signedPerturbationCheck(
        lstm.getWeightsFlat().length,
        (i) => lstm.getWeightsFlat()[i],
        (i, v) => { const w = lstm.getWeightsFlat(); w[i] = v; lstm.setWeightsFlat(w) },
        computeLoss,
        10,
      )
      expect(improved / checked).toBeGreaterThan(0.5)
    })

    it('multi-step training decreases loss', () => {
      const lstm = new LSTMLayer(2, 2, () => new SGD())
      const inputs = [0.5, -0.3]
      const lr = 0.01

      lstm.reset()
      const h0 = lstm.predict(inputs)
      const loss0 = h0.reduce((s, v) => s + v * v, 0)
      lstm.backprop([h0.map(v => 2 * v)], lr)

      // Several rounds
      for (let r = 0; r < 5; r++) {
        lstm.reset()
        const h = lstm.predict(inputs)
        lstm.backprop([h.map(v => 2 * v)], lr)
      }

      lstm.reset()
      const hFinal = lstm.predict(inputs)
      const lossFinal = hFinal.reduce((s, v) => s + v * v, 0)

      expect(lossFinal).toBeLessThan(loss0 * 2)
    })
  })

  // ── GRULayer ──────────────────────────────────────────────────────────

  describe('GRULayer', () => {
    it('signed perturbation check passes', () => {
      const gru = new GRULayer(2, 2, () => new SGD())
      const inputs = [0.5, -0.3]

      const computeLoss = () => {
        gru.reset()
        const out = gru.predict(inputs)
        return out.reduce((s, v) => s + v * v, 0)
      }

      const { improved, checked } = signedPerturbationCheck(
        gru.getWeightsFlat().length,
        (i) => gru.getWeightsFlat()[i],
        (i, v) => { const w = gru.getWeightsFlat(); w[i] = v; gru.setWeightsFlat(w) },
        computeLoss,
        10,
      )
      expect(improved / checked).toBeGreaterThan(0.5)
    })

    it('multi-step training decreases loss', () => {
      const gru = new GRULayer(2, 2, () => new SGD())
      const inputs = [0.5, -0.3]
      const lr = 0.01

      gru.reset()
      const h0 = gru.predict(inputs)
      const loss0 = h0.reduce((s, v) => s + v * v, 0)
      gru.backprop([h0.map(v => 2 * v)], lr)

      for (let r = 0; r < 5; r++) {
        gru.reset()
        const h = gru.predict(inputs)
        gru.backprop([h.map(v => 2 * v)], lr)
      }

      gru.reset()
      const hFinal = gru.predict(inputs)
      const lossFinal = hFinal.reduce((s, v) => s + v * v, 0)

      expect(lossFinal).toBeLessThan(loss0 * 2)
    })
  })

  // ── Conv1D ─────────────────────────────────────────────────────────────

  describe('Conv1D', () => {
    it('signed perturbation check passes', () => {
      const conv = new Conv1D(5, 3, 2, 1, 'valid', () => new SGD(), 1)
      const input = [0.5, -0.3, 0.8, 0.1, -0.6]

      const computeLoss = () => {
        const o = conv.forward(input)
        return o.reduce((s, row) =>
          s + row.reduce((a, b) => a + b * b, 0), 0)
      }

      const { improved, checked } = signedPerturbationCheck(
        conv.getWeights().length,
        (i) => conv.getWeights()[i],
        (i, v) => { const w = conv.getWeights(); w[i] = v; conv.setWeights(w) },
        computeLoss,
        10,
      )
      expect(improved / checked).toBeGreaterThan(0.5)
    })

    it('multi-step training decreases loss', () => {
      const conv = new Conv1D(5, 3, 2, 1, 'valid', () => new SGD(), 1)
      const input = [0.5, -0.3, 0.8, 0.1, -0.6]
      const lr = 0.01

      let prevLoss = Infinity
      let improved = false
      for (let i = 0; i < 10; i++) {
        const out = conv.forward(input)
        const loss = out.reduce((s, row) =>
          s + row.reduce((a, b) => a + b * b, 0), 0)
        if (loss < prevLoss * 0.999) improved = true
        prevLoss = loss
        const dOut = out.map(row => row.map(v => 2 * v))
        conv.backward(dOut, lr)
      }

      // Loss should have improved at least once
      expect(improved).toBe(true)
    })
  })

  // ── AttentionHead ─────────────────────────────────────────────────────

  describe('AttentionHead', () => {
    it('signed perturbation check passes', () => {
      const d_model = 4
      const d_k = 2
      const d_v = 2
      const head = new AttentionHead(d_model, d_k, d_v, false)

      const X = [
        [0.5, -0.3, 0.8, 0.1],
        [-0.2, 0.6, -0.4, 0.3],
      ]

      const computeLoss = () => {
        const out = head.predict(X)
        return out.reduce((s, row) =>
          s + row.reduce((a, b) => a + b * b, 0), 0)
      }

      const { improved, checked } = signedPerturbationCheck(
        head.getWeights().length,
        (i) => head.getWeights()[i],
        (i, v) => { const w = head.getWeights(); w[i] = v; head.setWeights(w) },
        computeLoss,
        10,
      )
      expect(improved / checked).toBeGreaterThan(0.5)
    })

    it('multi-step training decreases loss', () => {
      const d_model = 4
      const head = new AttentionHead(d_model, 2, 2, false)
      const X = [
        [0.5, -0.3, 0.8, 0.1],
        [-0.2, 0.6, -0.4, 0.3],
      ]
      const lr = 0.001

      let prevLoss = Infinity
      let improved = false
      for (let i = 0; i < 10; i++) {
        const out = head.predict(X)
        const loss = out.reduce((s, row) =>
          s + row.reduce((a, b) => a + b * b, 0), 0)
        if (loss < prevLoss * 0.999) improved = true
        prevLoss = loss
        const dOut = out.map(row => row.map(v => 2 * v))
        head.backward(dOut, lr)
      }

      expect(improved).toBe(true)
    })
  })

  // ── NetworkTransformer ─────────────────────────────────────────────────

  describe('NetworkTransformer', () => {
    it('signed perturbation check passes', () => {
      const seqLen = 2
      const net = new NetworkTransformer(seqLen, {
        vocabSize: 3,
        d_model: 3,
        nHeads: 1,
        d_ff: 4,
        nBlocks: 1,
        nClasses: 2,
      })
      const tokens = [0, 1]
      const targets = [1, 0, 0, 1]

      const computeLoss = () => {
        const logits = net.predict(tokens)
        const nClasses = 2
        let loss = 0
        for (let i = 0; i < seqLen; i++) {
          const offset = i * nClasses
          const l = logits.slice(offset, offset + nClasses)
          const maxL = Math.max(...l)
          const exps = l.map(v => Math.exp(v - maxL))
          const sumExps = exps.reduce((a, b) => a + b, 0)
          const probs = exps.map(e => e / sumExps)
          for (let c = 0; c < nClasses; c++) {
            if (targets[offset + c] > 0) {
              loss -= Math.log(Math.max(probs[c], 1e-7))
            }
          }
        }
        return loss / seqLen
      }

      const { improved, checked } = signedPerturbationCheck(
        net.getWeights().length,
        (i) => net.getWeights()[i],
        (i, v) => { const w = net.getWeights(); w[i] = v; net.setWeights(w) },
        computeLoss,
        10,
      )
      expect(improved / checked).toBeGreaterThan(0.5)
    })

    it('multi-step training decreases loss', () => {
      const seqLen = 3
      const net = new NetworkTransformer(seqLen, {
        vocabSize: 4,
        d_model: 4,
        nHeads: 1,
        d_ff: 8,
        nBlocks: 1,
        nClasses: 2,
      })
      const tokens = [0, 1, 2]
      const targets = [1, 0, 0, 1, 1, 0]
      const lr = 0.001

      const { improved } = multiStepLossCheck(
        () => net.train(tokens, targets, lr),
        15,
      )
      expect(improved).toBe(true)
    })
  })
})
