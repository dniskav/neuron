// Tests de comportamiento de VAE (T031, specs/006-coverage).
//
// No repite el oráculo de gradiente (tests/oracle/VAE.oracle.test.ts) ni el
// "reconstruye mejor que la media" de 004. Aquí se fijan valores calculados a mano:
//
//   ln 4 = 2·ln 2 = 1.3862943611198906  →  σ = exp(½·ln 4) = 2  y  exp(ln 4) = 4
//   sigmoide(0) = 0.5                   →  con pesos del decoder a cero, x̂ = 0.5 para cualquier z
//   recon (x = [1]) = (1 − 0.5)² / 1    = 0.25
//   KL por dimensión = −½ (1 + logVar − μ² − e^logVar)
//   ε fijado con un stub de Math.random (Box-Muller): u1 = e^(−½) da radio √(−2 ln u1) = 1,
//   y u2 = arccos(ε)/(2π) da cos(2π·u2) = ε. Así ε es exactamente el valor pedido.
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach. Los stubs de Math.random se
// instalan solo dentro de withRandom y se restauran al salir.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { VAE } from '../src/VAE'
import { relu, sigmoid } from '../src/activations'
import { seedRandom, restoreRandom } from './helpers/seed'

const LN4 = Math.log(4)   // 1.3862943611198906
const U1 = Math.exp(-0.5) // Box-Muller: sqrt(-2 ln(e^-0.5)) = 1

// Devuelve, por cada ε pedido, el par (u1, u2) que produce ese ε exacto. `used()` cuenta las llamadas.
function epsStub(eps: number[]): { fn: () => number; used: () => number } {
  const seq: number[] = []
  for (const e of eps) seq.push(U1, Math.acos(e) / (2 * Math.PI))
  let i = 0
  return {
    fn: () => {
      if (i >= seq.length) throw new Error('epsStub: más uniformes de los previstos')
      return seq[i++]
    },
    used: () => i,
  }
}

// Ejecuta body con Math.random reemplazado por fn y restaura el anterior (el sembrado) al salir.
function withRandom<T>(fn: () => number, body: () => T): T {
  const previous = Math.random
  Math.random = fn
  try {
    return body()
  } finally {
    Math.random = previous
  }
}

// VAE sin capas ocultas: encoder [1 → 2k] lineal, decoder [k → 1] sigmoide con pesos a cero.
function build(latent: number, klWeight = 1): VAE {
  const vae = new VAE(1, [], latent, [], undefined, klWeight)
  vae.decoder.setWeights(vae.decoder.getWeights().map(() => 0))
  return vae
}

// Pesos del encoder por neurona: [w, bias]. Orden de neuronas: μ₁…μₖ, logVar₁…logVarₖ.
function setEncoder(vae: VAE, neurons: Array<[number, number]>): void {
  vae.encoder.setWeights(neurons.flatMap(([w, b]) => [w, b]))
}

// Encoder con w = 0: μ y logVar valen lo mismo para cualquier x.
function constEncoder(vae: VAE, mu: number[], logVar: number[]): void {
  setEncoder(vae, [...mu, ...logVar].map((b): [number, number] => [0, b]))
}

describe('VAE (T031): valores a mano', () => {
  beforeEach(() => seedRandom())
  afterEach(restoreRandom)

  describe('forward: μ, logVar, ε y z', () => {
    it('μ y logVar salen de la capa lineal y z = μ + exp(logVar/2)·ε', () => {
      const vae = build(2)
      // x = [1]: μ₁ = 0.5·1 + 0 = 0.5; μ₂ = 0·1 − 1 = −1; logVar₁ = ln 4; logVar₂ = 0
      setEncoder(vae, [[0.5, 0], [0, -1], [0, LN4], [0, 0]])
      const stub = epsStub([0.5, -1])
      const f = withRandom(stub.fn, () => vae.forward([1]))

      expect(f.mu[0]).toBeCloseTo(0.5, 12)
      expect(f.mu[1]).toBeCloseTo(-1, 12)
      expect(f.logVar[0]).toBeCloseTo(LN4, 12)
      expect(f.logVar[1]).toBeCloseTo(0, 12)
      expect(f.eps[0]).toBeCloseTo(0.5, 12)
      expect(f.eps[1]).toBeCloseTo(-1, 12)
      // σ₁ = exp(½·ln 4) = 2, σ₂ = exp(0) = 1
      // z₁ = 0.5 + 2·0.5 = 1.5 ;  z₂ = −1 + 1·(−1) = −2
      expect(f.z[0]).toBeCloseTo(1.5, 12)
      expect(f.z[1]).toBeCloseTo(-2, 12)
      // 2 dimensiones × 2 uniformes de Box-Muller
      expect(stub.used()).toBe(4)
    })

    it('la desviación es exp(logVar/2): con logVar = ln 4 y ε = 1, z − μ = 2 (no 4)', () => {
      const vae = build(1)
      constEncoder(vae, [0], [LN4])
      const stub = epsStub([1])
      const f = withRandom(stub.fn, () => vae.forward([1]))
      // σ = exp(½·ln 4) = 2; z − μ = σ·1 = 2
      expect(f.z[0] - f.mu[0]).toBeCloseTo(2, 12)
    })

    it('cada dimensión de la latente consume su propio ε (no se reutiliza el primero)', () => {
      const vae = build(2)
      constEncoder(vae, [0, 0], [0, 0]) // σ = 1 y μ = 0, así que z = ε
      const stub = epsStub([1, -1])
      const f = withRandom(stub.fn, () => vae.forward([1]))
      expect(f.z[0]).toBeCloseTo(1, 12)
      expect(f.z[1]).toBeCloseTo(-1, 12)
      expect(stub.used()).toBe(4)
    })
  })

  describe('KL = −½ Σ (1 + logVar − μ² − e^logVar), leído de train()', () => {
    it('μ = 0 y logVar = 0 dan KL = 0 (recon = 0.25 con x = [1])', () => {
      const vae = build(2)
      constEncoder(vae, [0, 0], [0, 0])
      const r = withRandom(epsStub([0, 0]).fn, () => vae.train([1], 0.1))
      // por dimensión: −½ (1 + 0 − 0 − 1) = 0
      expect(r.klLoss).toBeCloseTo(0, 12)
      expect(r.reconLoss).toBeCloseTo(0.25, 12)
      expect(r.totalLoss).toBeCloseTo(0.25, 12)
    })

    it('μ = 1 y logVar = 0 dan 0.5 por dimensión: 0.5 con latente 1 y 1.0 con latente 2', () => {
      const one = build(1)
      constEncoder(one, [1], [0])
      const r1 = withRandom(epsStub([0]).fn, () => one.train([1], 0.1))
      // −½ (1 + 0 − 1 − 1) = −½ (−1) = 0.5
      expect(r1.klLoss).toBeCloseTo(0.5, 12)

      const two = build(2)
      constEncoder(two, [1, 1], [0, 0])
      const r2 = withRandom(epsStub([0, 0]).fn, () => two.train([1], 0.1))
      // suma sobre las dos dimensiones: 0.5 + 0.5
      expect(r2.klLoss).toBeCloseTo(1, 12)
    })

    it('mezcla: μ = (0.5, −1), logVar = (ln 4, 0) dan KL = 0.9318528194 + 0.5 = 1.4318528194', () => {
      const vae = build(2)
      constEncoder(vae, [0.5, -1], [LN4, 0])
      const r = withRandom(epsStub([0, 0]).fn, () => vae.train([1], 0.1))
      // dim 1: −½ (1 + ln 4 − 0.25 − 4) = 1.625 − ln4/2 = 1.625 − 0.6931471806 = 0.9318528194
      // dim 2: −½ (1 + 0 − 1 − 1) = 0.5
      expect(r.klLoss).toBeCloseTo(1.4318528194400547, 9)
    })
  })

  describe('klWeight (β) escala el término de KL en totalLoss', () => {
    for (const beta of [0, 0.25, 2]) {
      it(`β = ${beta}: totalLoss = 0.25 + β·1, y klLoss se reporta sin escalar`, () => {
        const vae = build(2, beta)
        constEncoder(vae, [1, 1], [0, 0]) // KL = 1
        const r = withRandom(epsStub([0, 0]).fn, () => vae.train([1], 0.1))
        expect(r.reconLoss).toBeCloseTo(0.25, 12)
        expect(r.klLoss).toBeCloseTo(1, 12)
        // totalLoss = recon + β·KL = 0.25 + β
        expect(r.totalLoss).toBeCloseTo(0.25 + beta, 12)
      })
    }

    it('rechaza klWeight negativo, NaN o infinito', () => {
      expect(() => new VAE(1, [], 1, [], undefined, -1)).toThrow()
      expect(() => new VAE(1, [], 1, [], undefined, Number.NaN)).toThrow()
      expect(() => new VAE(1, [], 1, [], undefined, Number.POSITIVE_INFINITY)).toThrow()
    })
  })

  describe('train: un paso a mano', () => {
    it('decoder: δ = 2(x − x̂)/n·σ′(x̂); pesos SGD w += lr·δ·z y b += lr·δ', () => {
      const vae = build(1) // β = 1
      // μ = 0.5, logVar = ln 4 → σ = 2; con ε = 0.5: z = 0.5 + 2·0.5 = 1.5
      constEncoder(vae, [0.5], [LN4])
      const r = withRandom(epsStub([0.5]).fn, () => vae.train([1], 0.5))

      // x̂ = σ(0) = 0.5. δ = 2·(1 − 0.5)/1 · σ′ = 1 · (0.5·0.5) = 0.25
      // w_dec = 0 + 0.5·0.25·z = 0.125·1.5 = 0.1875 ;  b_dec = 0 + 0.5·0.25 = 0.125
      const dec = vae.decoder.getWeights()
      expect(dec[0]).toBeCloseTo(0.1875, 12)
      expect(dec[1]).toBeCloseTo(0.125, 12)

      // totalLoss = recon + β·KL = 0.25 + (−½(1 + ln4 − 0.25 − 4)) = 0.25 + 0.9318528194
      expect(r.reconLoss).toBeCloseTo(0.25, 12)
      expect(r.totalLoss).toBeCloseTo(0.25 + 0.9318528194400547, 9)
    })

    it('50 pasos con semilla fija: pérdidas finitas y el decoder cambia', () => {
      const vae = build(2)
      const before = vae.decoder.getWeights()
      for (let i = 0; i < 50; i++) {
        const r = vae.train([0.3], 0.1)
        expect(Number.isFinite(r.totalLoss)).toBe(true)
        expect(Number.isFinite(r.reconLoss)).toBe(true)
        expect(Number.isFinite(r.klLoss)).toBe(true)
      }
      expect(vae.decoder.getWeights()).not.toEqual(before)
    })
  })

  describe('rama con options.activations (línea 78)', () => {
    // Encoder [1 → 1 → 2]: oculta [w=1, b=−2]; μ [w=3, b=0]; logVar [w=0, b=0].
    const WEIGHTS = [1, -2, 3, 0, 0, 0]

    it('con activations: la oculta usa relu y la salida es lineal (μ = 3 sin acotar)', () => {
      const vae = new VAE(1, [1], 1, [1], { activations: [relu, sigmoid] })
      vae.encoder.setWeights(WEIGHTS)
      // x = [3]: oculta = relu(3 − 2) = 1; μ = 3·1 + 0 = 3
      expect(vae.encode([3]).mu[0]).toBeCloseTo(3, 12)
      // x = [1]: oculta = relu(1 − 2) = 0; μ = 0
      expect(vae.encode([1]).mu[0]).toBeCloseTo(0, 12)
    })

    it('sin activations: la oculta es sigmoide, μ = 3·σ(1)', () => {
      const vae = new VAE(1, [1], 1, [1])
      vae.encoder.setWeights(WEIGHTS)
      // oculta = σ(3 − 2) = σ(1) = 0.7310585786300049; μ = 3·0.7310585786 = 2.1931757358900147
      expect(vae.encode([3]).mu[0]).toBeCloseTo(2.1931757358900147, 9)
    })
  })
})
