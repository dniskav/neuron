// Tests de TSNE (src/TSNE.ts). Algoritmo estocástico sin solución cerrada:
// se comprueban propiedades, no coordenadas.
//
// Caso a mano (n = 2): dos puntos tienen una sola vecina cada uno, así que
// P(1|0) = P(0|1) = 1 para cualquier sigma, y p01 = (1 + 1) / (2·2) = 0.5.
// En 2-D, q01 = inv / (2·inv) = 0.5 sea cual sea la distancia. Por tanto
// KL = 2 · 0.5 · ln(0.5 / (0.5 + 1e-12)) ≈ -2e-12, es decir ≈ 0 (|kl| < 1e-9).
//
// Datos de clústeres: 15 puntos por clúster en 5-D, ruido uniforme en [-1, 1)
// por eje (σ = √(1/3) ≈ 0.577). Centros (0,…,0) y (6,…,6): distancia entre
// centros 6·√5 ≈ 13.4, es decir ≈ 23 σ ≥ 10 σ.
//
// lr: se usa lr = 7 (y 5 / 10), NO el valor por defecto 200. Con n = 30 puntos
// lr = 200 es inestable (el embedding se expande a cientos de unidades, razón
// de separación ~0.1; la regla de estabilidad es lr <~ n / 12 · factor). Es un
// ajuste de hiperparámetros, no un defecto: 200 es el valor para n grande.
//
// Umbrales (medidos con 5 semillas de datos x perplexity {3.5, 5, 6.5} x
// lr {5, 7, 10}; el CI es Node 20/Linux):
//   - separación: razón R = (distancia entre centroides en 2-D) / (distancia
//     media entre pares del mismo clúster). Medido: peor caso 2.23, mejor 5.24;
//     control sin estructura 0.30. Umbral 1.5; el peor caso debe cumplir
//     R >= 1.5 / 0.7 = 2.14 (margen >= 30 %).
//     (La razón con el diámetro máximo en vez de la media da ~1-2.6: t-SNE
//     expande cada clúster, por lo que no se usa el máximo.)
//   - KL: kl(nIter = 300) <= 0.7 · kl(nIter = 5).
//
// Semilla: seedRandom en beforeEach y restoreRandom en afterEach; TSNE además
// recibe `seed` propio.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { TSNE } from '../src/TSNE'
import { mulberry32, seedRandom, restoreRandom } from './helpers/seed'

beforeEach(() => seedRandom())
afterEach(() => restoreRandom())

/** Dos clústeres de `m` puntos en 5-D con ruido uniforme en [-1, 1). */
function twoClusters(dataSeed: number, m = 15): number[][] {
  const rng = mulberry32(dataSeed)
  const X: number[][] = []
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < m; i++) {
      X.push(Array.from({ length: 5 }, () => c * 6 + (rng() * 2 - 1)))
    }
  }
  return X
}

function dist(a: number[], b: number[]): number {
  let s = 0
  for (let k = 0; k < a.length; k++) s += (a[k] - b[k]) ** 2
  return Math.sqrt(s)
}

function centroid(P: number[][]): number[] {
  return P[0].map((_, k) => P.reduce((s, r) => s + r[k], 0) / P.length)
}

/** Razón distancia entre centroides / distancia media entre pares del mismo clúster. */
function separation(Y: number[][], m: number): number {
  const A = Y.slice(0, m)
  const B = Y.slice(m)
  const between = dist(centroid(A), centroid(B))
  let sum = 0
  let count = 0
  for (const G of [A, B]) {
    for (let i = 0; i < G.length; i++) {
      for (let j = i + 1; j < G.length; j++) {
        sum += dist(G[i], G[j])
        count++
      }
    }
  }
  return between / (sum / count)
}

describe('TSNE shape and errors', () => {
  it('returns an n x nComponents embedding (default 2)', () => {
    const X = twoClusters(1)
    const t = new TSNE({ perplexity: 5, nIter: 20, seed: 1 })
    const Y = t.fitTransform(X)
    expect(Y.length).toBe(30)
    for (const row of Y) {
      expect(row.length).toBe(2)
      for (const v of row) expect(Number.isFinite(v)).toBe(true)
    }
    expect(t.embedding).toBe(Y)
  })

  it('honors nComponents = 3', () => {
    const t = new TSNE({ perplexity: 5, nIter: 10, nComponents: 3, seed: 1 })
    const Y = t.fitTransform(twoClusters(1))
    expect(Y.length).toBe(30)
    expect(Y.every(r => r.length === 3)).toBe(true)
  })

  it('fit() fills embedding and starts empty', () => {
    const t = new TSNE({ perplexity: 5, nIter: 5, seed: 1 })
    expect(t.embedding).toEqual([])
    t.fit(twoClusters(1))
    expect(t.embedding.length).toBe(30)
  })

  it('throws with fewer than 2 points', () => {
    expect(() => new TSNE({ perplexity: 1 }).fit([])).toThrow('need at least 2')
    expect(() => new TSNE({ perplexity: 1 }).fit([[1, 2]])).toThrow('need at least 2')
  })

  it('throws when perplexity >= n (boundary included)', () => {
    const X = [[0], [1], [2], [3]]
    expect(() => new TSNE({ perplexity: 4 }).fit(X)).toThrow(/perplexity \(4\) must be less than n \(4\)/)
    expect(() => new TSNE({ perplexity: 30 }).fit(X)).toThrow(/must be less than n/)
  })

  it('accepts perplexity just below n', () => {
    const X = [[0], [1], [2], [3]]
    expect(() => new TSNE({ perplexity: 3, nIter: 2, seed: 1 }).fit(X)).not.toThrow()
  })
})

describe('TSNE hand-computed case (n = 2)', () => {
  it('has kl() ~ 0 because P and Q are both [[0,.5],[.5,0]]', () => {
    const t = new TSNE({ perplexity: 1.5, nIter: 30, seed: 3 })
    t.fit([[0, 0], [1, 0]])
    expect(Math.abs(t.kl())).toBeLessThan(1e-9)
  })

  it('has kl() ~ 0 regardless of the separation of the two points', () => {
    const t = new TSNE({ perplexity: 1.5, nIter: 30, seed: 3 })
    t.fit([[0, 0], [0.5, 0.5]])
    expect(Math.abs(t.kl())).toBeLessThan(1e-9)
  })
})

describe('TSNE kl()', () => {
  it('is 0 before fit', () => {
    expect(new TSNE().kl()).toBe(0)
  })

  it('is finite, positive and at least 30 % lower after 300 iterations than after 5', () => {
    for (const dataSeed of [1, 2, 3, 4, 5]) {
      const X = twoClusters(dataSeed)
      const short = new TSNE({ perplexity: 5, lr: 7, nIter: 5, seed: dataSeed })
      short.fit(X)
      const long = new TSNE({ perplexity: 5, lr: 7, nIter: 300, seed: dataSeed })
      long.fit(X)
      expect(Number.isFinite(short.kl())).toBe(true)
      expect(Number.isFinite(long.kl())).toBe(true)
      expect(long.kl()).toBeGreaterThan(0)
      expect(long.kl()).toBeLessThanOrEqual(0.7 * short.kl())
    }
  })
})

describe('TSNE determinism', () => {
  it('same seed gives identical embedding and kl', () => {
    const X = twoClusters(7)
    const a = new TSNE({ perplexity: 5, nIter: 60, seed: 42 })
    const b = new TSNE({ perplexity: 5, nIter: 60, seed: 42 })
    a.fit(X)
    b.fit(X)
    expect(a.embedding).toEqual(b.embedding)
    expect(a.kl()).toBe(b.kl())
  })

  it('different seeds give different embeddings', () => {
    const X = twoClusters(7)
    const a = new TSNE({ perplexity: 5, nIter: 60, seed: 1 }).fitTransform(X)
    const b = new TSNE({ perplexity: 5, nIter: 60, seed: 2 }).fitTransform(X)
    expect(a).not.toEqual(b)
  })

  it('without seed, uses Math.random (seeded by the helper) reproducibly', () => {
    const X = twoClusters(7)
    seedRandom(99)
    const a = new TSNE({ perplexity: 5, nIter: 20 }).fitTransform(X)
    seedRandom(99)
    const b = new TSNE({ perplexity: 5, nIter: 20 }).fitTransform(X)
    expect(a).toEqual(b)
  })
})

describe('TSNE separates two well-separated clusters', () => {
  const variants: Array<{ perplexity: number; lr: number }> = []
  for (const perplexity of [3.5, 5, 6.5]) {
    for (const lr of [5, 7, 10]) variants.push({ perplexity, lr })
  }

  it('centroid distance >= 1.5 x mean intra-cluster distance, with 30 % margin (5 data seeds x 9 variants)', () => {
    for (const dataSeed of [1, 2, 3, 4, 5]) {
      const X = twoClusters(dataSeed)
      for (const v of variants) {
        const Y = new TSNE({ ...v, nIter: 300, seed: dataSeed + 10 }).fitTransform(X)
        expect(separation(Y, 15)).toBeGreaterThanOrEqual(1.5 / 0.7)
      }
    }
  })

  it('does not separate when there is no structure (control)', () => {
    // Una sola nube partida arbitrariamente por la mitad: R medida 0.30 << 1.5.
    const rng = mulberry32(5)
    const X = Array.from({ length: 30 }, () => Array.from({ length: 5 }, () => rng() * 2 - 1))
    const Y = new TSNE({ perplexity: 5, lr: 7, nIter: 300, seed: 3 }).fitTransform(X)
    expect(separation(Y, 15)).toBeLessThan(1.5)
  })
})
