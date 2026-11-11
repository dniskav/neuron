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
// Umbrales (medidos con 7 semillas de datos x perplexity {3.5, 5, 6.5} x
// lr {3.5, 4.9, 5, 7, 9.1, 10, 13}, es decir lr de referencia {5, 7, 10} x 0.7
// y x 1.3; el CI es Node 20/Linux):
//   - separación: razón R = (distancia entre centroides en 2-D) / (distancia
//     media entre pares del mismo clúster). Medido: peor caso 2.06 (lr 3.5,
//     perplexity 3.5), 2.23 en la rejilla lr {5, 7, 10}; mejor 5.66; control sin
//     estructura 0.30. Umbral 1.5 (el test afirma R >= 1.5 tal cual). Margen =
//     peor / umbral - 1 = 2.06 / 1.5 - 1 = 37 % >= 30 %. (Antes el test exigía
//     R >= 1.5 / 0.7 = 2.143, con lo que el margen real contra el peor caso 2.23
//     era ~4 %, no 30 %: el margen se aplicaba dos veces. T063.)
//     (La razón con el diámetro máximo en vez de la media da ~1-2.6: t-SNE
//     expande cada clúster, por lo que no se usa el máximo.)
//   - KL: kl(nIter = 300) <= 0.7 · kl(nIter = 5).
//
// Kernel e inicialización (T063): con nIter = 0 el bucle de descenso no corre,
// así que `embedding` es la inicialización y `kl()` se evalúa sobre ella. Tres
// puntos equidistantes (e1, e2, e3) con perplexity 2 dan P(j|i) = 1/2 para
// cualquier sigma (H = 1 bit = log2 2), luego p_ij = (1/2 + 1/2) / 6 = 1/6.
// Con q_ij = inv_ij / Z, inv = 1/(1 + d²), Z = Σ_{k≠l} inv_kl, se reimplementa
// a mano en el test la KL y un paso de gradiente (nIter = 1, lr = 1, momentum
// sin efecto en el primer paso, exageración x4):
//   y_i' = y_i - 4 Σ_j (4 p_ij - q_ij)(y_i - y_j) inv_ij
// La inicialización se reproduce con mulberry32 + Box-Muller x 0.01.
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

/** Inicialización de TSNE reproducida de forma independiente: Box-Muller x 0.01. */
function expectedInit(seed: number, n: number, dims: number): number[][] {
  const rng = mulberry32(seed)
  return Array.from({ length: n }, () =>
    Array.from({ length: dims }, () => {
      const u1 = Math.max(rng(), 1e-12)
      const u2 = rng()
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * 0.01
    })
  )
}

/** Matriz q_ij con el kernel Student-t 1/(1+d²), normalizada. */
function studentQ(Y: number[][]): { inv: number[][]; Q: number[][] } {
  const n = Y.length
  const inv = Y.map((a, i) => Y.map((b, j) => (i === j ? 0 : 1 / (1 + dist(a, b) ** 2))))
  const Z = inv.flat().reduce((s, v) => s + v, 0)
  return { inv, Q: inv.map(r => r.map(v => v / Z)) }
}

const E3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

describe('TSNE kernel and initialization (n = 3, hand-derived)', () => {
  it('initial embedding is 0.01 x N(0,1) from the seeded Box-Muller draw', () => {
    const Y = new TSNE({ perplexity: 2, nIter: 0, seed: 5 }).fitTransform(E3)
    const want = expectedInit(5, 3, 2)
    for (let i = 0; i < 3; i++) {
      for (let d = 0; d < 2; d++) expect(Y[i][d]).toBeCloseTo(want[i][d], 12)
    }
    // Cota independiente de la implementación: |z| <= sqrt(-2 ln 1e-12) = 7.43,
    // luego |y| < 0.075 con escala 0.01; y no puede ser casi cero.
    const big = new TSNE({ perplexity: 5, nIter: 0, seed: 9 }).fitTransform(twoClusters(1))
    const vals = big.flat()
    expect(Math.max(...vals.map(Math.abs))).toBeLessThan(0.075)
    const rms = Math.sqrt(vals.reduce((s, v) => s + v * v, 0) / vals.length)
    expect(rms).toBeGreaterThan(0.005)
    expect(rms).toBeLessThan(0.02)
  })

  it('kl() on the initial embedding matches sum p ln(p / q) with the Student-t kernel', () => {
    const t = new TSNE({ perplexity: 2, nIter: 0, seed: 5 })
    t.fit(E3)
    const { Q } = studentQ(expectedInit(5, 3, 2))
    let kl = 0
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) if (i !== j) kl += (1 / 6) * Math.log((1 / 6) / (Q[i][j] + 1e-12))
    }
    expect(t.kl()).toBeCloseTo(kl, 11)
  })

  it('one gradient step (lr = 1, exaggeration x4) matches the hand-derived update', () => {
    const Y0 = expectedInit(5, 3, 2)
    const { inv, Q } = studentQ(Y0)
    const Y1 = new TSNE({ perplexity: 2, nIter: 1, lr: 1, seed: 5 }).fitTransform(E3)
    for (let i = 0; i < 3; i++) {
      for (let d = 0; d < 2; d++) {
        let g = 0
        for (let j = 0; j < 3; j++) {
          if (j !== i) g += 4 * (4 * (1 / 6) - Q[i][j]) * inv[i][j] * (Y0[i][d] - Y0[j][d])
        }
        expect(Y1[i][d]).toBeCloseTo(Y0[i][d] - g, 10)
      }
    }
  })
})

describe('TSNE separates two well-separated clusters', () => {
  const variants: Array<{ perplexity: number; lr: number }> = []
  for (const perplexity of [3.5, 5, 6.5]) {
    // lr de referencia {5, 7, 10} y sus perturbaciones x0.7 / x1.3 (3.5, 13).
    for (const lr of [3.5, 5, 7, 10, 13]) variants.push({ perplexity, lr })
  }

  it('centroid distance >= 1.5 x mean intra-cluster distance (5 data seeds x 15 variants; worst case 2.06, margin 37 %)', () => {
    for (const dataSeed of [1, 2, 3, 4, 5]) {
      const X = twoClusters(dataSeed)
      for (const v of variants) {
        const Y = new TSNE({ ...v, nIter: 300, seed: dataSeed + 10 }).fitTransform(X)
        expect(separation(Y, 15)).toBeGreaterThanOrEqual(1.5)
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
