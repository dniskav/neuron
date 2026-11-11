// Tests de PCA (src/PCA.ts; iteración de potencias + deflación de Hotelling).
//
// Regla de valores (FR-004): cada cifra numérica lleva su derivación a mano en un
// comentario. Ninguna se copia de la salida del código.
//
// ─── Conjunto D (covarianza diagonal) ─────────────────────────────────────────
//   D = {(2,1), (2,−1), (−2,1), (−2,−1)}, media (0,0).
//   Σ = (1/n)·Σ xᵢxᵢᵀ:  Σx² = 4·4 = 16 → 16/4 = 4;  Σy² = 4·1 = 4 → 4/4 = 1;
//   Σxy = 2·1 + 2·(−1) + (−2)·1 + (−2)·(−1) = 0.   Luego Σ = diag(4, 1).
//   Autovalores 4 (eje e1) y 1 (eje e2). Ratios 4/5 = 0.8 y 1/5 = 0.2.
//   DS = D trasladado por (10, −3): media (10, −3), misma covarianza. Ejercita el centrado.
//
// ─── Conjunto L (línea y = x) ─────────────────────────────────────────────────
//   L = {(1,1), (2,2), (3,3)}, media (2,2). Centrados: (−1,−1), (0,0), (1,1).
//   Σ = (1/3)·[[2, 2], [2, 2]] = [[2/3, 2/3], [2/3, 2/3]].
//   Autovalores 4/3 (vector (1,1)/√2) y 0. Traza 4/3.
//   Primer componente ±(1/√2, 1/√2), λ₁ = 4/3, ratio 1 con un único componente.
//
// ─── Error de reconstrucción con un componente ─────────────────────────────────
//   PCA(1) sobre DS: x − x̂ = (y centrada)·e2, cuyo cuadrado es y² = 1 en cada
//   punto. Error cuadrático medio = 1 = λ₂ (varianza descartada).
//
// ─── Transformada de un punto ────────────────────────────────────────────────
//   (12, −2) − (10, −3) = (2, 1): en la base de ejes, z = (±2, ±1).
//
// ─── Tolerancias y mediciones (T051) ──────────────────────────────────────────
//   Antes de T051 el criterio de parada era |dot| − 1 < 1e-10, que deja los vectores a
//   ~1e-5 rad (el error del dot es cuadrático en el ángulo). Ahora se itera hasta
//   ‖v_nuevo − v‖ < 1e-14 y cada paso se re-ortogonaliza contra los componentes ya
//   hallados. Error máximo de reconstrucción con TODOS los componentes, |x − x̂|∞:
//
//     semilla        1        2        3        4        5      (conjunto DS)
//     antes     2.2e-5   1.7e-5   1.2e-5   9.0e-6   8.2e-6
//     después        0        0        0        0        0     (≤ 1.8e-15 en 18 ajustes)
//
//   También se midió en un conjunto 4-D de rango 3 (antes 2.06 por un cuarto eje no
//   ortogonal) y uno 3-D (antes ≤ 3.4e-5): después ≤ 1.8e-15 en 6 semillas. Se llega a
//   1e-9 con margen de ≥ 6 órdenes, así que ejes, componentes y reconstrucción se fijan
//   a 1e-9. Autovalores y ratios: 1e-9 (diferencia medida antes/después ≤ 4e-10).
//
// ─── Ratio con menos componentes que dimensiones (H2) ──────────────────────────
//   PCA(1) sobre DS: λ₁ = 4, varianza total = traza Σ = 4 + 1 = 5 ⇒ r₁ = 4/5 = 0.8
//   (antes el código dividía por la suma retenida 4 y daba 1). Si la traza se calculara
//   sin centrar sería (12²+12²+8²+8²)/4 + ((−2)²+(−4)²+(−2)²+(−4)²)/4 = 104 + 10 = 114
//   ⇒ r₁ = 4/114, distinto de 0.8. Con 2 componentes sobre DS la suma sigue siendo 1.
//
//
// ─── Datos con ejes y autovalores conocidos (T062) ────────────────────────────────
//   Filas de Hadamard H (Sylvester, H₁ = [1], H₂ₙ = [[H,H],[H,−H]]): entradas ±1,
//   filas mutuamente ortogonales de norma² = p. Ejes unitarios uₖ = Hₖ/√p. Para amplitudes
//   aₖ > 0 (K ejes) los datos son los 2K puntos ±aₖ·Hₖ: media 0 (vienen en pares ±) y
//     Σ = (1/2K)·Σₖ 2·aₖ²·Hₖ·Hₖᵀ = (1/K)·Σₖ aₖ²·p·uₖ·uₖᵀ   ⇒   λₖ = aₖ²·p/K, eje uₖ.
//   Con K = p: λₖ = aₖ². Con K < p el rango es K y los otros p − K autovalores son 0.
//   Las mediciones se hicieron aplicando a mano los mutantes de T046 a src/PCA.ts:
//     · deflación con signo cambiado u omitida, y sin la segunda pasada de
//       reortogonalización, y sin ortogonalizar el vector inicial: SIN efecto observable
//       (equivalentes: ver comentario de cada test), no se fuerza ningún test.
//     · sin ortogonalizar dentro del bucle (con o sin el vector inicial): lo detectan los
//       tests de «ortogonalización en el bucle» (ejes a 1e-5…1 de ortonormales, error de
//       reconstrucción ≥ 1e-6 frente a ≤ 1e-12 con el código correcto).
//// Semilla: seedRandom en beforeEach y restoreRandom en afterEach.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { PCA } from '../src/PCA'
import { seedRandom, restoreRandom } from './helpers/seed'

const D = [[2, 1], [2, -1], [-2, 1], [-2, -1]]
const DS = D.map(([x, y]) => [x + 10, y - 3])
const L = [[1, 1], [2, 2], [3, 3]]

const TOL_EIG = 9 // cifras de precisión de toBeCloseTo para autovalores y ratios
const TOL_AXIS = 1e-9 // tolerancia absoluta para componentes frente a ejes

/** Comprueba |actual − expected| < tol para cada elemento de un vector. */
function expectNearVec(actual: number[], expected: number[], tol: number): void {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(tol))
}

/** Error cuadrático medio por muestra entre dos matrices de filas. */
function meanSquaredRowError(A: number[][], B: number[][]): number {
  let s = 0
  A.forEach((row, i) => row.forEach((v, j) => { s += (v - B[i][j]) ** 2 }))
  return s / A.length
}

/** Filas de Hadamard (Sylvester) de orden p (potencia de 2): entradas ±1, ortogonales entre sí. */
function hadamard(p: number): number[][] {
  let H = [[1]]
  while (H.length < p) {
    const n = H.length
    const N: number[][] = []
    for (let i = 0; i < 2 * n; i++) {
      N.push(Array.from({ length: 2 * n }, (_, j) => H[i % n][j % n] * (i >= n && j >= n ? -1 : 1)))
    }
    H = N
  }
  return H
}

/** Los 2K puntos ±aₖ·Hₖ (aₖ > 0). Media 0, ejes uₖ = Hₖ/√p, λₖ = aₖ²·p/K (ver cabecera). */
function axisData(p: number, amps: number[]): number[][] {
  const H = hadamard(p)
  const X: number[][] = []
  amps.forEach((a, k) => {
    if (a > 0) {
      X.push(H[k].map(v => v * a))
      X.push(H[k].map(v => -v * a))
    }
  })
  return X
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)

/** max |cᵢ·cⱼ − δᵢⱼ| sobre los primeros m componentes. */
function maxOrthoError(C: number[][], m: number): number {
  let e = 0
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) e = Math.max(e, Math.abs(dot(C[i], C[j]) - (i === j ? 1 : 0)))
  return e
}

/** max |x − x̂| con todos los componentes. */
function maxReconstructionError(pca: PCA, X: number[][]): number {
  const R = pca.inverseTransform(pca.transform(X))
  let e = 0
  X.forEach((row, i) => row.forEach((v, j) => { e = Math.max(e, Math.abs(v - R[i][j])) }))
  return e
}

describe('PCA', () => {
  beforeEach(() => seedRandom(0x5eed))
  afterEach(() => restoreRandom())

  describe('constructor', () => {
    it('acepta nComponents entero positivo y empieza sin ajustar', () => {
      for (const k of [1, 2, 5]) {
        const pca = new PCA(k)
        expect(pca.components).toEqual([])
        expect(pca.explainedVariance).toEqual([])
        expect(pca.mean).toEqual([])
      }
    })

    it('rechaza nComponents no entero o no positivo', () => {
      for (const k of [0, -1, 1.5, NaN]) {
        expect(() => new PCA(k)).toThrow(/nComponents must be a positive integer/)
      }
    })
  })

  describe('fit: covarianza diag(4, 1)', () => {
    it('la media es (10, −3): suma 40/4 y −12/4', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expect(pca.mean).toEqual([10, -3])
    })

    it('autovalores 4 y 1 (varianza a lo largo de cada eje)', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expect(pca.explainedVariance[0]).toBeCloseTo(4, TOL_EIG)
      expect(pca.explainedVariance[1]).toBeCloseTo(1, TOL_EIG)
    })

    it('componentes = ejes e1 y e2, con signo libre (se compara |·|)', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const [c1, c2] = pca.components
      expectNearVec([Math.abs(c1[0]), Math.abs(c1[1])], [1, 0], TOL_AXIS)
      expectNearVec([Math.abs(c2[0]), Math.abs(c2[1])], [0, 1], TOL_AXIS)
    })

    it('ratios 4/5 = 0.8 y 1/5 = 0.2 con dos componentes', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const r = pca.explainedVarianceRatio()
      expect(r[0]).toBeCloseTo(0.8, TOL_EIG)
      expect(r[1]).toBeCloseTo(0.2, TOL_EIG)
    })
  })

  describe('fit: línea y = x', () => {
    it('primer componente ±(1/√2, 1/√2), λ₁ = 4/3, ratio 1 con un componente', () => {
      const pca = new PCA(1)
      pca.fit(L)
      const c = pca.components[0]
      const s = 1 / Math.SQRT2
      expectNearVec([Math.abs(c[0]), Math.abs(c[1])], [s, s], TOL_AXIS)
      expect(c[0] * c[1]).toBeGreaterThan(0) // misma dirección que (1,1)
      expect(pca.mean).toEqual([2, 2])
      expect(pca.explainedVariance[0]).toBeCloseTo(4 / 3, TOL_EIG)
      expect(pca.explainedVarianceRatio()[0]).toBeCloseTo(1, 9)
    })
  })

  describe('transform e inverseTransform', () => {
    it('transform de la media es el origen', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expectNearVec(pca.transform([[10, -3]])[0], [0, 0], 1e-12)
    })

    it('transform de (12, −2) da coordenadas (±2, ±1) en la base de ejes', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const z = pca.transform([[12, -2]])
      expect(z.length).toBe(1)
      expect(z[0].length).toBe(2)
      expect(Math.abs(Math.abs(z[0][0]) - 2)).toBeLessThan(TOL_AXIS)
      expect(Math.abs(Math.abs(z[0][1]) - 1)).toBeLessThan(TOL_AXIS)
    })

    it('transform devuelve n filas de nComponents columnas', () => {
      const pca = new PCA(1)
      pca.fit(DS)
      const z = pca.transform(DS)
      expect(z.length).toBe(4)
      z.forEach(row => expect(row.length).toBe(1))
    })

    it('inverseTransform([[0, 0]]) devuelve la media (sin proyección no hay desplazamiento)', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      expectNearVec(pca.inverseTransform([[0, 0]])[0], [10, -3], 1e-12)
    })

    it('inverseTransform(transform(X)) tiene forma n × nFeatures', () => {
      const pca = new PCA(2)
      pca.fit(DS)
      const R = pca.inverseTransform(pca.transform(DS))
      expect(R.length).toBe(4)
      R.forEach(row => expect(row.length).toBe(2))
    })

    it('reconstrucción con TODOS los componentes: error ≤ 1e-9 en 5 semillas (H1)', () => {
      // x̂ = Z·V + μ con V ortonormal 2×2 ⇒ x̂ = x. Antes de T051 el error era ~1e-5.
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const pca = new PCA(2)
        pca.fit(DS)
        const R = pca.inverseTransform(pca.transform(DS))
        DS.forEach((row, i) => expectNearVec(R[i], row, 1e-9))
      }
    })

    it('reconstrucción exacta también con datos de rango deficiente (4 columnas, rango 2)', () => {
      // Columnas (a, b, a+b, 2a−b): rango 2 ⇒ autovalores 3 y 4 son 0 y sus ejes
      // deben seguir siendo ortonormales para que x̂ = x.
      const X = [[0, 0], [1, 0], [0, 1], [2, 1], [1, 3]].map(([a, b]) => [a, b, a + b, 2 * a - b])
      const pca = new PCA(4)
      pca.fit(X)
      const R = pca.inverseTransform(pca.transform(X))
      X.forEach((row, i) => expectNearVec(R[i], row, 1e-9))
    })

    it('reconstrucción con 1 componente: error cuadrático medio = λ₂ = 1', () => {
      // Ver cabecera: x − x̂ = (y centrada)·e2, y² = 1 en cada punto.
      const pca = new PCA(1)
      pca.fit(DS)
      const R = pca.inverseTransform(pca.transform(DS))
      expect(meanSquaredRowError(DS, R)).toBeCloseTo(1, TOL_EIG)
    })
  })

  describe('ortogonalización en el bucle de iteración (T062)', () => {
    // Sin ortogonalizar cada paso, la iteración de potencias pura deja que el redondeo de la
    // deflación reintroduzca ejes anteriores: los ejes finales dejan de ser ortonormales.
    // (Deflación y reortogonalización por sí solas son redundantes entre sí: ver el informe.)

    it('8 ejes con autovalores 1e8, 1e6, …, 1e-6: ejes ortonormales y reconstrucción exacta', () => {
      // p = K = 8, aₖ = 1e4/10ᵏ ⇒ λₖ = aₖ² = 1e8/100ᵏ (cabecera). Σλ = 1e8·(1 − 100⁻⁸)/(1 − 1/100).
      // Ratio rₖ = 100⁻ᵏ / Σⱼ 100⁻ʲ. Ortonormalidad: VVᵀ = I (8 vectores unitarios ortogonales).
      // Medido: código correcto ≤ 5e-16; sin ortogonalizar en el bucle ≥ 2.9e-5 (6 semillas).
      const amps = [1e4, 1e3, 1e2, 10, 1, 0.1, 0.01, 0.001]
      const X = axisData(8, amps)
      const H = hadamard(8)
      const sumGeo = (1 - 100 ** -8) / (1 - 1 / 100)
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const pca = new PCA(8)
        pca.fit(X)
        expect(maxOrthoError(pca.components, 8)).toBeLessThan(1e-9)
        expect(maxReconstructionError(pca, X)).toBeLessThan(1e-8) // |x| ≤ 1e4·1
        // Los tres primeros ejes son H₀/√8, H₁/√8, H₂/√8 (|c·u| = 1; los demás, λ ≪ ruido de Σ)
        for (let k = 0; k < 3; k++) {
          expect(Math.abs(dot(pca.components[k], H[k])) / Math.sqrt(8)).toBeGreaterThan(1 - 1e-9)
        }
        // Ratios: los autovalores se ven a través del ruido relativo de Σ (~1e-16)
        const r = pca.explainedVarianceRatio()
        for (let k = 0; k < 8; k++) expect(Math.abs(r[k] - 100 ** -k / sumGeo)).toBeLessThan(1e-12)
        expect(pca.explainedVariance[0] / 1e8).toBeCloseTo(1, 9)
        expect(pca.explainedVariance[1] / 1e6).toBeCloseTo(1, 9)
      }
    })

    it('rango 3 en 4 columnas, escala 1e3 (caso de T051): cuarto eje ortogonal y x̂ = x', () => {
      // Puntos ±1e3·Hₖ, k = 1..3 (filas (1,1,1,1), (1,1,−1,−1), (1,−1,1,−1)); n = 6, media 0.
      // Σ = (1/6)·2·Σₖ 1e6·Hₖ·Hₖᵀ = (1e6/3)·4·(u₁u₁ᵀ + u₂u₂ᵀ + u₃u₃ᵀ) ⇒ λ = 4e6/3 (triple), 0.
      // Traza 4e6 ⇒ ratios 1/3, 1/3, 1/3 y 0. El eje nulo es u₄ = (1,−1,−1,1)/2.
      // Autovalor triple: la base de los tres primeros ejes es libre; se comprueba que sea
      // ortonormal, ⊥ u₄, y que x̂ = x. Medido (error máx. |x − x̂|, 6 semillas): correcto ≤ 1e-12; sin ortogonalizar ≥ 1e3.
      const X = axisData(4, [1e3, 1e3, 1e3, 0])
      const u4 = [0.5, -0.5, -0.5, 0.5]
      for (const seed of [1, 2, 3, 4, 5, 6]) {
        seedRandom(seed)
        const pca = new PCA(4)
        pca.fit(X)
        const C = pca.components
        expect(maxOrthoError(C, 3)).toBeLessThan(1e-9)
        for (let k = 0; k < 3; k++) expect(Math.abs(dot(C[k], u4))).toBeLessThan(1e-9)
        // El cuarto eje es ⊥ a los tres primeros (puede salir con norma ≈ 0 si λ₄ = 0 y el
        // ruido de Σ cae bajo el umbral de normalización: también cumple |c₄·cₖ| ≈ 0).
        for (let k = 0; k < 3; k++) expect(Math.abs(dot(C[3], C[k]))).toBeLessThan(1e-9)
        expect(maxReconstructionError(pca, X)).toBeLessThan(1e-6)
        pca.explainedVariance.slice(0, 3).forEach(l => expect(l / (4e6 / 3)).toBeCloseTo(1, 9))
        expect(Math.abs(pca.explainedVariance[3])).toBeLessThan(1e-6)
        const r = pca.explainedVarianceRatio()
        for (let k = 0; k < 3; k++) expect(r[k]).toBeCloseTo(1 / 3, 9)
      }
    })

    it('autovalores casi iguales (1.002001, 1.00100025, 1, 0.99900025): ejes ortonormales', () => {
      // p = K = 4, aₖ = (1.001, 1.0005, 1, 0.9995) ⇒ λₖ = aₖ² = 1.002001, 1.00100025, 1, 0.99900025
      // (cabecera), suma 4.0020015, ratio aₖ²/Σa². Huecos relativos ~5e-4 ⇒ ejes ≈ H₀…H₃/2.
      // Medido: ortonormalidad correcta ≤ 5e-16; sin ortogonalizar en el bucle ≥ 5.6e-8.
      const amps = [1.001, 1.0005, 1, 0.9995]
      const X = axisData(4, amps)
      const H = hadamard(4)
      const total = amps.reduce((s, a) => s + a * a, 0)
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const pca = new PCA(4)
        pca.fit(X)
        expect(maxOrthoError(pca.components, 4)).toBeLessThan(1e-10)
        expect(maxReconstructionError(pca, X)).toBeLessThan(1e-9)
        amps.forEach((a, k) => {
          expect(Math.abs(pca.explainedVariance[k] - a * a)).toBeLessThan(1e-8)
          expect(Math.abs(pca.explainedVarianceRatio()[k] - (a * a) / total)).toBeLessThan(1e-8)
          expect(Math.abs(dot(pca.components[k], H[k])) / 2).toBeGreaterThan(1 - 1e-8)
        })
      }
    })
  })

  describe('fitTransform', () => {
    it('equivale a fit seguido de transform con la misma semilla', () => {
      seedRandom(3)
      const a = new PCA(2).fitTransform(DS)
      seedRandom(3)
      const b = new PCA(2)
      b.fit(DS)
      expect(a).toEqual(b.transform(DS))
    })
  })

  describe('explainedVarianceRatio', () => {
    it('sin ajustar devuelve []', () => {
      expect(new PCA(1).explainedVarianceRatio()).toEqual([])
    })

    it('con menos componentes que dimensiones el ratio usa la varianza TOTAL: 0.8 (H2)', () => {
      // Ver cabecera: λ₁ = 4, traza Σ = 5 ⇒ r₁ = 4/5. Dividir por la suma retenida daría 1.
      const pca = new PCA(1)
      pca.fit(DS)
      const r = pca.explainedVarianceRatio()
      expect(r.length).toBe(1)
      expect(r[0]).toBeCloseTo(0.8, TOL_EIG)
    })

    it('datos sin varianza: ratio 0 (convención del código cuando el total es 0)', () => {
      const pca = new PCA(1)
      pca.fit([[1, 2], [1, 2], [1, 2]])
      expect(pca.explainedVariance).toEqual([0])
      expect(pca.explainedVarianceRatio()).toEqual([0])
    })
  })

  describe('errores', () => {
    it('fit lanza con menos de 2 muestras', () => {
      expect(() => new PCA(1).fit([[1, 2]])).toThrow(/at least 2 samples/)
    })

    it('fit lanza si nComponents supera el número de características', () => {
      expect(() => new PCA(3).fit(DS)).toThrow(/cannot exceed number of features \(2\)/)
    })

    it('transform lanza sin fit()', () => {
      expect(() => new PCA(1).transform([[0, 0]])).toThrow(/call fit\(\) before transform/)
    })

    it('inverseTransform lanza sin fit()', () => {
      expect(() => new PCA(1).inverseTransform([[0]])).toThrow(/call fit\(\) before inverseTransform/)
    })
  })

  describe('robustez: semillas y determinismo', () => {
    it('5 semillas dan los mismos autovalores y ejes', () => {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const pca = new PCA(2)
        pca.fit(DS)
        expect(pca.explainedVariance[0]).toBeCloseTo(4, TOL_EIG)
        expect(pca.explainedVariance[1]).toBeCloseTo(1, TOL_EIG)
        const [c1, c2] = pca.components
        expectNearVec([Math.abs(c1[0]), Math.abs(c1[1])], [1, 0], TOL_AXIS)
        expectNearVec([Math.abs(c2[0]), Math.abs(c2[1])], [0, 1], TOL_AXIS)
      }
    })

    it('misma semilla → mismos componentes y autovalores', () => {
      seedRandom(42)
      const a = new PCA(2)
      a.fit(DS)
      seedRandom(42)
      const b = new PCA(2)
      b.fit(DS)
      expect(a.components).toEqual(b.components)
      expect(a.explainedVariance).toEqual(b.explainedVariance)
    })
  })
})
