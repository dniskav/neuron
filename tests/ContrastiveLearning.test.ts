// Tests de la parte determinista de ContrastiveLearning (src/ContrastiveLearning.ts). Tarea T022 de
// specs/006-coverage. Sin tests de "aprende" (Q8): eso vive en tests/oracle/ContrastiveLearning.oracle.test.ts.
//
// Cifras a mano (no copiadas de la salida del código):
//   · Stub U1 = e^(-1/2): en Box-Muller sqrt(-2·ln U1) = 1, así que z = cos(2π·U2).
//     U2 = 0.5 → z = -1. Con eso addNoise(x, σ) = x - σ exacto.
//   · Guarda U1 = 0 → se usa 1e-10: sqrt(-2·ln 1e-10) = sqrt(46.05) ≈ 6.786 (finito; sin guarda sería Infinity).
//   · cosineSimilarity([1,2],[2,4]) = 10 / (√5·√20 + 1e-8) = 10 / (10 + 1e-8) = 1 - 1e-9 (aprox.).
//   · NT-Xent con proyecciones conocidas: se reemplaza `project` por la identidad en la instancia, así
//     las proyecciones son las entradas y la cuenta se hace a mano. Término por ancla i:
//       t_i = -ln( e^{s(i,pos)} / Σ_{k≠i} e^{s(i,k)} ),  s = cos/τ.  Pérdida = media de los 2N términos.
//     Caso A (N=2, τ=0.5): dos pares de vistas idénticas y ortogonales entre sí:
//        pares [[1,0],[1,0]] y [[0,1],[0,1]]. cos(positivo)=1 → s=2; cos(negativos)=0 → s=0.
//        Cada ancla: t = -ln( e² / (e² + 1 + 1) ) = ln(1 + 2e^-2) ≈ 0.2397. Pérdida = t.
//        Con τ=1: s=1 → t = ln(1 + 2e^-1) ≈ 0.5518 (distinta: la temperatura cuenta).
//     Caso B (N=2, τ=0.5): vistas a=[1,0], a'=[1,1], b=b'=[0,1]. cos: a·a'=√2/2, a·b=0, a'·b=√2/2, b·b'=1.
//        s(√2/2)=√2 ≈ 1.4142, s(1)=2, s(0)=0.
//        ancla a   (pos a'):  den = e^√2 + 1 + 1      → t1 = ln(1 + 2e^-√2)
//        ancla a'  (pos a):   den = e^√2 (a) + e^√2 (b) + e^√2 (b') → t2 = ln 3
//        ancla b   (pos b'):  den = 1 (a) + e^√2 (a') + e^2 (b') → t3 = ln(1 + (1 + e^√2)/e²)
//        ancla b'  (pos b):   igual que b (b = b')  → t3
//        pérdida = (t1 + t2 + 2·t3) / 4.
//     Todas iguales (colapso): z idéntico en las 2N vistas → cada término ln(2N-1) (HANDOFF N9).
//     Los +1e-8 y +1e-10 del código aportan < 1e-7; tolerancia 1e-6.
//
// Estadística con semilla (setup.ts fija Math.random): σ=0.5 sobre 10 000 muestras → media ±0.005, desv.
// ±0.0035 (tolerancia 0.02 y 3 %, ~4σ); tasa de dropout 0.3 sobre 10 000 → fracción ±0.0046 (tol. 0.02).

import { describe, it, expect } from 'vitest'
import { Augmenter, ContrastiveLearning } from '../src/ContrastiveLearning'

const E_HALF = Math.exp(-0.5) // U1 que da sqrt(-2·ln U1) = 1

function stubRandom(values: number[]): void {
  let k = 0
  Math.random = () => {
    if (k >= values.length) throw new Error('stubRandom: secuencia agotada')
    return values[k++]
  }
}

/** Instancia con `project` = identidad: las proyecciones son las entradas. */
function identityModel(temperature?: number): ContrastiveLearning {
  const c = new ContrastiveLearning(2, [2], 2, temperature === undefined ? {} : { temperature })
  c.project = (x: number[]) => x
  return c
}

describe('Augmenter.addNoise', () => {
  it('sigma 0 devuelve los mismos valores y no muta la entrada', () => {
    const x = [1, -2, 0.5]
    const out = Augmenter.addNoise(x, 0)
    expect(out).toEqual([1, -2, 0.5])
    expect(out).not.toBe(x)
    expect(x).toEqual([1, -2, 0.5])
  })

  it('entrada vacía devuelve vacío', () => {
    expect(Augmenter.addNoise([], 0.3)).toEqual([])
  })

  it('con U2 = 0.5 el ruido es exactamente -sigma (Box-Muller a mano)', () => {
    stubRandom([E_HALF, 0.5, E_HALF, 0.5])
    const out = Augmenter.addNoise([1, 2], 0.1)
    expect(out[0]).toBeCloseTo(0.9, 12)
    expect(out[1]).toBeCloseTo(1.9, 12)
  })

  it('la guarda u1 = 0 (usa 1e-10) da un ruido finito de ~6.786 sigmas', () => {
    stubRandom([0, 0]) // z = sqrt(-2 ln 1e-10) · cos(0) = 6.7860
    const out = Augmenter.addNoise([0], 1)
    expect(Number.isFinite(out[0])).toBe(true)
    expect(out[0]).toBeCloseTo(Math.sqrt(2 * 10 * Math.LN10), 9) // sqrt(46.0517) = 6.78581...
  })

  it('con semilla: media ~0 y desviación ~sigma (sigma 0.5, 10 000 muestras)', () => {
    const n = 10000
    const out = Augmenter.addNoise(new Array(n).fill(0), 0.5)
    const mean = out.reduce((a, v) => a + v, 0) / n
    const sd = Math.sqrt(out.reduce((a, v) => a + (v - mean) ** 2, 0) / n)
    expect(Math.abs(mean)).toBeLessThan(0.02)
    expect(Math.abs(sd - 0.5)).toBeLessThan(0.015)
  })

  it('sigma por defecto 0.05: desviación ~0.05', () => {
    const n = 10000
    const out = Augmenter.addNoise(new Array(n).fill(0))
    const sd = Math.sqrt(out.reduce((a, v) => a + v * v, 0) / n)
    expect(Math.abs(sd - 0.05)).toBeLessThan(0.0015)
  })

  it('misma semilla → misma salida (reproducible)', () => {
    // setup.ts siembra antes de cada test; aquí se reinicia la secuencia a mano con un stub idéntico.
    stubRandom([0.3, 0.7, 0.4, 0.2])
    const a = Augmenter.addNoise([1, 2], 0.2)
    stubRandom([0.3, 0.7, 0.4, 0.2])
    const b = Augmenter.addNoise([1, 2], 0.2)
    expect(a).toEqual(b)
  })
})

describe('Augmenter.dropoutFeatures', () => {
  it('tasa 0 deja todo igual', () => {
    const x = [1, 2, 3, 4]
    expect(Augmenter.dropoutFeatures(x, 0)).toEqual([1, 2, 3, 4])
  })

  it('tasa 1 anula todo', () => {
    expect(Augmenter.dropoutFeatures([1, 2, 3, 4], 1)).toEqual([0, 0, 0, 0])
  })

  it('anula solo donde random < tasa (el borde 0.1 < 0.1 no anula) y NO reescala los conservados', () => {
    // randoms [0.05, 0.1, 0.5, 0.09] con tasa 0.1: anula 1.º y 4.º; 2.º (0.1<0.1 falso) y 3.º se conservan.
    stubRandom([0.05, 0.1, 0.5, 0.09])
    const out = Augmenter.dropoutFeatures([10, 20, 30, 40], 0.1)
    // Sin reescala (el código no divide por 1-p): los valores conservados quedan intactos.
    expect(out).toEqual([0, 20, 30, 0])
  })

  it('tasa por defecto 0.1: con random 0.05 anula, con 0.2 conserva', () => {
    stubRandom([0.05, 0.2])
    expect(Augmenter.dropoutFeatures([7, 8])).toEqual([0, 8])
  })

  it('con semilla: fracción anulada ~tasa (0.3, 10 000 muestras) y el resto sin cambios', () => {
    const n = 10000
    const out = Augmenter.dropoutFeatures(new Array(n).fill(5), 0.3)
    const zeros = out.filter((v) => v === 0).length
    expect(Math.abs(zeros / n - 0.3)).toBeLessThan(0.02)
    expect(out.every((v) => v === 0 || v === 5)).toBe(true)
  })

  it('no muta la entrada', () => {
    const x = [1, 2, 3]
    Augmenter.dropoutFeatures(x, 1)
    expect(x).toEqual([1, 2, 3])
  })
})

describe('Augmenter.augment', () => {
  it('aplica primero ruido y luego dropout (orden del consumo de random)', () => {
    // ruido: 4 randoms [E,0.5,E,0.5] con sigma 0.1 → [0.9, 1.9]; dropout: [0.05, 0.5] con tasa 0.1 → [0, 1.9].
    stubRandom([E_HALF, 0.5, E_HALF, 0.5, 0.05, 0.5])
    const out = Augmenter.augment([1, 2], 0.1, 0.1)
    expect(out[0]).toBe(0)
    expect(out[1]).toBeCloseTo(1.9, 12)
  })

  it('sigma 0 y tasa 0 → copia idéntica', () => {
    expect(Augmenter.augment([3, 4, 5], 0, 0)).toEqual([3, 4, 5])
  })

  it('valores por defecto (0.05, 0.1): conserva longitud y queda cerca del original', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8]
    const out = Augmenter.augment(x)
    expect(out).toHaveLength(8)
    // cada componente: o anulada (0) o x ± ruido de 0.05·z con |z| < 8.
    out.forEach((v, i) => expect(v === 0 || Math.abs(v - x[i]) < 0.4).toBe(true))
  })
})

describe('Augmenter.makePair', () => {
  it('la primera vista es el original y la segunda es otra vista distinta del mismo x', () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8]
    const [a, b] = Augmenter.makePair(x)
    expect(a).toBe(x)
    expect(b).not.toBe(x)
    expect(b).toHaveLength(x.length)
    expect(b).not.toEqual(x) // ruido gaussiano continuo: casi seguro distinto (con la semilla fija, determinista)
    expect(x).toEqual([1, 2, 3, 4, 5, 6, 7, 8]) // el original no se muta
  })

  it('la segunda vista es augment(x) con los valores por defecto (secuencia stub)', () => {
    // sigma 0.05 con z = -1 → x - 0.05; dropout 0.1 con randoms [0.5, 0.01] → conserva 1.º, anula 2.º.
    stubRandom([E_HALF, 0.5, E_HALF, 0.5, 0.5, 0.01])
    const [, b] = Augmenter.makePair([1, 2])
    expect(b[0]).toBeCloseTo(0.95, 12)
    expect(b[1]).toBe(0)
  })
})

describe('ContrastiveLearning.cosineSimilarity', () => {
  const cos = ContrastiveLearning.cosineSimilarity

  it('ortogonales → 0', () => {
    expect(cos([1, 0], [0, 1])).toBe(0)
  })

  it('[1,2] y [2,4] → 10/(√5·√20 + 1e-8) = 1 - 1e-9 (el epsilon blando se nota)', () => {
    const v = cos([1, 2], [2, 4])
    expect(v).toBeCloseTo(10 / (10 + 1e-8), 12)
    expect(v).toBeLessThan(1 - 5e-10) // sin el 1e-8 sería ~1 exacto
    expect(v).toBeGreaterThan(1 - 2e-9)
  })

  it('opuestos → -1/(1 + 1e-8)', () => {
    expect(cos([1, 0], [-1, 0])).toBeCloseTo(-1 / (1 + 1e-8), 12)
  })

  it('invariante a la escala: cos([3,4],[6,8]) = 50/(5·10 + 1e-8)', () => {
    // dot = 18 + 32 = 50, |a| = 5, |b| = 10.
    expect(cos([3, 4], [6, 8])).toBeCloseTo(50 / (50 + 1e-8), 12)
  })

  it('vector cero → 0 (no NaN)', () => {
    expect(cos([0, 0], [1, 2])).toBe(0)
    expect(cos([0, 0], [0, 0])).toBe(0)
  })

  it('entradas vacías → 0', () => {
    expect(cos([], [])).toBe(0)
  })
})

describe('ContrastiveLearning constructor y formas', () => {
  it('encoderHidden vacío lanza', () => {
    expect(() => new ContrastiveLearning(4, [], 3)).toThrow('encoderHidden must have at least one element.')
  })

  it('temperatura por defecto 0.5 y configurable', () => {
    expect(new ContrastiveLearning(4, [6], 3).temperature).toBe(0.5)
    expect(new ContrastiveLearning(4, [6], 3, { temperature: 0.2 }).temperature).toBe(0.2)
  })

  it('encode devuelve encoderHidden[last] valores y project devuelve projectionDim', () => {
    const c = new ContrastiveLearning(4, [6, 5], 3)
    const x = [0.1, 0.2, 0.3, 0.4]
    expect(c.encode(x)).toHaveLength(5)
    expect(c.project(x)).toHaveLength(3)
  })

  it('project = projectionHead.predict(encode(x))', () => {
    const c = new ContrastiveLearning(4, [6, 5], 3)
    const x = [0.1, 0.2, 0.3, 0.4]
    expect(c.project(x)).toEqual(c.projectionHead.predict(c.encode(x)))
  })

  it('capa oculta de la cabeza = max(projectionDim, floor(encoderOut/2)); salida de projectionDim', () => {
    // encoderOut 8, projectionDim 3 → max(3, 4) = 4 ; encoderOut 4, projectionDim 3 → max(3, 2) = 3.
    const a = new ContrastiveLearning(4, [8], 3)
    expect(a.projectionHead.layers[0].neurons).toHaveLength(4)
    expect(a.projectionHead.layers[1].neurons).toHaveLength(3)
    const b = new ContrastiveLearning(4, [4], 3)
    expect(b.projectionHead.layers[0].neurons).toHaveLength(3)
    expect(b.projectionHead.layers[1].neurons).toHaveLength(3)
  })
})

describe('ContrastiveLearning.computeLoss (proyección identidad)', () => {
  it('Caso A (τ=0.5): ln(1 + 2e^-2)', () => {
    const c = identityModel()
    const loss = c.computeLoss([
      [[1, 0], [1, 0]],
      [[0, 1], [0, 1]],
    ])
    expect(loss).toBeCloseTo(Math.log(1 + 2 * Math.exp(-2)), 6)
    expect(loss).toBeCloseTo(0.2397, 3) // ln(1.27067)
  })

  it('Caso A con τ=1: ln(1 + 2e^-1) (la temperatura cuenta)', () => {
    const c = identityModel(1)
    const loss = c.computeLoss([
      [[1, 0], [1, 0]],
      [[0, 1], [0, 1]],
    ])
    expect(loss).toBeCloseTo(Math.log(1 + 2 * Math.exp(-1)), 6)
    expect(loss).toBeCloseTo(0.5518, 3)
  })

  it('Caso B (τ=0.5): (t1 + t2 + 2·t3)/4 a mano', () => {
    const c = identityModel()
    const loss = c.computeLoss([
      [[1, 0], [1, 1]],
      [[0, 1], [0, 1]],
    ])
    const r2 = Math.SQRT2
    const t1 = Math.log(1 + 2 * Math.exp(-r2))
    const t2 = Math.log(3)
    const t3 = Math.log(1 + (1 + Math.exp(r2)) / Math.exp(2))
    expect(loss).toBeCloseTo((t1 + t2 + 2 * t3) / 4, 6)
  })

  it('colapso (todas las proyecciones iguales) → ln(2N-1) (HANDOFF N9)', () => {
    const c = identityModel()
    const same: [number[], number[]] = [[1, 2], [1, 2]]
    expect(c.computeLoss([same, same])).toBeCloseTo(Math.log(3), 6) // N=2
    expect(c.computeLoss([same, same, same])).toBeCloseTo(Math.log(5), 6) // N=3
  })

  it('un solo par ortogonal → ln(1) = 0 (el único candidato es el positivo)', () => {
    const c = identityModel()
    expect(c.computeLoss([[[1, 0], [0, 1]]])).toBeCloseTo(0, 6)
  })

  it('la pérdida es >= 0 con el modelo real y pares con semilla', () => {
    const c = new ContrastiveLearning(4, [6, 4], 3)
    const pairs: [number[], number[]][] = [
      [[0.9, 0.1, 0.2, 0.7], [0.85, 0.15, 0.25, 0.65]],
      [[0.1, 0.8, 0.6, 0.2], [0.15, 0.75, 0.55, 0.25]],
    ]
    const loss = c.computeLoss(pairs)
    expect(Number.isFinite(loss)).toBe(true)
    expect(loss).toBeGreaterThanOrEqual(0)
  })
})

describe('ContrastiveLearning.trainStep (comportamiento determinista)', () => {
  const PAIRS: [number[], number[]][] = [
    [[0.9, 0.1, 0.2, 0.7], [0.85, 0.15, 0.25, 0.65]],
    [[0.1, 0.8, 0.6, 0.2], [0.15, 0.75, 0.55, 0.25]],
  ]
  const params = (c: ContrastiveLearning): number[] => [
    ...c.encoder.getWeights(),
    ...c.projectionHead.getWeights(),
  ]

  it('devuelve la pérdida ANTES de actualizar', () => {
    const c = new ContrastiveLearning(4, [6, 4], 3)
    const before = c.computeLoss(PAIRS)
    expect(c.trainStep(PAIRS, 1e-3)).toBeCloseTo(before, 12)
  })

  it('lr = 0 no cambia ningún peso', () => {
    const c = new ContrastiveLearning(4, [6, 4], 3)
    const p0 = params(c)
    c.trainStep(PAIRS, 0)
    expect(params(c)).toEqual(p0)
  })

  it('trainStep([], lr) devuelve NaN (0/0 en la media de 0 anclas): comportamiento ACTUAL', () => {
    // Caso degenerado fijado tal cual hoy. La tarea T041 (validación) lo cambia (lanzar error); al
    // hacerlo, reescribir este test. Los pesos además quedan NaN tras el paso (gradientes NaN).
    const c = new ContrastiveLearning(4, [6, 4], 3)
    expect(c.computeLoss([])).toBeNaN()
    expect(c.trainStep([], 1e-3)).toBeNaN()
  })
})
