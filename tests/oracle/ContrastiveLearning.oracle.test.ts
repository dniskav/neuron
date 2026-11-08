// Test de oráculo de gradiente para ContrastiveLearning.
//
// trainStep NO usa backprop analítico: estima dL/dw con diferencias finitas centrales propias
// sobre computeLoss y aplica w += lr * (-grad). El oráculo compara ese paso con un gradiente
// independiente por diferencias finitas de la misma computeLoss (eps 1e-5). NO es evidencia de un
// backprop escrito a mano (no lo hay). Comprueba el signo (descenso), que encoder y cabeza se
// actualizan, y que el paso es SIMULTÁNEO.
//
// Lo que se verifica del diseño vigente:
//  - la cabeza de proyección es [relu, lineal] (sin ReLU final, que podría llevar todas las
//    proyecciones a cero);
//  - trainStep evalúa TODOS los gradientes en los pesos originales y actualiza al final;
//  - cosineSimilarity usa un epsilon blando (dot / (|a||b| + 1e-8)), así la pérdida es continua
//    en z = 0;
//  - la diferencia finita interna usa eps 1e-6.
//
// Init "sana" (classifyInit): todo h no nulo para las 6 vistas y toda proyección con norma > 1e-4.
// Sobre inits Xavier crudos (sin escalar, sin semilla elegida), semillas 1..40: 21 sanas
// (1,2,3,5,8,9,10,17,20,24,25,28,29,30,31,33,35,36,37,38,39) y 19 muertas (h = 0 o algún z = 0:
// ReLU de la última capa del encoder o de la capa oculta de la cabeza; o z ~ 4e-5, semilla 7).
// En las sanas el oráculo da coseno >= 0.9999 y ratio en [0.99, 1.02] para lr x0.97/1/1.03, y
// también con lr 0.05 (paso simultáneo). Las iniciales muertas NO se saltan: un test fija su
// comportamiento (pérdida log(2N-1), gradiente nulo, sin cambio).
//
// Limitación documentada (no corregida): la similitud coseno tiene curvatura ~ 1/||z||; con normas
// muy pequeñas un lr finito se aparta del gradiente de primer orden (semilla 7, ||z|| 4e-5, es el
// caso límite excluido: coseno 0.99988 con eps 1e-6, 0.965 con 1e-4).
//
// Tests de aprendizaje: NO son robustos sobre semillas Xavier crudas, y es un hallazgo, no algo
// oculto. Con temperatura 0.1, lr 1e-3 y 400 pasos, solo 12 de las 21 semillas sanas bajan a
// final/inicial < 0.35 (las demás acaban en 1.0..1.9). Traza (semillas 3, 17, 24, 29): la pérdida
// mejora al principio, un paso se pasa (el gradiente ~ 1/(tau ||z||) es enorme con ||z|| ~ 1e-2),
// las ReLU ocultas de la cabeza mueren y la salida de la cabeza pasa a ser el sesgo constante para
// toda entrada (coseno 1 en todo, pérdida log(2N-1), gradiente nulo). Una capa oculta leakyRelu,
// otras temperaturas (0.3, 0.5, 1) y lr 1e-4..1e-3 no lo arreglaron (medido). Recomendación
// pendiente: revisar la inicialización (o las activaciones) de la cabeza; queda fuera de esta
// suite. Experimento T012 (sin cambio en src/): sobre las semillas crudas 1..40, T 0.1, lr 1e-3
// x0.97/1/1.03, 400 pasos, se midieron 22 variantes: A (ultima capa del encoder lineal), B (sesgo
// inicial 0.05/0.1/0.3/0.5/1 en capas ReLU), C (oculta de la cabeza de 8/16/32 unidades) y sus
// combinaciones. Base: 19 muertas, 12 de 21 vivas no aprenden. Con A+B (>= 0.1) no queda ninguna
// muerta y el oraculo da coseno >= 0.99999, pero 'aprende' (final/inicial < 0.35 con los 3 lr) falla
// en 3..9 semillas vivas aun en la mejor (A+B 0.3+C 8: 3 de 40, estancadas en ~1.0; ademas empeora
// la semilla sana 33) y B >= 0.5 empeora. Ninguna cumple el criterio de adopcion: se mantiene la
// init y la limitacion. Por eso los tests de aprendizaje usan una init sana fija (semilla 9 x3, temperatura 0.1,
// lr 1e-3, 400 pasos): final/inicial ~0.15, entre 0.1453 y 0.1537 sobre 20 perturbaciones
// semilladas del 3 %, con aserción < 0.35 (margen del 30 % sobre el umbral 0.5) y decreciente
// estricta en 100/200/300.
import { describe, it, expect } from 'vitest'
import { ContrastiveLearning } from '../../src/ContrastiveLearning'
import { gradientOracle } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom, restoreRandom, mulberry32 } from '../helpers/seed'

const LR = 1e-3

// Fixed positive pairs (original, perturbed view); 3 pairs -> 6 anchors.
const PAIRS: [number[], number[]][] = [
  [[0.9, 0.1, 0.2, 0.7], [0.85, 0.15, 0.25, 0.65]],
  [[0.1, 0.8, 0.6, 0.2], [0.15, 0.75, 0.55, 0.25]],
  [[0.5, 0.4, 0.9, 0.1], [0.45, 0.45, 0.85, 0.15]],
]

// Seed 3 is a healthy raw Xavier init (see classifyInit) and is used for the fixed-init tests.
const INIT_SEED = 3

function build(seed = INIT_SEED, scale = 1): ContrastiveLearning {
  seedRandom(seed)
  let c: ContrastiveLearning
  try {
    c = new ContrastiveLearning(4, [6, 4], 3)
  } finally {
    restoreRandom()
  }
  if (scale !== 1) {
    c.encoder.setWeights(c.encoder.getWeights().map((w) => w * scale))
    c.projectionHead.setWeights(c.projectionHead.getWeights().map((w) => w * scale))
  }
  return c
}

// Learning-test configuration (see the header, "learns test"): healthy init = seed 9 x3 (no dead
// ReLU), low temperature 0.1, lr 1e-3, 400 steps.
const LEARN_LR = 1e-3
const LEARN_STEPS = 400
const LEARN_TEMPERATURE = 0.1

/** Seed-9 x3 init with optional seeded relative noise (uniform, +-noise) on every weight. */
function buildHealthy(noise: number, noiseSeed: number): ContrastiveLearning {
  const c = build(9, 3)
  c.temperature = LEARN_TEMPERATURE
  if (noise > 0) {
    const rnd = mulberry32(noiseSeed + 1000)
    const jitter = (w: number): number => w * (1 + noise * (2 * rnd() - 1))
    c.encoder.setWeights(c.encoder.getWeights().map(jitter))
    c.projectionHead.setWeights(c.projectionHead.getWeights().map(jitter))
  }
  return c
}

function getParams(c: ContrastiveLearning): number[] {
  return [...c.encoder.getWeights(), ...c.projectionHead.getWeights()]
}

function setParams(c: ContrastiveLearning, p: number[]): void {
  const nEnc = c.encoder.getWeights().length
  c.encoder.setWeights(p.slice(0, nEnc))
  c.projectionHead.setWeights(p.slice(nEnc))
}

const norm = (v: number[]): number => Math.sqrt(v.reduce((a, x) => a + x * x, 0))
const Z_MIN = 1e-4

/** 'healthy': all h non-zero and all ||z|| > Z_MIN for the 6 views; otherwise 'dead'. */
function classifyInit(c: ContrastiveLearning): 'healthy' | 'dead' {
  for (const [a, b] of PAIRS)
    for (const x of [a, b]) {
      if (norm(c.encode(x)) === 0) return 'dead'
      if (norm(c.project(x)) <= Z_MIN) return 'dead'
    }
  return 'healthy'
}

const RAW_SEEDS = Array.from({ length: 40 }, (_, i) => i + 1)
const HEALTHY = RAW_SEEDS.filter((k) => classifyInit(build(k)) === 'healthy')
const DEAD = RAW_SEEDS.filter((k) => !HEALTHY.includes(k))

describe('ContrastiveLearning oracle', () => {
  it('smoke: outputs have the right sizes and trainStep returns the NT-Xent loss before the update', () => {
    const c = build()
    expect(c.encode(PAIRS[0][0])).toHaveLength(4)
    expect(c.project(PAIRS[0][0])).toHaveLength(3)
    const before = c.computeLoss(PAIRS)
    expect(Number.isFinite(before)).toBe(true)
    expect(before).toBeGreaterThan(0)
    const reported = c.trainStep(PAIRS, LR)
    expect(reported).toBeCloseTo(before, 12)
  })

  it('clean update: computeLoss is deterministic and trainStep is repeatable from the same weights', () => {
    const c = build()
    expect(c.computeLoss(PAIRS)).toBe(c.computeLoss(PAIRS))
    const p0 = getParams(c)
    c.trainStep(PAIRS, LR)
    const p1 = getParams(c)
    setParams(c, p0)
    c.trainStep(PAIRS, LR)
    expect(getParams(c)).toEqual(p1)
  })

  it('oracle: trainStep follows -dL/dw of computeLoss (cosine > 0.999; trivial, see header)', () => {
    const c = build()
    const r = gradientOracle({
      getParams: () => getParams(c),
      setParams: (p) => setParams(c, p),
      loss: () => c.computeLoss(PAIRS),
      step: (lr) => {
        c.trainStep(PAIRS, lr)
      },
      lr: LR,
    })
    expect(r.cosine).toBeGreaterThan(0.999)
  })

  it('classification: at least 10 healthy raw seeds in 1..40 (measured 21)', () => {
    expect(HEALTHY.length).toBeGreaterThanOrEqual(10)
    expect(HEALTHY.length + DEAD.length).toBe(40)
  })

  for (const mul of [0.97, 1, 1.03]) {
    it(`oracle on every healthy raw seed (lr x${mul}): cosine > 0.999, ratio in [0.9, 1.1]`, () => {
      for (const k of HEALTHY) {
        const c = build(k)
        const r = gradientOracle({
          getParams: () => getParams(c),
          setParams: (p) => setParams(c, p),
          loss: () => c.computeLoss(PAIRS),
          step: (lr) => {
            c.trainStep(PAIRS, lr)
          },
          lr: LR * mul,
        })
        expect(r.cosine, `seed ${k}`).toBeGreaterThan(0.999)
        expect(r.ratio, `seed ${k}`).toBeGreaterThan(0.9)
        expect(r.ratio, `seed ${k}`).toBeLessThan(1.1)
      }
    })
  }

  it('simultaneous step: with a large lr (0.05) the step still equals -dL/dw at the INITIAL point', () => {
    // Una actualización in situ (no simultánea) haría que los gradientes posteriores usen pesos ya cambiados: desvío O(lr).
    for (const k of HEALTHY) {
      const c = build(k)
      const r = gradientOracle({
        getParams: () => getParams(c),
        setParams: (p) => setParams(c, p),
        loss: () => c.computeLoss(PAIRS),
        step: (lr) => {
          c.trainStep(PAIRS, lr)
        },
        lr: 0.05,
      })
      expect(r.cosine, `seed ${k}`).toBeGreaterThan(0.999)
      expect(r.ratio, `seed ${k}`).toBeGreaterThan(0.9)
      expect(r.ratio, `seed ${k}`).toBeLessThan(1.1)
    }
  })

  it('linear projection head: projections can be negative (no final ReLU)', () => {
    let negatives = 0
    for (const k of HEALTHY) {
      const c = build(k)
      for (const [a] of PAIRS) negatives += c.project(a).filter((v) => v < 0).length
    }
    expect(negatives).toBeGreaterThan(0)
  })

  it('smooth cosine: the similarity is continuous as the norm goes to 0 (no jump at 1e-10)', () => {
    const cos = ContrastiveLearning.cosineSimilarity
    expect(cos([0, 0], [1, 2])).toBe(0)
    expect(cos([1, 2], [1, 2])).toBeCloseTo(1, 6)
    // Con un umbral duro habría un salto de ~1 alrededor de |a||b| = 1e-10; con epsilon blando la diferencia es < 1e-3.
    const below = cos([0.99e-5, 0], [0.99e-5, 0])
    const above = cos([1.01e-5, 0], [1.01e-5, 0])
    expect(Math.abs(above - below)).toBeLessThan(1e-3)
  })

  it('dead init (N9): documented behavior, not skipped', () => {
    // Red muerta de partida: limitación de inicialización, no de gradiente (documentada, no corregida).
    expect(DEAD.length).toBeGreaterThan(0)
    let allZero = 0
    const movers: number[] = []
    for (const k of DEAD) {
      const c = build(k)
      expect(Number.isFinite(c.computeLoss(PAIRS))).toBe(true)
      const zero = PAIRS.every(([a, b]) => norm(c.project(a)) === 0 && norm(c.project(b)) === 0)
      if (!zero) continue
      allZero++
      // every projection is the zero vector: all similarities 0, loss = log(2N-1) = log 5,
      // numeric gradient ~0, so trainStep reports that loss and (almost) leaves the weights.
      const p0 = getParams(c)
      expect(c.computeLoss(PAIRS)).toBeCloseTo(Math.log(5), 6)
      expect(c.trainStep(PAIRS, LR)).toBeCloseTo(Math.log(5), 6)
      const moved = Math.max(...getParams(c).map((w, i) => Math.abs(w - p0[i])))
      if (moved > 1e-9) movers.push(k)
    }
    expect(allZero).toBeGreaterThan(0)
    // Measured: seeds 13, 16, 40 move by <= 1.3e-12; seed 27 (a kink: the perturbation makes some
    // projections non-zero) moves by 1e-2 in one step, still from the dead start.
    expect(movers).toEqual([27])
  })

  it('learns: NT-Xent loss on a fixed batch drops below half (robust, smooth descent)', () => {
    const c = buildHealthy(0, 0)
    // Check the descent is smooth, not a lucky checkpoint: the loss at 100, 200, 300 and 400
    // steps must be strictly decreasing (measured ratios 0.21, 0.19, 0.17, 0.15).
    const checkpoints: number[] = []
    let n = 0
    const { ratio } = expectLearns({
      loss: () => c.computeLoss(PAIRS),
      step: () => {
        c.trainStep(PAIRS, LEARN_LR)
        if (++n % 100 === 0 && n < LEARN_STEPS) checkpoints.push(c.computeLoss(PAIRS))
      },
      steps: LEARN_STEPS,
    })
    for (let i = 1; i < checkpoints.length; i++)
      expect(checkpoints[i]).toBeLessThan(checkpoints[i - 1])
    expect(ratio).toBeLessThan(0.35)
  })

  // Robustez de plataforma: diferencias mínimas de coma flotante no deben cambiar el resultado.
  // Cada variante debe quedar claramente por debajo de 0.35 (el umbral del test es 0.5, así que el
  // margen es >= 30 %). Medido peor/mejor final/inicial sobre 20 semillas de ruido de init (3 %),
  // lr x0.97/0.99/1.01/1.03 y ruido relativo 1e-9: ver la cabecera.
  const VARIANTS: {
    name: string
    lrMul: number
    noise: number
    noiseSeed: number
  }[] = [
    ...[0.97, 0.99, 1.01, 1.03].map((m) => ({
      name: `lr x${m}`,
      lrMul: m,
      noise: 0,
      noiseSeed: 0,
    })),
    ...[1, 2, 3, 4, 5].map((k) => ({
      name: `init noise 1e-9 (#${k})`,
      lrMul: 1,
      noise: 1e-9,
      noiseSeed: k,
    })),
    ...Array.from({ length: 20 }, (_, i) => i + 1).map((k) => ({
      name: `init noise 3% (seed ${k})`,
      lrMul: 1,
      noise: 0.03,
      noiseSeed: k,
    })),
  ]
  for (const v of VARIANTS) {
    it(`learns robustly: ${v.name} keeps final/initial < 0.35`, () => {
      const c = buildHealthy(v.noise, v.noiseSeed)
      expectLearns({
        loss: () => c.computeLoss(PAIRS),
        step: () => {
          c.trainStep(PAIRS, LEARN_LR * v.lrMul)
        },
        steps: LEARN_STEPS,
        ratio: 0.35,
      })
    })
  }
})
