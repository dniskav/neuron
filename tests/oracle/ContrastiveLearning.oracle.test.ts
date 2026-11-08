// Gradient-oracle test for ContrastiveLearning (features 002 T031, 004 T006 / N9).
//
// trainStep does NOT use analytic backprop: it estimates dL/dw with its own internal central
// finite differences over computeLoss and applies w += lr * (-grad). The oracle compares that
// step with an independent finite-difference gradient of the same computeLoss (eps 1e-5). It is
// NOT evidence of a hand-written backprop (there is none). It checks the sign convention
// (descent), that encoder and head are both updated, and that the step is SIMULTANEOUS.
//
// Fixes covered here (004 T006, Q7):
//  - the projection head is [relu, linear] (no final ReLU: it could map every projection to zero);
//  - trainStep evaluates ALL gradients at the original weights and updates at the end (before it
//    updated in place, so later gradients used already-updated weights);
//  - cosineSimilarity uses a soft epsilon (dot / (|a||b| + 1e-8)) instead of a jump to 0 below a
//    norm threshold, so the loss is continuous at z = 0;
//  - the internal finite-difference step is 1e-6 (was 1e-4, inexact when ||z|| is 5e-3..6e-2).
//
// "Healthy" init (classifyInit): every encoder output h is non-zero for all 6 views and every
// projection has norm > 1e-4. Measured on raw Xavier inits (no scaling, no chosen seed), seeds
// 1..40: 21 healthy (1,2,3,5,8,9,10,17,20,24,25,28,29,30,31,33,35,36,37,38,39); the other 19
// start dead (h = 0 or some z = 0: ReLU of the encoder's last layer or of the head's hidden
// layer; or z ~ 4e-5, seed 7). On the healthy ones the oracle gives cosine >= 0.9999 and
// ratio in [0.99, 1.02] for lr x0.97/1/1.03, and also at lr 0.05 (simultaneous step).
// Dead inits are NOT skipped: a test pins their behavior (loss log(2N-1), zero gradient, no change).
//
// Limitation (c), documented not fixed: cosine similarity has curvature ~ 1/||z||; with very
// small norms a finite lr departs from the first-order gradient (seed 7, ||z|| 4e-5, is the
// excluded borderline case: cosine 0.99988 at eps 1e-6, 0.965 at 1e-4).
//
// Learns tests: NOT robust over raw Xavier seeds, and this is a finding, not hidden. With
// temperature 0.1, lr 1e-3, 400 steps, only 12 of the 21 healthy seeds reach final/initial
// < 0.35 (others end 1.0..1.9). Trace (seeds 3, 17, 24, 29): the loss first improves, a step
// overshoots (gradient ~ 1/(tau ||z||) is huge at ||z|| ~ 1e-2), the head hidden ReLUs die, the
// head output becomes the constant bias for every input (all cosines 1, loss log(2N-1), zero
// gradient). A leakyRelu head hidden layer, other temperatures (0.3, 0.5, 1) and lr 1e-4..1e-3
// did not fix it (measured). The cure belongs to init/activations (see the task report), which
// is outside T006. So the learns tests keep the fixed healthy init (seed 9 x3, temperature 0.1,
// lr 1e-3, 400 steps): final/initial ~0.15, 0.1453..0.1537 over 20 seeded 3 % perturbations,
// asserted < 0.35 (30 % margin on the 0.5 threshold) and strictly decreasing at 100/200/300.
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
    // An in-place update makes later gradients use already-updated weights: O(lr) departure.
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
    // The old guard returned 0 below |a||b| = 1e-10 and ~1 just above: a jump of ~1.
    const below = cos([0.99e-5, 0], [0.99e-5, 0])
    const above = cos([1.01e-5, 0], [1.01e-5, 0])
    expect(Math.abs(above - below)).toBeLessThan(1e-3)
  })

  it('dead init (N9): documented behavior, not skipped', () => {
    // N9: red muerta de partida; limitación de inicialización, no de gradiente
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

  // Platform robustness (CI ubuntu vs local macOS failed on the old test): tiny floating-point
  // differences must not change the outcome. Each variant must stay clearly under 0.35 (the
  // test threshold is 0.5, so the margin is >= 30 %). Measured worst/best final/initial over
  // 20 init noise seeds (3 %), lr x0.97/0.99/1.01/1.03 and 1e-9 relative noise: see the header.
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
