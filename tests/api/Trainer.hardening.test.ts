// T011 / I3 (Q1, Q2, Q3): Trainer no ignora opciones en silencio.
//
//  - `clipValue` ya no existe (Q1).
//  - Opciones sin efecto, dataset vacio y parametros invalidos LANZAN con el
//    nombre de la opcion, antes de llamar a network.train (Q2).
//  - `restoreBestWeights` (defecto false) restaura los pesos del mejor epoch (Q3).
//  - Con opciones validas el historial es IDENTICO al de la base (golden
//    capturado con el src anterior a T011, Math.random sembrado, 1e-12).
//
// Aprende: Network(2,4,2), 2 salidas, 5 semillas, lr x0.7/x1.3 (cifras abajo).

import { describe, it, expect, vi, afterEach } from 'vitest'
import { Trainer } from '../../src/Trainer'
import type { TrainableNetwork } from '../../src/Trainer'
import { Network } from '../../src/Network'
import { NetworkN } from '../../src/NetworkN'
import { validateArray } from '../../src/Validation'

function seed(a: number): () => number {
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
function useSeed(s: number) {
  vi.spyOn(Math, 'random').mockImplementation(seed(s))
}
afterEach(() => {
  vi.restoreAllMocks()
})

const xor = {
  inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
  targets: [[1, 0], [0, 1], [0, 1], [1, 0]],
}
const valSet = {
  inputs: [[0, 0.2], [0.9, 1], [1, 0.1]],
  targets: [[1, 0], [0, 1], [0, 1]],
}

/** Red completa (train/predict/getWeights/setWeights) con espia en train. */
function makeStub() {
  let w = [0.1, 0.2]
  const trainSpy = vi.fn((_i: number[], _t: number[], _lr: number) => 0.5)
  const net = {
    train: trainSpy,
    predict: (_i: number[]) => [0.5],
    getWeights: () => [...w],
    setWeights: (x: number[]) => {
      w = [...x]
    },
  }
  return { net, trainSpy }
}
const tiny = { inputs: [[0], [1]], targets: [[0], [1]] }

// ── Golden: historial intacto con opciones validas ────────────────────────────

const GOLDEN: Record<string, any> = {
  plain: {
    history: [0.261238190908316, 0.26058848313329186, 0.26064256732455227, 0.26010797576048966, 0.2603639123717286, 0.2601858637902328, 0.26009849624157677, 0.25998834370304136, 0.25996559594410246, 0.25948870694701454, 0.25983357300349424, 0.2598101658486984, 0.2597808067574966, 0.2597628680575087, 0.2593904869035907],
    stop: 'maxEpochs', best: -1, metricsLen: 0,
    weights: [0.13288108636907833, -0.0806499277445314, -0.016117510774013487, 0.48968800186616274, 0.23235385439453266, -0.013153361061048443, -0.4632838182994987, 0.03399878415149978, -0.006668968102584311, -0.3303030504425592, 0.1669794785603402, -0.017991016883241378, 0.31422084361073466, -0.08976316065674328, -0.29512916654691657, 0.33213096288611166, -0.10034260860521488, 0.27120567676649593, -0.1611195229075555, -0.2794205539196393, 0.027217449450620136, 0.05000514489952677],
  },
  decay: {
    history: [0.261238190908316, 0.2601506089113721, 0.2597093201152124, 0.2588296689242498, 0.2585637462277189, 0.25801294155606325, 0.25754897172836955, 0.2570974138204055, 0.25671788689753766, 0.25617678939483335, 0.2559789196315012, 0.255659219456618, 0.2553528766623395, 0.2550727805054968, 0.25472972580262454],
    stop: 'maxEpochs', best: -1, metricsLen: 0,
    weights: [0.13616565228009944, -0.07824001279712209, -0.010607160803414803, 0.49308682132877224, 0.23522826159557078, -0.007504691769785602, -0.4613919547961547, 0.03601051582790269, -0.0025167219907794073, -0.3277219099795164, 0.16958706615611935, -0.012931550288438043, 0.31749478905957434, -0.0851698826680484, -0.29239839118807515, 0.33587569061212813, -0.09412378943666072, 0.2706070062414467, -0.16253987342509257, -0.2800779751372773, 0.025721399120869827, 0.04894691288824968],
  },
  wd: {
    history: [0.2609856885740264, 0.2603192044093615, 0.26053794840880873, 0.2600021674549644, 0.26026156566306125, 0.2600717685490844, 0.2599844654647717, 0.2598820218450668, 0.2598776723638099, 0.25930370230040367, 0.25974576527475096, 0.25970951322183444, 0.2596874844764041, 0.2596784824437364, 0.2592534732147137],
    stop: 'maxEpochs', best: -1, metricsLen: 0,
    weights: [0.11067969667736453, -0.06766302336419368, -0.014078636383086263, 0.4084453804458707, 0.19375810193437912, -0.011823202837503315, -0.3874710502093726, 0.027931449425060657, -0.006593594196852153, -0.27618844531872055, 0.1392468261401562, -0.015633825559242844, 0.26292781555687333, -0.07365605089688981, -0.2461745666412915, 0.2777766661547649, -0.08277508234888763, 0.22604100045945597, -0.1354948888380132, -0.23377824086690663, 0.022201399030695113, 0.040855096330396475],
  },
  es: {
    history: [0.261238190908316, 0.26058848313329186, 0.26064256732455227, 0.26010797576048966, 0.2603639123717286, 0.2601858637902328, 0.26009849624157677, 0.25998834370304136, 0.25996559594410246, 0.25948870694701454, 0.25983357300349424, 0.2598101658486984, 0.2597808067574966, 0.2597628680575087, 0.2593904869035907, 0.2597877113279942, 0.25977215930355735, 0.2597551762960617],
    stop: 'earlyStopping', best: 0.2506953985583029, metricsLen: 0,
    weights: [0.13154608088446612, -0.0810248433442195, -0.01846009454338919, 0.48827343370144893, 0.23161279321898148, -0.015754279761817216, -0.46391068645725037, 0.033758791267127886, -0.008437307771129464, -0.3314341591084265, 0.16641932727140876, -0.020164303761341523, 0.3122489018084213, -0.09284941478929223, -0.2964462445436927, 0.33023896947493453, -0.10399700767937774, 0.2720464907213237, -0.15939469114388588, -0.27888494280351667, 0.028247464749771016, 0.05153240349763213],
  },
  all: {
    history: [0.2612128260628811, 0.26047397656646365, 0.26044107299372776, 0.2598317492619546, 0.2599692378770348, 0.259700985714453, 0.25951998540152144, 0.2593218323242421, 0.2592050465804944, 0.2586994343848119, 0.25889794881454087, 0.2587830384571752, 0.25866583321546166, 0.2585635885540394, 0.2581934129765103, 0.25840682999872566, 0.25830975166507275, 0.25821250120814554, 0.2581153555930008, 0.25776373646922557, 0.25794768860059347, 0.2576008841302998, 0.25751698020822356, 0.2574370721908825, 0.2575896513955951, 0.25751615461901, 0.25744169904779335, 0.25713478714661114, 0.25706366676937803, 0.25721178292224667, 0.25713815567020515, 0.2568618669365512],
    stop: 'earlyStopping', best: 0.2488641557177682, metricsLen: 32,
    accuracy: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.25, 0.25, 0.25, 0.75, 0.5, 0.75, 0.5, 0.75],
    weights: [0.12495985164989375, -0.07997226343242281, -0.022379107518415514, 0.470014031214236, 0.22141419754879763, -0.020436365969073694, -0.44980499696782733, 0.0313371168058871, -0.011718804735544793, -0.3222182570386463, 0.1590113540483557, -0.02356142313080521, 0.29735573597710246, -0.0970531260890588, -0.2907555198821037, 0.3145061431849101, -0.10983751164161573, 0.26642388166948683, -0.1488870930888025, -0.2666658112918896, 0.03145731445005627, 0.05635007796251857],
  },
  metrics: {
    history: [0.261238190908316, 0.26058848313329186, 0.26064256732455227, 0.26010797576048966, 0.2603639123717286, 0.2601858637902328, 0.26009849624157677, 0.25998834370304136, 0.25996559594410246, 0.25948870694701454],
    stop: 'maxEpochs', best: -1, metricsLen: 10,
    accuracy: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    weights: [0.13516156689744377, -0.078348142998625, -0.011932487037467843, 0.49183841815087914, 0.23502462469416377, -0.00915254848309807, -0.4623055052829999, 0.03580992397520051, -0.003949141344919042, -0.3288475185269418, 0.16943279073778794, -0.014455457913525881, 0.3172342267477006, -0.08542648722232987, -0.29230867135414, 0.33570328272890015, -0.09441626900749889, 0.2716721599428826, -0.16133495734196884, -0.27952770180104164, 0.026585053284506013, 0.050781555449674325],
  },
}
const CFGS: Record<string, any> = {
  plain: { epochs: 15, lr: 0.3 },
  decay: { epochs: 15, lr: 0.3, lrDecay: 0.95 },
  wd: { epochs: 15, lr: 0.3, weightDecay: 0.01 },
  es: { epochs: 300, lr: 0.3, earlyStopping: { patience: 5, minDelta: 0.001 }, val: true },
  all: { epochs: 40, lr: 0.3, lrDecay: 0.99, weightDecay: 0.001, earlyStopping: { patience: 8, minDelta: 0.0001 }, computeMetrics: true, val: true },
  metrics: { epochs: 10, lr: 0.3, computeMetrics: true },
}

function closeArr(a: number[], b: number[]) {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(1e-12)
}

describe('Trainer golden: opciones validas dan el mismo historial que la base', () => {
  for (const kind of ['Network', 'NetworkN']) {
    for (const key of Object.keys(CFGS)) {
      it(`${kind} / ${key}`, () => {
        useSeed(42)
        const net: any = kind === 'Network' ? new Network(2, 4, 2) : new NetworkN([2, 4, 2])
        const { val, ...opts } = CFGS[key]
        const t = new Trainer(net, opts)
        if (val) t.setValidationData(valSet)
        const h = t.train(xor)
        const g = GOLDEN[key]
        closeArr(h, g.history)
        expect(t.getStopReason()).toBe(g.stop)
        expect(t.getBestLoss()).toBeCloseTo(g.best, 12)
        expect(t.getMetrics().length).toBe(g.metricsLen)
        if (g.accuracy) closeArr(t.getMetrics().map((m) => m.accuracy), g.accuracy)
        closeArr(net.getWeights(), g.weights)
      })
    }
  }
})

// ── Q1: clipValue eliminado ───────────────────────────────────────────────────

describe('Trainer: clipValue eliminado (Q1)', () => {
  it('el Trainer no tiene la propiedad clipValue', () => {
    const t = new Trainer(new NetworkN([2, 4, 1]), {})
    expect('clipValue' in t).toBe(false)
  })
  it('clipValue ya no es una opcion del tipo', () => {
    // @ts-expect-error clipValue fue eliminado de TrainerOptions
    const t = new Trainer(new NetworkN([2, 4, 1]), { clipValue: 1 })
    expect('clipValue' in t).toBe(false)
  })
})

// ── Q2: errores con nombre de opcion y sin tocar la red ───────────────────────

describe('Trainer.train lanza ante opciones sin efecto o invalidas (Q2)', () => {
  it('earlyStopping sin setValidationData', () => {
    const { net, trainSpy } = makeStub()
    const t = new Trainer(net, { epochs: 3, earlyStopping: { patience: 2, minDelta: 0 } })
    expect(() => t.train(tiny)).toThrow(/earlyStopping.*setValidationData/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('earlyStopping con red sin getWeights/setWeights/predict', () => {
    const trainSpy = vi.fn(() => 0.5)
    const net: TrainableNetwork = { train: trainSpy }
    const t = new Trainer(net, { epochs: 3, earlyStopping: { patience: 2, minDelta: 0 } })
    t.setValidationData(tiny)
    expect(() => t.train(tiny)).toThrow(/earlyStopping.*getWeights\/setWeights\/predict/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('weightDecay con red sin getWeights/setWeights/predict', () => {
    const trainSpy = vi.fn(() => 0.5)
    const t = new Trainer({ train: trainSpy }, { epochs: 3, weightDecay: 0.01 })
    expect(() => t.train(tiny)).toThrow(/weightDecay.*getWeights\/setWeights\/predict/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('computeMetrics con red sin predict', () => {
    const trainSpy = vi.fn(() => 0.5)
    const t = new Trainer({ train: trainSpy }, { epochs: 3, computeMetrics: true })
    expect(() => t.train(tiny)).toThrow(/computeMetrics.*predict/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('dataset vacio', () => {
    const { net, trainSpy } = makeStub()
    const t = new Trainer(net, { epochs: 3 })
    expect(() => t.train({ inputs: [], targets: [] })).toThrow(/dataset.*empty/i)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('restoreBestWeights sin earlyStopping', () => {
    const { net, trainSpy } = makeStub()
    const t = new Trainer(net, { epochs: 3, restoreBestWeights: true })
    expect(() => t.train(tiny)).toThrow(/restoreBestWeights.*earlyStopping/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('restoreBestWeights con red sin getWeights/setWeights', () => {
    const trainSpy = vi.fn(() => 0.5)
    const t = new Trainer(
      { train: trainSpy },
      { epochs: 3, restoreBestWeights: true, earlyStopping: { patience: 2, minDelta: 0 } }
    )
    t.setValidationData(tiny)
    expect(() => t.train(tiny)).toThrow(/earlyStopping|restoreBestWeights/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  const bad: Array<[string, any, RegExp]> = [
    ['epochs 0', { epochs: 0 }, /epochs/],
    ['epochs negativo', { epochs: -3 }, /epochs/],
    ['epochs no entero', { epochs: 2.5 }, /epochs/],
    ['epochs NaN', { epochs: NaN }, /epochs/],
    // epochs: Infinity se prueba aparte con una red centinela (ver abajo): con
    // makeStub, quitar la validacion haria un bucle sin fin.
    ['lr 0', { lr: 0 }, /\blr\b/],
    ['lr negativo', { lr: -0.1 }, /\blr\b/],
    ['lr NaN', { lr: NaN }, /\blr\b/],
    ['lr Infinity', { lr: Infinity }, /\blr\b/],
    ['lrDecay 0', { lrDecay: 0 }, /lrDecay/],
    ['lrDecay > 1', { lrDecay: 1.01 }, /lrDecay/],
    ['lrDecay negativo', { lrDecay: -0.5 }, /lrDecay/],
    ['lrDecay NaN', { lrDecay: NaN }, /lrDecay/],
    ['weightDecay negativo', { weightDecay: -0.01 }, /weightDecay/],
    ['weightDecay NaN', { weightDecay: NaN }, /weightDecay/],
    ['weightDecay Infinity', { weightDecay: Infinity }, /weightDecay/],
    ['patience 0', { earlyStopping: { patience: 0, minDelta: 0 } }, /patience/],
    ['patience no entero', { earlyStopping: { patience: 1.5, minDelta: 0 } }, /patience/],
    ['patience NaN', { earlyStopping: { patience: NaN, minDelta: 0 } }, /patience/],
    ['minDelta NaN', { earlyStopping: { patience: 2, minDelta: NaN } }, /minDelta/],
    ['minDelta negativo', { earlyStopping: { patience: 2, minDelta: -0.1 } }, /minDelta/],
    ['minDelta Infinity', { earlyStopping: { patience: 2, minDelta: Infinity } }, /minDelta/],
  ]
  for (const [name, opts, re] of bad) {
    it(`parametro invalido: ${name}`, () => {
      const { net, trainSpy } = makeStub()
      const t = new Trainer(net, opts)
      t.setValidationData(tiny)
      expect(() => t.train(tiny)).toThrow(re)
      expect(trainSpy).not.toHaveBeenCalled()
    })
  }

  it('epochs Infinity lanza antes de entrenar, sin bucle infinito', () => {
    // La red centinela lanza en su primera llamada a train. Si faltara la
    // validacion de epochs, el bucle `epoch < Infinity` llamaria a train, el
    // test fallaria con el mensaje CENTINELA (no con 'Worker exited').
    const trainSpy = vi.fn((): number => {
      throw new Error('CENTINELA: network.train fue llamado')
    })
    const t = new Trainer({ train: trainSpy }, { epochs: Infinity })
    expect(() => t.train(tiny)).toThrow(/epochs/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('earlyStopping con validacion vacia lanza y no llama a network.train', () => {
    const { net, trainSpy } = makeStub()
    const t = new Trainer(net, { epochs: 3, earlyStopping: { patience: 2, minDelta: 0 } })
    t.setValidationData({ inputs: [], targets: [] })
    expect(() => t.train(tiny)).toThrow(/validation data is empty/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('sin earlyStopping, la validacion vacia no cambia el comportamiento', () => {
    const { net, trainSpy } = makeStub()
    const t = new Trainer(net, { epochs: 2 })
    expect(() => t.setValidationData({ inputs: [], targets: [] })).not.toThrow()
    expect(() => t.train(tiny)).not.toThrow()
    expect(trainSpy).toHaveBeenCalledTimes(4) // 2 epocas x 2 muestras
  })

  it('limites validos no lanzan: lrDecay 1, weightDecay 0, patience 1, epochs 1', () => {
    const { net } = makeStub()
    const t = new Trainer(net, {
      epochs: 1, lr: 1e-9, lrDecay: 1, weightDecay: 0,
      earlyStopping: { patience: 1, minDelta: 0 },
    })
    t.setValidationData(tiny)
    expect(() => t.train(tiny)).not.toThrow()
  })

  it('el dataset desigual sigue lanzando', () => {
    const { net, trainSpy } = makeStub()
    const t = new Trainer(net, { epochs: 2 })
    expect(() => t.train({ inputs: [[0]], targets: [[0], [1]] })).toThrow(/same length/)
    expect(trainSpy).not.toHaveBeenCalled()
  })

  it('validar no consume Math.random ni toca los pesos', () => {
    const rnd = vi.spyOn(Math, 'random')
    const net = new Network(2, 3, 2)
    const before = net.getWeights()
    rnd.mockClear()
    const t = new Trainer(net, { epochs: 3, lr: -1 })
    expect(() => t.train(xor)).toThrow(/lr/)
    expect(rnd).not.toHaveBeenCalled()
    expect(net.getWeights()).toEqual(before)
  })

  it('computeMetrics con objetivos no clasificatorios sigue siendo console.warn', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const net2 = new Network(2, 3, 2)
    const t2 = new Trainer(net2, { epochs: 2, computeMetrics: true })
    expect(() =>
      t2.train({ inputs: [[0, 0], [1, 1]], targets: [[0.3, 0.9], [2, 5]] })
    ).not.toThrow()
    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0][0])).toMatch(/computeMetrics/)
    expect(t2.getMetrics()).toEqual([])
  })
})

// ── Q3: restoreBestWeights ────────────────────────────────────────────────────

/** Envuelve Network y registra, al final de cada epoch, pesos y perdida de validacion. */
function recordRun(opts: { restore?: boolean; patience: number; epochs: number; s: number }) {
  useSeed(opts.s)
  const net = new Network(2, 4, 2)
  const nTrain = trainSet.inputs.length
  const weightsPerEpoch: number[][] = []
  const valLossPerEpoch: number[] = []
  let calls = 0
  const wrapped = {
    train: (i: number[], t: number[], lr: number) => {
      const r = net.train(i, t, lr)
      calls++
      if (calls % nTrain === 0) {
        weightsPerEpoch.push(net.getWeights())
        let tot = 0
        for (let k = 0; k < valData.inputs.length; k++) {
          const p = net.predict(valData.inputs[k])
          let s = 0
          for (let j = 0; j < p.length; j++) s += (valData.targets[k][j] - p[j]) ** 2
          tot += s / p.length
        }
        valLossPerEpoch.push(tot / valData.inputs.length)
      }
      return r
    },
    predict: (i: number[]) => net.predict(i),
    getWeights: () => net.getWeights(),
    setWeights: (w: number[]) => net.setWeights(w),
  }
  const t = new Trainer(wrapped, {
    epochs: opts.epochs,
    lr: 2.0,
    earlyStopping: { patience: opts.patience, minDelta: 0 },
    restoreBestWeights: opts.restore,
  })
  t.setValidationData(valData)
  t.train(trainSet)
  let best = 0
  for (let e = 1; e < valLossPerEpoch.length; e++) if (valLossPerEpoch[e] < valLossPerEpoch[best]) best = e
  return { net, t, weightsPerEpoch, valLossPerEpoch, best }
}
// Entrena con un conjunto y valida con otro cuya etiqueta contradice parte del primero:
// la validacion mejora al principio y luego empeora.
const trainSet = {
  inputs: [[0, 0], [1, 1], [0, 1], [1, 0]],
  targets: [[1, 0], [0, 1], [0, 1], [1, 0]],
}
const valData = {
  inputs: [[0.1, 0.1], [0.9, 0.9], [0.5, 0.5], [0.2, 0.8]],
  targets: [[1, 0], [0, 1], [1, 0], [1, 0]],
}

describe('Trainer.restoreBestWeights (Q3)', () => {
  it('true: termina por earlyStopping con los pesos del mejor epoch', () => {
    const r = recordRun({ restore: true, patience: 6, epochs: 400, s: 7 })
    expect(r.t.getStopReason()).toBe('earlyStopping')
    const last = r.weightsPerEpoch.length - 1
    expect(r.best).toBeLessThan(last) // el escenario realmente empeora despues
    expect(r.best).toBeGreaterThanOrEqual(1) // y no es trivialmente el primer epoch
    expect(r.net.getWeights()).toEqual(r.weightsPerEpoch[r.best])
    expect(r.net.getWeights()).not.toEqual(r.weightsPerEpoch[last])
    expect(r.t.getBestLoss()).toBeCloseTo(r.valLossPerEpoch[r.best], 12)
  })

  it('false (defecto): deja los pesos del ultimo epoch', () => {
    const r = recordRun({ patience: 6, epochs: 400, s: 7 })
    const last = r.weightsPerEpoch.length - 1
    expect(r.best).toBeLessThan(last)
    expect(r.net.getWeights()).toEqual(r.weightsPerEpoch[last])
    const r2 = recordRun({ restore: false, patience: 6, epochs: 400, s: 7 })
    expect(r2.net.getWeights()).toEqual(r2.weightsPerEpoch[r2.weightsPerEpoch.length - 1])
  })

  it('true: tambien restaura cuando termina por maxEpochs', () => {
    const r = recordRun({ restore: true, patience: 100000, epochs: 40, s: 7 })
    expect(r.t.getStopReason()).toBe('maxEpochs')
    expect(r.weightsPerEpoch.length).toBe(40)
    expect(r.best).toBeLessThan(39)
    expect(r.net.getWeights()).toEqual(r.weightsPerEpoch[r.best])
  })

  it('el historial no cambia por restaurar (misma longitud y valores)', () => {
    const a = recordRun({ restore: true, patience: 6, epochs: 400, s: 7 })
    const b = recordRun({ restore: false, patience: 6, epochs: 400, s: 7 })
    expect(a.t.getHistory()).toEqual(b.t.getHistory())
    expect(a.t.getBestLoss()).toBe(b.t.getBestLoss())
  })

  it('con minDelta alto solo cuenta como mejor lo que mejora por mas de minDelta', () => {
    // best se actualiza solo con valLoss < best - minDelta: con un minDelta enorme
    // solo el primer epoch es "mejor" y se restauran sus pesos.
    useSeed(5)
    const net = new Network(2, 4, 2)
    const first: number[][] = []
    let calls = 0
    const wrapped = {
      train: (i: number[], t: number[], lr: number) => {
        const r = net.train(i, t, lr)
        calls++
        if (calls === trainSet.inputs.length) first.push(net.getWeights())
        return r
      },
      predict: (i: number[]) => net.predict(i),
      getWeights: () => net.getWeights(),
      setWeights: (w: number[]) => net.setWeights(w),
    }
    const t = new Trainer(wrapped, {
      epochs: 10, lr: 0.5,
      earlyStopping: { patience: 100, minDelta: 100 },
      restoreBestWeights: true,
    })
    t.setValidationData(valData)
    t.train(trainSet)
    expect(net.getWeights()).toEqual(first[0])
  })

  it('un snapshot con valores no finitos NO se restaura: setWeights lanza y el error se propaga', () => {
    // Borde documentado: la restauracion pasa por setWeights, que valida (T001-T010).
    let w = [1, 2]
    let n = 0
    const net = {
      train: (_i: number[], _t: number[], _lr: number) => {
        n++
        w = n === 1 ? [NaN, 2] : [9, 9]
        return 0.1
      },
      predict: (_i: number[]) => [0.5],
      getWeights: () => [...w],
      setWeights: (x: number[]) => {
        validateArray(x, 2, 'Stub.setWeights')
        w = [...x]
      },
    }
    const t = new Trainer(net, {
      epochs: 3,
      earlyStopping: { patience: 10, minDelta: 0 },
      restoreBestWeights: true,
    })
    t.setValidationData({ inputs: [[0]], targets: [[0.5]] })
    expect(() => t.train({ inputs: [[0]], targets: [[0]] })).toThrow(/Stub\.setWeights/)
    expect(w).toEqual([9, 9]) // no se escribio nada
  })

  it('un modelo que divergio a NaN despues del mejor epoch si se restaura', () => {
    let w = [1, 2]
    let n = 0
    const net = {
      train: (_i: number[], _t: number[], _lr: number) => {
        n++
        w = n === 1 ? [1, 2] : [NaN, NaN]
        return 0.1
      },
      predict: (_i: number[]) => (n === 1 ? [0.5] : [NaN]),
      getWeights: () => [...w],
      setWeights: (x: number[]) => {
        validateArray(x, 2, 'Stub.setWeights')
        w = [...x]
      },
    }
    const t = new Trainer(net, {
      epochs: 4,
      earlyStopping: { patience: 10, minDelta: 0 },
      restoreBestWeights: true,
    })
    t.setValidationData({ inputs: [[0]], targets: [[0.5]] })
    t.train({ inputs: [[0]], targets: [[0]] })
    expect(w).toEqual([1, 2])
  })
})

// ── Aprende: Trainer + Network(2,4,2) con 2 salidas ───────────────────────────
// Problema: clase 0 si x0 > x1, si no clase 1 (one-hot), 20 puntos de una malla.
// Medido (lr base 0.5 x0.7/x1/x1.3, semillas 1..5, 300 epochs): peor razon
// final/inicial de la perdida = 0.0055, peor accuracy final = 1.0. Umbrales:
// razon < 0.7 y accuracy >= 0.75.

describe('Trainer aprende con Network(2,4,2) (T007 integrada)', () => {
  const pts: number[][] = []
  for (let a = 0; a <= 4; a++) for (let b = 0; b <= 4; b++) if (a !== b) pts.push([a / 4, b / 4])
  const data = {
    inputs: pts,
    targets: pts.map((p) => (p[0] > p[1] ? [1, 0] : [0, 1])),
  }
  for (const mult of [0.7, 1, 1.3]) {
    for (const s of [1, 2, 3, 4, 5]) {
      it(`lr x${mult}, semilla ${s}`, () => {
        useSeed(s)
        const net = new Network(2, 4, 2)
        const t = new Trainer(net, { epochs: 300, lr: 0.5 * mult, computeMetrics: true })
        const h = t.train(data)
        const ratio = h[h.length - 1] / h[0]
        const m = t.getMetrics()
        const acc = m[m.length - 1].accuracy
        expect(ratio).toBeLessThan(0.7)
        expect(acc).toBeGreaterThanOrEqual(0.75)
      })
    }
  }
})
