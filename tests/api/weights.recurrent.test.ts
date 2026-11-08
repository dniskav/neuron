// T004 (I2 + I6, Q12): LSTMLayer, GRULayer y NetworkLSTM.
//
// - getWeights() devuelve COPIAS profundas (matrices y sesgos).
// - setWeights(data) valida la forma ANTES de escribir y COPIA elemento a
//   elemento en los arrays existentes (sin reasignar referencias, sin alias
//   entre modelos).
// - setWeightsFlat valida la longitud total antes de escribir.
//
// Los valores válidos no cambian: el golden de getWeightsFlat con semilla fija
// es el mismo que en la base (capturado antes de tocar src/).

import { describe, it, expect } from 'vitest'
import { LSTMLayer } from '../../src/LSTMLayer'
import { GRULayer } from '../../src/GRU'
import { NetworkLSTM } from '../../src/NetworkLSTM'
import { seedRandom } from '../helpers/seed'

const H = 2 // hiddenSize
const IN = 2 // inputSize
const COMB = IN + H // columnas de cada W de puerta

// Pesos planos de la base con semilla fija (antes de T004).
const GOLD_LSTM: number[] = [0.2425241061934615, -0.24671718699653025, 0.5218122061699301, -0.45984821691067784, -0.14039550148338517, 0.17794433923475306, -0.21456843225290098, -0.22184758087922987, 1, 1, 0.07012078956642942, -0.15287462481333983, 0.5734771893942693, -0.06589724755809809, 0.19907699059679654, -0.10102669557239498, -0.21945223893855365, 0.05655762269703621, 0, 0, 0.39172362242953424, 0.3174042960248416, -0.36168475526371036, -0.30425084502904925, -0.17981409216963884, -0.1989414820289893, -0.5170233007291418, -0.2406247867850135, 0, 0, -0.510084143158857, -0.15453784770090523, 0.022384982696402888, -0.2813588427735337, 0.5104565418917715, -0.5620236646429332, 0.25726668672098146, -0.467946842450953, 0, 0]
const GOLD_GRU: number[] = [0.2425241061934615, -0.24671718699653025, 0.5218122061699301, -0.45984821691067784, -0.14039550148338517, 0.17794433923475306, -0.21456843225290098, -0.22184758087922987, 0, 0, 0.07012078956642942, -0.15287462481333983, 0.5734771893942693, -0.06589724755809809, 0.19907699059679654, -0.10102669557239498, -0.21945223893855365, 0.05655762269703621, 0, 0, 0.39172362242953424, 0.3174042960248416, -0.36168475526371036, -0.30425084502904925, -0.17981409216963884, -0.1989414820289893, -0.5170233007291418, -0.2406247867850135, 0, 0]
const GOLD_NET: number[] = [0.2425241061934615, -0.24671718699653025, 0.5218122061699301, -0.45984821691067784, -0.14039550148338517, 0.17794433923475306, -0.21456843225290098, -0.22184758087922987, 1, 1, 0.07012078956642942, -0.15287462481333983, 0.5734771893942693, -0.06589724755809809, 0.19907699059679654, -0.10102669557239498, -0.21945223893855365, 0.05655762269703621, 0, 0, 0.39172362242953424, 0.3174042960248416, -0.36168475526371036, -0.30425084502904925, -0.17981409216963884, -0.1989414820289893, -0.5170233007291418, -0.2406247867850135, 0, 0, -0.510084143158857, -0.15453784770090523, 0.022384982696402888, -0.2813588427735337, 0.5104565418917715, -0.5620236646429332, 0.25726668672098146, -0.467946842450953, 0, 0, 0.31002748139403513, 0.5530458180738679, 0, 0.3019191139439164, 0.16802478103116028, 0]

type Gate = { W: number[][]; b: number[] }
type LayerWeights = Record<string, Gate>

interface LayerModel {
  getWeights(): LayerWeights
  setWeights(data: LayerWeights): void
  getWeightsFlat(): number[]
  setWeightsFlat(weights: number[]): void
}

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T
}

// Copia de la estructura con todos los valores desplazados (arrays nuevos).
function shifted(w: LayerWeights, delta: number): LayerWeights {
  const out: LayerWeights = {}
  for (const [name, g] of Object.entries(w)) {
    out[name] = { W: g.W.map(r => r.map(v => v + delta)), b: g.b.map(v => v + delta) }
  }
  return out
}

function gateOf(m: LayerModel, name: string): Gate {
  return (m as unknown as Record<string, Gate>)[name]
}

// Un episodio corto con gradiente no nulo (sanity: cambia los pesos).
function trainLSTM(l: LSTMLayer): void {
  l.reset()
  l.predict([0.3, -0.7])
  l.predict([0.9, 0.1])
  l.backprop([[0.5, -0.2], [0.1, 0.4]], 0.5)
}

function trainGRU(g: GRULayer): void {
  g.reset()
  g.predict([0.3, -0.7])
  g.predict([0.9, 0.1])
  g.backprop([[0.5, -0.2], [0.1, 0.4]], 0.5)
}

interface BatteryOpts<M extends LayerModel> {
  cls: string
  gates: [string, string, string]
  make: () => M
  train: (m: M) => void
  total: number
  gold: number[]
}

function layerBattery<M extends LayerModel>(o: BatteryOpts<M>): void {
  const { cls, gates, make, train, total, gold } = o
  const [G0, G1, G2] = gates

  describe(cls, () => {
    it('golden: getWeightsFlat con semilla fija igual que en la base', () => {
      seedRandom()
      expect(make().getWeightsFlat()).toEqual(gold)
    })

    it('getWeights devuelve copias: mutarlas no cambia getWeightsFlat del modelo', () => {
      const m = make()
      const before = m.getWeightsFlat()
      const w = m.getWeights()
      w[G0].W[0][0] += 1
      w[G1].b[0] += 1
      w[G2].W[1][COMB - 1] -= 1
      expect(m.getWeightsFlat()).toEqual(before)
    })

    it('getWeights no se mueve al entrenar el modelo', () => {
      const m = make()
      const before = m.getWeightsFlat()
      const w = m.getWeights()
      const snap = clone(w)
      train(m)
      expect(m.getWeightsFlat()).not.toEqual(before) // sanity: el entrenamiento sí cambia pesos
      expect(w).toEqual(snap)
    })

    it('setWeights escribe en los mismos arrays (sin reasignar referencias)', () => {
      const a = make()
      const b = make()
      const refW = gateOf(b, G0).W
      const refB = gateOf(b, G0).b
      b.setWeights(a.getWeights())
      expect(gateOf(b, G0).W).toBe(refW)
      expect(gateOf(b, G0).b).toBe(refB)
    })

    it('alias: b.setWeights(a.getWeights()) y entrenar b deja a idéntico', () => {
      const a = make()
      const b = make()
      b.setWeights(a.getWeights())
      const aBefore = a.getWeightsFlat()
      const bBefore = b.getWeightsFlat()
      expect(bBefore).toEqual(aBefore)
      train(b)
      expect(b.getWeightsFlat()).not.toEqual(bBefore) // sanity
      expect(a.getWeightsFlat()).toEqual(aBefore)
    })

    it('alias: los datos pasados a setWeights no se mueven al entrenar b', () => {
      const a = make()
      const b = make()
      const data = a.getWeights()
      b.setWeights(data)
      const snap = clone(data)
      train(b)
      expect(data).toEqual(snap)
    })

    it('setWeights copia valores: mutar los datos de entrada despues no cambia el modelo', () => {
      const a = make()
      const b = make()
      const data = a.getWeights()
      b.setWeights(data)
      const before = b.getWeightsFlat()
      data[G0].W[0][0] = 99
      data[G2].b[1] = 99
      expect(b.getWeightsFlat()).toEqual(before)
    })

    it('ida y vuelta: set(get()) deja los pesos idénticos y escribe valores nuevos', () => {
      const m = make()
      const w = m.getWeightsFlat()
      m.setWeights(m.getWeights())
      expect(m.getWeightsFlat()).toEqual(w)

      const n = make()
      n.setWeights(shifted(m.getWeights(), 0.25))
      expect(n.getWeightsFlat()).toEqual(w.map(v => v + 0.25))
    })

    // Formas erróneas: cada una lanza con mensaje claro y el modelo queda intacto.
    // Los demás valores están desplazados (+0.5) para que una escritura parcial
    // previa al error se detecte en la comprobación de estado intacto.
    const badShapes: Array<{ label: string; defect: (d: LayerWeights) => void; msg: string }> = [
      {
        label: 'una fila de menos',
        defect: d => { d[G0].W = d[G0].W.slice(0, -1) },
        msg: `${cls}.setWeights(${G0}.W): expected ${H} rows, got ${H - 1}`,
      },
      {
        label: 'una columna de menos en una fila',
        defect: d => { d[G1].W[1] = d[G1].W[1].slice(0, -1) },
        msg: `${cls}.setWeights(${G1}.W): row 1 expected ${COMB} cols, got ${COMB - 1}`,
      },
      {
        label: 'un sesgo de más',
        defect: d => { d[G2].b.push(0.5) },
        msg: `${cls}.setWeights(${G2}.b): expected array of length ${H}, got ${H + 1}`,
      },
      {
        label: 'falta una puerta',
        defect: d => { delete d[G2] },
        msg: `${cls}.setWeights: missing gate ${G2}`,
      },
      {
        label: 'NaN en un sesgo',
        defect: d => { d[G0].b[0] = NaN },
        msg: `${cls}.setWeights(${G0}.b): invalid value at index 0: NaN`,
      },
    ]

    for (const bad of badShapes) {
      it(`setWeights lanza con ${bad.label} y nombra ${cls}.setWeights`, () => {
        const m = make()
        const d = shifted(m.getWeights(), 0.5)
        bad.defect(d)
        expect(() => m.setWeights(d)).toThrow(bad.msg)
      })

      it(`setWeights deja el modelo intacto tras ${bad.label}`, () => {
        const m = make()
        const before = m.getWeightsFlat()
        const d = shifted(m.getWeights(), 0.5)
        bad.defect(d)
        expect(() => m.setWeights(d)).toThrow()
        expect(m.getWeightsFlat()).toEqual(before)
      })
    }

    it(`setWeightsFlat con longitud corta lanza y deja intacto`, () => {
      const m = make()
      const before = m.getWeightsFlat()
      const short = before.slice(0, -1).map(v => v + 0.5)
      expect(() => m.setWeightsFlat(short)).toThrow(
        `${cls}.setWeightsFlat: expected array of length ${total}, got ${total - 1}`,
      )
      expect(m.getWeightsFlat()).toEqual(before)
    })

    it(`setWeightsFlat con longitud larga lanza y deja intacto`, () => {
      const m = make()
      const before = m.getWeightsFlat()
      const long = [...before.map(v => v + 0.5), 0.5]
      expect(() => m.setWeightsFlat(long)).toThrow(
        `${cls}.setWeightsFlat: expected array of length ${total}, got ${total + 1}`,
      )
      expect(m.getWeightsFlat()).toEqual(before)
    })
  })
}

layerBattery<LSTMLayer>({
  cls: 'LSTMLayer',
  gates: ['forgetGate', 'inputGate', 'cellGate'],
  make: () => new LSTMLayer(IN, H),
  train: trainLSTM,
  total: 4 * (H * COMB + H),
  gold: GOLD_LSTM,
})

layerBattery<GRULayer>({
  cls: 'GRULayer',
  gates: ['resetGate', 'updateGate', 'newGate'],
  make: () => new GRULayer(IN, H),
  train: trainGRU,
  total: 3 * (H * COMB + H),
  gold: GOLD_GRU,
})

// ─── NetworkLSTM ──────────────────────────────────────────────────────────────

type LSTMWeights = ReturnType<LSTMLayer['getWeights']>
type NetWeights = {
  lstm: LSTMWeights
  dense: Array<Array<{ weights: number[]; bias: number }>>
}

function shiftedNet(w: NetWeights, delta: number): NetWeights {
  return {
    lstm: shifted(w.lstm, delta) as LSTMWeights,
    dense: w.dense.map(layer =>
      layer.map(n => ({ weights: n.weights.map(v => v + delta), bias: n.bias + delta })),
    ),
  }
}

function trainNet(net: NetworkLSTM): void {
  net.resetState()
  net.predict([0.3, -0.7])
  net.predict([0.9, 0.1])
  net.train([[0.2, 0.8], [0.6, 0.4]], 0.5)
}

describe('NetworkLSTM', () => {
  const NET_TOTAL = 4 * (H * COMB + H) + (H + 1) * 2 // LSTM + una capa densa de 2 neuronas

  it('golden: getWeightsFlat con semilla fija igual que en la base', () => {
    seedRandom()
    expect(new NetworkLSTM(IN, H, [2]).getWeightsFlat()).toEqual(GOLD_NET)
  })

  it('getWeights devuelve copias profundas (lstm y dense): mutarlas no cambia el modelo', () => {
    const net = new NetworkLSTM(IN, H, [2])
    const before = net.getWeightsFlat()
    const w = net.getWeights()
    w.lstm.forgetGate.W[0][0] += 1
    w.lstm.outputGate.b[1] -= 1
    w.dense[0][0].weights[0] += 1
    w.dense[0][1].bias += 1
    expect(net.getWeightsFlat()).toEqual(before)
  })

  it('getWeights no se mueve al entrenar la red', () => {
    const net = new NetworkLSTM(IN, H, [2])
    const before = net.getWeightsFlat()
    const w = net.getWeights()
    const snap = clone(w)
    trainNet(net)
    expect(net.getWeightsFlat()).not.toEqual(before) // sanity
    expect(w).toEqual(snap)
  })

  it('setWeights escribe en los mismos arrays (lstm y dense, sin reasignar)', () => {
    const a = new NetworkLSTM(IN, H, [2])
    const b = new NetworkLSTM(IN, H, [2])
    const refLstm = b.lstm.forgetGate.W
    const refDense = b.denseLayers[0].neurons[0].weights
    b.setWeights(a.getWeights())
    expect(b.lstm.forgetGate.W).toBe(refLstm)
    expect(b.denseLayers[0].neurons[0].weights).toBe(refDense)
  })

  it('alias: b.setWeights(a.getWeights()) y entrenar b deja a idéntico', () => {
    const a = new NetworkLSTM(IN, H, [2])
    const b = new NetworkLSTM(IN, H, [2])
    b.setWeights(a.getWeights())
    const aBefore = a.getWeightsFlat()
    const bBefore = b.getWeightsFlat()
    expect(bBefore).toEqual(aBefore)
    trainNet(b)
    expect(b.getWeightsFlat()).not.toEqual(bBefore) // sanity
    expect(a.getWeightsFlat()).toEqual(aBefore)
  })

  it('alias: los datos pasados a setWeights no se mueven al entrenar b', () => {
    const a = new NetworkLSTM(IN, H, [2])
    const b = new NetworkLSTM(IN, H, [2])
    const data = a.getWeights()
    b.setWeights(data)
    const snap = clone(data)
    trainNet(b)
    expect(data).toEqual(snap)
  })

  it('setWeights copia valores: mutar los datos de entrada despues no cambia el modelo', () => {
    const a = new NetworkLSTM(IN, H, [2])
    const b = new NetworkLSTM(IN, H, [2])
    const data = a.getWeights()
    b.setWeights(data)
    const before = b.getWeightsFlat()
    data.lstm.forgetGate.W[0][0] = 99
    data.dense[0][0].weights[0] = 99
    expect(b.getWeightsFlat()).toEqual(before)
  })

  it('ida y vuelta: set(get()) deja los pesos idénticos y escribe valores nuevos', () => {
    const net = new NetworkLSTM(IN, H, [2])
    const w = net.getWeightsFlat()
    net.setWeights(net.getWeights())
    expect(net.getWeightsFlat()).toEqual(w)

    const other = new NetworkLSTM(IN, H, [2])
    other.setWeightsFlat(w.map(v => v + 0.25))
    const copy = new NetworkLSTM(IN, H, [2])
    copy.setWeights(other.getWeights())
    expect(copy.getWeightsFlat()).toEqual(w.map(v => v + 0.25))
  })

  // Formas erróneas: cada una lanza con mensaje claro y la red queda intacta.
  const badShapes: Array<{ label: string; defect: (d: NetWeights) => NetWeights; msg: string }> = [
    {
      label: 'ninguna capa densa',
      defect: d => ({ ...d, dense: [] }),
      msg: 'NetworkLSTM.setWeights: expected 1 dense layers, got 0',
    },
    {
      label: 'una neurona con un peso de menos',
      defect: d => {
        d.dense[0][1].weights = d.dense[0][1].weights.slice(0, -1)
        return d
      },
      msg: 'NetworkLSTM.setWeights(dense[0][1].weights): expected array of length 2, got 1',
    },
    {
      label: 'una neurona de menos en una capa',
      defect: d => ({ ...d, dense: [d.dense[0].slice(0, 1)] }),
      msg: 'NetworkLSTM.setWeights: dense layer 0: expected 2 neurons, got 1',
    },
    {
      label: 'sesgo NaN en una neurona',
      defect: d => {
        d.dense[0][0].bias = NaN
        return d
      },
      msg: 'NetworkLSTM.setWeights(dense[0][0].bias): expected finite number, got NaN',
    },
    {
      label: 'una fila de menos en una puerta del lstm',
      defect: d => {
        d.lstm.cellGate.W = d.lstm.cellGate.W.slice(0, -1)
        return d
      },
      msg: `LSTMLayer.setWeights(cellGate.W): expected ${H} rows, got ${H - 1}`,
    },
    {
      label: 'falta el bloque lstm',
      defect: d => ({ ...d, lstm: undefined as unknown as LSTMWeights }),
      msg: 'NetworkLSTM.setWeights: missing lstm',
    },
  ]

  for (const bad of badShapes) {
    it(`setWeights lanza con ${bad.label}`, () => {
      const net = new NetworkLSTM(IN, H, [2])
      const d = bad.defect(shiftedNet(net.getWeights(), 0.5))
      expect(() => net.setWeights(d)).toThrow(bad.msg)
    })

    it(`setWeights deja la red intacta tras ${bad.label}`, () => {
      const net = new NetworkLSTM(IN, H, [2])
      const before = net.getWeightsFlat()
      const d = bad.defect(shiftedNet(net.getWeights(), 0.5))
      expect(() => net.setWeights(d)).toThrow()
      expect(net.getWeightsFlat()).toEqual(before)
    })
  }

  it('setWeightsFlat con longitud corta lanza y deja intacta la red', () => {
    const net = new NetworkLSTM(IN, H, [2])
    const before = net.getWeightsFlat()
    const short = before.slice(0, -1).map(v => v + 0.5)
    expect(() => net.setWeightsFlat(short)).toThrow(
      `NetworkLSTM.setWeightsFlat: expected array of length ${NET_TOTAL}, got ${NET_TOTAL - 1}`,
    )
    expect(net.getWeightsFlat()).toEqual(before)
  })

  it('setWeightsFlat con longitud larga lanza y deja intacta la red', () => {
    const net = new NetworkLSTM(IN, H, [2])
    const before = net.getWeightsFlat()
    const long = [...before.map(v => v + 0.5), 0.5]
    expect(() => net.setWeightsFlat(long)).toThrow(
      `NetworkLSTM.setWeightsFlat: expected array of length ${NET_TOTAL}, got ${NET_TOTAL + 1}`,
    )
    expect(net.getWeightsFlat()).toEqual(before)
  })
})
