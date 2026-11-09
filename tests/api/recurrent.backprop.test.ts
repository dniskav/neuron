// T013 (Q10): backprop recurrente y train lanzan Error ante forma incorrecta.
//
// - LSTMLayer.backprop, GRULayer.backprop y NetworkLSTM.train validan la longitud
//   de la secuencia contra la trayectoria, el tamaño de cada fila (hSize para
//   dh_seq, tamaño de salida para targets) y dcFinal (solo LSTM).
// - Trayectoria vacía + secuencia vacía sigue sin error (ceros / no-op).
// - Un error NO vacía la trayectoria: el reintento válido da los mismos pesos
//   que sin el error intermedio (golden).
//
// Cifras: IN = 2 e H = 3 (inputSize != hSize, para detectar validar contra
// inputSize); NetworkLSTM(2, 3, [2]) con 2 salidas.

import { describe, it, expect } from 'vitest'
import { LSTMLayer } from '../../src/LSTMLayer'
import { GRULayer } from '../../src/GRU'
import { NetworkLSTM } from '../../src/NetworkLSTM'
import { withSeed } from '../helpers/seed'

const IN = 2
const H = 3
const SEED = 0x13
const X1 = [0.3, -0.7]
const X2 = [0.9, 0.1]
const DH_OK = [[0.5, -0.2, 0.1], [0.1, 0.4, -0.3]]
const DC_OK = [0.2, -0.1, 0.05]
const T_OK = [[0.2, 0.8], [0.6, 0.4]]

function lstm(): LSTMLayer {
  return withSeed(SEED, () => new LSTMLayer(IN, H))
}
function gru(): GRULayer {
  return withSeed(SEED, () => new GRULayer(IN, H))
}
function net(): NetworkLSTM {
  return withSeed(SEED, () => new NetworkLSTM(IN, H, [2]))
}

// ─── LSTMLayer ────────────────────────────────────────────────────────────────

function lstmEpisode(l: LSTMLayer): void {
  l.reset()
  l.predict(X1)
  l.predict(X2)
}

// Ejecuta un episodio; si se pasa `bad`, la llamada inválida debe lanzar y
// después se hace el backprop válido sobre la MISMA trayectoria.
function runLSTM(bad: ((l: LSTMLayer) => void) | null) {
  const l = lstm()
  lstmEpisode(l)
  if (bad) expect(() => bad(l)).toThrow()
  const r = l.backprop(DH_OK, 0.5, DC_OK)
  return { w: l.getWeightsFlat(), r }
}

describe('LSTMLayer.backprop valida la forma (Q10)', () => {
  it('longitud de dh_seq distinta de la trayectoria (menos y más) lanza con ambas longitudes', () => {
    const a = lstm()
    lstmEpisode(a)
    expect(() => a.backprop([DH_OK[0]], 0.5)).toThrow(/LSTMLayer\.backprop.*length 2.*got 1/)
    expect(() => a.backprop([...DH_OK, DH_OK[0]], 0.5)).toThrow(/LSTMLayer\.backprop.*length 2.*got 3/)
  })

  it('sin predict previo y con secuencia no vacía lanza (esperada 0)', () => {
    const a = lstm()
    expect(() => a.backprop(DH_OK, 0.5)).toThrow(/LSTMLayer\.backprop.*length 0.*got 2/)
  })

  it('dh_seq que no es un array lanza', () => {
    const a = lstm()
    lstmEpisode(a)
    expect(() => a.backprop(null as unknown as number[][], 0.5)).toThrow(/LSTMLayer\.backprop.*expected/)
  })

  it('fila dh_seq[t] con longitud inputSize (no hSize) lanza', () => {
    const a = lstm()
    lstmEpisode(a)
    const bad = [[0.5, -0.2], [0.1, 0.4]] // IN = 2, H = 3
    expect(() => a.backprop(bad, 0.5)).toThrow(/LSTMLayer\.backprop\(dh_seq\[0\]\).*length 3.*got 2/)
  })

  it('fila dh_seq[t] más larga que hSize lanza', () => {
    const a = lstm()
    lstmEpisode(a)
    const bad = [DH_OK[0], [0.1, 0.4, -0.3, 0]]
    expect(() => a.backprop(bad, 0.5)).toThrow(/LSTMLayer\.backprop\(dh_seq\[1\]\).*length 3.*got 4/)
  })

  it('dcFinal de longitud distinta de hSize lanza', () => {
    const a = lstm()
    lstmEpisode(a)
    expect(() => a.backprop(DH_OK, 0.5, [1, 2])).toThrow(/LSTMLayer\.backprop\(dcFinal\).*length 3.*got 2/)
  })

  it('dcFinal de longitud hSize es válido', () => {
    const a = lstm()
    lstmEpisode(a)
    expect(() => a.backprop(DH_OK, 0.5, DC_OK)).not.toThrow()
  })

  it('episodio vacío sin error: devuelve ceros y no cambia los pesos', () => {
    const a = lstm()
    const before = a.getWeightsFlat()
    expect(a.backprop([], 0.5)).toEqual({ dh0: [0, 0, 0], dc0: [0, 0, 0] })
    expect(a.getWeightsFlat()).toEqual(before)
  })

  describe('un error no vacía la trayectoria: el reintento válido da los mismos pesos (golden)', () => {
    it('golden: el episodio válido sí cambia los pesos (sanity)', () => {
      expect(runLSTM(null).w).not.toEqual(lstm().getWeightsFlat())
    })

    it('retorno {dh0, dc0} intacto: claves y longitudes hSize', () => {
      const r = runLSTM(null).r
      expect(Object.keys(r).sort()).toEqual(['dc0', 'dh0'])
      expect(r.dh0).toHaveLength(H)
      expect(r.dc0).toHaveLength(H)
    })

    it.each([
      ['longitud', (l: LSTMLayer) => l.backprop([DH_OK[0]], 0.5)],
      ['fila', (l: LSTMLayer) => l.backprop([[1, 2], [3, 4]], 0.5)],
      ['dcFinal', (l: LSTMLayer) => l.backprop(DH_OK, 0.5, [1, 2])],
    ])('error de %s intermedio: reintento válido == golden', (_name, bad) => {
      expect(runLSTM(bad)).toEqual(runLSTM(null))
    })
  })
})

// ─── GRULayer ─────────────────────────────────────────────────────────────────

function gruEpisode(g: GRULayer): void {
  g.reset()
  g.predict(X1)
  g.predict(X2)
}

function runGRU(bad: ((g: GRULayer) => void) | null): number[] {
  const g = gru()
  gruEpisode(g)
  if (bad) expect(() => bad(g)).toThrow()
  g.backprop(DH_OK, 0.5)
  return g.getWeightsFlat()
}

describe('GRULayer.backprop valida la forma (Q10)', () => {
  it('longitud de dh_seq distinta de la trayectoria lanza con ambas longitudes', () => {
    const g = gru()
    gruEpisode(g)
    expect(() => g.backprop([DH_OK[0]], 0.5)).toThrow(/GRULayer\.backprop.*length 2.*got 1/)
    expect(() => g.backprop([...DH_OK, DH_OK[0]], 0.5)).toThrow(/GRULayer\.backprop.*length 2.*got 3/)
  })

  it('sin predict previo y con secuencia no vacía lanza (esperada 0)', () => {
    const g = gru()
    expect(() => g.backprop(DH_OK, 0.5)).toThrow(/GRULayer\.backprop.*length 0.*got 2/)
  })

  it('dh_seq que no es un array lanza', () => {
    const g = gru()
    gruEpisode(g)
    expect(() => g.backprop(null as unknown as number[][], 0.5)).toThrow(/GRULayer\.backprop.*expected/)
  })

  it('fila dh_seq[t] con longitud inputSize (no hSize) lanza', () => {
    const g = gru()
    gruEpisode(g)
    expect(() => g.backprop([[0.5, -0.2], [0.1, 0.4]], 0.5)).toThrow(
      /GRULayer\.backprop\(dh_seq\[0\]\).*length 3.*got 2/,
    )
  })

  it('episodio vacío sin error: no cambia los pesos', () => {
    const g = gru()
    const before = g.getWeightsFlat()
    expect(() => g.backprop([], 0.5)).not.toThrow()
    expect(g.getWeightsFlat()).toEqual(before)
  })

  describe('un error no vacía la trayectoria: golden', () => {
    it('golden: el episodio válido sí cambia los pesos (sanity)', () => {
      expect(runGRU(null)).not.toEqual(gru().getWeightsFlat())
    })

    it.each([
      ['longitud', (g: GRULayer) => g.backprop([DH_OK[0]], 0.5)],
      ['fila', (g: GRULayer) => g.backprop([[1, 2], [3, 4]], 0.5)],
    ])('error de %s intermedio: reintento válido == golden', (_name, bad) => {
      expect(runGRU(bad)).toEqual(runGRU(null))
    })
  })
})

// ─── NetworkLSTM.train ────────────────────────────────────────────────────────

function netEpisode(n: NetworkLSTM): void {
  n.resetState()
  n.predict(X1)
  n.predict(X2)
}

function runNet(bad: ((n: NetworkLSTM) => void) | null): number[] {
  const n = net()
  netEpisode(n)
  if (bad) expect(() => bad(n)).toThrow()
  n.train(T_OK, 0.5)
  return n.getWeightsFlat()
}

describe('NetworkLSTM.train valida la forma (Q10)', () => {
  it('número de targets distinto de la trayectoria lanza con ambas longitudes', () => {
    const n = net()
    netEpisode(n)
    expect(() => n.train([T_OK[0]], 0.5)).toThrow(/NetworkLSTM\.train.*length 2.*got 1/)
    expect(() => n.train([...T_OK, T_OK[0]], 0.5)).toThrow(/NetworkLSTM\.train.*length 2.*got 3/)
  })

  it('sin predict previo y con targets no vacíos lanza (esperada 0)', () => {
    const n = net()
    n.resetState()
    expect(() => n.train(T_OK, 0.5)).toThrow(/NetworkLSTM\.train.*length 0.*got 2/)
  })

  it('targets que no es un array lanza', () => {
    const n = net()
    netEpisode(n)
    expect(() => n.train(null as unknown as number[][], 0.5)).toThrow(/NetworkLSTM\.train.*expected/)
  })

  it('fila targets[t] con longitud distinta del tamaño de salida lanza', () => {
    const n = net()
    netEpisode(n)
    expect(() => n.train([[0.2], [0.6, 0.4]], 0.5)).toThrow(
      /NetworkLSTM\.train\(targets\[0\]\).*length 2.*got 1/,
    )
  })

  it('episodio vacío sin error: no-op (no cambia los pesos)', () => {
    const n = net()
    n.resetState()
    const before = n.getWeightsFlat()
    expect(() => n.train([], 0.5)).not.toThrow()
    expect(n.getWeightsFlat()).toEqual(before)
  })

  it('si la trayectoria del LSTM no coincide, lanza sin tocar las capas densas', () => {
    const n = net()
    n.resetState()
    n.predict(X1)
    n.lstm.predict(X2) // trayectoria LSTM de 2 pasos; NetworkLSTM solo conoce 1
    const denseBefore = n.getWeights().dense
    expect(() => n.train([T_OK[0]], 0.5)).toThrow(/LSTMLayer\.backprop.*length 2.*got 1/)
    expect(n.getWeights().dense).toEqual(denseBefore)
  })

  describe('un error no vacía la trayectoria: golden', () => {
    it('golden: el episodio válido sí cambia los pesos (sanity)', () => {
      expect(runNet(null)).not.toEqual(net().getWeightsFlat())
    })

    it.each([
      ['número de targets', (n: NetworkLSTM) => n.train([T_OK[0]], 0.5)],
      ['fila', (n: NetworkLSTM) => n.train([[0.2], [0.6, 0.4]], 0.5)],
    ])('error de %s intermedio: reintento válido == golden', (_name, bad) => {
      expect(runNet(bad)).toEqual(runNet(null))
    })
  })
})

// ─── NetworkLSTM: denseStructure vacío (T021, MENOR 2) ───────────────────────
//
// Sin capas densas, train() revienta con TypeError al leer la última capa densa.
// El constructor lo rechaza con un Error que nombra NetworkLSTM. Con una capa
// densa el comportamiento no cambia (ver los tests de arriba).

describe('NetworkLSTM rechaza denseStructure vacío (T021)', () => {
  it('new NetworkLSTM(IN, H, []) lanza un Error que nombra NetworkLSTM y denseStructure', () => {
    expect(() => new NetworkLSTM(IN, H, [])).toThrow(/NetworkLSTM.*denseStructure/)
  })

  it('el mensaje indica que hace falta al menos una capa densa', () => {
    expect(() => new NetworkLSTM(IN, H, [])).toThrow(/at least one dense layer/)
  })

  it('con una capa densa la construcción y el primer train siguen funcionando', () => {
    const n = new NetworkLSTM(IN, H, [2])
    netEpisode(n)
    expect(() => n.train(T_OK, 0.5)).not.toThrow()
  })
})
