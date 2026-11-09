// T035 (FR-008 / US6): getWeightsFlat()/getWeights() de LSTMLayer, GRULayer,
// NetworkLSTM, NetworkTransformer y NetworkTransformerRL sin `push(...array)`
// sobre arrays que superan el límite de argumentos de la VM.
//
// Medido en este worktree (Node v26.12.1, script t035_limit.cjs):
//   límite de push(...array) = 124669 argumentos.
//
// Dimensiones de los casos grandes (todas deducidas a mano):
//   - LSTMLayer(150000, 1): cada puerta tiene 1 fila de 150001 pesos (> límite).
//     Flat = 4 · (1·150001 + 1) = 600008.
//   - GRULayer(150000, 1): 3 puertas con 1 fila de 150001 pesos (> límite).
//     Flat = 3 · (1·150001 + 1) = 450006.
//   - NetworkLSTM(1, 200, [2]): el LSTM tiene 4 · (200·201 + 200) = 161600 pesos
//     (> límite; el spread que falla es `push(...this.lstm.getWeightsFlat())`).
//     Capa densa: 2 neuronas · (200 + 1) = 402. Total = 162002.
//   - NetworkTransformer(seqLen 2, vocab 3, d_model 144, nHeads 4, d_ff 576, nBlocks 1, nClasses 2):
//     bloque = MHA 4·dM² (= 82944, por debajo del límite: es el spread de
//     TransformerBlock, tarea T034) + norm1 2dM + ff1 dFF·dM + b1 dFF + ff2 dM·dFF
//     + b2 dM + norm2 2dM = 82944 + 288 + 82944 + 576 + 82944 + 144 + 288 = 250128
//     (> límite; el spread que falla es `push(...block.getWeights())`).
//     Total = vocab·dM + seqLen·dM + bloque + nClasses·dM + nClasses
//           = 432 + 288 + 250128 + 288 + 2 = 251138.
//   - NetworkTransformerRL(seqLen 2, inputDim 3, d_model 144, nHeads 4, d_ff 576, nBlocks 1, nActions 2):
//     Total = inputDim·dM + bloque + nActions·dM + nActions = 432 + 250128 + 288 + 2 = 250850.
//
// Golden de tamaño pequeño: capturado con src/ SIN CAMBIAR (base fix/v0.4),
// semilla 0x5eed, antes de tocar src/. Longitudes deducidas a mano en cada test.

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { LSTMLayer } from '../../src/LSTMLayer'
import { GRULayer } from '../../src/GRU'
import { NetworkLSTM } from '../../src/NetworkLSTM'
import { NetworkTransformer } from '../../src/NetworkTransformer'
import { NetworkTransformerRL } from '../../src/NetworkTransformerRL'
import { seedRandom, restoreRandom } from '../helpers/seed'

const SEED = 0x5eed

beforeEach(() => {
  seedRandom(SEED)
})

afterEach(() => {
  restoreRandom()
})

// ── Golden de tamaño pequeño ────────────────────────────────────────────────

// LSTMLayer(2, 2): 4 puertas · (2 · (2+2) + 2) = 40.
const GOLD_LSTM: number[] = [0.2425241061934615,-0.24671718699653025,0.5218122061699301,-0.45984821691067784,-0.14039550148338517,0.17794433923475306,-0.21456843225290098,-0.22184758087922987,1,1,0.07012078956642942,-0.15287462481333983,0.5734771893942693,-0.06589724755809809,0.19907699059679654,-0.10102669557239498,-0.21945223893855365,0.05655762269703621,0,0,0.39172362242953424,0.3174042960248416,-0.36168475526371036,-0.30425084502904925,-0.17981409216963884,-0.1989414820289893,-0.5170233007291418,-0.2406247867850135,0,0,-0.510084143158857,-0.15453784770090523,0.022384982696402888,-0.2813588427735337,0.5104565418917715,-0.5620236646429332,0.25726668672098146,-0.467946842450953,0,0]

// GRULayer(2, 2): 3 puertas · (2 · (2+2) + 2) = 30.
const GOLD_GRU: number[] = [0.2425241061934615,-0.24671718699653025,0.5218122061699301,-0.45984821691067784,-0.14039550148338517,0.17794433923475306,-0.21456843225290098,-0.22184758087922987,0,0,0.07012078956642942,-0.15287462481333983,0.5734771893942693,-0.06589724755809809,0.19907699059679654,-0.10102669557239498,-0.21945223893855365,0.05655762269703621,0,0,0.39172362242953424,0.3174042960248416,-0.36168475526371036,-0.30425084502904925,-0.17981409216963884,-0.1989414820289893,-0.5170233007291418,-0.2406247867850135,0,0]

// NetworkLSTM(2, 2, [3, 1]): LSTM 40 + Layer(3, 2) = 3·(2+1) = 9 + Layer(1, 3) = 1·(3+1) = 4. Total 53.
const GOLD_NET_LSTM: number[] = [0.2425241061934615,-0.24671718699653025,0.5218122061699301,-0.45984821691067784,-0.14039550148338517,0.17794433923475306,-0.21456843225290098,-0.22184758087922987,1,1,0.07012078956642942,-0.15287462481333983,0.5734771893942693,-0.06589724755809809,0.19907699059679654,-0.10102669557239498,-0.21945223893855365,0.05655762269703621,0,0,0.39172362242953424,0.3174042960248416,-0.36168475526371036,-0.30425084502904925,-0.17981409216963884,-0.1989414820289893,-0.5170233007291418,-0.2406247867850135,0,0,-0.510084143158857,-0.15453784770090523,0.022384982696402888,-0.2813588427735337,0.5104565418917715,-0.5620236646429332,0.25726668672098146,-0.467946842450953,0,0,0.31002748139403513,0.5530458180738679,0,0.3019191139439164,0.16802478103116028,0,-0.41309359613536245,0.49073770320806775,0,0.3592710022288117,-0.41013857518238434,-0.4651620714950429,0]

// NetworkTransformer(2, {vocab 3, d_model 2, nHeads 1, d_ff 2, nBlocks 1, nClasses 2}):
// tokenEmb 3·2 = 6, posEmb 2·2 = 4, bloque 4·4 + 2·2·2 + 2 + 5·2 = 36, outputProj 2·2 = 4, outputBias 2. Total 52.
const GOLD_TRANSFORMER: number[] = [0.2970301552492711,-0.3021656094581601,0.6390868233361524,-0.5631967452799197,-0.17194867040822623,0.21793641687092896,-0.2627915869642741,-0.2717066869124676,0.0858800773994133,-0.18723241270605134,0.7023632465706944,-0.08070731598560256,0.2438185232454982,-0.12373192727593013,-0.2687730041553952,0.06926865833629564,0.47976149757350733,0.3887392837140825,-0.44297154906975134,-0.37262966206588527,-0.2202263871886997,-0.24365255982204664,-0.6332216359579678,-0.29470397354463995,-0.6247229383119833,-0.18926943640757818,1,1,0,0,0.027415892753608905,-0.3445927997075579,0.6251790317502329,-0.6883356008721389,0,0,0.3150860551414287,-0.5731154953756927,0.31002748139403513,0.5530458180738679,0,0,1,1,0,0,0.3019191139439164,0.16802478103116028,-0.41309359613536245,0.49073770320806775,0,0]

// NetworkTransformerRL(2, 2, {d_model 2, nHeads 1, d_ff 2, nBlocks 1, nActions 2}):
// inputProj 2·2 = 4, bloque 36, outputProj 2·2 = 4, outputBias 2. Total 46.
const GOLD_TRANSFORMER_RL: number[] = [0.2970301552492711,-0.3021656094581601,0.6390868233361524,-0.5631967452799197,-0.17194867040822623,0.21793641687092896,-0.2627915869642741,-0.2717066869124676,0.0858800773994133,-0.18723241270605134,0.7023632465706944,-0.08070731598560256,0.2438185232454982,-0.12373192727593013,-0.2687730041553952,0.06926865833629564,0.47976149757350733,0.3887392837140825,-0.44297154906975134,-0.37262966206588527,1,1,0,0,-0.2202263871886997,-0.24365255982204664,-0.6332216359579678,-0.29470397354463995,0,0,-0.6247229383119833,-0.18926943640757818,0.027415892753608905,-0.3445927997075579,0,0,1,1,0,0,0.6251790317502329,-0.6883356008721389,0.3150860551414287,-0.5731154953756927,0,0]

describe('golden de tamaño pequeño (semilla fija, igual que la base)', () => {
  it('LSTMLayer(2, 2).getWeightsFlat() es el golden (longitud 40)', () => {
    const flat = new LSTMLayer(2, 2).getWeightsFlat()
    expect(flat.length).toBe(4 * (2 * (2 + 2) + 2))
    expect(flat).toEqual(GOLD_LSTM)
  })

  it('GRULayer(2, 2).getWeightsFlat() es el golden (longitud 30)', () => {
    const flat = new GRULayer(2, 2).getWeightsFlat()
    expect(flat.length).toBe(3 * (2 * (2 + 2) + 2))
    expect(flat).toEqual(GOLD_GRU)
  })

  it('NetworkLSTM(2, 2, [3, 1]).getWeightsFlat() es el golden (LSTM, luego densas; longitud 53)', () => {
    const flat = new NetworkLSTM(2, 2, [3, 1]).getWeightsFlat()
    expect(flat.length).toBe(40 + 3 * (2 + 1) + 1 * (3 + 1))
    expect(flat).toEqual(GOLD_NET_LSTM)
  })

  it('NetworkTransformer(2, {vocab 3, d_model 2, ...}).getWeights() es el golden (longitud 52)', () => {
    const net = new NetworkTransformer(2, { vocabSize: 3, d_model: 2, nHeads: 1, d_ff: 2, nBlocks: 1, nClasses: 2 })
    expect(net.getWeights().length).toBe(3 * 2 + 2 * 2 + 36 + 2 * 2 + 2)
    expect(net.getWeights()).toEqual(GOLD_TRANSFORMER)
  })

  it('NetworkTransformerRL(2, 2, {d_model 2, ...}).getWeights() es el golden (longitud 46)', () => {
    const net = new NetworkTransformerRL(2, 2, { d_model: 2, nHeads: 1, d_ff: 2, nBlocks: 1, nActions: 2 })
    expect(net.getWeights().length).toBe(2 * 2 + 36 + 2 * 2 + 2)
    expect(net.getWeights()).toEqual(GOLD_TRANSFORMER_RL)
  })
})

// ── Tamaño grande: superan el límite de argumentos (ver cabecera) ───────────

// Índice de la primera diferencia (-1 si son idénticos, -2 si difiere la longitud).
// Un toEqual fallido sobre cientos de miles de números serializa el diff completo
// y tarda minutos; esta comparación falla al instante con un mensaje corto.
function firstMismatch(actual: number[], expected: number[]): number {
  if (actual.length !== expected.length) return -2
  for (let i = 0; i < actual.length; i++) if (actual[i] !== expected[i]) return i
  return -1
}

describe('tamaño grande: sin RangeError por push(...array)', () => {
  it('LSTMLayer(150000, 1): flat de longitud 600008 y round-trip', () => {
    const layer = new LSTMLayer(150000, 1)
    const flat = layer.getWeightsFlat()
    expect(flat.length).toBe(600008)
    layer.setWeightsFlat(flat)
    expect(firstMismatch(layer.getWeightsFlat(), flat)).toBe(-1)
  }, 60000)

  it('GRULayer(150000, 1): flat de longitud 450006 y round-trip', () => {
    const layer = new GRULayer(150000, 1)
    const flat = layer.getWeightsFlat()
    expect(flat.length).toBe(450006)
    layer.setWeightsFlat(flat)
    expect(firstMismatch(layer.getWeightsFlat(), flat)).toBe(-1)
  }, 60000)

  it('NetworkLSTM(1, 200, [2]): flat de longitud 162002 (LSTM 161600 + densa 402) y round-trip', () => {
    const net = new NetworkLSTM(1, 200, [2])
    const flat = net.getWeightsFlat()
    expect(flat.length).toBe(4 * (200 * (1 + 200) + 200) + 2 * (200 + 1))
    net.setWeightsFlat(flat)
    expect(firstMismatch(net.getWeightsFlat(), flat)).toBe(-1)
  }, 60000)

  it('NetworkTransformer(2, d_model 144, nBlocks 1): getWeights de longitud 251138 y round-trip', () => {
    const net = new NetworkTransformer(2, { vocabSize: 3, d_model: 144, nHeads: 4, d_ff: 576, nBlocks: 1, nClasses: 2 })
    const flat = net.getWeights()
    expect(flat.length).toBe(3 * 144 + 2 * 144 + 250128 + 2 * 144 + 2)
    net.setWeights(flat)
    expect(firstMismatch(net.getWeights(), flat)).toBe(-1)
  }, 60000)

  it('NetworkTransformerRL(2, 3, d_model 144, nBlocks 1): getWeights de longitud 250850 y round-trip', () => {
    const net = new NetworkTransformerRL(2, 3, { d_model: 144, nHeads: 4, d_ff: 576, nBlocks: 1, nActions: 2 })
    const flat = net.getWeights()
    expect(flat.length).toBe(3 * 144 + 250128 + 2 * 144 + 2)
    net.setWeights(flat)
    expect(firstMismatch(net.getWeights(), flat)).toBe(-1)
  }, 60000)
})
