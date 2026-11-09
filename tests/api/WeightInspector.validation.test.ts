import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { NetworkN } from '../../src/NetworkN'
import { WeightInspector } from '../../src/WeightInspector'
import { seedRandom, restoreRandom } from '../helpers/seed'

// Validación de entradas de WeightInspector (T054, principio V: fallar con claridad).
// Red de referencia: NetworkN([2, 2, 1]) con los mismos pesos fijados que tests/WeightInspector.test.ts.

const FLAT_WEIGHTS = [1, -1, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0]

function buildNet(): NetworkN {
  const net = new NetworkN([2, 2, 1])
  net.setWeights([...FLAT_WEIGHTS])
  return net
}

describe('WeightInspector: deadThreshold inválido', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  it('inspect lanza con NaN', () => {
    expect(() => WeightInspector.inspect(buildNet(), NaN)).toThrow(/WeightInspector\.inspect/)
  })

  it('inspect lanza con un umbral negativo', () => {
    expect(() => WeightInspector.inspect(buildNet(), -0.1)).toThrow(/WeightInspector\.inspect/)
  })

  it('inspectAll lanza con NaN', () => {
    expect(() => WeightInspector.inspectAll(buildNet(), NaN)).toThrow(/WeightInspector\.inspectAll/)
  })

  it('inspectAll lanza con un umbral negativo', () => {
    expect(() => WeightInspector.inspectAll(buildNet(), -1)).toThrow(/WeightInspector\.inspectAll/)
  })

  it('countDeadReLUs lanza con NaN y con un umbral negativo', () => {
    expect(() => WeightInspector.countDeadReLUs([[0]], NaN)).toThrow(/WeightInspector\.countDeadReLUs/)
    expect(() => WeightInspector.countDeadReLUs([[0]], -1e-6)).toThrow(/WeightInspector\.countDeadReLUs/)
  })

  it('print lanza con NaN (lo propaga inspect)', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    expect(() => WeightInspector.print(buildNet(), NaN)).toThrow(/WeightInspector\.inspect/)
    expect(spy).not.toHaveBeenCalled()
    vi.restoreAllMocks()
  })

  it('acepta el umbral 0: ningún peso es muerto con |w| < 0', () => {
    // Mismo caso válido que el golden: 0 es un umbral permitido.
    const stats = WeightInspector.inspect(buildNet(), 0)
    expect(stats[0].deadCount).toBe(0)
    expect(stats[1].deadCount).toBe(0)
    expect(WeightInspector.inspectAll(buildNet(), 0).deadCount).toBe(0)
    expect(WeightInspector.countDeadReLUs([[0], [0]], 0)).toBe(0)
  })
})

describe('WeightInspector.countDeadReLUs: activaciones inválidas', () => {
  it('lanza con filas de distinta longitud', () => {
    expect(() => WeightInspector.countDeadReLUs([[0, 1], [0]])).toThrow(/WeightInspector\.countDeadReLUs/)
  })

  it('lanza con una fila más larga que la primera', () => {
    expect(() => WeightInspector.countDeadReLUs([[0], [0, 1]])).toThrow(/WeightInspector\.countDeadReLUs/)
  })

  it('lanza con NaN en las activaciones', () => {
    expect(() => WeightInspector.countDeadReLUs([[0, NaN], [0, 0]])).toThrow(/WeightInspector\.countDeadReLUs/)
  })

  it('lanza con Infinity en las activaciones', () => {
    expect(() => WeightInspector.countDeadReLUs([[0, 1], [Infinity, 0]])).toThrow(/WeightInspector\.countDeadReLUs/)
  })
})

describe('WeightInspector: pesos NaN (comportamiento documentado, no se lanza)', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  // setWeights rechaza NaN (validateArray), así que el NaN se escribe directamente en el sesgo.
  // Red [2, 1]: una neurona con pesos [0.5, 0.25] y sesgo NaN → 3 parámetros: [0.5, 0.25, NaN].
  //   mean y std son NaN (la suma contamina todo).
  //   min y max ignoran los NaN (las comparaciones con NaN son falsas): min = 0.25, max = 0.5.
  //   Un NaN no es muerto (|NaN| < umbral es falso): solo 0.25 cuenta con umbral 0.3.
  it('mean y std son NaN; min y max ignoran el NaN; el NaN no cuenta como muerto', () => {
    const net = new NetworkN([2, 1])
    net.layers[0].neurons[0].weights = [0.5, 0.25]
    net.layers[0].neurons[0].bias = NaN
    const s = WeightInspector.inspectAll(net, 0.3)
    expect(s.totalParams).toBe(3)
    expect(s.mean).toBeNaN()
    expect(s.std).toBeNaN()
    expect(s.min).toBe(0.25)
    expect(s.max).toBe(0.5)
    expect(s.deadCount).toBe(1)
  })
})
