import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { NetworkN } from '../src/NetworkN'
import { WeightInspector } from '../src/WeightInspector'
import { seedRandom, restoreRandom } from './helpers/seed'

// ── Red de referencia con pesos fijados a mano (T013, FR-004) ─────────────────
// NetworkN([2, 2, 1]). El orden de getWeights/setWeights es, por capa y por neurona,
// sus pesos seguidos del sesgo (el mismo orden que usa WeightInspector).
//
//   Capa 0 (6 parámetros):
//     n0: w = [ 1, -1], b =  0.5
//     n1: w = [-0.5, 0.25], b = -0.25
//     plano: [1, -1, 0.5, -0.5, 0.25, -0.25]
//   Capa 1 (3 parámetros):
//     n0: w = [0.75, -0.75], b = 0
//     plano: [0.75, -0.75, 0]
//
// Derivaciones (estadísticas poblacionales, división entre n):
//   Capa 0: suma = 1-1+0.5-0.5+0.25-0.25 = 0 → media 0.
//           suma de cuadrados = 1+1+0.25+0.25+0.0625+0.0625 = 2.625
//           varianza = 2.625/6 = 0.4375 = 7/16 → desviación = √7/4 ≈ 0.6614378
//           min = -1, max = 1.
//   Capa 1: suma = 0 → media 0. Cuadrados = 0.5625+0.5625+0 = 1.125
//           varianza = 1.125/3 = 0.375 = 3/8 → desviación = √(3/8) ≈ 0.6123724
//           min = -0.75, max = 0.75.
//   Global (9): suma = 0 → media 0. Cuadrados = 2.625+1.125 = 3.75
//           varianza = 3.75/9 = 5/12 → desviación = √(5/12) ≈ 0.6454972
//           min = -1, max = 1.
//
// Muertos (|w| < umbral, estricto):
//   umbral 0.5: capa 0 → {0.25, -0.25} = 2 (los ±0.5 están justo en el umbral y viven);
//               capa 1 → {0} = 1; global = 3.
//   umbral por defecto 1e-3: capa 0 → 0; capa 1 → {0} = 1; global = 1.
const FLAT_WEIGHTS = [1, -1, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0]

function buildNet(): NetworkN {
  const net = new NetworkN([2, 2, 1])
  net.setWeights([...FLAT_WEIGHTS])
  return net
}

describe('WeightInspector.inspect (por capa)', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  it('media, desviación, mínimo, máximo y nº de parámetros por capa (sesgos incluidos)', () => {
    const stats = WeightInspector.inspect(buildNet(), 0.5)
    expect(stats).toHaveLength(2)

    // Capa 0: 6 parámetros (4 pesos + 2 sesgos). Si se excluyeran los sesgos serían 4.
    expect(stats[0].totalParams).toBe(6)
    expect(stats[0].mean).toBeCloseTo(0, 12)
    expect(stats[0].std).toBeCloseTo(Math.sqrt(7) / 4, 12) // √(2.625/6)
    expect(stats[0].min).toBe(-1)
    expect(stats[0].max).toBe(1)

    // Capa 1: 3 parámetros (2 pesos + 1 sesgo).
    expect(stats[1].totalParams).toBe(3)
    expect(stats[1].mean).toBeCloseTo(0, 12)
    expect(stats[1].std).toBeCloseTo(Math.sqrt(3 / 8), 12) // √(1.125/3)
    expect(stats[1].min).toBe(-0.75)
    expect(stats[1].max).toBe(0.75)
  })

  it('cuenta muertos con |w| < umbral: un peso exactamente igual al umbral vive', () => {
    // Los pesos ±0.5 de la capa 0 valen exactamente 0.5 y no son muertos.
    const stats = WeightInspector.inspect(buildNet(), 0.5)
    expect(stats[0].deadCount).toBe(2) // 0.25 y -0.25
    expect(stats[1].deadCount).toBe(1) // 0
  })

  it('compara la magnitud: un peso negativo con |w| < umbral cuenta como muerto', () => {
    // Con umbral 0.3 solo son muertos 0.25 y -0.25. Sin valor absoluto, -0.5 y -1
    // también caerían por debajo de 0.3 y el recuento daría 4.
    const stats = WeightInspector.inspect(buildNet(), 0.3)
    expect(stats[0].deadCount).toBe(2)
  })

  it('umbral por defecto 1e-3: solo el cero exacto es muerto', () => {
    const stats = WeightInspector.inspect(buildNet())
    expect(stats[0].deadCount).toBe(0)
    expect(stats[1].deadCount).toBe(1)
  })
})

describe('WeightInspector.inspectAll (global)', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  it('agrega los 9 parámetros de toda la red: media 0, desviación √(5/12)', () => {
    const g = WeightInspector.inspectAll(buildNet(), 0.5)
    expect(g.totalParams).toBe(9)
    expect(g.mean).toBeCloseTo(0, 12)
    expect(g.std).toBeCloseTo(Math.sqrt(5 / 12), 12) // √(3.75/9); con n-1 daría √(0.46875)
    expect(g.min).toBe(-1)
    expect(g.max).toBe(1)
    expect(g.deadCount).toBe(3) // 2 de la capa 0 + 1 de la capa 1
  })

  it('con el umbral por defecto solo cuenta el cero exacto', () => {
    expect(WeightInspector.inspectAll(buildNet()).deadCount).toBe(1)
  })
})

describe('WeightInspector.print', () => {
  beforeEach(() => seedRandom())
  afterEach(() => {
    vi.restoreAllMocks()
    restoreRandom()
  })

  // Cada fila se separa en tokens: etiqueta, media, desviación, mín, máx, muertos, parámetros.
  // Valores esperados (umbral por defecto 1e-3):
  //   Capa 0: media 0, desv. √7/4 = 0.6614, mín -1, máx 1, muertos 0/6, 6 parámetros.
  //   Capa 1: media 0, desv. √(3/8) = 0.6124, mín -0.75, máx 0.75, muertos 1/3, 3 parámetros.
  //   Global: media 0, desv. √(5/12) = 0.6455, mín -1, máx 1, muertos 1/9, 9 parámetros.
  it('imprime una fila por capa y una global con los valores calculados a mano', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    WeightInspector.print(buildNet())

    const lines = spy.mock.calls.map((args) => String(args[0]))
    const rowTokens = (label: string) => {
      const row = lines.find((l) => l.startsWith(label))
      expect(row, `fila "${label}" ausente`).toBeDefined()
      return row!.trim().split(/\s+/)
    }

    expect(rowTokens('Layer 0')).toEqual(['Layer', '0', '0.0000', '0.6614', '-1.0000', '1.0000', '0/6', '6'])
    expect(rowTokens('Layer 1')).toEqual(['Layer', '1', '0.0000', '0.6124', '-0.7500', '0.7500', '1/3', '3'])
    expect(rowTokens('Global')).toEqual(['Global', '0.0000', '0.6455', '-1.0000', '1.0000', '1/9', '9'])
  })
})

describe('WeightInspector.countDeadReLUs', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  it('cuenta las neuronas que valen 0 en todas las muestras (activaciones a mano)', () => {
    // Filas = muestras, columnas = neuronas.
    //   neurona 0: 0, 0, 0 → siempre 0 → muerta.
    //   neurona 1: 0.5, 0.2, 0 → viva.
    //   neurona 2: 0, 0, 0.3 → viva.
    const acts = [
      [0, 0.5, 0],
      [0, 0.2, 0],
      [0, 0, 0.3],
    ]
    expect(WeightInspector.countDeadReLUs(acts)).toBe(1)
  })

  it('una neurona con alguna activación por encima del umbral no es muerta', () => {
    // Neurona 1 vale 0.5 en la primera muestra: viva aunque las demás sean 0.
    expect(WeightInspector.countDeadReLUs([[0, 0.5], [0, 0]])).toBe(1)
  })

  it('umbral por defecto 1e-6: activaciones por debajo cuentan como muertas', () => {
    // Neurona 0: 1e-7 y 0 → muerta. Neurona 1: 0.5 → viva.
    expect(WeightInspector.countDeadReLUs([[1e-7, 0.5], [0, 0.5]])).toBe(1)
  })

  it('un valor igual al umbral no es muerto (comparación estricta)', () => {
    // |0.5| < 0.5 es falso: la neurona vive aunque la segunda muestra valga 0.
    expect(WeightInspector.countDeadReLUs([[0.5], [0]], 0.5)).toBe(0)
  })

  it('compara la magnitud: una activación negativa grande no es muerta', () => {
    // Neurona 0: |-2| = 2 → viva. Neurona 1: 0 y 0 → muerta. Sin valor absoluto daría 2.
    expect(WeightInspector.countDeadReLUs([[-2, 0], [0, 0]])).toBe(1)
  })

  it('sin muestras devuelve 0', () => {
    expect(WeightInspector.countDeadReLUs([])).toBe(0)
  })
})
