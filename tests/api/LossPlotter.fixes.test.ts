import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { LossPlotter } from '../../src/LossPlotter'
import { restoreRandom, seedRandom } from '../helpers/seed'

// T053: correcciones de LossPlotter hallazgos de T012. LossPlotter es determinista;
// la semilla se fija por la constitución IV.
//
// Derivación a mano (valores por defecto: width 60, height 15 → plotH 15, fila media = floor(15/2) = 7,
// que es la línea 8 del render porque la línea 0 es el título):
//   serie constante [0.5]: min = max = 0.5 → media real = (0.5 + 0.5) / 2 = 0.5 → '0.5000'
//   serie constante [2, 2]: media = 2 → '2.000'; serie [0, 0]: media = 0 → '0.000'
//   serie 0..124999: min = 0 → '0.000'; max = 124999 ≥ 1e4 → toExponential(1) = '1.2e+5';
//     last = 124999 → '1.2e+5'; n = 125000

// Golden del render ANTERIOR al cambio de T053 (series no constantes); actualizado en T020 por la alineación del marco.
const GOLDEN: Array<{ o: { width?: number; height?: number; title?: string }; s: number[]; r: string }> = [
  {"o":{},"s":[1,0.5,0.25],"r":"┌ Loss Curve ──────────────────────────────────────────────┐\n 1.000 ┤*                                                  │\n       │                                                   │\n       │                                                   │\n       │                                                   │\n       │                                                   │\n       │                                                   │\n       │                                                   │\n0.6250 ┤                                                   │\n       │                                                   │\n       │                         *                         │\n       │                                                   │\n       │                                                   │\n       │                                                   │\n       │                                                   │\n0.2500 ┤                                                  *│\n       └───────────────────────────────────────────────────┘\n        0                                                 2  epoch\n min=0.2500  max=1.000  last=0.2500  n=3"}, // golden actualizado en T020 por la alineación del marco (longitud de la línea de título)
  {"o":{"width":40,"height":6,"title":"A"},"s":[3,1,2,0.5,0.7,0.1],"r":"┌ A ───────────────────────────────────┐\n 3.000 ┤*                              │\n       │                               │\n       │            *                  │\n 1.550 ┤      *                        │\n       │                  *     *      │\n0.1000 ┤                              *│\n       └───────────────────────────────┘\n        0                             5  epoch\n min=0.1000  max=3.000  last=0.1000  n=6"}, // golden actualizado en T020 por la alineación del marco (longitud de la línea de título)
  {"o":{"width":20,"height":4},"s":[0.9,0.1,0.4,0.3,0.8,0.05,0.6],"r":"┌ Loss Curve ──────┐\n 0.9000 ┤*     *   │\n        │         *│\n 0.4750 ┤   * *    │\n0.05000 ┤  *     * │\n        └──────────┘\n         0        6  epoch\n min=0.05000  max=0.9000  last=0.6000  n=7"}, // golden actualizado en T020 por la alineación del marco (etiqueta por 0.05000: una columna menos de dibujo)
  {"o":{"width":80,"height":9,"title":"Wide"},"s":[5,4,3.5,3,2.9,2,1.5,1.2,1.1,1],"r":"┌ Wide ────────────────────────────────────────────────────────────────────────┐\n 5.000 ┤*                                                                      │\n       │                                                                       │\n       │        *                                                              │\n       │                *                                                      │\n 3.000 ┤                       *       *                                       │\n       │                                                                       │\n       │                                       *                               │\n       │                                               *                       │\n 1.000 ┤                                                      *       *       *│\n       └───────────────────────────────────────────────────────────────────────┘\n        0                                                                     9  epoch\n min=1.000  max=5.000  last=1.000  n=10"}, // golden actualizado en T020 por la alineación del marco (longitud de la línea de título)
]

function midLabel(plotter: LossPlotter): string {
  return plotter.render().split('\n')[8].slice(0, 8)
}

describe('LossPlotter correcciones (T053)', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  describe('golden: el render de series no constantes no cambia', () => {
    for (const [i, g] of GOLDEN.entries()) {
      it(`caso ${i + 1}: ${g.s.length} valores, opciones ${JSON.stringify(g.o)}`, () => {
        const plotter = new LossPlotter(g.o)
        plotter.addMultiple(g.s)
        expect(plotter.render()).toBe(g.r)
      })
    }
  })

  describe('etiqueta media con serie constante', () => {
    it('[0.5] → la etiqueta media es la media real 0.5000 (antes salía 1.000)', () => {
      const p = new LossPlotter()
      p.add(0.5)
      expect(midLabel(p)).toBe('0.5000 ┤')
    })

    it('[2, 2] → 2.000 y [0, 0] → 0.000', () => {
      const a = new LossPlotter()
      a.addMultiple([2, 2])
      expect(midLabel(a)).toBe('2.000 ┤'.padStart(8))

      const b = new LossPlotter()
      b.addMultiple([0, 0])
      expect(midLabel(b)).toBe('0.000 ┤'.padStart(8))
    })

    it('el eje es monótono: con [0.5] las tres etiquetas valen 0.5000', () => {
      const p = new LossPlotter()
      p.add(0.5)
      const lines = p.render().split('\n')
      expect(lines[1].slice(0, 8)).toBe('0.5000 ┤')
      expect(lines[8].slice(0, 8)).toBe('0.5000 ┤')
      expect(lines[15].slice(0, 8)).toBe('0.5000 ┤')
    })
  })

  describe('series largas (sin desbordar la pila)', () => {
    it('125000 valores: render no lanza y el pie lleva min, max y last correctos', () => {
      const p = new LossPlotter()
      const data = Array.from({ length: 125000 }, (_, i) => i)
      p.addMultiple(data)
      const footer = p.render().split('\n').slice(-1)[0]
      expect(footer).toBe(' min=0.000  max=1.2e+5  last=1.2e+5  n=125000')
    })

    it('mínimo en el último valor y máximo en el primero (extremos del bucle)', () => {
      const p = new LossPlotter()
      p.addMultiple(Array.from({ length: 125000 }, (_, i) => 125000 - i))
      const footer = p.render().split('\n').slice(-1)[0]
      // max = 125000 → '1.3e+5' (1.25e5.toExponential(1)); min = 1 → '1.000'
      expect(footer).toBe(` min=1.000  max=${(125000).toExponential(1)}  last=1.000  n=125000`)
    })
  })

  describe('add / addMultiple lanzan con valores no finitos', () => {
    for (const bad of [NaN, Infinity, -Infinity]) {
      it(`add(${bad}) lanza LossPlotter.add y no modifica el estado`, () => {
        const p = new LossPlotter()
        expect(() => p.add(bad)).toThrow(/LossPlotter\.add:/)
        expect(p.render()).toBe('[Loss Curve] — no data yet')
      })

      it(`addMultiple([1, ${bad}]) lanza LossPlotter.addMultiple y no añade ni el 1`, () => {
        const p = new LossPlotter()
        expect(() => p.addMultiple([1, bad])).toThrow(/LossPlotter\.addMultiple:/)
        expect(p.render()).toBe('[Loss Curve] — no data yet')
      })
    }

    it('un valor inválido no corrompe datos válidos previos', () => {
      const p = new LossPlotter()
      p.addMultiple([1, 2])
      expect(() => p.add(NaN)).toThrow()
      expect(() => p.addMultiple([3, Infinity])).toThrow()
      expect(p.render().split('\n').slice(-1)[0]).toBe(' min=1.000  max=2.000  last=2.000  n=2')
    })
  })

  describe('constructor valida width y height', () => {
    const invalid = [0, -5, NaN, Infinity, 10.5]
    for (const bad of invalid) {
      it(`width ${bad} lanza`, () => {
        expect(() => new LossPlotter({ width: bad })).toThrow(/LossPlotter\.constructor:.*width/)
      })
      it(`height ${bad} lanza`, () => {
        expect(() => new LossPlotter({ height: bad })).toThrow(/LossPlotter\.constructor:.*height/)
      })
    }

    it('enteros positivos pequeños se aceptan y se recortan a los mínimos (plotW 4, plotH 3)', () => {
      const p = new LossPlotter({ width: 5, height: 1 })
      p.addMultiple([1, 0])
      const lines = p.render().split('\n')
      // plotH = max(3, 1) = 3 filas; título + 3 filas + eje x + epoch + pie = 7
      expect(lines).toHaveLength(7)
    })

    it('sin opciones y con opciones parciales sigue funcionando', () => {
      expect(() => new LossPlotter()).not.toThrow()
      expect(() => new LossPlotter({ title: 'x' })).not.toThrow()
      expect(() => new LossPlotter({ width: 30 })).not.toThrow()
    })
  })
})
