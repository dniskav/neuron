import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LossPlotter } from '../src/LossPlotter'
import { restoreRandom, seedRandom } from './helpers/seed'

// LossPlotter es determinista (no usa Math.random). La semilla se fija igualmente
// en beforeEach/afterEach para cumplir la constitución IV.
//
// ─── Derivación a mano de la serie [1, 0.5, 0.25], valores por defecto ────────
//   width 60, height 15, title 'Loss Curve'
//   yLabelW = 8; plotW = max(4, 60 - 8 - 1) = 51; plotH = max(3, 15) = 15
//   min = 0.25, max = 1, range = 0.75, n = 3; epochs automáticos = [0, 1, 2]
//   columna = round(idx / (n - 1) · (plotW - 1)) = round(0, 25, 50) = 0, 25, 50
//   fila    = round((1 - (l - min) / range) · (plotH - 1)):
//               l = 1    → norm 1   → round(0)      = 0
//               l = 0.5  → norm 1/3 → round(9.333)  = 9
//               l = 0.25 → norm 0   → round(14)     = 14
//   etiquetas (toPrecision(4), padStart(6) + ' ┤') en la fila 0 (max = '1.000'),
//     la fila floor(15/2) = 7 (medio = 0.25 + 0.75/2 = 0.625 → '0.6250') y la 14 (min = '0.2500')
//   cada fila de la gráfica ocupa 8 (etiqueta) + 51 (celdas) + 1 ('│') = 60 = width
//   eje x: 7 espacios + '└' + 51 × '─' + '┘'
//   eje epoch: 8 espacios + '0' + (51 - 1 - 1 = 49) espacios + '2' + '  epoch'
//   pie: ' min=0.2500  max=1.000  last=0.2500  n=3'
//   total de líneas: 1 (título) + 15 (filas) + 1 (eje x) + 1 (epoch) + 1 (pie) = 19
//
// Nota: la línea del título mide 61 (un carácter más que las filas). Es cosmético y no
// se fija aquí; el test solo comprueba que empieza por '┌' y contiene el título.

const PLOT_W = 51
const TICK_LABELS: Record<number, string> = {
  0: ' 1.000 ┤',
  7: '0.6250 ┤',
  14: '0.2500 ┤',
}
// fila → columna de la marca '*' (derivado arriba)
const STAR_COL: Record<number, number> = { 0: 0, 9: 25, 14: 50 }
const NO_LABEL = '       │'

function cells(col: number): string {
  return ' '.repeat(col) + '*' + ' '.repeat(PLOT_W - 1 - col)
}

const EXPECTED_BODY: string[] = [
  ...Array.from({ length: 15 }, (_, r) => {
    const label = TICK_LABELS[r] ?? NO_LABEL
    const body = r in STAR_COL ? cells(STAR_COL[r]) : ' '.repeat(PLOT_W)
    return label + body + '│'
  }),
  '       └' + '─'.repeat(PLOT_W) + '┘',
  ' '.repeat(8) + '0' + ' '.repeat(PLOT_W - 2) + '2' + '  epoch',
  ' min=0.2500  max=1.000  last=0.2500  n=3',
]

function bodyOf(text: string): string[] {
  return text.split('\n').slice(1)
}

describe('LossPlotter', () => {
  beforeEach(() => seedRandom())
  afterEach(() => {
    vi.restoreAllMocks()
    restoreRandom()
  })

  describe('render: serie [1, 0.5, 0.25] con valores por defecto', () => {
    it('reproduce la gráfica derivada a mano (filas, ejes, epoch y pie)', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.5, 0.25])
      const lines = plotter.render().split('\n')

      expect(lines).toHaveLength(19)
      expect(lines.slice(1)).toEqual(EXPECTED_BODY)
    })

    it('título: empieza por ┌ y contiene el título configurado', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.5, 0.25])
      const title = plotter.render().split('\n')[0]

      expect(title.startsWith('┌')).toBe(true)
      expect(title).toContain(' Loss Curve ')
    })

    it('coloca la marca de cada pérdida en la fila y columna calculadas a mano', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.5, 0.25])
      const lines = plotter.render().split('\n')

      // filas de la gráfica = líneas 1..15; la columna es la posición tras la etiqueta de 8 caracteres
      const stars: Array<[number, number]> = []
      for (let r = 0; r < 15; r++) {
        const col = lines[1 + r].indexOf('*')
        if (col !== -1) stars.push([r, col - 8])
      }
      expect(stars).toEqual([
        [0, 0],
        [9, 25],
        [14, 50],
      ])
    })

    it('las etiquetas del eje Y son max (fila 0), medio (fila 7) y min (fila 14)', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.5, 0.25])
      const lines = plotter.render().split('\n')

      expect(lines[1].startsWith(' 1.000 ┤')).toBe(true)
      expect(lines[8].startsWith('0.6250 ┤')).toBe(true)
      expect(lines[15].startsWith('0.2500 ┤')).toBe(true)
    })

    it('cada fila de la gráfica y el eje x miden exactamente el ancho configurado (60)', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.5, 0.25])
      const lines = plotter.render().split('\n')

      for (let r = 1; r <= 15; r++) expect(lines[r]).toHaveLength(60)
      expect(lines[16]).toHaveLength(60)
    })
  })

  describe('add / addMultiple', () => {
    it('addMultiple incluye el primer valor (n = 3) y numera las épocas desde 0', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.5, 0.25])
      const lines = plotter.render().split('\n')

      expect(lines[18]).toBe(' min=0.2500  max=1.000  last=0.2500  n=3')
      // primera época '0', última '2': relleno = 51 - 1 - 1 = 49
      expect(lines[17]).toBe(' '.repeat(8) + '0' + ' '.repeat(49) + '2' + '  epoch')
    })

    it('add con epoch explícito usa esas épocas en el eje X', () => {
      const plotter = new LossPlotter()
      plotter.add(1, 10)
      plotter.add(0.5, 20)
      plotter.add(0.25, 30)
      const lines = plotter.render().split('\n')

      // primera época '10' (2 caracteres), última '30' (2 caracteres): relleno = 51 - 2 - 2 = 47
      expect(lines[17]).toBe(' '.repeat(8) + '10' + ' '.repeat(47) + '30' + '  epoch')
      expect(lines[18]).toBe(' min=0.2500  max=1.000  last=0.2500  n=3')
    })

    it('la última pérdida del pie es la última añadida, no la mínima', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.25, 0.5])
      const footer = plotter.render().split('\n').slice(-1)[0]

      expect(footer).toBe(' min=0.2500  max=1.000  last=0.5000  n=3')
    })
  })

  describe('render: ancho y alto configurados', () => {
    it('width 40 y height 6 producen 6 filas de 40 columnas, eje x de 40 y 10 líneas', () => {
      // plotW = 40 - 8 - 1 = 31; plotH = 6 (el último índice de fila es 5)
      // columnas: round(0, 0.5, 1) · 30 = 0, 15, 30
      // filas: round(0) = 0 ; round((1 - 1/3) · 5) = round(3.333) = 3 ; round(5) = 5
      const plotter = new LossPlotter({ width: 40, height: 6 })
      plotter.addMultiple([1, 0.5, 0.25])
      const lines = plotter.render().split('\n')

      expect(lines).toHaveLength(10)
      for (let r = 1; r <= 6; r++) expect(lines[r]).toHaveLength(40)
      expect(lines[7]).toHaveLength(40)

      const stars: Array<[number, number]> = []
      for (let r = 0; r < 6; r++) {
        const col = lines[1 + r].indexOf('*')
        if (col !== -1) stars.push([r, col - 8])
      }
      expect(stars).toEqual([
        [0, 0],
        [3, 15],
        [5, 30],
      ])
    })

    it('usa el título configurado', () => {
      const plotter = new LossPlotter({ title: 'Train' })
      plotter.addMultiple([1, 0.5])
      const title = plotter.render().split('\n')[0]

      expect(title.startsWith('┌ Train ')).toBe(true)
      expect(title).not.toContain('Loss Curve')
    })
  })

  describe('render: formato de números', () => {
    it('valores ≥ 1e4 usan notación exponencial con un decimal', () => {
      // _fmtNum: 20000.toExponential(1) = '2.0e+4'; 15000.toExponential(1) = '1.5e+4'
      const plotter = new LossPlotter()
      plotter.addMultiple([20000, 15000])
      const lines = plotter.render().split('\n')

      expect(lines[1].startsWith('2.0e+4 ┤')).toBe(true)
      expect(lines[15].startsWith('1.5e+4 ┤')).toBe(true)
      expect(lines[18]).toBe(' min=1.5e+4  max=2.0e+4  last=1.5e+4  n=2')
    })

    it('valores 0 < |n| < 1e-3 usan notación exponencial; el cero usa toPrecision', () => {
      // 0.0002.toExponential(1) = '2.0e-4'; 0.0001.toExponential(1) = '1.0e-4'
      const small = new LossPlotter()
      small.addMultiple([0.0002, 0.0001])
      expect(small.render().split('\n')[18]).toBe(' min=1.0e-4  max=2.0e-4  last=1.0e-4  n=2')

      // _fmtNum(0): |0| < 1e-3 pero n === 0 → toPrecision(4) = '0.000'; max 0.5 → '0.5000'
      const zero = new LossPlotter()
      zero.addMultiple([0.5, 0])
      expect(zero.render().split('\n')[18]).toBe(' min=0.000  max=0.5000  last=0.000  n=2')
    })
  })

  describe('render: sin datos', () => {
    it('sin pérdidas devuelve el aviso con el título', () => {
      expect(new LossPlotter().render()).toBe('[Loss Curve] — no data yet')
      expect(new LossPlotter({ title: 'T' }).render()).toBe('[T] — no data yet')
    })

    it('addMultiple([]) no añade datos', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([])
      expect(plotter.render()).toBe('[Loss Curve] — no data yet')
    })
  })

  describe('print', () => {
    it('imprime exactamente el texto de render (un solo console.log)', () => {
      const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 0.5, 0.25])
      plotter.print()

      expect(spy).toHaveBeenCalledTimes(1)
      expect(spy).toHaveBeenCalledWith(plotter.render())
      expect(bodyOf(String(spy.mock.calls[0][0]))).toEqual(EXPECTED_BODY)
    })
  })

  describe('reset', () => {
    it('vacía los datos: render vuelve al aviso y una nueva pérdida numera desde la época 0', () => {
      const plotter = new LossPlotter()
      plotter.addMultiple([1, 2])
      plotter.reset()
      expect(plotter.render()).toBe('[Loss Curve] — no data yet')

      plotter.add(3)
      const lines = plotter.render().split('\n')
      expect(lines[18]).toBe(' min=3.000  max=3.000  last=3.000  n=1')
      // una sola época '0': primera = última = '0', relleno = max(1, 51 - 1 - 1) = 49
      expect(lines[17]).toBe(' '.repeat(8) + '0' + ' '.repeat(49) + '0  epoch')
    })
  })
})
