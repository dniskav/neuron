// T037 (feature 006, FR-009 / US5): LogisticRegression.train, SoftmaxRegression.train,
// LinearRegression.fitGD y LinearRegression.fitNormal validan con validateDataset (src/Validation.ts).
//
// - fitNormal antes no validaba filas irregulares ni NaN: devolvía pesos NaN. Ahora lanza.
// - Las cuatro validan ANTES de modificar pesos o sesgo: un modelo ya ajustado queda intacto
//   tras el error (se compara weights/bias antes y después del throw).
// - Los mensajes existentes se conservan (los fija tests/api/Regression.incremental.test.ts).
//
// Valores a mano: y = 2x + 1 con X = [0, 1, 2], y = [1, 3, 5] → Xa = [[0,1],[1,1],[2,1]],
// XᵀX = [[5,3],[3,3]], Xᵀy = [13, 9], det = 6, W = (1/6)·[[3,−3],[−3,5]]·[13, 9] = [2, 1].
import { describe, it, expect } from 'vitest'
import { LogisticRegression, SoftmaxRegression } from '../../src/LogisticRegression'
import { LinearRegression } from '../../src/LinearRegression'

// ─── fitNormal: entradas inválidas lanzan y no tocan el modelo ───────────────

describe('fitNormal: datos inválidos lanzan con validateDataset', () => {
  const cases: { name: string; X: unknown; y: unknown; msg: RegExp }[] = [
    {
      name: 'fila de X más corta',
      X: [[1, 2], [3]],
      y: [1, 2],
      msg: /LinearRegression\.fitNormal: X\[1\]: expected array of length 2, got 1/,
    },
    {
      name: 'fila de X más larga',
      X: [[1, 2], [3, 4, 5]],
      y: [1, 2],
      msg: /LinearRegression\.fitNormal: X\[1\]: expected array of length 2, got 3/,
    },
    {
      name: 'X[0] no es un array',
      X: [1, [2]],
      y: [1, 2],
      msg: /LinearRegression\.fitNormal: X\[0\]: expected array, got number/,
    },
    {
      name: 'filas sin rasgos',
      X: [[], []],
      y: [1, 2],
      msg: /LinearRegression\.fitNormal: X rows need at least 1 feature/,
    },
    {
      name: 'NaN en X',
      X: [[1, 2], [3, NaN]],
      y: [1, 2],
      msg: /LinearRegression\.fitNormal: X\[1\]: invalid value at index 1: NaN/,
    },
    {
      name: 'Infinity en X',
      X: [[Infinity, 2], [3, 4]],
      y: [1, 2],
      msg: /LinearRegression\.fitNormal: X\[0\]: invalid value at index 0: Infinity/,
    },
    {
      name: 'NaN en y',
      X: [[0], [1], [2]],
      y: [1, NaN, 5],
      msg: /LinearRegression\.fitNormal: y: invalid value at index 1: NaN/,
    },
    {
      name: 'Infinity en y',
      X: [[0], [1]],
      y: [Infinity, 1],
      msg: /LinearRegression\.fitNormal: y: invalid value at index 0: Infinity/,
    },
    {
      name: 'y no numérica',
      X: [[0], [1]],
      y: ['1', 0],
      msg: /LinearRegression\.fitNormal: y: invalid value at index 0: 1/,
    },
  ]

  for (const c of cases) {
    it(`${c.name} lanza y deja un modelo ya ajustado intacto`, () => {
      const m = new LinearRegression()
      m.weights = [0.1, 0.2, 0.3]
      const before = [...m.weights]
      expect(() => m.fitNormal(c.X as number[][], c.y as number[])).toThrow(c.msg)
      expect(m.weights).toEqual(before)
    })

    it(`${c.name} lanza en un modelo nuevo y lo deja sin ajustar`, () => {
      const m = new LinearRegression()
      expect(() => m.fitNormal(c.X as number[][], c.y as number[])).toThrow(c.msg)
      expect(m.weights).toEqual([])
    })
  }

  it('tras un error, un ajuste válido da la solución exacta W = [2, 1]', () => {
    const m = new LinearRegression()
    expect(() => m.fitNormal([[1, 2], [3]], [1, 2])).toThrow()
    m.fitNormal([[0], [1], [2]], [1, 3, 5])
    expect(m.weights[0]).toBeCloseTo(2, 9)
    expect(m.weights[1]).toBeCloseTo(1, 9)
  })

  it('X e y de longitud distinta conservan el mensaje "elements" de fitNormal', () => {
    expect(() => new LinearRegression().fitNormal([[1], [2]], [1])).toThrow(
      'LinearRegression.fitNormal: X has 2 rows but y has 1 elements',
    )
  })
})

// ─── Un error de datos no cambia pesos ni sesgo (los tres de fitGD/train) ───

describe('un error de datos deja intacto un modelo existente (pesos y sesgo)', () => {
  it('LogisticRegression.train: y fuera de [0, 1] y X irregular', () => {
    const m = new LogisticRegression()
    m.weights = [0.1, 0.2]
    m.bias = 0.3
    expect(() => m.train([[1, 2], [3, 4]], [0, 1.5], 0.1, 1)).toThrow(
      'LogisticRegression.train: y[1] = 1.5 is outside [0, 1]',
    )
    expect(() => m.train([[-1, 2], [3]], [0, 1], 0.1, 1)).toThrow(
      'LogisticRegression.train: X[1]: expected array of length 2, got 1',
    )
    expect(m.weights).toEqual([0.1, 0.2])
    expect(m.bias).toBe(0.3)
  })

  it('SoftmaxRegression.train: etiqueta no entera, negativa y X irregular', () => {
    const m = new SoftmaxRegression()
    m.weights = [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
    m.biases = [0.5, 0.6]
    const w = m.weights.map((r) => [...r])
    const b = [...m.biases]
    const X = [[1, 2], [3, 4]]
    expect(() => m.train(X, [0, 1.5], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: label y[1] = 1.5 must be an integer >= 0',
    )
    expect(() => m.train(X, [-1, 0], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: label y[0] = -1 must be an integer >= 0',
    )
    expect(() => m.train([[1, 2], [3]], [0, 1], 0.1, 1)).toThrow(
      'SoftmaxRegression.train: X[1]: expected array of length 2, got 1',
    )
    expect(m.weights).toEqual(w)
    expect(m.biases).toEqual(b)
  })

  it('LinearRegression.fitGD: X irregular e y con NaN', () => {
    const m = new LinearRegression()
    m.weights = [0.1, 0.2, 0.3]
    expect(() => m.fitGD([[1, 2], [3]], [1, 2], 0.1, 1)).toThrow(
      'LinearRegression.fitGD: X[1]: expected array of length 2, got 1',
    )
    expect(() => m.fitGD([[1, 2], [3, 4]], [1, NaN], 0.1, 1)).toThrow(
      'LinearRegression.fitGD: y: invalid value at index 1: NaN',
    )
    expect(m.weights).toEqual([0.1, 0.2, 0.3])
  })
})
