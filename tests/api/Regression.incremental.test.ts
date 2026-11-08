// T014 (feature 005, Q7): entrenamiento incremental y validacion de datos en
// LogisticRegression.train, SoftmaxRegression.train y LinearRegression.fitGD.
//
// - SoftmaxRegression.train y LinearRegression.fitGD continuan desde los pesos existentes
//   si su forma coincide con los datos (como ya hacia LogisticRegression.train) y solo
//   inicializan si no hay pesos o la forma no coincide. Softmax con modelo existente y
//   etiqueta >= K lanza. Sin modelo, K = max(y) + 1 (bucle, no Math.max(...y)).
// - Las tres validan filas de X (longitud constante, finitas) y y (finita); Softmax exige
//   etiquetas enteras >= 0; Logistic exige y en [0, 1]. Todo ANTES de tocar pesos o Math.random.
//
// Golden (modelo SIN entrenar, 3 epocas, semilla de tests/setup.ts): constantes capturadas con
// el src de la base ANTES del cambio, tolerancia 1e-9; demuestran que el orden de consumo de
// Math.random y la trayectoria no cambian.
//
// Test de 'aprende' (Softmax, 3 clases separables, pasos de 1 epoca): validado con lr x0.7/x1.3
// y 5 semillas (1..5). Cifras medidas (peor razon final/inicial; umbral 0.5, margen exigido
// >= 30 % => <= 0.35): con 20 pasos el peor caso (lr x0.7, 5 semillas) es 0.049 (umbral usado 0.35).
import { describe, it, expect, vi, afterEach } from 'vitest'
import { LogisticRegression, SoftmaxRegression } from '../../src/LogisticRegression'
import { LinearRegression } from '../../src/LinearRegression'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

afterEach(() => {
  vi.restoreAllMocks()
})

function expectClose(actual: number[], expected: number[], tol = 1e-9) {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) {
    expect(Math.abs(actual[i] - expected[i]), `indice ${i}`).toBeLessThanOrEqual(tol)
  }
}

// ─── Golden: modelo sin entrenar (identico a la base) ────────────────────────

const GX = [
  [1, 0.5, -0.2],
  [0.3, -1, 0.8],
  [-0.7, 0.4, 0.1],
  [0.9, 0.9, -0.5],
  [-0.2, -0.6, 0.7],
]

describe('golden: modelo sin entrenar sigue la trayectoria de la base', () => {
  it('LogisticRegression.train', () => {
    const m = new LogisticRegression()
    const losses = m.train(GX, [1, 0, 0, 1, 0], 0.3, 3)
    expectClose(losses, [0.8237701864701601, 0.5168969805550299, 0.35875897871212425])
    expectClose(m.weights, [1.1444512082713532, 0.7165854315020591, -0.17747291341615923])
    expect(Math.abs(m.bias - -0.46985139111545826)).toBeLessThanOrEqual(1e-9)
  })

  it('SoftmaxRegression.train', () => {
    const m = new SoftmaxRegression()
    const losses = m.train(GX, [0, 2, 1, 2, 0], 0.3, 3)
    expectClose(losses, [1.1573556366820084, 1.0628220166424058, 1.0069687525669304])
    expectClose(m.weights[0], [0.3006829093624269, -0.447339423795317, 0.8768958545284842])
    expectClose(m.weights[1], [-1.2754320928807337, 0.05989926960672038, 0.12352888458138947])
    expectClose(m.weights[2], [0.3639608917621621, -0.4737597176854799, 0.08834622936992782])
    expectClose(m.biases, [-0.01791098733389032, -0.23135034422472778, 0.24926133155861804])
  })

  it('LinearRegression.fitGD', () => {
    const m = new LinearRegression()
    const losses = m.fitGD(GX, [1.5, -0.5, 0.25, 2, -1], 0.1, 3)
    expectClose(losses, [1.5125, 1.0083655639999998, 0.6782867828770117])
    expectClose(m.weights, [
      0.31241333616000005, 0.37361937312000004, -0.24040464624000002, 0.20824846720000004,
    ])
  })

  it('Softmax: el modelo sin entrenar consume Math.random 3 x 3 veces, una por peso', () => {
    const spy = vi.spyOn(Math, 'random')
    new SoftmaxRegression().train(GX, [0, 2, 1, 2, 0], 0.3, 2)
    expect(spy).toHaveBeenCalledTimes(9)
  })
})

// ─── Igualdad exacta train(5) + train(5) == train(10) desde pesos a mano ─────

const IX = [
  [1, 0.5],
  [0.3, -1],
  [-0.7, 0.4],
  [0.9, 0.9],
  [-0.2, -0.6],
  [0.5, -0.3],
]

describe('train(5) + train(5) == train(10) desde pesos fijados a mano (exacto)', () => {
  it('LogisticRegression', () => {
    const y = [1, 0, 0, 1, 0, 1]
    const a = new LogisticRegression()
    a.weights = [0.2, -0.4]
    a.bias = 0.1
    const b = new LogisticRegression()
    b.weights = [0.2, -0.4]
    b.bias = 0.1
    const la = [...a.train(IX, y, 0.3, 5), ...a.train(IX, y, 0.3, 5)]
    const lb = b.train(IX, y, 0.3, 10)
    expect(a.weights).toEqual(b.weights)
    expect(a.bias).toEqual(b.bias)
    expect(la).toEqual(lb)
  })

  it('SoftmaxRegression', () => {
    const y = [0, 2, 1, 2, 0, 1]
    const init = () => {
      const m = new SoftmaxRegression()
      m.weights = [
        [0.2, -0.4],
        [-0.3, 0.25],
        [0.5, 0.1],
      ]
      m.biases = [0.1, 0, -0.1]
      return m
    }
    const a = init()
    const b = init()
    const spy = vi.spyOn(Math, 'random')
    const la = [...a.train(IX, y, 0.3, 5), ...a.train(IX, y, 0.3, 5)]
    const lb = b.train(IX, y, 0.3, 10)
    expect(spy).not.toHaveBeenCalled()
    expect(a.weights).toEqual(b.weights)
    expect(a.biases).toEqual(b.biases)
    expect(la).toEqual(lb)
  })

  it('LinearRegression.fitGD', () => {
    const y = [1, -2, 0.5, 3, -1, 0.25]
    const a = new LinearRegression()
    a.weights = [0.2, -0.4, 0.1]
    const b = new LinearRegression()
    b.weights = [0.2, -0.4, 0.1]
    const spy = vi.spyOn(Math, 'random')
    const la = [...a.fitGD(IX, y, 0.05, 5), ...a.fitGD(IX, y, 0.05, 5)]
    const lb = b.fitGD(IX, y, 0.05, 10)
    expect(spy).not.toHaveBeenCalled()
    expect(a.weights).toEqual(b.weights)
    expect(la).toEqual(lb)
  })
})

// ─── Pesos a mano + train parten desde ahi ───────────────────────────────────

describe('pesos a mano + train: parten desde ellos (no los pisan)', () => {
  it('LinearRegression.fitGD: un paso a mano desde w=[1], b=0', () => {
    // X=[[1]], y=[3], lr=0.25: residuo = 1 - 3 = -2; dW = (2/1)*[1,1]*(-2) = [-4,-4]
    // w = 1 - 0.25*(-4) = 2 ; b = 0 - 0.25*(-4) = 1.
    const m = new LinearRegression()
    m.weights = [1, 0]
    const losses = m.fitGD([[1]], [3], 0.25, 1)
    expect(losses).toEqual([4])
    expect(m.weights[0]).toBeCloseTo(2, 12)
    expect(m.weights[1]).toBeCloseTo(1, 12)
  })

  it('SoftmaxRegression.train: un paso a mano desde W=[[1],[0]], b=0', () => {
    // x=[1], y=0, lr=0.5. scores=[1,0]; P0=e/(e+1), P1=1/(e+1). delta=[P0-1, P1]=[-P1, P1].
    // W0 = 1 + 0.5*P1 ; W1 = 0 - 0.5*P1 ; b0 = 0.5*P1 ; b1 = -0.5*P1.
    const p1 = 1 / (Math.E + 1)
    const m = new SoftmaxRegression()
    m.weights = [[1], [0]]
    m.biases = [0, 0]
    const spy = vi.spyOn(Math, 'random')
    m.train([[1]], [0], 0.5, 1)
    expect(spy).not.toHaveBeenCalled()
    expect(m.weights[0][0]).toBeCloseTo(1 + 0.5 * p1, 12)
    expect(m.weights[1][0]).toBeCloseTo(-0.5 * p1, 12)
    expect(m.biases[0]).toBeCloseTo(0.5 * p1, 12)
    expect(m.biases[1]).toBeCloseTo(-0.5 * p1, 12)
  })

  it('SoftmaxRegression.train: con K=3 en el modelo y solo etiquetas 0 y 1 en los datos, K sigue siendo 3', () => {
    const m = new SoftmaxRegression()
    m.weights = [[0.1], [0.2], [0.3]]
    m.biases = [0, 0, 0]
    m.train([[1], [2]], [0, 1], 0.1, 1)
    expect(m.weights).toHaveLength(3)
    expect(m.biases).toHaveLength(3)
    expect(m.predictProba([1])).toHaveLength(3)
  })

  it('LogisticRegression.train: parte de los pesos a mano (comportamiento previo)', () => {
    const m = new LogisticRegression()
    m.weights = [0, 0]
    m.bias = 0
    m.train([[2, -1]], [1], 0.2, 1)
    expect(m.weights[0]).toBeCloseTo(0.2, 12)
    expect(m.weights[1]).toBeCloseTo(-0.1, 12)
    expect(m.bias).toBeCloseTo(0.1, 12)
  })
})

// ─── Cambio de nFeatures: reinicia con la forma nueva ────────────────────────

describe('cambio de nFeatures reinicia con la forma nueva', () => {
  it('LinearRegression.fitGD: pesos de 2 rasgos (3 valores), datos de 1 rasgo', () => {
    const m = new LinearRegression()
    m.weights = [5, 5, 5]
    m.fitGD([[1], [2], [3]], [2, 4, 6], 0.1, 1)
    expect(m.weights).toHaveLength(2)
    // Reinicia a cero (no a 5): un paso con lr 0.1 desde 0 queda lejos de 5.
    expect(Math.abs(m.weights[0])).toBeLessThan(2)
  })

  it('LinearRegression.fitGD: pesos de 1 rasgo, datos de 3 rasgos', () => {
    const m = new LinearRegression()
    m.weights = [5, 5]
    m.fitGD(GX, [1, 2, 3, 4, 5], 0.1, 1)
    expect(m.weights).toHaveLength(4)
  })

  it('SoftmaxRegression.train: modelo de 2 rasgos, datos de 3 rasgos; K por los datos', () => {
    const m = new SoftmaxRegression()
    m.weights = [
      [1, 1],
      [1, 1],
    ]
    m.biases = [0, 0]
    // La etiqueta 2 no lanza: la forma no coincide, asi que se reinicia con K = 3.
    m.train(GX, [0, 2, 1, 2, 0], 0.1, 1)
    expect(m.weights).toHaveLength(3)
    m.weights.forEach((w) => expect(w).toHaveLength(3))
    expect(m.biases).toHaveLength(3)
  })

  it('SoftmaxRegression.train: biases de longitud distinta de K no coinciden y se reinicia', () => {
    const m = new SoftmaxRegression()
    m.weights = [[1], [1]]
    m.biases = [0]
    m.train([[1], [2]], [0, 1], 0.1, 1)
    expect(m.biases).toHaveLength(2)
    expect(m.weights).toHaveLength(2)
  })

  it('LogisticRegression.train: pesos de 2 rasgos, datos de 3 rasgos', () => {
    const m = new LogisticRegression()
    m.weights = [5, 5]
    m.bias = 5
    m.train(GX, [1, 0, 0, 1, 0], 0.1, 1)
    expect(m.weights).toHaveLength(3)
    expect(Math.abs(m.bias)).toBeLessThan(2)
  })
})

// ─── K = max(y) + 1 ──────────────────────────────────────────────────────────

describe('SoftmaxRegression: K sin modelo', () => {
  it('K = max(y) + 1', () => {
    const m = new SoftmaxRegression()
    m.train([[1], [2]], [0, 3], 0.1, 1)
    expect(m.weights).toHaveLength(4)
    expect(m.biases).toHaveLength(4)
  })

  it('K se calcula con un bucle: y de 400000 elementos no desborda la pila', () => {
    const n = 400000
    const row = [1]
    const X = new Array<number[]>(n).fill(row)
    const y = new Array<number>(n).fill(0)
    y[n - 1] = 1
    const m = new SoftmaxRegression()
    m.train(X, y, 0.001, 1)
    expect(m.weights).toHaveLength(2)
  }, 60000)

  it('sin modelo y con una sola clase lanza (necesita >= 2 clases)', () => {
    expect(() => new SoftmaxRegression().train([[1], [2]], [0, 0], 0.1, 1)).toThrow(
      /SoftmaxRegression\.train: need at least 2 classes/
    )
  })
})

// ─── Validacion de datos, con texto ──────────────────────────────────────────

describe('validacion de datos en train / fitGD', () => {
  const okX = [[1, 2], [3, 4]]

  type Fresh = { name: string; run: (X: unknown, y: unknown) => unknown; okY: number[] }
  const cases: Fresh[] = [
    {
      name: 'LogisticRegression.train',
      run: (X, y) => new LogisticRegression().train(X as number[][], y as number[], 0.1, 1),
      okY: [0, 1],
    },
    {
      name: 'SoftmaxRegression.train',
      run: (X, y) => new SoftmaxRegression().train(X as number[][], y as number[], 0.1, 1),
      okY: [0, 1],
    },
    {
      name: 'LinearRegression.fitGD',
      run: (X, y) => new LinearRegression().fitGD(X as number[][], y as number[], 0.1, 1),
      okY: [0.5, 1.5],
    },
  ]

  for (const c of cases) {
    describe(c.name, () => {
      const re = (rest: string) => new RegExp(c.name.replace('.', '\\.') + rest)

      it('fila de X mas corta lanza con la longitud esperada y recibida', () => {
        expect(() => c.run([[1, 2], [3]], c.okY)).toThrow(re('.*expected array of length 2, got 1'))
      })

      it('fila de X mas larga lanza', () => {
        expect(() => c.run([[1, 2], [3, 4, 5]], c.okY)).toThrow(re('.*expected array of length 2, got 3'))
      })

      it('fila de X que no es un array lanza', () => {
        expect(() => c.run([[1, 2], 7], c.okY)).toThrow(re('.*expected array'))
      })

      it('X con NaN lanza e indica la posicion', () => {
        expect(() => c.run([[1, 2], [3, NaN]], c.okY)).toThrow(re('.*invalid value at index 1'))
      })

      it('X con Infinity lanza', () => {
        expect(() => c.run([[Infinity, 2], [3, 4]], c.okY)).toThrow(re('.*invalid value at index 0'))
      })

      it('X con una fila sin rasgos lanza', () => {
        expect(() => c.run([[], []], c.okY)).toThrow(re('.*at least 1 feature'))
      })

      it('y con NaN lanza', () => {
        expect(() => c.run(okX, [0, NaN])).toThrow(re('.*invalid value at index 1'))
      })

      it('y con Infinity lanza', () => {
        expect(() => c.run(okX, [Infinity, 1])).toThrow(re('.*invalid value at index 0'))
      })

      it('y que no es numerica (string) lanza', () => {
        expect(() => c.run(okX, ['1', 0])).toThrow(re('.*invalid value at index 0'))
      })

      it('un error de datos no consume Math.random', () => {
        const spy = vi.spyOn(Math, 'random')
        expect(() => c.run([[1, 2], [3, NaN]], c.okY)).toThrow()
        expect(spy).not.toHaveBeenCalled()
      })
    })
  }

  it('LogisticRegression: y fuera de [0, 1] lanza (1.5 y -0.1) y 0, 0.5 y 1 valen', () => {
    const run = (y: number[]) => new LogisticRegression().train(okX, y, 0.1, 1)
    expect(() => run([0, 1.5])).toThrow(/LogisticRegression\.train: y\[1\] = 1\.5 is outside \[0, 1\]/)
    expect(() => run([-0.1, 1])).toThrow(/LogisticRegression\.train: y\[0\] = -0\.1 is outside \[0, 1\]/)
    expect(() => run([0, 1])).not.toThrow()
    expect(() => run([0.5, 1])).not.toThrow()
  })

  it('SoftmaxRegression: etiqueta no entera lanza', () => {
    expect(() => new SoftmaxRegression().train(okX, [0, 1.5], 0.1, 1)).toThrow(
      /SoftmaxRegression\.train: label y\[1\] = 1\.5 must be an integer >= 0/
    )
  })

  it('SoftmaxRegression: etiqueta negativa lanza', () => {
    expect(() => new SoftmaxRegression().train(okX, [-1, 1], 0.1, 1)).toThrow(
      /SoftmaxRegression\.train: label y\[0\] = -1 must be an integer >= 0/
    )
  })

  it('SoftmaxRegression con modelo existente y etiqueta >= K lanza y no cambia nada', () => {
    const m = new SoftmaxRegression()
    m.weights = [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
    m.biases = [0.5, 0.6]
    const w = m.weights.map((r) => [...r])
    const b = [...m.biases]
    expect(() => m.train(okX, [0, 2], 0.1, 1)).toThrow(
      /SoftmaxRegression\.train: label y\[1\] = 2 is >= 2 classes of the existing model/
    )
    expect(m.weights).toEqual(w)
    expect(m.biases).toEqual(b)
  })

  it('un error de datos deja intacto un modelo existente (los tres)', () => {
    const lg = new LogisticRegression()
    lg.weights = [0.1, 0.2]
    lg.bias = 0.3
    expect(() => lg.train([[1, 2], [3, NaN]], [0, 1], 0.1, 1)).toThrow()
    expect(lg.weights).toEqual([0.1, 0.2])
    expect(lg.bias).toBe(0.3)

    const ln = new LinearRegression()
    ln.weights = [0.1, 0.2, 0.3]
    expect(() => ln.fitGD([[1, 2], [3, 4]], [1, NaN], 0.1, 1)).toThrow()
    expect(ln.weights).toEqual([0.1, 0.2, 0.3])

    const sm = new SoftmaxRegression()
    sm.weights = [[0.1, 0.2], [0.3, 0.4]]
    sm.biases = [0, 0]
    expect(() => sm.train([[1, 2], [3]], [0, 1], 0.1, 1)).toThrow()
    expect(sm.weights).toEqual([[0.1, 0.2], [0.3, 0.4]])
  })
})

// ─── Aprende en pasos incrementales ──────────────────────────────────────────

describe('SoftmaxRegression aprende en pasos incrementales de 1 epoca', () => {
  const X = [
    [1, 0],
    [0.9, 0.1],
    [0, 1],
    [0.1, 0.9],
    [-1, -1],
    [-0.9, -1.1],
  ]
  const Y = [0, 0, 1, 1, 2, 2]
  const LEARN_STEPS = 20
  const meanCE = (m: SoftmaxRegression) => {
    let s = 0
    for (let i = 0; i < X.length; i++) s -= Math.log(m.predictProba(X[i])[Y[i]])
    return s / X.length
  }

  for (const lrScale of [0.7, 1, 1.3]) {
    for (const seed of [1, 2, 3, 4, 5]) {
      it(`lr x${lrScale}, semilla ${seed}: la CE baja por debajo de 0.35 de la inicial`, () => {
        seedRandom(seed)
        const m = new SoftmaxRegression()
        m.train(X, Y, 1e-12, 1) // punto inicial aleatorio de la semilla (paso practicamente nulo)
        expectLearns({
          loss: () => meanCE(m),
          step: () => void m.train(X, Y, 0.5 * lrScale, 1),
          steps: LEARN_STEPS,
          ratio: 0.35, // umbral de la constitucion 0.5 con margen >= 30 %
        })
      })
    }
  }
})
