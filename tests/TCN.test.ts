// Tests de comportamiento de CausalConv1D y TCN (feature 006, T028).
//
// Complementan tests/oracle/TCN.oracle.test.ts (gradiente y "aprende" con MSE): aquí se
// fijan causalidad, campo receptivo, respuesta al impulso y residual con valores calculados
// a mano (la derivación de cada cifra va en un comentario) y se comprueba que train()
// baja la pérdida con 5 semillas y lr x0.7 / x1.3.
//
// Convención de src/TCN.ts: kernels[f][k][c] multiplica a la entrada en la posición
// t + k·d del signal con relleno izquierdo, es decir, a x[t - (K-1-k)·d].
// kernels[f][K-1] es el peso del instante actual y kernels[f][0] el del más antiguo.
//
// Cifras de cabecera del aprendizaje (ver el último test):
//   tarea y[t] = x[t-1], T = 16, TCN(1, 4, 2, 2, 1), steps = 300, lr base 0.3,
//   5 semillas x lr {0.7, 1.3} = {0.21, 0.39}; umbral de expectLearns 0.5; el peor caso
//   debe quedar por debajo de 0.7 · 0.5 = 0.35 (margen del 30 %). Peor caso medido: 0.012.
//   Zona inestable: con lr base 0.5 el ×1.3 (0.65) deja la semilla 4 en 0.73 y lr 1.0 hace
//   crecer la pérdida; por eso la base es 0.3 y no 0.5. El gradiente de esta configuración
//   (con nivel sin residual) sí sigue -dL/dw (coseno 1.0 frente a diferencias finitas).
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { CausalConv1D, TCN } from '../src/TCN'
import { seedRandom, restoreRandom, withSeed } from './helpers/seed'
import { expectLearns } from './helpers/learns'

beforeEach(() => seedRandom())
afterEach(() => restoreRandom())

// Fija pesos de salida de TCN (campos privados: se accede con `as any`, como en el oráculo).
function setOutput(net: TCN, W: number[][], B: number[]): void {
  const n = net as any
  n._outputW = W
  n._outputB = B
}

// TCN con todos los kernels a 1 y sesgos a 0, proyección identidad (W = 1, b = 0).
// Con entrada constante 1 y residual en cada nivel, las activaciones se calculan a mano
// (ver el test "entrada constante").
function allOnesTcn(): TCN {
  const net = new TCN(1, 1, 2, 3, 1)
  for (const layer of net.layers) {
    layer.kernels = [[[1], [1]]]
    layer.biases = [0]
  }
  setOutput(net, [[1]], [0])
  return net
}

describe('CausalConv1D', () => {
  it('rechaza dimensiones no positivas', () => {
    expect(() => new CausalConv1D(0, 1, 1, 1)).toThrow('all dimensions must be positive')
    expect(() => new CausalConv1D(1, 0, 1, 1)).toThrow('all dimensions must be positive')
    expect(() => new CausalConv1D(1, 1, 0, 1)).toThrow('all dimensions must be positive')
    expect(() => new CausalConv1D(1, 1, 1, 0)).toThrow('all dimensions must be positive')
  })

  it('backward sin forward previo lanza un error explícito', () => {
    const conv = new CausalConv1D(1, 1, 2, 1)
    expect(() => conv.backward([[0]], 0.1)).toThrow('call forward() first')
  })

  it('respuesta al impulso con K=2 y d=2: el kernel aparece desplazado', () => {
    // y[t] = b + a·x[t-2] + c·x[t], con a = 2 (k=0), c = 3 (k=1), b = 0.5 y x = impulso en t=0.
    // t=0: 0.5 + 3·1 = 3.5 | t=1: 0.5 | t=2: 0.5 + 2·1 = 2.5 | t=3..5: 0.5
    const conv = new CausalConv1D(1, 1, 2, 2)
    conv.kernels = [[[2], [3]]]
    conv.biases = [0.5]
    const x = [[1], [0], [0], [0], [0], [0]]
    expect(conv.forward(x)).toEqual([[3.5], [0.5], [2.5], [0.5], [0.5], [0.5]])
  })

  it('campo receptivo con K=3 y d=2: un impulso solo llega a t = s+4, s+2, s', () => {
    // Pesos a 1 y sesgo 0: y[t] = x[t-4] + x[t-2] + x[t].
    // Impulso en s = 3: aparece en t = 3 (x[t]), t = 5 (x[t-2]) y t = 7 (x[t-4]).
    // Campo receptivo de una capa: (K-1)·d + 1 = 5 posiciones; t = 2 y t = 8 quedan a 0.
    const conv = new CausalConv1D(1, 1, 3, 2)
    conv.kernels = [[[1], [1], [1]]]
    conv.biases = [0]
    const x = Array.from({ length: 10 }, (_, t) => [t === 3 ? 1 : 0])
    expect(conv.forward(x).map((r) => r[0])).toEqual([0, 0, 0, 1, 0, 1, 0, 1, 0, 0])
  })

  it('causal: cambiar x[s] no altera las salidas en t < s y Δy[s] = Σ del cambio', () => {
    // Pesos a 1, sesgo 0, K=3, d=2, 2 canales. x[s] entra en y[s] con k = K-1 (peso 1):
    // cambiar el canal 0 en +7 y el canal 1 en -4 mueve y[s][f] en 7 - 4 = 3 para cada f.
    const conv = new CausalConv1D(2, 2, 3, 2)
    conv.kernels = [
      [[1, 1], [1, 1], [1, 1]],
      [[1, 1], [1, 1], [1, 1]],
    ]
    conv.biases = [0, 0]
    const x = Array.from({ length: 8 }, (_, t) => [t * 0.1 + 0.2, 0.5 - t * 0.3])
    const y = conv.forward(x)
    for (let s = 0; s < x.length; s++) {
      const alt = x.map((r) => [...r])
      alt[s] = [alt[s][0] + 7, alt[s][1] - 4]
      const y2 = conv.forward(alt)
      for (let t = 0; t < s; t++) expect(y2[t]).toEqual(y[t])
      for (let f = 0; f < 2; f++) expect(y2[s][f] - y[s][f]).toBeCloseTo(3, 9)
    }
  })
})

describe('TCN', () => {
  it('rechaza levels y outputSize no positivos', () => {
    expect(() => new TCN(1, 1, 2, 0, 1)).toThrow('levels must be positive')
    expect(() => new TCN(1, 1, 2, 2, 0)).toThrow('outputSize must be positive')
  })

  it('dilaciones 1, 2, 4 y campo receptivo (K-1)·(2^L-1)+1', () => {
    // K=2, L=3: 1·(2³-1)+1 = 8. K=3, L=2: 2·(2²-1)+1 = 7. K=2, L=1: 1·1+1 = 2.
    const net = new TCN(1, 1, 2, 3, 1)
    expect(net.layers.map((l) => l.dilation)).toEqual([1, 2, 4])
    expect(net.receptiveField).toBe(8)
    expect(new TCN(1, 1, 3, 2, 1).receptiveField).toBe(7)
    expect(new TCN(1, 1, 2, 1, 1).receptiveField).toBe(2)
  })

  it('residual: con convoluciones a cero la salida es la propia entrada', () => {
    // inputChannels = channels = 1 en ambos niveles → cada nivel suma su entrada a relu(0) = 0.
    // Proyección W = 1, b = 0 → y[t] = x[t]. Incluye negativos: el residual no pasa por la ReLU.
    const net = new TCN(1, 1, 1, 2, 1)
    for (const layer of net.layers) {
      layer.kernels = [[[0]]]
      layer.biases = [0]
    }
    setOutput(net, [[1]], [0])
    const x = [[1.5], [-2], [0], [3]]
    expect(net.forward(x)).toEqual(x)
  })

  it('residual se suma DESPUÉS de la ReLU (orden real, T058): TCN(1,1,1,1,1) con x = -2 → -2', () => {
    // Derivación a mano, un solo nivel (K = 1, d = 1), kernel 0, sesgo 0, proyección W = 1, b = 0:
    //   conv     = 0·(-2) + 0          = 0
    //   relu     = relu(0)             = 0
    //   residual = relu + x            = 0 + (-2) = -2   (suma después de la ReLU)
    //   proyección = 1·(-2) + 0        = -2
    // Si el residual se sumara ANTES de la ReLU: relu(0 + (-2)) = 0 → salida 0, y el test fallaría.
    const net = new TCN(1, 1, 1, 1, 1)
    net.layers[0].kernels = [[[0]]]
    net.layers[0].biases = [0]
    setOutput(net, [[1]], [0])
    expect(net.forward([[-2]])).toEqual([[-2]])
  })

  it('sin residual en el primer nivel (inputChannels ≠ channels): la entrada no llega a la salida', () => {
    // TCN(2, 3, 1, 1, 1): 2 ≠ 3 → no hay residual. Convolución a cero → relu(0) = 0 → y = b = 0.5.
    const net = new TCN(2, 3, 1, 1, 1)
    net.layers[0].kernels = [[[0, 0]], [[0, 0]], [[0, 0]]]
    net.layers[0].biases = [0, 0, 0]
    setOutput(net, [[1, 1, 1]], [0.5])
    expect(net.forward([[3, -7], [1, 2]])).toEqual([[0.5], [0.5]])
  })

  it('respuesta al impulso con ReLU y sesgo (1 → 2 canales, K=2, un nivel)', () => {
    // Sin residual (1 ≠ 2). Kernel f=0: [2, 3]; f=1: [-1, 4]; sesgos 0.
    // conv_f[t] = k0·x[t-1] + k1·x[t]. Con x = impulso en t=0:
    //   conv_0 = [3, 2, 0, 0]  → relu = [3, 2, 0, 0]
    //   conv_1 = [4, -1, 0, 0] → relu = [4, 0, 0, 0]  (el -1 se anula)
    // y[t] = relu_0[t] + relu_1[t] + 0.5 = [7.5, 2.5, 0.5, 0.5]
    const net = new TCN(1, 2, 2, 1, 1)
    net.layers[0].kernels = [
      [[2], [3]],
      [[-1], [4]],
    ]
    net.layers[0].biases = [0, 0]
    setOutput(net, [[1, 1]], [0.5])
    expect(net.forward([[1], [0], [0], [0]])).toEqual([[7.5], [2.5], [0.5], [0.5]])
  })

  it('entrada constante: valores a mano y ventana completa en t = 7', () => {
    // Kernels a 1, sesgos 0, x = 1 para todo t, residual en los 3 niveles (1 = 1).
    // Nivel 0 (d=1): h0[t] = relu(x[t-1] + x[t]) + x[t] → [2, 3, 3, ...]
    // Nivel 1 (d=2): h1[t] = relu(h0[t-2] + h0[t]) + h0[t] → [4, 6, 8, 9, 9, ...]
    // Nivel 2 (d=4): h2[t] = relu(h1[t-4] + h1[t]) + h1[t] → [8, 12, 16, 18, 22, 24, 26, 27, ...]
    // A partir de t = 7 el campo receptivo (8 posiciones) cabe entero en la secuencia: 27.
    const net = allOnesTcn()
    const y = net.forward(Array.from({ length: 8 }, () => [1]))
    expect(y.map((r) => r[0])).toEqual([8, 12, 16, 18, 22, 24, 26, 27])
  })

  it('campo receptivo: y[11] no depende de x[3] (distancia 8) y sí de x[4] (distancia 7)', () => {
    // Con RF = 8, y[11] depende de x[4..11]. y[11] = 27 (ver test anterior, ventana completa).
    const net = allOnesTcn()
    const x = Array.from({ length: 12 }, () => [1])
    const y = net.forward(x)
    expect(y[11][0]).toBe(27)

    const far = x.map((r) => [...r])
    far[3] = [100]
    expect(net.forward(far)[11]).toEqual(y[11])

    // Pesos y activaciones positivos: subir x[4] sube y[11] estrictamente.
    const near = x.map((r) => [...r])
    near[4] = [100]
    expect(net.forward(near)[11][0]).toBeGreaterThan(27)
  })

  it('causal: cambiar x[s] no altera las salidas en t < s (TCN con pesos aleatorios)', () => {
    const net = new TCN(2, 3, 2, 3, 2)
    const x = Array.from({ length: 10 }, (_, t) => [(t % 3) - 1, 0.5 - 0.1 * t])
    const y = net.forward(x)
    for (let s = 0; s < x.length; s++) {
      const alt = x.map((r) => [...r])
      alt[s] = [alt[s][0] + 5, alt[s][1] - 3]
      const y2 = net.forward(alt)
      for (let t = 0; t < s; t++) expect(y2[t]).toEqual(y[t])
      // No trivial: la salida en s sí cambia.
      expect(y2[s]).not.toEqual(y[s])
    }
  })

  it('train baja la pérdida: predecir x[t-1] con 5 semillas y lr x0.7 / x1.3', () => {
    // Tarea causal: y[t] = x[t-1] (y[0] = 0). Pérdida = MSE sobre T·outputSize.
    // Para cada semilla y factor de lr se exige expectLearns (final < 0.5 · inicial) y
    // además que el peor caso de final/inicial quede por debajo de 0.7 · 0.5 = 0.35.
    const T = 16
    const LR = 0.3
    const STEPS = 300
    const x = Array.from({ length: T }, (_, t) => [Math.sin(1.3 * t + 0.4)])
    const target = Array.from({ length: T }, (_, t) => [t === 0 ? 0 : x[t - 1][0]])
    const lossOf = (net: TCN): number => {
      const y = net.forward(x)
      let s = 0
      y.forEach((r, t) => {
        s += (r[0] - target[t][0]) ** 2
      })
      return s / T
    }

    let worst = 0
    for (const seed of [1, 2, 3, 4, 5]) {
      for (const f of [0.7, 1.3]) {
        withSeed(seed, () => {
          const net = new TCN(1, 4, 2, 2, 1)
          const r = expectLearns({
            loss: () => lossOf(net),
            step: () => {
              net.train(x, target, LR * f)
            },
            steps: STEPS,
            ratio: 0.5,
          })
          worst = Math.max(worst, r.ratio)
        })
      }
    }
    expect(worst).toBeLessThan(0.7 * 0.5)
  })
})
