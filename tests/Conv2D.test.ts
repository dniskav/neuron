// T032 (feature 006, coverage): tests de comportamiento de Conv2D (src/Conv2D.ts).
//
// Valores esperados calculados A MANO, no copiados de la salida del codigo.
// Convencion del modulo (correlacion, el kernel NO se voltea):
//   out[h][w][f] = bias[f] + sum_{kh,kw,c} K[f][kh][kw][c] · padded[h·s + kh][w·s + kw][c]
// Orden de getWeights/setWeights: kernels en orden [f][kh][kw][c], luego biases[f].
// Padding 'same': top = left = floor((k−1)/2); bottom = right = (k−1) − top; ceros fuera.
// Forma de salida: valid → floor((H − k)/s) + 1 ; same → ceil(H/s).
//
// No se repiten aqui: oraculo de gradiente y dInput (tests/oracle/Conv2D.oracle.test.ts)
// ni el relleno 'same' con kernel par frente a la replica antigua (tests/api/Conv2D.padding.test.ts).
// La semilla de Math.random la fija tests/setup.ts en cada test.
import { describe, it, expect } from 'vitest'
import { Conv2D } from '../src/Conv2D'

// Filas [H][W] de un solo canal → [H][W][1]
const cube1 = (rows: number[][]): number[][][] => rows.map((r) => r.map((v) => [v]))

describe('Conv2D forward valid a mano', () => {
  it('kernel 2×2 asimetrico sobre 3×3: correlacion por ventanas', () => {
    // X = [[1,2,3],[4,5,6],[7,8,9]], K = [[2,−1],[0,3]], bias = −1, valid → 2×2
    // Pesos (orden f,kh,kw,c): 2, −1, 0, 3 ; bias −1.
    // ventana (0,0) = [[1,2],[4,5]]: 2·1 + (−1)·2 + 0·4 + 3·5 − 1 = 2 − 2 + 0 + 15 − 1 = 14
    // ventana (0,1) = [[2,3],[5,6]]: 2·2 + (−1)·3 + 0·5 + 3·6 − 1 = 4 − 3 + 0 + 18 − 1 = 18
    // ventana (1,0) = [[4,5],[7,8]]: 2·4 + (−1)·5 + 0·7 + 3·8 − 1 = 8 − 5 + 0 + 24 − 1 = 26
    // ventana (1,1) = [[5,6],[8,9]]: 2·5 + (−1)·6 + 0·8 + 3·9 − 1 = 10 − 6 + 0 + 27 − 1 = 30
    // Si se volteara el kernel (convolucion), daria [[8,12],[20,24]]: distinto.
    const conv = new Conv2D(3, 3, 1, 2, 1, { padding: 'valid' })
    conv.setWeights([2, -1, 0, 3, -1])
    const y = conv.forward(cube1([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    expect(y).toEqual([
      [[14], [18]],
      [[26], [30]],
    ])
  })

  it('el sesgo se suma en cada posicion (kernel nulo → salida = bias)', () => {
    // K = 0 → cada posicion vale solo bias: f0 = 3, f1 = −1. Salida 2×2×2.
    const conv = new Conv2D(3, 3, 1, 2, 2, { padding: 'valid' })
    conv.setWeights([0, 0, 0, 0, 0, 0, 0, 0, 3, -1])
    const y = conv.forward(cube1([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    expect(y).toEqual([
      [[3, -1], [3, -1]],
      [[3, -1], [3, -1]],
    ])
  })

  it('varios canales de entrada y de salida', () => {
    // Entrada 2×2×2: X[0][0]=[1,2], X[0][1]=[3,4], X[1][0]=[5,6], X[1][1]=[7,8]. Kernel 2×2 valid → 1×1×2.
    // Filtro 0: K[0][0]=[1,0], K[0][1]=[0,1], K[1][0]=[2,0], K[1][1]=[0,−1], bias 0.
    //   X00·[1,0] = 1 ; X01·[0,1] = 4 ; X10·[2,0] = 10 ; X11·[0,−1] = −8 → 1+4+10−8 = 7
    // Filtro 1: K[0][0]=[0,1], K[0][1]=[1,1], K[1][0]=[0,0], K[1][1]=[3,0], bias 2.
    //   X00·[0,1] = 2 ; X01·[1,1] = 7 ; X10·[0,0] = 0 ; X11·[3,0] = 21 → 30, +2 = 32
    // Pesos: f0 (1,0, 0,1, 2,0, 0,−1), f1 (0,1, 1,1, 0,0, 3,0), biases (0, 2): 16 + 2 = 18.
    const conv = new Conv2D(2, 2, 2, 2, 2, { padding: 'valid' })
    conv.setWeights([1, 0, 0, 1, 2, 0, 0, -1, 0, 1, 1, 1, 0, 0, 3, 0, 0, 2])
    const x = [
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ]
    expect(conv.forward(x)).toEqual([[[7, 32]]])
  })
})

describe('Conv2D relleno y forma de salida', () => {
  it("'same' con k=2 rellena solo abajo y derecha (top = left = 0)", () => {
    // K = [[0,0],[0,1]] → out[h][w] = X[h+1][w+1] (0 si se sale del tablero).
    // X = [[1,2,3],[4,5,6],[7,8,9]]:
    //   (0,0)=X[1][1]=5 ; (0,1)=X[1][2]=6 ; (0,2)=fuera=0
    //   (1,0)=X[2][1]=8 ; (1,1)=X[2][2]=9 ; (1,2)=fuera=0
    //   fila 2 = fuera = 0,0,0
    // Si el relleno estuviera arriba/izquierda (invertido), daria X tal cual: [[1,2,3],...]
    const conv = new Conv2D(3, 3, 1, 2, 1, { padding: 'same' })
    expect(conv.outputShape()).toEqual([3, 3, 1])
    conv.setWeights([0, 0, 0, 1, 0])
    const y = conv.forward(cube1([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    expect(y).toEqual([
      [[5], [6], [0]],
      [[8], [9], [0]],
      [[0], [0], [0]],
    ])
  })

  it("'same' con k=3 y todo unos cuenta las celdas dentro del tablero", () => {
    // Ventana 3×3 de unos, centrada: esquina ve 2×2 = 4 celdas, borde ve 2×3 = 6, centro ve 3×3 = 9.
    // Salida: [[4,6,4],[6,9,6],[4,6,4]].
    const conv = new Conv2D(3, 3, 1, 3, 1, { padding: 'same' })
    conv.setWeights([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    const ones = cube1([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    expect(conv.forward(ones)).toEqual([
      [[4], [6], [4]],
      [[6], [9], [6]],
      [[4], [6], [4]],
    ])
  })

  it('forma de salida valid y same segun H, k y stride (formulas a mano)', () => {
    // valid: floor((H−k)/s) + 1 ; same: ceil(H/s). Cada fila: [H, W, k, s] → valid, same.
    // (5,6,3,1): valid 3×4, same 5×6
    // (5,6,3,2): valid floor(2/2)+1=2 × floor(3/2)+1=2 → 2×2 ; same ceil(5/2)=3 × ceil(6/2)=3 → 3×3
    // (4,4,2,2): valid floor(2/2)+1=2 × 2 → 2×2 ; same ceil(4/2)=2 × 2 → 2×2
    // (7,7,3,3): valid floor(4/3)+1=2 × 2 → 2×2 ; same ceil(7/3)=3 × 3 → 3×3
    const F = 2
    const cases: Array<{ H: number; W: number; k: number; s: number; valid: number[]; same: number[] }> = [
      { H: 5, W: 6, k: 3, s: 1, valid: [3, 4, F], same: [5, 6, F] },
      { H: 5, W: 6, k: 3, s: 2, valid: [2, 2, F], same: [3, 3, F] },
      { H: 4, W: 4, k: 2, s: 2, valid: [2, 2, F], same: [2, 2, F] },
      { H: 7, W: 7, k: 3, s: 3, valid: [2, 2, F], same: [3, 3, F] },
    ]
    for (const c of cases) {
      const label = `H=${c.H} W=${c.W} k=${c.k} s=${c.s}`
      const cv = new Conv2D(c.H, c.W, 1, c.k, F, { padding: 'valid', stride: c.s })
      const cs = new Conv2D(c.H, c.W, 1, c.k, F, { padding: 'same', stride: c.s })
      expect(cv.outputShape(), `valid ${label}`).toEqual(c.valid)
      expect(cs.outputShape(), `same ${label}`).toEqual(c.same)
      // forward real: las dimensiones de la salida coinciden con la formula
      const x = Array.from({ length: c.H }, () =>
        Array.from({ length: c.W }, () => [1]),
      )
      const yv = cv.forward(x)
      expect([yv.length, yv[0].length, yv[0][0].length], `valid fwd ${label}`).toEqual(c.valid)
      const ys = cs.forward(x)
      expect([ys.length, ys[0].length, ys[0][0].length], `same fwd ${label}`).toEqual(c.same)
    }
  })
})

describe('Conv2D stride', () => {
  it('stride 2 sobre 4×4 con ventana 2×2 de unos: suma de bloques no solapados', () => {
    // X[h][w] = 4h + w + 1 (1..16). Ventanas que empiezan en (0,0),(0,2),(2,0),(2,2):
    //   bloque (0,0): 1+2+5+6 = 14 ; bloque (0,2): 3+4+7+8 = 22
    //   bloque (2,0): 9+10+13+14 = 46 ; bloque (2,2): 11+12+15+16 = 54
    // Forma valid: floor((4−2)/2)+1 = 2 → 2×2. Si el stride se ignorara daria 3×3 de ventanas solapadas.
    const conv = new Conv2D(4, 4, 1, 2, 1, { padding: 'valid', stride: 2 })
    expect(conv.outputShape()).toEqual([2, 2, 1])
    conv.setWeights([1, 1, 1, 1, 0])
    const x = cube1([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ])
    expect(conv.forward(x)).toEqual([
      [[14], [22]],
      [[46], [54]],
    ])
  })
})

describe('Conv2D pesos getWeights / setWeights', () => {
  it('orden de serializacion: kernels [f][kh][kw][c] y luego biases', () => {
    // 2 filtros, kernel 2×2, 2 canales → 8 valores por filtro: 2·2·2·2 = 16, + 2 sesgos = 18.
    // Valores 1..18 en orden: kernels[f][kh][kw][c] = índice f·8 + kh·4 + kw·2 + c (+1).
    //   kernels[0][0][1] = índices 2,3 → [3,4] ; kernels[1][1][1] = índices 14,15 → [15,16]
    //   kernels[1][0][0] = índices 8,9 → [9,10] ; biases = índices 16,17 → [17,18]
    const conv = new Conv2D(3, 3, 2, 2, 2, { padding: 'valid' })
    const W = Array.from({ length: 18 }, (_, i) => i + 1)
    expect(conv.getWeights()).toHaveLength(18)
    conv.setWeights(W)
    expect(conv.getWeights()).toEqual(W)
    expect(conv.kernels[0][0][1]).toEqual([3, 4])
    expect(conv.kernels[1][0][0]).toEqual([9, 10])
    expect(conv.kernels[1][1][1]).toEqual([15, 16])
    expect(conv.biases).toEqual([17, 18])
  })

  it('getWeights devuelve una copia: mutarla no altera el modelo', () => {
    const conv = new Conv2D(3, 3, 1, 2, 1, { padding: 'valid' })
    conv.setWeights([2, -1, 0, 3, -1])
    const w = conv.getWeights()
    w[0] = 999
    w[4] = 999
    expect(conv.getWeights()).toEqual([2, -1, 0, 3, -1])
    expect(conv.forward(cube1([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))[0][0]).toEqual([14])
  })
})

describe('Conv2D errores explicitos', () => {
  it('dimensiones, canales o filtros no positivos lanzan', () => {
    expect(() => new Conv2D(0, 3, 1, 2, 1)).toThrow(/dimensions and filters must be positive/)
    expect(() => new Conv2D(3, 0, 1, 2, 1)).toThrow(/dimensions and filters must be positive/)
    expect(() => new Conv2D(3, 3, 0, 2, 1)).toThrow(/dimensions and filters must be positive/)
    expect(() => new Conv2D(3, 3, 1, 2, 0)).toThrow(/dimensions and filters must be positive/)
  })

  it('kernelSize no positivo lanza (escalar o par)', () => {
    expect(() => new Conv2D(3, 3, 1, 0, 1)).toThrow(/kernelSize must be positive/)
    expect(() => new Conv2D(3, 3, 1, [2, 0], 1)).toThrow(/kernelSize must be positive/)
  })

  it('forward con altura distinta lanza con la altura esperada', () => {
    const conv = new Conv2D(3, 3, 1, 2, 1)
    expect(() => conv.forward(cube1([[1, 2, 3], [4, 5, 6]]))).toThrow(
      /Conv2D\.forward: expected height 3, got 2/,
    )
  })

  it('forward con anchura distinta lanza con la anchura esperada', () => {
    const conv = new Conv2D(3, 3, 1, 2, 1)
    expect(() => conv.forward(cube1([[1, 2], [3, 4], [5, 6]]))).toThrow(
      /Conv2D\.forward: expected width 3, got 2/,
    )
  })

  it('backward sin forward previo lanza', () => {
    const conv = new Conv2D(3, 3, 1, 2, 1)
    expect(() => conv.backward([[[0]]], 0.1)).toThrow(/Conv2D\.backward: call forward\(\) first/)
  })
})
