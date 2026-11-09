import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { HopfieldNetwork } from '../src/HopfieldNetwork'
import { seedRandom, restoreRandom } from './helpers/seed'

// ─── Derivaciones a mano (N = 4 salvo que se indique) ────────────────────────
//
// Regla de Hebb: W_ij = (1/N)·Σ_p p_i·p_j, W_ii = 0, simétrica.
//
// Patrón p = [+1, −1, +1, +1]. Pares i<j:
//   (0,1): (1/4)(+1)(−1) = −1/4    (0,2): (1/4)(+1)(+1) = +1/4
//   (0,3): (1/4)(+1)(+1) = +1/4    (1,2): (1/4)(−1)(+1) = −1/4
//   (1,3): (1/4)(−1)(+1) = −1/4    (2,3): (1/4)(+1)(+1) = +1/4
//
// Energía E(s) = −½ Σ_ij W_ij s_i s_j = −Σ_{i<j} W_ij s_i s_j (W simétrica).
//   E(p)  = −Σ_{i<j} (1/4)·(p_i p_j)² = −6/4 = −1.5
//   s = p con el bit 0 invertido = [−1, −1, +1, +1]:
//     s_i s_j por par: (0,1)=+1, (0,2)=−1, (0,3)=−1, (1,2)=−1, (1,3)=−1, (2,3)=+1
//     Σ_{i<j} W_ij s_i s_j = −1/4 −1/4 −1/4 +1/4 +1/4 +1/4 = 0  →  E = 0
//
// Recall de s (bit 0 invertido), campo local de la neurona 0:
//   h_0 = W_01·s_1 + W_02·s_2 + W_03·s_3 = (−1/4)(−1) + (1/4)(+1) + (1/4)(+1) = 3/4 > 0  →  s_0 = +1
//   h_1 = (−1/4)(−1) + (−1/4)(+1) + (−1/4)(+1) = −1/4  →  s_1 = −1 (ya correcto)
//   h_2 = (+1/4)(−1) + (−1/4)(−1) + (+1/4)(+1) = +1/4  →  s_2 = +1 (ya correcto)
//   h_3 = (+1/4)(−1) + (−1/4)(−1) + (+1/4)(+1) = +1/4  →  s_3 = +1 (ya correcto)
//   Resultado independiente del orden: p.
//
// Dos patrones p = [+1, −1, +1, +1] y q = [+1, +1, −1, +1]:
//   W_03 = (1·1 + 1·1)/4 = +1/2;   W_12 = ((−1)(+1) + (+1)(−1))/4 = −1/2
//   W_01 = ((+1)(−1) + (+1)(+1))/4 = 0;  W_02 = ((+1)(+1) + (+1)(−1))/4 = 0
//   W_13 = ((−1)(+1) + (+1)(+1))/4 = 0;  W_23 = ((+1)(+1) + (−1)(+1))/4 = 0
//   E(p) = −(W_03·1·1 + W_12·(−1)·(+1)) = −(1/2 + 1/2) = −1
//
// Caso N = 2 con p = [+1, −1]: W_01 = (+1)(−1)/2 = −1/2.
//   E(s) = −W_01 s_0 s_1 = (1/2)·s_0·s_1.
//   Desde [+1, +1]: E = +1/2. Los dos puntos fijos son [+1, −1] y [−1, +1], con E = −1/2.
//   Recall asíncrono llega a uno de ellos (depende del orden); la síncrona oscila entre [−1,−1] y [+1,+1].

describe('HopfieldNetwork', () => {
  beforeEach(() => {
    seedRandom(42)
    // El aviso de capacidad (T052) saltaría con N pequeño; se silencia para no ensuciar la salida.
    vi.spyOn(console, 'warn').mockImplementation(() => {})
  })
  afterEach(() => {
    restoreRandom()
    vi.restoreAllMocks()
  })

  describe('constructor', () => {
    it('creates an n×n zero matrix and no stored patterns', () => {
      const net = new HopfieldNetwork(4)
      expect(net.n).toBe(4)
      expect(net.storedPatterns).toBe(0)
      expect(net.weights).toHaveLength(4)
      for (const row of net.weights) {
        expect(row).toEqual([0, 0, 0, 0])
      }
    })

    it.each([0, -3, 1.5, NaN, Infinity])('rejects n = %s', (n) => {
      expect(() => new HopfieldNetwork(n)).toThrow('n must be a positive integer')
    })
  })

  describe('store (Hebbian rule)', () => {
    it('builds W_ij = p_i·p_j / N for one pattern, with zero diagonal', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      // Valores a mano: ver cabecera (pares i<j: −1/4, +1/4, +1/4, −1/4, −1/4, +1/4)
      expect(net.weights).toEqual([
        [0, -0.25, 0.25, 0.25],
        [-0.25, 0, -0.25, -0.25],
        [0.25, -0.25, 0, 0.25],
        [0.25, -0.25, 0.25, 0],
      ])
    })

    it('keeps W symmetric and the diagonal at zero', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      for (let i = 0; i < 4; i++) {
        expect(net.weights[i][i]).toBe(0)
        for (let j = 0; j < 4; j++) {
          expect(net.weights[i][j]).toBe(net.weights[j][i])
        }
      }
    })

    it('accumulates the Hebbian sum over two patterns', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      net.store([1, 1, -1, 1])
      // Valores a mano: W_03 = +1/2, W_12 = −1/2, resto de pares = 0 (ver cabecera)
      expect(net.weights[0][3]).toBeCloseTo(0.5, 12)
      expect(net.weights[3][0]).toBeCloseTo(0.5, 12)
      expect(net.weights[1][2]).toBeCloseTo(-0.5, 12)
      expect(net.weights[2][1]).toBeCloseTo(-0.5, 12)
      expect(net.weights[0][1]).toBeCloseTo(0, 12)
      expect(net.weights[0][2]).toBeCloseTo(0, 12)
      expect(net.weights[1][3]).toBeCloseTo(0, 12)
      expect(net.weights[2][3]).toBeCloseTo(0, 12)
      expect(net.storedPatterns).toBe(2)
    })

    it('rejects a pattern of the wrong length', () => {
      const net = new HopfieldNetwork(4)
      expect(() => net.store([1, -1, 1])).toThrow('does not match network size 4')
    })

    it.each([
      ['0', [1, 0, 1, 1]],
      ['2', [1, 2, 1, 1]],
      ['NaN', [1, NaN, 1, 1]],
    ])('rejects a non-bipolar value (%s)', (_label, pattern) => {
      const net = new HopfieldNetwork(4)
      expect(() => net.store(pattern)).toThrow('pattern values must be +1 or -1')
    })

    it('leaves weights and counter untouched when the pattern is rejected', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      const before = JSON.stringify(net.weights)
      expect(() => net.store([1, 0, 1, 1])).toThrow()
      expect(JSON.stringify(net.weights)).toBe(before)
      expect(net.storedPatterns).toBe(1)
    })
  })

  describe('recall', () => {
    it('leaves a stored pattern unchanged (fixed point)', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      expect(net.recall([1, -1, 1, 1])).toEqual([1, -1, 1, 1])
    })

    it('restores a stored pattern with one bit inverted', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      // Entrada: p con el bit 0 invertido. Campo h_0 = 3/4 → recupera +1 (ver cabecera).
      expect(net.recall([-1, -1, 1, 1])).toEqual([1, -1, 1, 1])
    })

    it('gives the same result over several seeds (independent of update order)', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      for (let seed = 1; seed <= 5; seed++) {
        seedRandom(seed)
        expect(net.recall([-1, -1, 1, 1])).toEqual([1, -1, 1, 1])
      }
    })

    it('does not mutate the input and returns a new array', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      const input = [-1, -1, 1, 1]
      const out = net.recall(input)
      expect(input).toEqual([-1, -1, 1, 1])
      expect(out).not.toBe(input)
    })

    it('with maxIter 0 returns a copy of the input without updating', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      expect(net.recall([-1, -1, 1, 1], 0)).toEqual([-1, -1, 1, 1])
    })

    it('keeps a neuron unchanged when its local field is exactly 0', () => {
      // N = 1: W = [[0]], el campo es siempre 0 → el estado no cambia (ni a +1).
      const net = new HopfieldNetwork(1)
      net.store([1])
      expect(net.recall([-1])).toEqual([-1])
      expect(net.recall([1])).toEqual([1])
    })

    it('converges to a fixed point (stable state), not an oscillation', () => {
      // N = 2, p = [+1, −1]: desde [+1, +1] el resultado es [+1, −1] o [−1, +1]
      // (ambos con E = −1/2), según el orden de actualización.
      const net = new HopfieldNetwork(2)
      net.store([1, -1])
      for (let seed = 1; seed <= 5; seed++) {
        seedRandom(seed)
        const out = net.recall([1, 1])
        const isStable = out.every((s, i) => {
          const h = net.weights[i][0] * out[0] + net.weights[i][1] * out[1]
          return h === 0 || Math.sign(h) === s
        })
        expect(isStable).toBe(true)
        expect([[1, -1], [-1, 1]]).toContainEqual(out)
      }
    })

    it('rejects an input of the wrong length', () => {
      const net = new HopfieldNetwork(4)
      expect(() => net.recall([1, -1])).toThrow('does not match network size 4')
    })
  })

  describe('energy', () => {
    it('gives −1.5 for the stored pattern (N = 4, one pattern)', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      // E(p) = −6·(1/4) = −1.5 (ver cabecera)
      expect(net.energy([1, -1, 1, 1])).toBeCloseTo(-1.5, 12)
    })

    it('gives 0 for the pattern with one bit inverted', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      // E(s) = −Σ_{i<j} W_ij s_i s_j = 0 (ver cabecera)
      expect(net.energy([-1, -1, 1, 1])).toBeCloseTo(0, 12)
    })

    it('gives −1 for the two stored patterns (N = 4, two patterns)', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      net.store([1, 1, -1, 1])
      // E(p) = E(q) = −1 (ver cabecera)
      expect(net.energy([1, -1, 1, 1])).toBeCloseTo(-1, 12)
      expect(net.energy([1, 1, -1, 1])).toBeCloseTo(-1, 12)
    })

    it('is lower at the stored pattern than at the corrupted state', () => {
      const net = new HopfieldNetwork(4)
      net.store([1, -1, 1, 1])
      expect(net.energy([1, -1, 1, 1])).toBeLessThan(net.energy([-1, -1, 1, 1]))
    })

    it('rejects a state of the wrong length', () => {
      const net = new HopfieldNetwork(4)
      expect(() => net.energy([1, -1])).toThrow('does not match network size 4')
    })
  })

  describe('binarize / unbinarize', () => {
    it('maps 0 → −1 and 1 → +1', () => {
      expect(HopfieldNetwork.binarize([0, 1, 1, 0])).toEqual([-1, 1, 1, -1])
    })

    it('maps −1 → 0 and +1 → 1', () => {
      expect(HopfieldNetwork.unbinarize([-1, 1, 1, -1])).toEqual([0, 1, 1, 0])
    })

    it('round-trips a 0/1 pattern', () => {
      const bits = [1, 0, 0, 1, 1, 0]
      expect(HopfieldNetwork.unbinarize(HopfieldNetwork.binarize(bits))).toEqual(bits)
    })
  })
})
