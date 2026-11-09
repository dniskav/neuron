import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { HopfieldNetwork } from '../../src/HopfieldNetwork'
import { seedRandom, restoreRandom } from '../helpers/seed'

// T052: validación de entradas y aviso de capacidad de HopfieldNetwork.
//
// Capacidad: max(1, floor(0.138·N)). N = 4 → floor(0.552) = 0 → 1 (avisa en el 2.º);
// N = 20 → floor(2.76) = 2 (avisa en el 3.º); N = 8 → floor(1.104) = 1 (avisa en el 2.º);
// N = 100 → floor(13.8) = 13 (avisa en el 14.º).
//
// Golden (N = 6, patrones a = [1,-1,1,-1,1,1], b = [-1,-1,1,1,-1,1]), capturado
// con la base antes del cambio; ver derivaciones: W_ij = (a_i a_j + b_i b_j)/6.
//   W_03 = (1·(-1) + (-1)(1))/6 = -1/3;  W_04 = (1·1 + (-1)(-1))/6 = 1/3
//   E(a) = -Σ_{i<j} W_ij a_i a_j = -(6 pares con |W|=1/3, todos alineados) = -2

describe('HopfieldNetwork validation (T052)', () => {
  beforeEach(() => seedRandom(7))
  afterEach(() => {
    restoreRandom()
    vi.restoreAllMocks()
  })

  const a = [1, -1, 1, -1, 1, 1]
  const b = [-1, -1, 1, 1, -1, 1]

  describe('golden of valid cases (unchanged)', () => {
    it('keeps store/recall/energy/binarize/unbinarize results', () => {
      vi.spyOn(console, 'warn').mockImplementation(() => {})
      const net = new HopfieldNetwork(6)
      net.store(a)
      net.store(b)
      const third = 1 / 3
      expect(net.weights[0][3]).toBeCloseTo(-third, 12)
      expect(net.weights[0][4]).toBeCloseTo(third, 12)
      expect(net.weights[1][5]).toBeCloseTo(-third, 12)
      expect(net.weights[2][5]).toBeCloseTo(third, 12)
      expect(net.weights[0][1]).toBe(0)
      expect(net.storedPatterns).toBe(2)
      expect(net.recall(a)).toEqual(a)
      expect(net.recall(b)).toEqual(b)
      expect(net.recall([1, -1, 1, -1, 1, -1])).toEqual(a)
      expect(net.recall([1, 1, 1, 1, 1, 1], 0)).toEqual([1, 1, 1, 1, 1, 1])
      expect(net.energy(a)).toBeCloseTo(-2, 12)
      expect(net.energy(b)).toBeCloseTo(-2, 12)
      // E(1,...,1) = -Σ_{i<j} W_ij; pares no nulos: 03:-1/3, 04:+1/3, 12:-1/3, 15:-1/3, 25:+1/3, 34:-1/3
      // → suma = -2/3 → E = +2/3 (coincide con el golden de la base)
      expect(net.energy([1, 1, 1, 1, 1, 1])).toBeCloseTo(2 / 3, 12)
      // binarize conserva el umbral actual: solo 0 → -1; todo otro finito → +1
      expect(HopfieldNetwork.binarize([0, 1, 1, 0, -3, 2, 0.5, -0.5])).toEqual([-1, 1, 1, -1, 1, 1, 1, 1])
      // unbinarize: solo -1 → 0; todo otro finito → 1
      expect(HopfieldNetwork.unbinarize([-1, 1, 1, -1, 0, 2, 0.5])).toEqual([0, 1, 1, 0, 1, 1, 1])
    })
  })

  describe('recall / energy reject invalid input', () => {
    const bad: [string, number[]][] = [
      ['length', [1, -1]],
      ['zero', [1, 0, 1, 1]],
      ['two', [1, 2, 1, 1]],
      ['NaN', [1, NaN, 1, 1]],
      ['Infinity', [1, Infinity, 1, 1]],
    ]

    it.each(bad)('recall throws (%s)', (_l, v) => {
      const net = new HopfieldNetwork(4)
      expect(() => net.recall(v)).toThrow(/^HopfieldNetwork\.recall: /)
    })

    it.each(bad)('energy throws (%s)', (_l, v) => {
      const net = new HopfieldNetwork(4)
      expect(() => net.energy(v)).toThrow(/^HopfieldNetwork\.energy: /)
    })

    it.each(bad.slice(1))('store throws with the method prefix (%s)', (_l, v) => {
      const net = new HopfieldNetwork(4)
      expect(() => net.store(v)).toThrow(/^HopfieldNetwork\.store: /)
      expect(net.storedPatterns).toBe(0)
    })

    it('store throws on wrong length with the method prefix', () => {
      expect(() => new HopfieldNetwork(4).store([1, -1])).toThrow(/^HopfieldNetwork\.store: .*length/)
    })
  })

  describe('binarize / unbinarize reject non-finite values', () => {
    it.each([NaN, Infinity, -Infinity])('binarize throws on %s', (v) => {
      expect(() => HopfieldNetwork.binarize([0, v, 1])).toThrow(/^HopfieldNetwork\.binarize: /)
    })

    it.each([NaN, Infinity, -Infinity])('unbinarize throws on %s', (v) => {
      expect(() => HopfieldNetwork.unbinarize([-1, v, 1])).toThrow(/^HopfieldNetwork\.unbinarize: /)
    })
  })

  describe('capacity warning', () => {
    const pats = (n: number, k: number) =>
      Array.from({ length: k }, (_, p) => Array.from({ length: n }, (_, i) => ((i >> p) & 1 ? 1 : -1)))

    it('does not warn up to floor(0.138·N) patterns (N = 20: 2)', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const net = new HopfieldNetwork(20)
      const [p1, p2] = pats(20, 2)
      net.store(p1)
      net.store(p2)
      expect(spy).not.toHaveBeenCalled()
    })

    it('warns when storedPatterns exceeds floor(0.138·N) (N = 20: 3rd pattern)', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const net = new HopfieldNetwork(20)
      for (const p of pats(20, 3)) net.store(p)
      expect(spy).toHaveBeenCalledTimes(1)
      expect(String(spy.mock.calls[0][0])).toContain('HopfieldNetwork.store')
    })

    it('warns only once per network, not at every store', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const net = new HopfieldNetwork(20)
      for (const p of pats(20, 5)) net.store(p)
      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('warns once per network instance', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const n1 = new HopfieldNetwork(8)
      const n2 = new HopfieldNetwork(8)
      const [p1, p2] = pats(8, 2)
      n1.store(p1) // 1 > floor(1.104) = 1 ? no
      expect(spy).not.toHaveBeenCalled()
      n1.store(p2) // 2 > 1
      n2.store(p1)
      n2.store(p2)
      expect(spy).toHaveBeenCalledTimes(2)
    })

    it('N = 4: one pattern never warns, the 2nd does', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const net = new HopfieldNetwork(4)
      const [p1, p2] = pats(4, 2)
      net.store(p1)
      expect(spy).not.toHaveBeenCalled()
      net.store(p2)
      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('N = 100: threshold 13, warns at the 14th pattern', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const net = new HopfieldNetwork(100)
      const all = pats(100, 14)
      for (const p of all.slice(0, 13)) net.store(p)
      expect(spy).not.toHaveBeenCalled()
      net.store(all[13])
      expect(spy).toHaveBeenCalledTimes(1)
    })

    it('does not warn for a rejected pattern', () => {
      const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      const net = new HopfieldNetwork(4)
      expect(() => net.store([1, 0, 1, 1])).toThrow()
      expect(spy).not.toHaveBeenCalled()
    })
  })
})
