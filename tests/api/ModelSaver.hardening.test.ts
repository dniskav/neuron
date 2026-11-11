// T006 (I2 / Q11): ModelSaver guarda metadatos de estructura y valida al cargar.
//
// Formato nuevo: { format: 1, length, structure?, weights, timestamp }.
// `structure` solo si el modelo expone `structure` como array de números.
// NO se guarda el nombre de la clase (los empaquetadores lo minimizan).
//
// fromJSON valida SIN tocar el modelo: JSON válido, weights numéricos y
// finitos, longitud igual a la del modelo (también para JSON antiguo sin
// metadatos), length coherente, structure igual a la del modelo si ambos
// existen. Cada error lleva `ModelSaver.fromJSON` y las cantidades.

import { describe, it, expect } from 'vitest'
import { ModelSaver, type Serializable } from '../../src/ModelSaver'
import { NetworkN } from '../../src/NetworkN'
import { Network } from '../../src/Network'
import { EmbeddingMatrix } from '../../src/MatMul'

// Modelo espía: registra cuántas veces se escribieron pesos.
class SpyModel implements Serializable {
  setCalls = 0
  w: number[]
  structure?: number[]
  constructor(w: number[], structure?: number[]) {
    this.w = [...w]
    if (structure !== undefined) this.structure = structure
  }
  getWeights(): number[] { return [...this.w] }
  setWeights(weights: number[]): void {
    this.setCalls++
    this.w = [...weights]
  }
}

// Reemplaza el campo `weights` de un JSON guardado por otro valor.
function withWeights(json: string, weights: unknown): string {
  const d = JSON.parse(json)
  d.weights = weights
  return JSON.stringify(d)
}

function patch(json: string, fields: Record<string, unknown>): string {
  return JSON.stringify({ ...JSON.parse(json), ...fields })
}

describe('ModelSaver.toJSON: metadatos', () => {
  it('NetworkN guarda format, length y structure', () => {
    const net = new NetworkN([2, 4, 1])
    const d = JSON.parse(ModelSaver.toJSON(net))
    expect(d.format).toBe(1)
    expect(d.length).toBe(net.getWeights().length)
    expect(d.structure).toEqual([2, 4, 1])
    expect(d.weights).toEqual(net.getWeights())
    expect(typeof d.timestamp).toBe('number')
  })

  it('las claves son exactamente format, length, structure, weights, timestamp (sin nombre de clase)', () => {
    const d = JSON.parse(ModelSaver.toJSON(new NetworkN([2, 4, 1])))
    expect(Object.keys(d).sort()).toEqual(['format', 'length', 'structure', 'timestamp', 'weights'])
  })

  it('EmbeddingMatrix (sin structure) no añade la clave structure', () => {
    const emb = new EmbeddingMatrix(5, 3)
    const d = JSON.parse(ModelSaver.toJSON(emb))
    expect(d.format).toBe(1)
    expect(d.length).toBe(15)
    expect('structure' in d).toBe(false)
  })

  it('Network (sin structure) no añade la clave structure', () => {
    const d = JSON.parse(ModelSaver.toJSON(new Network(2, 4, 1)))
    expect('structure' in d).toBe(false)
    expect(d.length).toBe(17)
  })

  it('structure solo se guarda si es un array de números', () => {
    const bad = new SpyModel([1, 2, 3], ['a', 'b'] as unknown as number[])
    const d = JSON.parse(ModelSaver.toJSON(bad))
    expect('structure' in d).toBe(false)
    expect(d.length).toBe(3)
  })
})

describe('ModelSaver: ida y vuelta intacta', () => {
  it('NetworkN restaura exactamente los pesos', () => {
    const net = new NetworkN([2, 4, 1])
    const orig = net.getWeights()
    const json = ModelSaver.toJSON(net)
    net.setWeights(orig.map(v => v + 1))
    ModelSaver.fromJSON(net, json)
    expect(net.getWeights()).toEqual(orig)
  })

  it('EmbeddingMatrix restaura exactamente los pesos', () => {
    const emb = new EmbeddingMatrix(5, 3)
    const orig = emb.getWeights()
    const json = ModelSaver.toJSON(emb)
    emb.setWeights(orig.map(v => v + 1))
    ModelSaver.fromJSON(emb, json)
    expect(emb.getWeights()).toEqual(orig)
  })

  it('Network restaura exactamente los pesos', () => {
    const net = new Network(2, 4, 1)
    const orig = net.getWeights()
    const json = ModelSaver.toJSON(net)
    net.setWeights(orig.map(v => v + 1))
    ModelSaver.fromJSON(net, json)
    expect(net.getWeights()).toEqual(orig)
  })
})

describe('ModelSaver.fromJSON: JSON antiguo (sin metadatos)', () => {
  it('se carga si la longitud coincide', () => {
    const net = new NetworkN([2, 4, 1])
    const orig = net.getWeights()
    const old = JSON.stringify({ weights: orig, timestamp: 123 })
    net.setWeights(orig.map(v => v + 1))
    ModelSaver.fromJSON(net, old)
    expect(net.getWeights()).toEqual(orig)
  })

  it('se carga en EmbeddingMatrix si la longitud coincide', () => {
    const emb = new EmbeddingMatrix(5, 3)
    const orig = emb.getWeights()
    const old = JSON.stringify({ weights: orig.map(v => v * 2), timestamp: 1 })
    ModelSaver.fromJSON(emb, old)
    expect(emb.getWeights()).toEqual(orig.map(v => v * 2))
  })

  it('con longitud distinta lanza y deja el modelo intacto', () => {
    const net = new NetworkN([2, 4, 1])
    const before = net.getWeights()
    const old = JSON.stringify({ weights: before.slice(0, 5), timestamp: 1 })
    expect(() => ModelSaver.fromJSON(net, old)).toThrow(/ModelSaver\.fromJSON/)
    expect(() => ModelSaver.fromJSON(net, old)).toThrow(/17/)
    expect(() => ModelSaver.fromJSON(net, old)).toThrow(/5/)
    expect(net.getWeights()).toEqual(before)
  })
})

describe('ModelSaver.fromJSON: rechazos con modelo intacto', () => {
  it('modelo de otra longitud lanza con ambas cantidades', () => {
    const src = new NetworkN([2, 4, 1])
    const json = ModelSaver.toJSON(src)
    const dst = new NetworkN([2, 3, 1])
    const before = dst.getWeights()
    expect(() => ModelSaver.fromJSON(dst, json)).toThrow(/ModelSaver\.fromJSON/)
    expect(() => ModelSaver.fromJSON(dst, json)).toThrow(/expects 13 weights and the file has 17/)
    expect(dst.getWeights()).toEqual(before)
  })

  it('misma longitud y distinta structure lanza (NetworkN [2,4,1] vs [2,3,2], ambos 17 pesos)', () => {
    const src = new NetworkN([2, 4, 1])
    const json = ModelSaver.toJSON(src)
    const dst = new NetworkN([2, 3, 2])
    expect(dst.getWeights().length).toBe(17)
    const before = dst.getWeights()
    expect(() => ModelSaver.fromJSON(dst, json)).toThrow(/ModelSaver\.fromJSON/)
    expect(() => ModelSaver.fromJSON(dst, json)).toThrow(/\[2,4,1\]/)
    expect(() => ModelSaver.fromJSON(dst, json)).toThrow(/\[2,3,2\]/)
    expect(dst.getWeights()).toEqual(before)
  })

  it('weights con null (NaN al guardar) lanza y nombra el índice', () => {
    const net = new NetworkN([2, 4, 1])
    const json = withWeights(ModelSaver.toJSON(net), net.getWeights().map((v, i) => (i === 3 ? NaN : v)))
    const before = net.getWeights()
    expect(() => ModelSaver.fromJSON(net, json)).toThrow(/ModelSaver\.fromJSON/)
    expect(() => ModelSaver.fromJSON(net, json)).toThrow(/weights\[3\]/)
    expect(net.getWeights()).toEqual(before)
  })

  it('weights con un string lanza', () => {
    const net = new NetworkN([2, 4, 1])
    const w = net.getWeights()
    w[0] = 'abc' as unknown as number
    const before = net.getWeights()
    expect(() => ModelSaver.fromJSON(net, withWeights(ModelSaver.toJSON(net), w))).toThrow(/weights\[0\]/)
    expect(net.getWeights()).toEqual(before)
  })

  it('length incoherente con weights lanza (length mayor)', () => {
    const net = new NetworkN([2, 4, 1])
    const json = patch(ModelSaver.toJSON(net), { length: net.getWeights().length + 1 })
    const before = net.getWeights()
    expect(() => ModelSaver.fromJSON(net, json)).toThrow(/ModelSaver\.fromJSON/)
    expect(net.getWeights()).toEqual(before)
  })

  it('length incoherente con weights lanza (length menor)', () => {
    const net = new NetworkN([2, 4, 1])
    const json = patch(ModelSaver.toJSON(net), { length: net.getWeights().length - 1 })
    const before = net.getWeights()
    expect(() => ModelSaver.fromJSON(net, json)).toThrow(/ModelSaver\.fromJSON/)
    expect(net.getWeights()).toEqual(before)
  })

  it('texto que no es JSON lanza con mensaje claro y deja el modelo intacto', () => {
    const net = new NetworkN([2, 4, 1])
    const before = net.getWeights()
    expect(() => ModelSaver.fromJSON(net, 'esto no es json {')).toThrow(/ModelSaver\.fromJSON.*JSON/)
    expect(net.getWeights()).toEqual(before)
  })

  it('structure malformada en el archivo lanza', () => {
    const net = new NetworkN([2, 4, 1])
    const json = patch(ModelSaver.toJSON(net), { structure: 'x' })
    expect(() => ModelSaver.fromJSON(net, json)).toThrow(/ModelSaver\.fromJSON.*structure/)
  })

  it('format desconocido lanza', () => {
    const net = new NetworkN([2, 4, 1])
    const json = patch(ModelSaver.toJSON(net), { format: 2 })
    expect(() => ModelSaver.fromJSON(net, json)).toThrow(/ModelSaver\.fromJSON.*unsupported format/)
  })

  it('no escribe pesos antes de validar (modelo espía)', () => {
    const src = new SpyModel([1, 2, 3, 4])
    const json = ModelSaver.toJSON(src)
    const dst = new SpyModel([9, 9, 9])
    expect(() => ModelSaver.fromJSON(dst, json)).toThrow(/ModelSaver\.fromJSON/)
    expect(dst.setCalls).toBe(0)
    expect(dst.getWeights()).toEqual([9, 9, 9])
  })
})

describe('ModelSaver: archivo y modelo sin structure', () => {
  it('modelo sin structure acepta archivo con structure si la longitud coincide (solo longitud)', () => {
    const src = new NetworkN([2, 4, 1])
    const json = ModelSaver.toJSON(src)
    const dst = new Network(2, 4, 1)
    expect(() => ModelSaver.fromJSON(dst, json)).not.toThrow()
    expect(dst.getWeights()).toEqual(src.getWeights())
  })

  it('archivo sin structure se carga en NetworkN aunque el modelo tenga structure', () => {
    const net = new NetworkN([2, 4, 1])
    const orig = net.getWeights()
    const json = JSON.stringify({ format: 1, length: orig.length, weights: orig, timestamp: 1 })
    net.setWeights(orig.map(v => v + 1))
    ModelSaver.fromJSON(net, json)
    expect(net.getWeights()).toEqual(orig)
  })
})

describe('ModelSaver: saveToFile / loadFromFile con callbacks', () => {
  it('ida y vuelta intacta con NetworkN', () => {
    const net = new NetworkN([2, 4, 1])
    const orig = net.getWeights()
    let stored = ''
    ModelSaver.saveToFile(net, 'm.json', (_p, d) => { stored = d })
    net.setWeights(orig.map(v => v + 1))
    ModelSaver.loadFromFile(net, 'm.json', () => stored)
    expect(net.getWeights()).toEqual(orig)
  })

  it('loadFromFile de un archivo de otra longitud lanza y deja el modelo intacto', () => {
    const src = new NetworkN([2, 4, 1])
    let stored = ''
    ModelSaver.saveToFile(src, 'm.json', (_p, d) => { stored = d })
    const dst = new NetworkN([3, 3, 1])
    const before = dst.getWeights()
    expect(() => ModelSaver.loadFromFile(dst, 'm.json', () => stored)).toThrow(/ModelSaver\.fromJSON/)
    expect(dst.getWeights()).toEqual(before)
  })
})
