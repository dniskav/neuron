import { describe, it, expect } from 'vitest'
import { DatasetLoader } from '../../src/DatasetLoader'

// T050 (hallazgos H1 y H2 de T008, principio V): celdas numéricas vacías y filas
// irregulares LANZAN en vez de producir NaN o datos desalineados en silencio.
// Los números de fila son de DATOS, base 1 (la cabecera no cuenta).

describe('DatasetLoader — golden de entradas válidas (no debe cambiar)', () => {
  it('CSV válido con salto de línea final, comillas y CRLF', () => {
    // Filas: (1, "a,b", 10) y (2, "c""d", 20) con fila final vacía por el '\n' del final.
    // Categorías de t por código: 'a,b' (0x61) < 'c"d' (0x63) → 0 y 1.
    const csv = 'x,t,y\n1,"a,b",10\n2,"c""d",20\n'
    const r = DatasetLoader.fromCSV(csv, { featureCols: ['x', 't'], targetCols: ['y'] })
    expect(r.inputs).toEqual([[1, 1, 0], [2, 0, 1]])
    expect(r.targets).toEqual([[10], [20]])
    expect(r.featureNames).toEqual(['x', 't_a,b', 't_c"d'])
    expect(r.categoricalMaps).toEqual({ t: { 'a,b': 0, 'c"d': 1 } })
    expect(r.numRows).toBe(2)

    // Mismo resultado con CRLF y varias líneas vacías al final.
    const crlf = DatasetLoader.fromCSV(csv.replace(/\n/g, '\r\n') + '\r\n\r\n', {
      featureCols: ['x', 't'], targetCols: ['y'],
    })
    expect(crlf).toEqual(r)
  })

  it('JSON válido (cadena y array) con números y texto', () => {
    const recs = [{ a: 1.5, c: 'u', b: 0 }, { a: -2, c: 'v', b: 1 }]
    const opts = { featureCols: ['a', 'c'], targetCols: ['b'] }
    const r = DatasetLoader.fromJSON(JSON.stringify(recs), opts)
    expect(r.inputs).toEqual([[1.5, 1, 0], [-2, 0, 1]])
    expect(r.targets).toEqual([[0], [1]])
    expect(DatasetLoader.fromJSON(recs, opts)).toEqual(r)
  })

  it('una celda vacía en columna de TEXTO sigue siendo una categoría válida', () => {
    // Columna c = {"", z}: no es numérica; '' < 'z' → ''=0, z=1.
    const r = DatasetLoader.fromCSV('c,y\n,1\nz,2\n', { featureCols: ['c'], targetCols: ['y'] })
    expect(r.inputs).toEqual([[1, 0], [0, 1]])
  })
})

describe('DatasetLoader.fromCSV — celdas numéricas vacías (H1)', () => {
  it('objetivo vacío: fila 1, columna "b"', () => {
    // Fila de datos 1 = "1," → b vacío.
    expect(() => DatasetLoader.fromCSV('a,b\n1,\n2,3\n', { featureCols: ['a'], targetCols: ['b'] }))
      .toThrow('DatasetLoader.fromCSV: empty value in numeric column "b" at row 1')
  })

  it('entrada vacía en la fila 2, columna "a"', () => {
    // Fila de datos 2 = ",3" → a vacía.
    expect(() => DatasetLoader.fromCSV('a,b\n1,2\n,3\n', { featureCols: ['a'], targetCols: ['b'] }))
      .toThrow('DatasetLoader.fromCSV: empty value in numeric column "a" at row 2')
  })

  it('una columna que no se usa puede tener celdas vacías', () => {
    const r = DatasetLoader.fromCSV('a,b,z\n1,2,\n3,4,\n', { featureCols: ['a'], targetCols: ['b'] })
    expect(r.inputs).toEqual([[1], [3]])
  })
})

describe('DatasetLoader.fromCSV — filas irregulares (H2)', () => {
  const opts = { featureCols: ['a'], targetCols: ['b'] }

  it('fila corta: número de fila y longitudes esperada/obtenida', () => {
    // Fila 2 = "3" → 1 campo, se esperaban 2.
    expect(() => DatasetLoader.fromCSV('a,b\n1,2\n3\n', opts))
      .toThrow('DatasetLoader.fromCSV: row 2 has 1 fields, expected 2')
  })

  it('fila larga: antes se descartaba el excedente', () => {
    // Fila 1 = "1,2,9" → 3 campos, se esperaban 2.
    expect(() => DatasetLoader.fromCSV('a,b\n1,2,9\n3,4\n', opts))
      .toThrow('DatasetLoader.fromCSV: row 1 has 3 fields, expected 2')
  })

  it('la fila vacía final por salto de línea no cuenta como fila corta', () => {
    expect(DatasetLoader.fromCSV('a,b\n1,2\n3,4\n\n', opts).numRows).toBe(2)
  })

  it('una comilla con coma interior es UN campo (no fila larga)', () => {
    expect(DatasetLoader.fromCSV('a,b\n"1,5",2\n', { featureCols: ['b'], targetCols: ['a'] }).numRows).toBe(1)
  })
})

describe('DatasetLoader.fromJSON — celdas vacías (H1)', () => {
  const opts = { featureCols: ['a'], targetCols: ['b'] }

  it('null en una columna numérica lanza con fila y columna', () => {
    // Registro 2: a = null.
    expect(() => DatasetLoader.fromJSON([{ a: 1, b: 2 }, { a: null, b: 3 }], opts))
      .toThrow('DatasetLoader.fromJSON: empty value in numeric column "a" at row 2')
  })

  it('clave ausente en una columna numérica lanza', () => {
    expect(() => DatasetLoader.fromJSON('[{"a":1,"b":2},{"a":5}]', opts))
      .toThrow('DatasetLoader.fromJSON: empty value in numeric column "b" at row 2')
  })
})
