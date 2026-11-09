import { describe, it, expect } from 'vitest'
import { DatasetLoader } from '../src/DatasetLoader'

// DatasetLoader no usa aleatoriedad. tests/setup.ts ya fija Math.random antes de
// cada test (seedRandom de tests/helpers/seed.ts), así que no hace falta nada más.
//
// Valores esperados: derivados a mano de cada CSV/JSON (comentarios junto a cada
// expect). No se copian de la salida del módulo.
//
// Nota: no se fijan como correctos dos comportamientos sospechosos (celda numérica
// vacía → NaN; filas irregulares aceptadas en silencio). Ver el informe de T008.

// ── CSV de referencia ───────────────────────────────────────────────────────
// Cabecera: x1, x2, color, label (la línea "# ..." se salta).
// color es texto con valores {red, blue, red}. Orden por código UTF-16:
// "blue" < "red" → blue = 0, red = 1, así que one-hot(color) tiene 2 posiciones.
//   fila 1: x1=1, x2=2, red  → entrada [1, 2, 0, 1]   objetivo [0]
//   fila 2: x1=3, x2=4, blue → entrada [3, 4, 1, 0]   objetivo [1]
//   fila 3: x1=5, x2=6, red  → entrada [5, 6, 0, 1]   objetivo [1]
const CSV_REF = [
  '# dataset de referencia',
  'x1,x2,color,label',
  '1,2,red,0',
  '3,4,blue,1',
  '5,6,red,1',
].join('\n')

const OPTS_REF = { featureCols: ['x1', 'x2', 'color'], targetCols: ['label'] }

// Opciones para CSV de dos columnas (a, b) usados en los casos de error.
const OPTS_REF_AB = { featureCols: ['a'], targetCols: ['b'] }

describe('DatasetLoader.fromCSV — CSV de 3 filas con salida a mano', () => {
  it('entradas, objetivos, nombres expandidos, mapa categórico y numRows', () => {
    const r = DatasetLoader.fromCSV(CSV_REF, OPTS_REF)

    // Entradas y objetivos según la derivación de arriba.
    expect(r.inputs).toEqual([[1, 2, 0, 1], [3, 4, 1, 0], [5, 6, 0, 1]])
    expect(r.targets).toEqual([[0], [1], [1]])

    // Columnas one-hot: color_blue (índice 0) y color_red (índice 1).
    expect(r.featureNames).toEqual(['x1', 'x2', 'color_blue', 'color_red'])
    expect(r.targetNames).toEqual(['label'])

    // Solo color es texto; label es numérica y no aparece en el mapa.
    expect(r.categoricalMaps).toEqual({ color: { blue: 0, red: 1 } })

    // 3 filas de datos (sin contar la cabecera ni el comentario).
    expect(r.numRows).toBe(3)
  })

  it('CRLF, líneas vacías y comentarios dan el mismo resultado', () => {
    // Mismo CSV con finales \r\n, una línea vacía al principio y otra al final.
    // Derivación: las líneas vacías y "#" se descartan; el resto es igual a CSV_REF.
    const crlf = '\r\n' + CSV_REF.replace(/\n/g, '\r\n') + '\r\n\r\n'
    const r = DatasetLoader.fromCSV(crlf, OPTS_REF)

    expect(r.inputs).toEqual([[1, 2, 0, 1], [3, 4, 1, 0], [5, 6, 0, 1]])
    expect(r.targets).toEqual([[0], [1], [1]])
    expect(r.numRows).toBe(3)
  })

  it('comillas: coma dentro de un campo y comillas escapadas ("")', () => {
    // "hola, mundo" es un solo campo (la coma va entre comillas).
    // "dijo ""sí""" es el texto: dijo "sí"  (las "" son una comilla escapada).
    // Orden de categorías: 'd' (0x64) < 'h' (0x68) → dijo "sí" = 0, hola, mundo = 1.
    // Fila 1: 'hola, mundo' → índice 1 → [0, 1]. Fila 2: 'dijo "sí"' → índice 0 → [1, 0].
    const csv = 'text,label\n"hola, mundo",1\n"dijo ""sí""",2'
    const r = DatasetLoader.fromCSV(csv, { featureCols: ['label'], targetCols: ['text'] })

    expect(r.inputs).toEqual([[1], [2]])
    expect(r.targets).toEqual([[0, 1], [1, 0]])
    expect(r.targetNames).toEqual(['text_dijo "sí"', 'text_hola, mundo'])
  })

  it('categorías ordenadas por código UTF-16 (no por aparición ni numéricamente)', () => {
    // Valores en orden de aparición: b, 10, 2, A.
    // Orden por código: '1' (0x31) < '2' (0x32) < 'A' (0x41) < 'b' (0x62),
    // es decir "10" < "2" < "A" < "b" → 10 = 0, 2 = 1, A = 2, b = 3.
    // Un orden numérico daría 2 < 10 y pondría "2" antes que "10".
    const csv = 'c,y\nb,1\n10,2\n2,3\nA,4'
    const r = DatasetLoader.fromCSV(csv, { featureCols: ['c'], targetCols: ['y'] })

    expect(r.categoricalMaps).toEqual({ c: { '10': 0, '2': 1, A: 2, b: 3 } })
    // Fila "b" → posición 3; "10" → 0; "2" → 1; "A" → 2.
    expect(r.inputs).toEqual([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    expect(r.targets).toEqual([[1], [2], [3], [4]])
  })

  it('una sola celda no numérica convierte toda la columna en categórica', () => {
    // Columna b = {x, 1}: "x" no es numérico, así que toda la columna es texto.
    // Orden: '1' (0x31) < 'x' (0x78) → 1 = 0, x = 1.
    // Fila 1: x → [0, 1]. Fila 2: 1 → [1, 0].
    const csv = 'a,b\n7,x\n8,1'
    const r = DatasetLoader.fromCSV(csv, { featureCols: ['a'], targetCols: ['b'] })

    expect(r.inputs).toEqual([[7], [8]])
    expect(r.targets).toEqual([[0, 1], [1, 0]])
    expect(r.categoricalMaps).toEqual({ b: { '1': 0, x: 1 } })
  })

  it('números con exponente y signo se parsean como valor', () => {
    // 1.5e2 = 1.5 · 10² = 150. -0.5 = -0.5.
    const r = DatasetLoader.fromCSV('a,b\n1.5e2,-0.5', { featureCols: ['a'], targetCols: ['b'] })

    expect(r.inputs).toEqual([[150]])
    expect(r.targets).toEqual([[-0.5]])
  })

  it('espacios alrededor de cabeceras y campos se recortan', () => {
    // Cabeceras "x1 " y " y" → "x1" e "y". Campos " 1 " y " 2 " → 1 y 2.
    const r = DatasetLoader.fromCSV('x1 , y\n 1 , 2 ', { featureCols: ['x1'], targetCols: ['y'] })

    expect(r.featureNames).toEqual(['x1'])
    expect(r.inputs).toEqual([[1]])
    expect(r.targets).toEqual([[2]])
  })

  it('salta comentarios y líneas vacías en medio del CSV', () => {
    // Filas de datos: (1,2) y (3,4). Las demás líneas no cuentan.
    const csv = 'x,y\n# comentario\n1,2\n\n3,4\n# fin'
    const r = DatasetLoader.fromCSV(csv, { featureCols: ['x'], targetCols: ['y'] })

    expect(r.inputs).toEqual([[1], [3]])
    expect(r.targets).toEqual([[2], [4]])
    expect(r.numRows).toBe(2)
  })
})

describe('DatasetLoader.fromCSV — errores', () => {
  it('lanza si no hay fila de datos (solo cabecera, solo comentarios o vacío)', () => {
    // Se exigen al menos 2 filas no vacías y no comentadas: cabecera + 1 dato.
    expect(() => DatasetLoader.fromCSV('a,b', OPTS_REF_AB))
      .toThrow('CSV must have a header row and at least one data row')
    expect(() => DatasetLoader.fromCSV('a,b\n# nada\n\n', OPTS_REF_AB))
      .toThrow('CSV must have a header row and at least one data row')
    expect(() => DatasetLoader.fromCSV('', OPTS_REF_AB))
      .toThrow('CSV must have a header row and at least one data row')
  })

  it('lanza si una columna de entrada no existe en la cabecera', () => {
    expect(() => DatasetLoader.fromCSV('a,b\n1,2', { featureCols: ['zz'], targetCols: ['b'] }))
      .toThrow('column "zz" not found in header [a, b]')
  })

  it('lanza si una columna objetivo no existe en la cabecera', () => {
    expect(() => DatasetLoader.fromCSV('a,b\n1,2', { featureCols: ['a'], targetCols: ['nope'] }))
      .toThrow('column "nope" not found')
  })

  it('con encodeStrings: false, un valor de texto lanza con el nombre de la columna', () => {
    expect(() => DatasetLoader.fromCSV('a,b\n1,x', { featureCols: ['a'], targetCols: ['b'], encodeStrings: false }))
      .toThrow('column "b" contains non-numeric values')
  })
})

describe('DatasetLoader.fromJSON', () => {
  // Registros con cabecera = claves de la primera fila: x, c, y.
  // c (texto) = {cat, fish} → cat = 0, fish = 1.
  // y (texto) = {cat, dog}  → cat = 0, dog = 1.
  //   fila 1: x=1, c=cat,  y=dog → entrada [1, 1, 0]  objetivo [0, 1]
  //   fila 2: x=2, c=fish, y=cat → entrada [2, 0, 1]  objetivo [1, 0]
  const RECORDS = [
    { x: 1, c: 'cat', y: 'dog' },
    { x: 2, c: 'fish', y: 'cat' },
  ]
  const OPTS_JSON = { featureCols: ['x', 'c'], targetCols: ['y'] }

  it('one-hot de texto y numéricos, con cadena JSON y con array ya parseado', () => {
    const desdeCadena = DatasetLoader.fromJSON(JSON.stringify(RECORDS), OPTS_JSON)
    const desdeArray = DatasetLoader.fromJSON(RECORDS, OPTS_JSON)

    for (const r of [desdeCadena, desdeArray]) {
      expect(r.inputs).toEqual([[1, 1, 0], [2, 0, 1]])
      expect(r.targets).toEqual([[0, 1], [1, 0]])
      expect(r.featureNames).toEqual(['x', 'c_cat', 'c_fish'])
      expect(r.targetNames).toEqual(['y_cat', 'y_dog'])
      expect(r.categoricalMaps).toEqual({ c: { cat: 0, fish: 1 }, y: { cat: 0, dog: 1 } })
      expect(r.numRows).toBe(2)
    }
  })

  it('lanza con array vacío, con un objeto en lugar de array y con columna inexistente', () => {
    expect(() => DatasetLoader.fromJSON('[]', OPTS_JSON))
      .toThrow('expected a non-empty JSON array of objects')
    expect(() => DatasetLoader.fromJSON('{"x":1}', OPTS_JSON))
      .toThrow('expected a non-empty JSON array of objects')
    expect(() => DatasetLoader.fromJSON(RECORDS, { featureCols: ['nope'], targetCols: ['y'] }))
      .toThrow('column "nope" not found')
  })
})
