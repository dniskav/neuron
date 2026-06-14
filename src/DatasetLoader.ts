// ─── DatasetLoader ────────────────────────────────────────────────────────────
//
// Parsers that convert raw CSV or JSON text into DataPair objects ready to
// feed into a DataLoader. The library has no I/O dependency — you supply the
// raw string (read it however you like: fs.readFileSync, fetch, etc.).
//
// ── CSV ───────────────────────────────────────────────────────────────────────
//   const raw = fs.readFileSync('iris.csv', 'utf8')
//   const { inputs, targets } = DatasetLoader.fromCSV(raw, {
//     featureCols: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
//     targetCols:  ['species'],                  // string → one-hot automatically
//   })
//   const loader = new DataLoader({ inputs, targets }, 32)
//
// ── JSON ──────────────────────────────────────────────────────────────────────
//   const raw = fs.readFileSync('data.json', 'utf8')
//   const { inputs, targets } = DatasetLoader.fromJSON(raw, {
//     featureCols: ['x1', 'x2'],
//     targetCols:  ['label'],
//   })
//
// ── Column encoding ───────────────────────────────────────────────────────────
// Numeric columns are parsed directly.
// String columns are one-hot encoded automatically — the mapping is returned
// as `categoricalMaps` so you can decode predictions later.
//
// ─────────────────────────────────────────────────────────────────────────────

import type { DataPair } from './DataLoader'

// ── Public types ──────────────────────────────────────────────────────────────

export interface DatasetLoaderOptions {
  /** Column names to use as input features. */
  featureCols: string[]
  /** Column names to use as targets / labels. */
  targetCols: string[]
  /**
   * When true, string values in feature/target columns are one-hot encoded.
   * When false, non-numeric values throw an error. Default: true.
   */
  encodeStrings?: boolean
}

/**
 * Maps a column name to its {value → one-hot index} dictionary.
 * Useful for decoding model predictions back to class names.
 */
export type CategoricalMap = Record<string, Record<string, number>>

export interface DatasetLoaderResult extends DataPair {
  /**
   * For each string column that was one-hot encoded, maps the column name to
   * the {category → index} dictionary used during encoding.
   */
  categoricalMaps: CategoricalMap
  /** Column names in the order they appear in each input vector. */
  featureNames: string[]
  /** Column names (or expanded one-hot names) in the order they appear in each target vector. */
  targetNames: string[]
  /** Total number of rows parsed. */
  numRows: number
}

// ── DatasetLoader ─────────────────────────────────────────────────────────────
export class DatasetLoader {
  // ── CSV ─────────────────────────────────────────────────────────────────────
  /**
   * Parse a CSV string into a DataPair.
   *
   * - The first non-empty row is treated as a header.
   * - Numeric values are parsed with parseFloat.
   * - String values are one-hot encoded (one column → N binary columns).
   * - Empty rows and comment lines (starting with #) are skipped.
   *
   * @param csv     - raw CSV text
   * @param options - which columns to use as features / targets
   */
  static fromCSV(csv: string, options: DatasetLoaderOptions): DatasetLoaderResult {
    const rows = DatasetLoader._parseCSV(csv)
    if (rows.length < 2) throw new Error('DatasetLoader.fromCSV: CSV must have a header row and at least one data row.')

    const header = rows[0]
    const dataRows = rows.slice(1)

    return DatasetLoader._buildDataPair(header, dataRows, options)
  }

  // ── JSON ─────────────────────────────────────────────────────────────────────
  /**
   * Parse a JSON string (array of objects) into a DataPair.
   *
   * Expected format:
   *   [{ "col1": 1.0, "col2": "cat", "label": "dog" }, ...]
   *
   * @param json    - raw JSON text or a pre-parsed array of objects
   * @param options - which columns to use as features / targets
   */
  static fromJSON(
    json: string | Record<string, unknown>[],
    options: DatasetLoaderOptions,
  ): DatasetLoaderResult {
    const records: Record<string, unknown>[] =
      typeof json === 'string' ? JSON.parse(json) : json

    if (!Array.isArray(records) || records.length === 0) {
      throw new Error('DatasetLoader.fromJSON: expected a non-empty JSON array of objects.')
    }

    const header = Object.keys(records[0])
    const dataRows = records.map(row => header.map(col => String(row[col] ?? '')))

    return DatasetLoader._buildDataPair(header, dataRows, options)
  }

  // ── Private: shared pipeline ──────────────────────────────────────────────
  private static _buildDataPair(
    header: string[],
    dataRows: string[][],
    options: DatasetLoaderOptions,
  ): DatasetLoaderResult {
    const { featureCols, targetCols, encodeStrings = true } = options

    // Validate column references
    for (const col of [...featureCols, ...targetCols]) {
      if (!header.includes(col)) {
        throw new Error(`DatasetLoader: column "${col}" not found in header [${header.join(', ')}].`)
      }
    }

    // Determine which columns are categorical (string) vs numeric
    const catMaps: CategoricalMap = {}

    const buildEncoder = (cols: string[]) => {
      for (const col of cols) {
        const colIdx = header.indexOf(col)
        const values = dataRows.map(row => row[colIdx])
        const isNumeric = values.every(v => v === '' || !isNaN(Number(v)))

        if (!isNumeric) {
          if (!encodeStrings) {
            throw new Error(`DatasetLoader: column "${col}" contains non-numeric values. Set encodeStrings: true to one-hot encode them.`)
          }
          // Build sorted unique categories for determinism
          const unique = [...new Set(values)].sort()
          catMaps[col] = Object.fromEntries(unique.map((v, i) => [v, i]))
        }
      }
    }

    buildEncoder(featureCols)
    buildEncoder(targetCols)

    // Encode a single column value to a vector segment
    const encodeValue = (col: string, raw: string): number[] => {
      if (catMaps[col]) {
        const categories = catMaps[col]
        const n = Object.keys(categories).length
        const vec = new Array(n).fill(0)
        const idx = categories[raw]
        if (idx !== undefined) vec[idx] = 1
        return vec
      }
      return [parseFloat(raw)]
    }

    // Build expanded column name lists (for one-hot cols: col_cat0, col_cat1 …)
    const expandNames = (cols: string[]): string[] =>
      cols.flatMap(col => {
        if (catMaps[col]) {
          return Object.keys(catMaps[col]).map(cat => `${col}_${cat}`)
        }
        return [col]
      })

    const featureNames = expandNames(featureCols)
    const targetNames  = expandNames(targetCols)

    // Encode all rows
    const inputs: number[][] = []
    const targets: number[][] = []

    for (const row of dataRows) {
      const input: number[] = featureCols.flatMap(col => {
        const raw = row[header.indexOf(col)]
        return encodeValue(col, raw)
      })
      const target: number[] = targetCols.flatMap(col => {
        const raw = row[header.indexOf(col)]
        return encodeValue(col, raw)
      })
      inputs.push(input)
      targets.push(target)
    }

    return {
      inputs,
      targets,
      categoricalMaps: catMaps,
      featureNames,
      targetNames,
      numRows: dataRows.length,
    }
  }

  // ── Private: RFC 4180-compatible CSV parser ───────────────────────────────
  private static _parseCSV(csv: string): string[][] {
    const rows: string[][] = []
    const lines = csv.split(/\r?\n/)

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) continue
      rows.push(DatasetLoader._parseCSVRow(trimmed))
    }

    return rows
  }

  private static _parseCSVRow(line: string): string[] {
    const fields: string[] = []
    let current = ''
    let inQuotes = false

    for (let i = 0; i < line.length; i++) {
      const ch = line[i]

      if (inQuotes) {
        if (ch === '"' && line[i + 1] === '"') {
          // Escaped quote inside quoted field
          current += '"'
          i++
        } else if (ch === '"') {
          inQuotes = false
        } else {
          current += ch
        }
      } else {
        if (ch === '"') {
          inQuotes = true
        } else if (ch === ',') {
          fields.push(current.trim())
          current = ''
        } else {
          current += ch
        }
      }
    }

    fields.push(current.trim())
    return fields
  }
}
