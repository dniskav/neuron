import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { seedRandom, restoreRandom } from './helpers/seed'
import {
  confusionMatrix,
  precision,
  recall,
  f1Score,
  accuracy,
  rocCurve,
  auc,
  mae,
  rmse,
  r2Score,
  perplexity,
  printConfusionMatrix,
  classificationReport,
} from '../src/Metrics'

// Metrics no es estocástico. La semilla se fija igualmente para cumplir la
// Constitución IV (entorno determinista en todos los tests).
beforeEach(() => seedRandom(0x5eed))
afterEach(() => {
  restoreRandom()
  vi.restoreAllMocks()
})

// Todas las cifras esperadas están deducidas a mano en el comentario de cada expect.

// ─── Classification ───────────────────────────────────────────────────────────

describe('accuracy', () => {
  it('cuenta aciertos sobre el total', () => {
    // Aciertos: pos 0 (0=0), pos 1 (1=1), pos 3 (0=0). Fallo: pos 2 (1≠0). 3/4 = 0.75
    expect(accuracy([0, 1, 1, 0], [0, 1, 0, 0])).toBe(0.75)
  })

  it('predicción perfecta → 1; ninguna acierta → 0', () => {
    // Todas coinciden: 3/3 = 1
    expect(accuracy([0, 1, 2], [0, 1, 2])).toBe(1)
    // Ninguna coincide (0≠1, 1≠2, 2≠0): 0/3 = 0
    expect(accuracy([0, 1, 2], [1, 2, 0])).toBe(0)
  })
})

describe('confusionMatrix', () => {
  it('K = max(etiquetas)+1 y C[i][j] cuenta (real i, predicho j)', () => {
    // yTrue=[0,1,1], yPred=[0,1,0]. K = max(0,1,1,0,1,0)+1 = 2.
    // Pares (real,pred): (0,0), (1,1), (1,0).
    // C[0][0]=1, C[0][1]=0, C[1][0]=1, C[1][1]=1  →  [[1,0],[1,1]]
    expect(confusionMatrix([0, 1, 1], [0, 1, 0])).toEqual([
      [1, 0],
      [1, 1],
    ])
  })

  it('numClasses explícito añade filas y columnas vacías', () => {
    // K = 3. Pares (real,pred): (0,0), (1,2), (2,2).
    // C[0][0]=1, C[1][2]=1, C[2][2]=1, resto 0.
    expect(confusionMatrix([0, 1, 2], [0, 2, 2], 3)).toEqual([
      [1, 0, 0],
      [0, 0, 1],
      [0, 0, 1],
    ])
  })
})

describe('precision / recall / f1 binarios', () => {
  // yTrue = [1,1,1,0,0,0,0,0], yPred = [1,0,0,1,0,0,0,0]
  // i0: real 1, pred 1 → TP.  i1, i2: real 1, pred 0 → FN (2).
  // i3: real 0, pred 1 → FP (1).  i4..i7: real 0, pred 0 → TN.
  // Para positiveClass = 1: TP=1, FP=1, FN=2.
  //   P = 1/(1+1) = 0.5.   R = 1/(1+2) = 1/3.   F1 = 2·P·R/(P+R) = (1/3)/(5/6) = 2/5.
  const yT = [1, 1, 1, 0, 0, 0, 0, 0]
  const yP = [1, 0, 0, 1, 0, 0, 0, 0]

  it('precision con positiveClass = 1 vale 1/2', () => {
    expect(precision(yT, yP, 1)).toBeCloseTo(1 / 2, 12)
  })

  it('recall con positiveClass = 1 vale 1/3', () => {
    expect(recall(yT, yP, 1)).toBeCloseTo(1 / 3, 12)
  })

  it('f1 con positiveClass = 1 vale 2/5', () => {
    expect(f1Score(yT, yP, 1)).toBeCloseTo(2 / 5, 12)
  })

  it('positiveClass = 0 usa la clase 0 como positiva', () => {
    // Predichos 0: i1, i2, i4, i5, i6, i7 (6). De ellos reales 0: i4..i7 (4). P0 = 4/6 = 2/3.
    // Reales 0: i3..i7 (5). Predichos 0 entre ellos: i4..i7 (4). R0 = 4/5.
    expect(precision(yT, yP, 0)).toBeCloseTo(2 / 3, 12)
    expect(recall(yT, yP, 0)).toBeCloseTo(4 / 5, 12)
  })

  it('sin positiveClass: precision y recall son medias macro sobre K=2', () => {
    // Macro precision = (P0 + P1)/2 = (2/3 + 1/2)/2 = 7/12.
    // Macro recall    = (R0 + R1)/2 = (4/5 + 1/3)/2 = 17/30.
    expect(precision(yT, yP)).toBeCloseTo(7 / 12, 12)
    expect(recall(yT, yP)).toBeCloseTo(17 / 30, 12)
  })

  it('sin positiveClass: f1 macro = media de las F1 por clase (convención sklearn average="macro")', () => {
    // Clase 0: TP=4 (i4..i7), FP=2 (i1,i2), FN=1 (i3). P0 = 4/6 = 2/3, R0 = 4/5.
    //   F1_0 = 2·(2/3)·(4/5) / ((2/3)+(4/5)) = (16/15) / (22/15) = 8/11.
    // Clase 1: P1 = 1/2, R1 = 1/3, F1_1 = 2/5 (como en el caso binario de arriba).
    // Macro F1 = (8/11 + 2/5) / 2 = ((40 + 22)/55) / 2 = 62/110 = 31/55 ≈ 0.5636.
    // Cambio decidido (T036, Q2 del usuario): antes se calculaba 2PR/(P+R) con P y R macro (119/207).
    expect(f1Score(yT, yP)).toBeCloseTo(31 / 55, 12)
  })

  it('f1 vale 0 cuando precision + recall = 0', () => {
    // yTrue=[0,0], yPred=[1,1], positiveClass 0: no hay predicción 0 (P=0) ni TP (R=0). p+r = 0 → 0.
    expect(f1Score([0, 0], [1, 1], 0)).toBe(0)
  })
})

describe('rocCurve / auc', () => {
  it('clasificador perfecto: AUC = 1', () => {
    // Pares (pos, neg) con pos > neg: positivos {0.8, 0.9} frente a negativos {0.1, 0.2}: 4 de 4.
    // AUC = 4/4 = 1 (Mann-Whitney), y la regla trapezoidal da lo mismo.
    expect(auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])).toBeCloseTo(1, 12)
  })

  it('clasificador invertido: AUC = 0', () => {
    // Positivos {0.1, 0.2} frente a negativos {0.8, 0.9}: ningún par pos > neg. AUC = 0/4 = 0.
    expect(auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])).toBeCloseTo(0, 12)
  })

  it('caso intermedio: AUC = 0.75', () => {
    // yTrue=[1,0,1,0], scores=[0.9,0.8,0.7,0.6]. Positivos {0.9, 0.7}, negativos {0.8, 0.6}.
    // Pares: (0.9,0.8) sí, (0.9,0.6) sí, (0.7,0.8) no, (0.7,0.6) sí → 3/4 = 0.75.
    expect(auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])).toBeCloseTo(0.75, 12)
  })

  it('rocCurve en el caso intermedio: puntos (fpr, tpr) y umbrales a mano', () => {
    // Umbrales desc: 0.9, 0.8, 0.7, 0.6, más uno por encima del máximo: 0.9+1 = 1.9.
    // P=2, N=2. Para cada umbral t, pred = score ≥ t:
    //   t=1.9: ninguno           → (fpr 0,   tpr 0)
    //   t=0.9: 0.9(+)            → (fpr 0,   tpr 1/2)
    //   t=0.8: 0.9(+), 0.8(-)    → (fpr 1/2, tpr 1/2)
    //   t=0.7: + 0.7(+)          → (fpr 1/2, tpr 1)
    //   t=0.6: + 0.6(-)          → (fpr 1,   tpr 1)
    const curve = rocCurve([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
    expect(curve.map((p) => p.threshold)).toEqual([1.9, 0.9, 0.8, 0.7, 0.6])
    expect(curve.map((p) => p.fpr)).toEqual([0, 0, 0.5, 0.5, 1])
    expect(curve.map((p) => p.tpr)).toEqual([0, 0.5, 0.5, 1, 1])
  })
})

// ─── Regression ───────────────────────────────────────────────────────────────

describe('mae / rmse', () => {
  // yTrue=[1,2,3], yPred=[2,2,5]. Errores absolutos: 1, 0, 2.
  it('mae = (1+0+2)/3 = 1', () => {
    expect(mae([1, 2, 3], [2, 2, 5])).toBe(1)
  })

  it('rmse = √((1+0+4)/3) = √(5/3)', () => {
    // Errores al cuadrado: 1, 0, 4. Media = 5/3. Raíz = √(5/3) ≈ 1.2909944487.
    expect(rmse([1, 2, 3], [2, 2, 5])).toBeCloseTo(Math.sqrt(5 / 3), 12)
  })
})

describe('r2Score', () => {
  // yTrue=[1,2,3]: media 2. SS_tot = (1-2)² + 0 + (3-2)² = 2.

  it('predicción perfecta → R² = 1', () => {
    // SS_res = 0 → R² = 1 - 0/2 = 1
    expect(r2Score([1, 2, 3], [1, 2, 3])).toBe(1)
  })

  it('predecir siempre la media → R² = 0', () => {
    // yPred=[2,2,2]: SS_res = 1 + 0 + 1 = 2 → R² = 1 - 2/2 = 0
    expect(r2Score([1, 2, 3], [2, 2, 2])).toBe(0)
  })

  it('caso intermedio: yPred=[1,2,2] → R² = 1/2', () => {
    // SS_res = 0 + 0 + (3-2)² = 1 → R² = 1 - 1/2 = 0.5
    expect(r2Score([1, 2, 3], [1, 2, 2])).toBe(0.5)
  })
})

// ─── Language / Sequences ─────────────────────────────────────────────────────

describe('perplexity', () => {
  it('distribución uniforme 1/V → perplexity = V', () => {
    // V=4, T=3, P(real)=1/4 en cada paso. PP = exp(-(1/3)·3·ln(1/4)) = 4.
    const probs = [
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
    ]
    expect(perplexity([0, 1, 2], probs)).toBeCloseTo(4, 10)
  })

  it('caso no uniforme: PP = √(8/3)', () => {
    // P(real)=0.5 en t=0 y 0.75 en t=1. Producto = 3/8.
    // PP = exp(-(1/2)·ln(3/8)) = (3/8)^(-1/2) = √(8/3) ≈ 1.6329931619.
    expect(perplexity([0, 1], [[0.5, 0.5], [0.25, 0.75]])).toBeCloseTo(Math.sqrt(8 / 3), 10)
  })

  it('probabilidad 0 se recorta a eps = 1e-15: PP finita ≈ 1e15', () => {
    // P=0 → max(1e-15, 0) = 1e-15. PP = exp(-ln(1e-15)) = 1e15.
    const p = perplexity([0], [[0, 1]])
    expect(Number.isFinite(p)).toBe(true)
    expect(Math.abs(p / 1e15 - 1)).toBeLessThan(1e-9)
  })
})

// ─── Visualisation ────────────────────────────────────────────────────────────

describe('printConfusionMatrix', () => {
  it('imprime la matriz con el formato de columnas de width 6', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    printConfusionMatrix([
      [1, 0],
      [1, 1],
    ])

    // K=2, etiquetas '0','1', colW = max(6, 1) = 6. Cabecera = 6 espacios + 2 + '0'.padStart(6) + 2 + '1'.padStart(6).
    // Ancho de la cabecera = 6+2+6+2+6 = 22. Separador = 22 guiones.
    // Fila 0 (real 0): '0' + valores 1, 0. Fila 1 (real 1): '1' + valores 1, 1.
    const header = ' '.repeat(6) + '  ' + '0'.padStart(6) + '  ' + '1'.padStart(6)
    const expected = [
      '',
      'Confusion Matrix (rows = actual, cols = predicted):',
      header,
      '─'.repeat(22),
      '0'.padStart(6) + '  ' + '1'.padStart(6) + '  ' + '0'.padStart(6),
      '1'.padStart(6) + '  ' + '1'.padStart(6) + '  ' + '1'.padStart(6),
      '',
    ]
    expect(spy.mock.calls.map((c) => c[0])).toEqual(expected)
  })
})

describe('classificationReport', () => {
  it('imprime precision, recall, f1 y soporte por clase y macro', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(() => {})
    // yTrue=[0,0,1,1], yPred=[0,1,1,1].
    // Clase 0: TP=1, FP=0, FN=1 → P=1, R=1/2, F1=2·1·(1/2)/(3/2)=2/3, soporte 2.
    // Clase 1: TP=2, FP=1, FN=0 → P=2/3, R=1, F1=2·(2/3)·1/(5/3)=4/5, soporte 2.
    // Macro: P=(1+2/3)/2=5/6≈0.8333, R=(1/2+1)/2=3/4=0.7500, F1=(2/3+4/5)/2=11/15≈0.7333, soporte 4.
    classificationReport([0, 0, 1, 1], [0, 1, 1, 1])

    const rows = spy.mock.calls.map((c) => c[0] as string)
    const row = (label: string): string | undefined => rows.find((r) => r.startsWith(label))

    // colW = max(10, len('class_0')) = 10. Cada cifra: toFixed(4).padStart(10); soporte: String(n).padStart(10).
    expect(row('class_0')).toBe(
      'class_0'.padEnd(10) + '1.0000'.padStart(10) + '0.5000'.padStart(10) + '0.6667'.padStart(10) + '2'.padStart(10),
    )
    expect(row('class_1')).toBe(
      'class_1'.padEnd(10) + '0.6667'.padStart(10) + '1.0000'.padStart(10) + '0.8000'.padStart(10) + '2'.padStart(10),
    )
    expect(row('macro avg')).toBe(
      'macro avg'.padEnd(10) + '0.8333'.padStart(10) + '0.7500'.padStart(10) + '0.7333'.padStart(10) + '4'.padStart(10),
    )
  })
})

// ─── Degenerate inputs (T036: lanzan en vez de devolver NaN) ──────────────────

describe('degenerate inputs', () => {
  it('mae con longitudes distintas lanza Metrics.mae (antes devolvía NaN)', () => {
    // yTrue tiene 2 elementos e yPred 1: se rechaza antes de sumar (ver tests/api/Metrics.validation.test.ts).
    expect(() => mae([1, 2], [1])).toThrow(/^Metrics\.mae:/)
  })
})
