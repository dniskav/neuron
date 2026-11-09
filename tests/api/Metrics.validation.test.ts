import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { seedRandom, restoreRandom } from '../helpers/seed'
import {
  accuracy,
  confusionMatrix,
  precision,
  recall,
  f1Score,
  rocCurve,
  auc,
  mae,
  rmse,
  r2Score,
  classificationReport,
} from '../../src/Metrics'

// T036 (US5 / Q2, principio V): Metrics LANZA ante entradas degeneradas en vez de
// devolver NaN o un valor sin sentido. Todos los mensajes empiezan por "Metrics.<función>:".
// Metrics no es estocástico; la semilla se fija igualmente (Constitución IV).

beforeEach(() => seedRandom(0x5eed))
afterEach(() => {
  restoreRandom()
  vi.restoreAllMocks()
})

// ─── Longitudes distintas y vectores vacíos ───────────────────────────────────

describe('longitudes distintas', () => {
  it('accuracy, mae, rmse y r2Score lanzan con yTrue y yPred de distinta longitud', () => {
    // Antes: accuracy([0,1,1],[0,1]) = 0.667 (yPred[2] es undefined ≠ 1); mae/rmse daban NaN.
    expect(() => accuracy([0, 1, 1], [0, 1])).toThrow(/^Metrics\.accuracy:/)
    expect(() => mae([1, 2], [1])).toThrow(/^Metrics\.mae:/)
    expect(() => rmse([1, 2, 3], [1, 2])).toThrow(/^Metrics\.rmse:/)
    expect(() => r2Score([1, 2], [1, 2, 3])).toThrow(/^Metrics\.r2Score:/)
  })

  it('confusionMatrix, precision, recall, f1Score y classificationReport lanzan con longitudes distintas', () => {
    expect(() => confusionMatrix([0, 1, 1], [0, 1])).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => precision([0, 1, 1], [0, 1])).toThrow(/^Metrics\.precision:/)
    expect(() => recall([0, 1, 1], [0, 1])).toThrow(/^Metrics\.recall:/)
    expect(() => f1Score([0, 1, 1], [0, 1])).toThrow(/^Metrics\.f1Score:/)
    vi.spyOn(console, 'log').mockImplementation(() => {})
    expect(() => classificationReport([0, 1, 1], [0, 1])).toThrow(/^Metrics\.classificationReport:/)
  })

  it('rocCurve y auc lanzan con yTrue y yScores de distinta longitud', () => {
    expect(() => rocCurve([0, 1, 1], [0.1, 0.9])).toThrow(/^Metrics\.rocCurve:/)
    expect(() => auc([0, 1], [0.1, 0.9, 0.5])).toThrow(/^Metrics\.auc:/)
  })
})

describe('vectores vacíos', () => {
  it('todas las funciones de pares lanzan con entradas vacías', () => {
    // Antes: accuracy([],[]) = 0; confusionMatrix([],[]) = []; precision([],[]) = 0/-Infinity = -0.
    expect(() => accuracy([], [])).toThrow(/^Metrics\.accuracy:/)
    expect(() => confusionMatrix([], [])).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => precision([], [])).toThrow(/^Metrics\.precision:/)
    expect(() => recall([], [])).toThrow(/^Metrics\.recall:/)
    expect(() => f1Score([], [])).toThrow(/^Metrics\.f1Score:/)
    expect(() => rocCurve([], [])).toThrow(/^Metrics\.rocCurve:/)
    expect(() => auc([], [])).toThrow(/^Metrics\.auc:/)
    expect(() => mae([], [])).toThrow(/^Metrics\.mae:/)
    expect(() => rmse([], [])).toThrow(/^Metrics\.rmse:/)
    expect(() => r2Score([], [])).toThrow(/^Metrics\.r2Score:/)
  })
})

// ─── Etiquetas ────────────────────────────────────────────────────────────────

describe('etiquetas negativas o no enteras', () => {
  it('confusionMatrix, precision, recall y f1Score lanzan con una etiqueta negativa', () => {
    // Antes: confusionMatrix lanzaba TypeError ("Cannot read properties of undefined");
    // precision/recall/f1 ignoraban la muestra en silencio.
    expect(() => confusionMatrix([0, -1], [0, 1])).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => confusionMatrix([0, 1], [0, -1])).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => precision([0, -1], [0, 1])).toThrow(/^Metrics\.precision:/)
    expect(() => recall([0, 1], [0, -1])).toThrow(/^Metrics\.recall:/)
    expect(() => f1Score([0, -1], [0, 1])).toThrow(/^Metrics\.f1Score:/)
  })

  it('confusionMatrix, precision, recall y f1Score lanzan con etiquetas no enteras o NaN', () => {
    expect(() => confusionMatrix([0, 1.5], [0, 1])).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => confusionMatrix([0, 1], [0, NaN])).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => precision([0, 1], [0, 2.5])).toThrow(/^Metrics\.precision:/)
    expect(() => recall([0.5, 1], [0, 1])).toThrow(/^Metrics\.recall:/)
    expect(() => f1Score([0, 1], [0, Infinity])).toThrow(/^Metrics\.f1Score:/)
  })

  it('positiveClass negativo o no entero lanza en precision, recall y f1Score', () => {
    expect(() => precision([0, 1], [0, 1], -1)).toThrow(/^Metrics\.precision:/)
    expect(() => recall([0, 1], [0, 1], 0.5)).toThrow(/^Metrics\.recall:/)
    expect(() => f1Score([0, 1], [0, 1], -1)).toThrow(/^Metrics\.f1Score:/)
  })

  it('etiquetas válidas (enteros ≥ 0) no lanzan', () => {
    // yTrue=[0,2], yPred=[0,2]: K = 3. Matriz identidad parcial: C[0][0]=1, C[2][2]=1.
    expect(confusionMatrix([0, 2], [0, 2])).toEqual([
      [1, 0, 0],
      [0, 0, 0],
      [0, 0, 1],
    ])
  })
})

// ─── numClasses ───────────────────────────────────────────────────────────────

describe('numClasses', () => {
  it('lanza si una etiqueta es ≥ numClasses (antes: TypeError al indexar la matriz)', () => {
    // numClasses = 2 → índices válidos 0 y 1; la etiqueta 2 de yTrue queda fuera.
    expect(() => confusionMatrix([0, 2], [0, 0], 2)).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => confusionMatrix([0, 0], [0, 3], 3)).toThrow(/^Metrics\.confusionMatrix:/)
  })

  it('lanza si numClasses no es un entero positivo', () => {
    expect(() => confusionMatrix([0, 1], [0, 1], 0)).toThrow(/^Metrics\.confusionMatrix:/)
    expect(() => confusionMatrix([0, 1], [0, 1], 2.5)).toThrow(/^Metrics\.confusionMatrix:/)
  })
})

// ─── auc con una sola clase ───────────────────────────────────────────────────

describe('auc con una sola clase', () => {
  it('lanza si todas las etiquetas de yTrue son iguales', () => {
    // Antes: auc([1,1,1], …) = 0 sin aviso (sin negativos, FPR = 0 en todo el recorrido).
    expect(() => auc([1, 1, 1], [0.1, 0.5, 0.9])).toThrow(/^Metrics\.auc:/)
    expect(() => auc([0, 0], [0.2, 0.3])).toThrow(/^Metrics\.auc:/)
  })

  it('con las dos clases presentes calcula el área (no lanza)', () => {
    // Positivos {0.8, 0.9}, negativos {0.1, 0.2}: AUC = 1 (4 de 4 pares ordenados bien).
    expect(auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])).toBeCloseTo(1, 12)
  })
})

// ─── r2Score con yTrue constante ──────────────────────────────────────────────

describe('r2Score con yTrue constante', () => {
  // SS_tot = 0 (todos los yTrue valen 2). Convención sklearn (force_finite):
  // R² = 1 si SS_res = 0 (predicción perfecta), 0 si no.
  it('predicción perfecta sobre yTrue constante → 1', () => {
    // SS_res = 0+0+0 = 0 → 1.
    expect(r2Score([2, 2, 2], [2, 2, 2])).toBe(1)
  })

  it('predicción distinta sobre yTrue constante → 0 (antes devolvía 1)', () => {
    // yPred=[1,2,3]: SS_res = (2-1)² + (2-2)² + (2-3)² = 1 + 0 + 1 = 2 > 0 → 0.
    expect(r2Score([2, 2, 2], [1, 2, 3])).toBe(0)
    // yPred=[2,2,3]: SS_res = 0 + 0 + 1 = 1 > 0 → 0.
    expect(r2Score([2, 2, 2], [2, 2, 3])).toBe(0)
  })
})

// ─── Gran número de etiquetas (antes Math.max(...yTrue, ...yPred) desbordaba) ─

describe('200 000 etiquetas', () => {
  it('no desbordan la pila y dan los valores a mano', () => {
    const n = 200000
    const yT = Array.from({ length: n }, (_, i) => i % 2) // 0,1,0,1,…: 100 000 de cada clase
    const yP = new Array<number>(n).fill(0) // todas las predicciones son 0

    // Matriz: real 0 → pred 0 = 100000; real 1 → pred 0 = 100000; columna 1 vacía.
    expect(confusionMatrix(yT, yP)).toEqual([
      [100000, 0],
      [100000, 0],
    ])
    // Precision clase 0: TP/(TP+FP) = 100000/(100000+100000) = 1/2. Clase 1: sin predicciones → 0.
    // Macro = (1/2 + 0)/2 = 1/4.
    expect(precision(yT, yP)).toBeCloseTo(1 / 4, 12)
    // Recall clase 0: 100000/(100000+0) = 1. Clase 1: 0/100000 = 0. Macro = (1+0)/2 = 1/2.
    expect(recall(yT, yP)).toBeCloseTo(1 / 2, 12)
    // F1 clase 0: 2·(1/2)·1/((1/2)+1) = 2/3. Clase 1: P+R = 0 → 0. Media = (2/3 + 0)/2 = 1/3.
    expect(f1Score(yT, yP)).toBeCloseTo(1 / 3, 12)
  })
})
