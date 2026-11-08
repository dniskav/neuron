// Test de oráculo de gradiente para Word2Vec.
//
// Perdida derivada del codigo: softmax COMPLETO sobre el vocabulario (no hay muestreo
// negativo ni nada aleatorio en train()). Skip-gram: L = -log softmax(h . W2)[ctx] con
// h = embeddings[centro]; CBOW: h = media de los embeddings del contexto, objetivo el
// centro. train() hace una actualizacion SGD por par (skip-gram) o por posicion (CBOW),
// en orden, y devuelve la perdida media por epoca.
//
// Aleatoriedad: solo buildVocab() usa Math.random (inicializacion de embeddings y W2).
// Los tests llaman buildVocab() antes; train() ya no consume Math.random. La semilla
// global (tests/setup.ts) cubre el resto. Por tanto el oraculo es valido: perdida
// determinista y reproducible, sin excepcion de Constitucion II.
//
// Convencion: el paso real es w -= lr * dL/dw, es decir w + lr*g con g = -dL/dw; un modelo
// correcto da coseno ~ +1. La perdida del oraculo es la SUMA de las perdidas de todos los
// pares de la frase (train() hace un paso por par, secuencialmente, sin promediar). Con lr
// pequeno la secuencia coincide con el gradiente de la suma a primer orden.
//
// dh de _skipgramStep/_cbowStep usa la W2 PREVIA: todos los gradientes se evalúan en el mismo
// punto y solo después se actualiza W2. Medido (5 semillas de init, 300 épocas, corpus de prueba):
//   - skip-gram con lr=0.1 no diverge: la pérdida final queda en ~0.52 (test de abajo).
//   - Los oráculos de TODA la frase (varios pares, un paso SGD secuencial por par) NO pueden
//     alcanzar coseno > 0.999 a lr alto aunque el código sea correcto: la secuencia de pasos
//     difiere del gradiente de la suma en O(lr) por el orden de los pares. Cosenos medidos
//     (peor de 5 semillas): skip-gram lr 0.05: 0.9982; lr 0.1: 0.9925; CBOW lr 0.05: 0.9994;
//     lr 0.1: 0.9975. Por eso el oráculo de frase se mantiene a lr=1e-3 (coseno ~1), y la
//     detección a lr alto se hace con pruebas de UN SOLO paso de UN par (lr=0.5, error máximo
//     1e-9 por componente contra diferencias finitas), que son exactas, y con la prueba de no
//     divergencia de skip-gram a lr=0.1.
//
// Helpers usados sin modificar: gradientOracle (bindParams, expectGradientMatches) y expectLearns.
import { describe, it, expect } from 'vitest'
import { Word2Vec } from '../../src/Word2Vec'
import { bindParams, expectGradientMatches } from '../helpers/gradientOracle'
import { expectLearns } from '../helpers/learns'
import { seedRandom } from '../helpers/seed'

type Internals = {
  _W2: number[][]
  _skipgramStep(center: number, target: number, lr: number): number
  _cbowStep(center: number, context: number[], lr: number): number
}

const SENTENCES = [['a', 'b', 'c', 'd']]
const DIM = 3

// Referencia independiente: -log softmax(h . W2)[target].
function ce(h: number[], W2: number[][], target: number): number {
  const V = W2[0].length
  const s = Array.from({ length: V }, (_, j) => h.reduce((acc, hd, d) => acc + hd * W2[d][j], 0))
  const z = s.reduce((acc, v) => acc + Math.exp(v), 0)
  return -Math.log(Math.exp(s[target]) / z)
}

// Suma de perdidas de todos los pares (centro, contexto) de la frase, con ventana 1 (skip-gram).
function skipgramSum(m: Word2Vec, W2: number[][], sent: string[]): number {
  const idx = sent.map((w) => m.vocab.get(w)!)
  let total = 0
  for (let t = 0; t < idx.length; t++) {
    for (const off of [-1, 1]) {
      const pos = t + off
      if (pos < 0 || pos >= idx.length) continue
      total += ce(m.embeddings[idx[t]], W2, idx[pos])
    }
  }
  return total
}

// Suma de perdidas CBOW (ventana 1): h = media del contexto, objetivo el centro.
function cbowSum(m: Word2Vec, W2: number[][], sent: string[]): number {
  const idx = sent.map((w) => m.vocab.get(w)!)
  let total = 0
  for (let t = 0; t < idx.length; t++) {
    const ctx = [-1, 1].map((o) => idx[t + o]).filter((v) => v !== undefined)
    const h = Array.from(
      { length: DIM },
      (_, d) => ctx.reduce((s, c) => s + m.embeddings[c][d], 0) / ctx.length,
    )
    total += ce(h, W2, idx[t])
  }
  return total
}

function build(model: 'skipgram' | 'cbow') {
  const m = new Word2Vec(DIM, { windowSize: 1, model })
  m.buildVocab(SENTENCES) // unico consumidor de Math.random
  const W2 = (m as unknown as Internals)._W2
  return { m, W2 }
}

function setup(model: 'skipgram' | 'cbow') {
  const { m, W2 } = build(model)
  const sum = model === 'skipgram' ? skipgramSum : cbowSum
  const binder = bindParams(m.embeddings, W2)
  return {
    m,
    loss: () => sum(m, W2, SENTENCES[0]),
    step: (lr: number) => void m.train(SENTENCES, lr, 1),
    getParams: binder.getParams,
    setParams: binder.setParams,
  }
}

describe('Word2Vec oraculo', () => {
  it('humo: train devuelve una perdida finita por epoca y deja embeddings finitos', () => {
    const m = new Word2Vec(DIM, { windowSize: 1 })
    const losses = m.train(SENTENCES, 0.05, 3)
    expect(losses).toHaveLength(3)
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(m.embeddings.flat().every(Number.isFinite)).toBe(true)
    expect(m.embeddings).toHaveLength(4)
  })

  it('comportamiento: con embeddings y W2 en cero la perdida inicial es log(V)', () => {
    const { m, W2 } = build('skipgram')
    m.embeddings.forEach((r) => r.fill(0))
    W2.forEach((r) => r.fill(0))
    const losses = m.train(SENTENCES, 0, 1) // lr=0: no cambia nada
    expect(losses[0]).toBeCloseTo(Math.log(4), 10)
  })

  it('oraculo skip-gram: el paso sigue -dL/dw (embeddings y W2)', () => {
    expectGradientMatches(setup('skipgram'))
  })

  it('oraculo CBOW: el paso sigue -dL/dw (embeddings y W2)', () => {
    expectGradientMatches(setup('cbow'))
  })

  it('expectLearns skip-gram: la perdida baja a menos de la mitad', () => {
    const { m, W2 } = build('skipgram')
    expectLearns({
      loss: () => skipgramSum(m, W2, SENTENCES[0]),
      step: () => void m.train(SENTENCES, 0.03, 1),
      steps: 300,
    })
  })

  it('expectLearns: la perdida devuelta por train baja entre la primera y la ultima epoca', () => {
    const m = new Word2Vec(DIM, { windowSize: 1 })
    const losses = m.train(SENTENCES, 0.03, 300)
    expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.5)
  })

  it('expectLearns CBOW: la perdida baja a menos de la mitad', () => {
    const { m, W2 } = build('cbow')
    expectLearns({
      loss: () => cbowSum(m, W2, SENTENCES[0]),
      step: () => void m.train(SENTENCES, 0.1, 1),
      steps: 300,
    })
  })
  // Un paso de UN par: delta = -lr * grad de la perdida de ese par, exacto (sin efecto de orden).
  function singleStep(model: 'skipgram' | 'cbow', seed: number, lr: number) {
    seedRandom(seed)
    const { m, W2 } = build(model)
    const it = m as unknown as Internals
    const pairLoss = () =>
      model === 'skipgram'
        ? ce(m.embeddings[1], W2, 2)
        : ce(
            Array.from({ length: DIM }, (_, d) => (m.embeddings[0][d] + m.embeddings[3][d]) / 2),
            W2,
            2,
          )
    const { getParams, setParams } = bindParams(m.embeddings, W2)
    const p0 = [...getParams()]
    const eps = 1e-5
    const grad = p0.map((_, i) => {
      const plus = [...p0]
      plus[i] += eps
      const minus = [...p0]
      minus[i] -= eps
      setParams(plus)
      const lp = pairLoss()
      setParams(minus)
      const lm = pairLoss()
      return (lp - lm) / (2 * eps)
    })
    setParams([...p0])
    if (model === 'skipgram') it._skipgramStep(1, 2, lr)
    else it._cbowStep(2, [0, 3], lr)
    const p1 = getParams()
    let maxErr = 0
    p0.forEach((v, i) => {
      maxErr = Math.max(maxErr, Math.abs(p1[i] - v - -lr * grad[i]))
    })
    return maxErr
  }

  for (const model of ['skipgram', 'cbow'] as const) {
    it(`un paso ${model} (lr=0.5, y x0.97/x1.03, 5 semillas): delta = -lr*grad exacto, 1e-9`, () => {
      for (const lr of [0.5, 0.5 * 0.97, 0.5 * 1.03]) {
        for (const seed of [1, 2, 3, 4, 5]) {
          expect(singleStep(model, seed, lr)).toBeLessThan(1e-9)
        }
      }
    })
  }

  it('skip-gram lr=0.1 no diverge (antes 1.35 -> 18.4; ahora ~0.52): final < 0.5 * inicial', () => {
    for (const lr of [0.1, 0.1 * 0.97, 0.1 * 1.03]) {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const m = new Word2Vec(DIM, { windowSize: 1 })
        const losses = m.train(SENTENCES, lr, 300)
        expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.5)
      }
    }
  })

  it('CBOW converge con lr=0.1 (5 semillas): final < 0.5 * inicial', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      seedRandom(seed)
      const m = new Word2Vec(DIM, { windowSize: 1, model: 'cbow' })
      const losses = m.train(SENTENCES, 0.1, 300)
      expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.5)
    }
  })
})
