// Tests de comportamiento de Word2Vec (src/Word2Vec.ts). El gradiente ya lo cubre
// tests/oracle/Word2Vec.oracle.test.ts; aqui se fijan tokenize, vocabulario, similitud,
// vecinos, ventana, minCount, el promedio de CBOW y que el modelo aprende.
//
// Regla de valores: cada cifra lleva su calculo a mano en un comentario.
// Semilla: tests/setup.ts fija Math.random antes de cada test; los tests de "aprende"
// repiten con 5 semillas explicitas. buildVocab es el unico consumidor de Math.random.
import { describe, it, expect } from 'vitest'
import { Word2Vec } from '../src/Word2Vec'
import { seedRandom } from './helpers/seed'

type Internals = {
  _W2: number[][]
  _skipgramStep(center: number, target: number, lr: number): number
  _cbowStep(center: number, context: number[], lr: number): number
}
const internals = (m: Word2Vec) => m as unknown as Internals

/** Modelo con vocabulario a,b,c,... y embeddings fijados a mano (dim = longitud de cada vector). */
function fixed(vectors: Record<string, number[]>): Word2Vec {
  const words = Object.keys(vectors)
  const dim = vectors[words[0]].length
  const m = new Word2Vec(dim)
  m.buildVocab([words])
  words.forEach((w) => (m.embeddings[m.vocab.get(w)!] = [...vectors[w]]))
  return m
}

describe('Word2Vec.tokenize', () => {
  it('pasa a minusculas, quita puntuacion y separa por espacios', () => {
    // "Hello, World! 42." -> "hello  world  42 " -> [hello, world, 42]
    expect(Word2Vec.tokenize('Hello, World! 42.')).toEqual(['hello', 'world', '42'])
  })

  it('conserva apostrofes y guiones dentro de la palabra', () => {
    expect(Word2Vec.tokenize("It's well-known")).toEqual(["it's", 'well-known'])
  })

  it('el guion bajo y los signos separan palabras; cadena vacia o solo signos da []', () => {
    // 'a_b' -> 'a b' (el guion bajo no esta en la lista permitida)
    expect(Word2Vec.tokenize('a_b')).toEqual(['a', 'b'])
    expect(Word2Vec.tokenize('')).toEqual([])
    expect(Word2Vec.tokenize('  ?! ,, ')).toEqual([])
  })
})

describe('Word2Vec.buildVocab', () => {
  it('asigna indices por orden de primera aparicion y dimensiona las matrices', () => {
    const m = new Word2Vec(4)
    m.buildVocab([['x', 'y', 'x'], ['z', 'y']])
    // orden de primera aparicion: x=0, y=1, z=2
    expect([...m.vocab.entries()]).toEqual([['x', 0], ['y', 1], ['z', 2]])
    expect(m.vocabSize).toBe(3)
    expect(m.embeddings).toHaveLength(3)
    expect(m.embeddings.every((r) => r.length === 4)).toBe(true)
    // W2 es [dim x vocab] = 4 x 3
    expect(internals(m)._W2).toHaveLength(4)
    expect(internals(m)._W2.every((r) => r.length === 3)).toBe(true)
  })

  it('la inicializacion respeta la cota Xavier |w| <= sqrt(1/fan)', () => {
    const m = new Word2Vec(4)
    m.buildVocab([['x', 'y', 'z', 'w']])
    // embeddings: sqrt(1/4) = 0.5 ; W2: sqrt(1/vocab) = sqrt(1/4) = 0.5
    expect(m.embeddings.flat().every((v) => Math.abs(v) <= 0.5)).toBe(true)
    expect(internals(m)._W2.flat().every((v) => Math.abs(v) <= 0.5)).toBe(true)
  })

  it('minCount descarta palabras raras', () => {
    const m = new Word2Vec(3, { minCount: 2 })
    // frecuencias: x=2, y=2, z=1 -> z se descarta
    m.buildVocab([['x', 'y', 'x'], ['z', 'y']])
    expect([...m.vocab.keys()]).toEqual(['x', 'y'])
    expect(m.vocabSize).toBe(2)
    expect(m.vocab.has('z')).toBe(false)
  })

  it('minCount por defecto (1) conserva todo', () => {
    const m = new Word2Vec(3)
    m.buildVocab([['x', 'y', 'x'], ['z', 'y']])
    expect(m.vocabSize).toBe(3)
  })

  it('lanza si el vocabulario queda vacio (minCount alto o corpus vacio)', () => {
    expect(() => new Word2Vec(3, { minCount: 5 }).buildVocab([['x', 'y']])).toThrow(/empty/)
    expect(() => new Word2Vec(3).buildVocab([])).toThrow(/empty/)
  })
})

describe('Word2Vec.getEmbedding / similarity', () => {
  it('getEmbedding devuelve un vector de dimension embeddingDim', () => {
    const m = new Word2Vec(5)
    m.buildVocab([['a', 'b']])
    expect(m.getEmbedding('a')).toHaveLength(5)
    expect(m.getEmbedding('b')).toBe(m.embeddings[1])
  })

  it('palabra desconocida lanza, tambien en similarity, mostSimilar y analogy', () => {
    const m = new Word2Vec(3)
    m.buildVocab([['a', 'b', 'c']])
    expect(() => m.getEmbedding('zzz')).toThrow(/unknown word "zzz"/)
    expect(() => m.similarity('a', 'zzz')).toThrow(/unknown/)
    expect(() => m.mostSimilar('zzz')).toThrow(/unknown/)
    expect(() => m.analogy('a', 'b', 'zzz')).toThrow(/unknown/)
  })

  it('coseno a mano con embeddings fijados', () => {
    const m = fixed({ p: [3, 4], q: [4, 3], r: [0, 5], s: [-3, -4], t: [6, 8], z: [0, 0] })
    // p.q = 12+12 = 24 ; |p|=|q|=5 -> 24/25 = 0.96
    expect(m.similarity('p', 'q')).toBeCloseTo(0.96, 12)
    // p.r = 20 ; |p|=5, |r|=5 -> 20/25 = 0.8
    expect(m.similarity('p', 'r')).toBeCloseTo(0.8, 12)
    // p.s = -9-16 = -25 -> -25/25 = -1
    expect(m.similarity('p', 's')).toBeCloseTo(-1, 12)
    // t = 2p: 18+32 = 50 ; |t|=10 -> 50/(5*10) = 1 (invariante a la escala; sin normalizar daria 50)
    expect(m.similarity('p', 't')).toBeCloseTo(1, 12)
    // vector cero: denominador < 1e-12 -> 0 (no NaN)
    expect(m.similarity('p', 'z')).toBe(0)
    expect(m.similarity('z', 'z')).toBe(0)
  })

  it('similitud de una palabra con ella misma = 1 y es simetrica', () => {
    const m = new Word2Vec(6)
    m.buildVocab([['a', 'b', 'c']])
    expect(m.similarity('a', 'a')).toBeCloseTo(1, 12)
    expect(m.similarity('a', 'b')).toBeCloseTo(m.similarity('b', 'a'), 14)
  })
})

describe('Word2Vec.mostSimilar / analogy', () => {
  // dim 2. Cosenos con a=[1,0]:
  //   e=[4,3] -> 4/(1*5) = 0.8 ; b=[3,4] -> 3/5 = 0.6 ; c=[0,5] -> 0 ; d=[-2,0] -> -2/2 = -1
  const m = fixed({ a: [1, 0], b: [3, 4], c: [0, 5], d: [-2, 0], e: [4, 3] })

  it('orden descendente por coseno y excluye la propia palabra', () => {
    const r = m.mostSimilar('a')
    expect(r.map((x) => x.word)).toEqual(['e', 'b', 'c', 'd'])
    expect(r.map((x) => x.score)).toEqual([
      expect.closeTo(0.8, 12),
      expect.closeTo(0.6, 12),
      expect.closeTo(0, 12),
      expect.closeTo(-1, 12),
    ])
  })

  it('topK limita el numero de resultados y conserva los mejores', () => {
    expect(m.mostSimilar('a', 2).map((x) => x.word)).toEqual(['e', 'b'])
    expect(m.mostSimilar('a', 1).map((x) => x.word)).toEqual(['e'])
    expect(m.mostSimilar('a', 100)).toHaveLength(4) // 5 palabras - la propia
  })

  it('analogy: positive1 - negative + positive2, excluyendo las tres palabras de entrada', () => {
    const w = fixed({
      king: [3, 1], man: [1, 1], woman: [1, 3], queen: [3, 3], x: [0, -1],
    })
    // target = [3,1]-[1,1]+[1,3] = [3,3]
    // queen: cos = 1 ; x: (0*3 + -1*3)/(1*sqrt(18)) = -3/4.2426 = -0.7071
    // 'man'=[1,1] tambien tendria cos 1 con el objetivo: si no se excluyera ganaria o empataria
    const r = w.analogy('king', 'man', 'woman')
    expect(r.map((x) => x.word)).toEqual(['queen', 'x'])
    expect(r[0].score).toBeCloseTo(1, 12)
    expect(r[1].score).toBeCloseTo(-Math.SQRT1_2, 12)
    expect(w.analogy('king', 'man', 'woman', 1).map((x) => x.word)).toEqual(['queen'])
  })
})

describe('Word2Vec.train: estructura', () => {
  it('devuelve una perdida por epoca y construye el vocabulario si falta', () => {
    const m = new Word2Vec(3, { windowSize: 1 })
    const losses = m.train([['a', 'b', 'c']], 0.01, 4)
    expect(losses).toHaveLength(4)
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(m.vocabSize).toBe(3)
  })

  it('con pesos en cero y lr=0 la perdida media es log(V) en cada epoca', () => {
    const m = new Word2Vec(2, { windowSize: 1 })
    m.buildVocab([['a', 'b', 'c', 'd']])
    m.embeddings.forEach((r) => r.fill(0))
    internals(m)._W2.forEach((r) => r.fill(0))
    // scores = 0 -> probs = 1/4 -> loss = -log(1/4) = log 4 por par; media = log 4
    const losses = m.train([['a', 'b', 'c', 'd']], 0, 3)
    // (tolerancia 1e-9: la perdida usa log(p + 1e-12))
    losses.forEach((l) => expect(l).toBeCloseTo(Math.log(4), 9))
  })

  it('no reconstruye el vocabulario si ya existe', () => {
    const m = new Word2Vec(3)
    m.buildVocab([['a', 'b', 'c']])
    const emb = m.embeddings
    m.train([['a', 'b', 'c']], 0.01, 1)
    expect(m.embeddings).toBe(emb)
  })

  it('sin pares (frase de una sola palabra) la perdida de la epoca es 0', () => {
    const m = new Word2Vec(3)
    expect(m.train([['solo']], 0.1, 2)).toEqual([0, 0])
  })

  it('palabras fuera del vocabulario se ignoran en el entrenamiento', () => {
    const m = new Word2Vec(2, { windowSize: 1 })
    m.buildVocab([['a', 'b']])
    // 'zz' no esta en el vocabulario: la frase queda [a, b] (2 indices)
    const calls: number[][] = []
    const it = internals(m)
    const orig = it._skipgramStep.bind(m)
    it._skipgramStep = (c, t, lr) => (calls.push([c, t]), orig(c, t, lr))
    m.train([['a', 'zz', 'b']], 0.01, 1)
    // pares: (a->b) y (b->a)
    expect(calls).toEqual([[0, 1], [1, 0]])
  })
})

describe('Word2Vec.train: ventana, minCount y modelo', () => {
  const SENT = [['a', 'b', 'c', 'd', 'e']]

  function skipPairs(windowSize: number) {
    const m = new Word2Vec(2, { windowSize })
    m.buildVocab(SENT)
    const calls: number[][] = []
    const it = internals(m)
    it._skipgramStep = (c, t) => (calls.push([c, t]), 0)
    m.train(SENT, 0.1, 1)
    return calls
  }

  it('skip-gram: windowSize 1 genera 2*(5-1) = 8 pares; windowSize 2 genera 2*(4+3) = 14', () => {
    // ventana 1: cada adyacente en ambos sentidos = 4 adyacencias * 2 = 8
    // ventana 2: distancia 1 -> 4 pares * 2 ; distancia 2 -> 3 pares * 2 ; 8 + 6 = 14
    expect(skipPairs(1)).toHaveLength(8)
    expect(skipPairs(2)).toHaveLength(14)
    // ventana 1: el centro 0 ('a') solo predice 'b'
    expect(skipPairs(1).filter(([c]) => c === 0)).toEqual([[0, 1]])
    // ventana 2: el centro 2 ('c') predice a,b,d,e en orden de offset
    expect(skipPairs(2).filter(([c]) => c === 2).map(([, t]) => t)).toEqual([0, 1, 3, 4])
  })

  it('ventana por defecto = 2', () => {
    const m = new Word2Vec(2)
    m.buildVocab(SENT)
    let n = 0
    internals(m)._skipgramStep = () => (n++, 0)
    m.train(SENT, 0.1, 1)
    expect(n).toBe(14)
  })

  it('CBOW: una llamada por posicion con el contexto de la ventana', () => {
    for (const [w, expected] of [
      // ventana 1: posiciones 0..4 -> [b], [a,c], [b,d], [c,e], [d]
      [1, [[1], [0, 2], [1, 3], [2, 4], [3]]],
      // ventana 2: [b,c], [a,c,d], [a,b,d,e], [b,c,e], [c,d]
      [2, [[1, 2], [0, 2, 3], [0, 1, 3, 4], [1, 2, 4], [2, 3]]],
    ] as [number, number[][]][]) {
      const m = new Word2Vec(2, { windowSize: w, model: 'cbow' })
      m.buildVocab(SENT)
      const calls: [number, number[]][] = []
      internals(m)._cbowStep = (c, ctx) => (calls.push([c, [...ctx]]), 0)
      m.train(SENT, 0.1, 1)
      expect(calls.map(([c]) => c)).toEqual([0, 1, 2, 3, 4])
      expect(calls.map(([, ctx]) => ctx)).toEqual(expected)
    }
  })

  it('CBOW no llama a skip-gram y viceversa', () => {
    const cb = new Word2Vec(2, { windowSize: 1, model: 'cbow' })
    cb.buildVocab(SENT)
    let sg = 0
    internals(cb)._skipgramStep = () => (sg++, 0)
    cb.train(SENT, 0.1, 1)
    expect(sg).toBe(0)

    const sk = new Word2Vec(2, { windowSize: 1 })
    sk.buildVocab(SENT)
    let cbn = 0
    internals(sk)._cbowStep = () => (cbn++, 0)
    sk.train(SENT, 0.1, 1)
    expect(cbn).toBe(0)
  })

  it('minCount: las palabras raras no generan pares ni aparecen como contexto', () => {
    const m = new Word2Vec(2, { windowSize: 2, minCount: 2 })
    // x=2, y=2, z=1 -> z fuera. Frases filtradas: [x,y,x] y [y]
    const sents = [['x', 'y', 'x'], ['z', 'y']]
    m.buildVocab(sents)
    const calls: number[][] = []
    internals(m)._skipgramStep = (c, t) => (calls.push([c, t]), 0)
    m.train(sents, 0.1, 1)
    // [x,y,x] (idx 0,1,0), ventana 2: t0 -> 1,0 ; t1 -> 0,0 ; t2 -> 0,1 ; total 6. [y] no tiene contexto
    expect(calls).toEqual([[0, 1], [0, 0], [1, 0], [1, 0], [0, 0], [0, 1]])
  })

  it('model por defecto = skipgram', () => {
    const m = new Word2Vec(2)
    m.buildVocab(SENT)
    let n = 0
    internals(m)._skipgramStep = () => (n++, 0)
    m.train(SENT, 0.1, 1)
    expect(n).toBeGreaterThan(0)
  })
})

describe('Word2Vec: un paso a mano (embeddings y W2 conocidos)', () => {
  // Estado: dim 2, V=3 (palabras a,b,c). W2 = [[1,0,0],[-0.5,0,0]].
  // Con h = [2,4]: scores_j = 2*W2[0][j] + 4*W2[1][j] = [2-2, 0, 0] = [0,0,0] -> probs = 1/3.
  function state(model: 'skipgram' | 'cbow') {
    const m = new Word2Vec(2, { model })
    m.buildVocab([['a', 'b', 'c']])
    internals(m)._W2.splice(0, 2, [1, 0, 0], [-0.5, 0, 0])
    return m
  }
  // objetivo = 1 -> err = probs - onehot = [1/3, -2/3, 1/3]
  // dh[0] = 1*(1/3) + 0 + 0 = 1/3 ; dh[1] = -0.5*(1/3) = -1/6
  // W2 con lr=0.6: W2[d][j] -= 0.6*h[d]*err[j]
  //   fila0: h=2 -> 1.2*err = [0.4,-0.8,0.4] -> [1-0.4, 0+0.8, 0-0.4] = [0.6, 0.8, -0.4]
  //   fila1: h=4 -> 2.4*err = [0.8,-1.6,0.8] -> [-0.5-0.8, 1.6, -0.8] = [-1.3, 1.6, -0.8]
  const W2_AFTER = [[0.6, 0.8, -0.4], [-1.3, 1.6, -0.8]]

  it('skip-gram: perdida log 3 y actualizacion de W2 y del embedding del centro', () => {
    const m = state('skipgram')
    m.embeddings[0] = [2, 4] // centro a ; objetivo b
    const loss = internals(m)._skipgramStep(0, 1, 0.6)
    expect(loss).toBeCloseTo(Math.log(3), 9) // log(p + 1e-12)
    // embedding centro: [2 - 0.6*(1/3), 4 - 0.6*(-1/6)] = [1.8, 4.1]
    expect(m.embeddings[0][0]).toBeCloseTo(1.8, 12)
    expect(m.embeddings[0][1]).toBeCloseTo(4.1, 12)
    internals(m)._W2.forEach((row, d) => row.forEach((v, j) => expect(v).toBeCloseTo(W2_AFTER[d][j], 12)))
  })

  it('CBOW: h es la media del contexto (k=2) y el gradiente se reparte entre k', () => {
    const m = state('cbow')
    m.embeddings[0] = [1, 3]
    m.embeddings[2] = [3, 5]
    // h = ([1,3] + [3,5]) / 2 = [2,4] ; centro b (idx 1)
    const loss = internals(m)._cbowStep(1, [0, 2], 0.6)
    expect(loss).toBeCloseTo(Math.log(3), 9) // log(p + 1e-12)
    // cada contexto: -= lr*dh/k = 0.6*[1/3, -1/6]/2 = [0.1, -0.05]
    expect(m.embeddings[0][0]).toBeCloseTo(0.9, 12)
    expect(m.embeddings[0][1]).toBeCloseTo(3.05, 12)
    expect(m.embeddings[2][0]).toBeCloseTo(2.9, 12)
    expect(m.embeddings[2][1]).toBeCloseTo(5.05, 12)
    internals(m)._W2.forEach((row, d) => row.forEach((v, j) => expect(v).toBeCloseTo(W2_AFTER[d][j], 12)))
  })

  it('CBOW con un solo contexto equivale al paso skip-gram (k=1, h = ese embedding)', () => {
    const a = state('skipgram')
    a.embeddings[0] = [2, 4]
    const la = internals(a)._skipgramStep(0, 1, 0.6)
    const b = state('cbow')
    b.embeddings[0] = [2, 4]
    const lb = internals(b)._cbowStep(1, [0], 0.6)
    expect(lb).toBeCloseTo(la, 12)
    expect(b.embeddings[0]).toEqual(a.embeddings[0])
    expect(internals(b)._W2).toEqual(internals(a)._W2)
  })
})

describe('Word2Vec: aprende', () => {
  // Corpus con dos grupos de sustitutos: cada animal aparece en "the X ran fast" y cada
  // vehiculo en "a Y drove slow"; 3 repeticiones de cada frase. Las palabras de un grupo
  // comparten contextos exactos, asi que sus vectores deben acercarse y alejarse de los del otro.
  const A = ['cat', 'dog', 'fox', 'cow']
  const B = ['car', 'bus', 'van', 'truck']
  const corpus: string[][] = []
  for (let r = 0; r < 3; r++) for (const x of A) corpus.push(['the', x, 'ran', 'fast'])
  for (let r = 0; r < 3; r++) for (const y of B) corpus.push(['a', y, 'drove', 'slow'])

  function gap(m: Word2Vec): { intra: number; inter: number } {
    const all = [...A, ...B]
    let intra = 0, ni = 0, inter = 0, nx = 0
    for (let i = 0; i < 8; i++) {
      for (let j = i + 1; j < 8; j++) {
        const s = m.similarity(all[i], all[j])
        if ((i < 4) === (j < 4)) { intra += s; ni++ } else { inter += s; nx++ }
      }
    }
    return { intra: intra / ni, inter: inter / nx }
  }

  // Umbral = diferencia intra - inter. Medido (fuera de este test, 5 semillas, lr x0.7/x1.3):
  // skip-gram lr 0.03, 100 epocas: intra >= 0.97, inter <= -0.11 -> diferencia >= 1.08.
  // Se exige 0.5 * 1.3 = 0.65 (30 % sobre el umbral 0.5). Un modelo sin entrenar da ~0.
  it('skip-gram (lr 0.03, x0.7, x1.3; 5 semillas): similitud intra-grupo supera a inter-grupo', () => {
    for (const lr of [0.03 * 0.7, 0.03, 0.03 * 1.3]) {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const m = new Word2Vec(8, { windowSize: 2, model: 'skipgram' })
        m.train(corpus, lr, 100)
        const { intra, inter } = gap(m)
        expect(intra - inter).toBeGreaterThan(0.5 * 1.3)
        expect(intra).toBeGreaterThan(0.8)
      }
    }
  })

  // CBOW: diferencia medida >= 0.40 con lr 0.1 y 100 epocas; umbral 0.2 * 1.3 = 0.26.
  it('CBOW (lr 0.1, x0.7, x1.3; 5 semillas): similitud intra-grupo supera a inter-grupo', () => {
    for (const lr of [0.1 * 0.7, 0.1, 0.1 * 1.3]) {
      for (const seed of [1, 2, 3, 4, 5]) {
        seedRandom(seed)
        const m = new Word2Vec(8, { windowSize: 2, model: 'cbow' })
        m.train(corpus, lr, 100)
        const { intra, inter } = gap(m)
        expect(intra - inter).toBeGreaterThan(0.2 * 1.3)
      }
    }
  })

  it('sin entrenar, intra e inter son indistinguibles (la diferencia es mucho menor)', () => {
    seedRandom(1)
    const m = new Word2Vec(8, { windowSize: 2 })
    m.buildVocab(corpus)
    const { intra, inter } = gap(m)
    expect(Math.abs(intra - inter)).toBeLessThan(0.5)
  })

  it('mostSimilar tras entrenar: el vecino mas cercano de "cat" es un animal', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      seedRandom(seed)
      const m = new Word2Vec(8, { windowSize: 2 })
      m.train(corpus, 0.03, 100)
      expect(A).toContain(m.mostSimilar('cat', 1)[0].word)
    }
  })

  // El corpus tiene entropia irreducible (cada palabra tiene varios objetivos posibles), asi
  // que la perdida no se puede reducir a la mitad. Inicial ~ log 12 = 2.48 (12 palabras).
  // Medido (5 semillas, lr x0.7/x1.3, 100 epocas): final/inicial skip-gram ~0.55, CBOW ~0.29.
  // Umbral 0.8 (el peor medido, 0.57, deja >= 30 % de margen).
  // Timeout 30 s: con cobertura v8 tarda ~2.0 s medido (T045); el de 5 s por defecto se agota a veces.
  it('la perdida de train baja por debajo de 0.8 de la inicial (skip-gram lr 0.03, CBOW lr 0.1)', () => {
    for (const [model, base] of [['skipgram', 0.03], ['cbow', 0.1]] as const) {
      for (const mult of [0.7, 1, 1.3]) {
        for (const seed of [1, 2, 3, 4, 5]) {
          seedRandom(seed)
          const m = new Word2Vec(8, { windowSize: 2, model })
          const losses = m.train(corpus, base * mult, 100)
          expect(losses[losses.length - 1]).toBeLessThan(losses[0] * 0.8)
        }
      }
    }
  }, 30000)
})
