import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { Tokenizer } from '../src/Tokenizer'
import { seedRandom, restoreRandom } from './helpers/seed'

// ─── Corpus de referencia (T007) ──────────────────────────────────────────────
// Modo 'word', lowercase por defecto.
//   "the cat sat" → the, cat, sat
//   "the dog sat" → the, dog, sat
// Frecuencias: the = 2, sat = 2, cat = 1, dog = 1.
// Orden: frecuencia descendente y, en empate, alfabético (localeCompare):
//   sat(2), the(2), cat(1), dog(1)
// Con las 4 especiales delante:
//   <PAD>=0 <UNK>=1 <BOS>=2 <EOS>=3 sat=4 the=5 cat=6 dog=7  → vocabSize 8
const CORPUS = ['the cat sat', 'the dog sat']
const VOCAB = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'sat', 'the', 'cat', 'dog']

function fittedWord(options: ConstructorParameters<typeof Tokenizer>[0] = {}): Tokenizer {
  return new Tokenizer(options).fit(CORPUS)
}

describe('Tokenizer', () => {
  beforeEach(() => seedRandom())
  afterEach(() => restoreRandom())

  // ── tokenize ───────────────────────────────────────────────────────────────
  describe('tokenize', () => {
    it("word: minúsculas y puntuación como token propio", () => {
      // "The Cat!" → lowercase "the cat!" → [the] [cat] [!]
      expect(fittedWord().tokenize('The Cat!')).toEqual(['the', 'cat', '!'])
    })

    it("word: el apóstrofo separa 'don't' en tres tokens", () => {
      // "don't stop" → [don] ['] [t] [stop]: el espacio no es token
      expect(fittedWord().tokenize("don't stop")).toEqual(['don', "'", 't', 'stop'])
    })

    it('lowercase: false conserva las mayúsculas', () => {
      // Sin lowercase, [A-Za-z] sigue casando (flag i), pero el texto no se baja.
      const tok = new Tokenizer({ lowercase: false })
      expect(tok.tokenize('The')).toEqual(['The'])
    })

    it("char: cada carácter es un token, incluido el espacio", () => {
      // "Ab c" → lowercase "ab c" → [a] [b] [ ] [c]  (4 tokens)
      expect(new Tokenizer({ mode: 'char' }).tokenize('Ab c')).toEqual(['a', 'b', ' ', 'c'])
    })

    it('whitespace: parte solo por espacios y deja la puntuación pegada', () => {
      // "  hi there!\tok\n" → split(/\s+/) → ["", hi, there!, ok, ""] → filter → [hi, there!, ok]
      expect(new Tokenizer({ mode: 'whitespace' }).tokenize('  hi there!\tok\n')).toEqual([
        'hi',
        'there!',
        'ok',
      ])
    })
  })

  // ── fit ────────────────────────────────────────────────────────────────────
  describe('fit', () => {
    it('isFitted pasa de false a true', () => {
      const tok = new Tokenizer()
      expect(tok.isFitted).toBe(false)
      tok.fit(CORPUS)
      expect(tok.isFitted).toBe(true)
    })

    it('construye el vocabulario deducido a mano (8 entradas, orden por frecuencia y alfabético)', () => {
      const tok = fittedWord()
      // vocabSize = 4 especiales + {sat, the, cat, dog} = 8
      expect(tok.vocabSize).toBe(8)
      expect(tok.getVocabulary()).toEqual(VOCAB)
    })

    it('asigna los IDs a mano: sat=4, the=5, cat=6, dog=7; especiales 0..3', () => {
      const tok = fittedWord()
      expect(tok.tokenToId('<PAD>')).toBe(0)
      expect(tok.tokenToId('<UNK>')).toBe(1)
      expect(tok.tokenToId('<BOS>')).toBe(2)
      expect(tok.tokenToId('<EOS>')).toBe(3)
      expect(tok.tokenToId('sat')).toBe(4)
      expect(tok.tokenToId('the')).toBe(5)
      expect(tok.tokenToId('cat')).toBe(6)
      expect(tok.tokenToId('dog')).toBe(7)
    })

    it('tokenToId devuelve undefined fuera del vocabulario e idToToken fuera de rango', () => {
      const tok = fittedWord()
      expect(tok.tokenToId('bird')).toBeUndefined()
      expect(tok.idToToken(6)).toBe('cat')
      expect(tok.idToToken(8)).toBeUndefined()
    })

    it('fit() otra vez reconstruye el vocabulario desde cero', () => {
      const tok = new Tokenizer().fit(['x y']).fit(['z'])
      // Solo las 4 especiales y 'z' (freq 1); 'x' e 'y' ya no existen.
      expect(tok.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'z'])
      expect(tok.tokenToId('x')).toBeUndefined()
    })

    it('maxVocab = 6 deja 6 − 4 especiales = 2 tokens, los más frecuentes (sat, the)', () => {
      const tok = fittedWord({ maxVocab: 6 })
      expect(tok.vocabSize).toBe(6)
      expect(tok.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'sat', 'the'])
      // cat y dog quedan fuera → UNK (id 1): "cat sat" → [1, 4]
      expect(tok.encode('cat sat')).toEqual([1, 4])
    })

    it('specialTokens añade especiales tras PAD/UNK/BOS/EOS, antes del vocabulario', () => {
      // 4 especiales + <CLS> (id 4) + 'a' (id 5); vocabSize 6
      const tok = new Tokenizer({ specialTokens: ['<CLS>'] }).fit(['a'])
      expect(tok.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CLS>', 'a'])
      expect(tok.tokenToId('<CLS>')).toBe(4)
      expect(tok.tokenToId('a')).toBe(5)
    })

    it('un especial repetido (<PAD>) no ocupa otro ID ni desplaza a los siguientes', () => {
      // <PAD> ya está en 0 → se ignora; <CLS> sigue en 4 y 'a' en 5 (no 6)
      const tok = new Tokenizer({ specialTokens: ['<PAD>', '<CLS>'] }).fit(['a'])
      expect(tok.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CLS>', 'a'])
      expect(tok.tokenToId('<PAD>')).toBe(0)
      expect(tok.tokenToId('<CLS>')).toBe(4)
      expect(tok.tokenToId('a')).toBe(5)
    })

    it('modo char: cada letra es un token, ordenado alfabéticamente (a=4, b=5)', () => {
      // "ba" → [b] [a], ambos freq 1 → orden alfabético: a, b
      const tok = new Tokenizer({ mode: 'char' }).fit(['ba'])
      expect(tok.tokenToId('a')).toBe(4)
      expect(tok.tokenToId('b')).toBe(5)
      expect(tok.vocabSize).toBe(6)
    })

    it('corpus vacío deja solo las 4 especiales', () => {
      const tok = new Tokenizer().fit([])
      expect(tok.vocabSize).toBe(4)
      expect(tok.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>'])
    })
  })

  // ── encode ─────────────────────────────────────────────────────────────────
  describe('encode', () => {
    it('convierte tokens conocidos a IDs: "the cat" → [5, 6]', () => {
      expect(fittedWord().encode('the cat')).toEqual([5, 6])
    })

    it('token no visto en fit → <UNK> (id 1): "the bird" → [5, 1]', () => {
      expect(fittedWord().encode('the bird')).toEqual([5, 1])
    })

    it('addBOS y addEOS añaden 2 (BOS) al inicio y 3 (EOS) al final', () => {
      const tok = fittedWord()
      expect(tok.encode('the cat', { addBOS: true, addEOS: true })).toEqual([2, 5, 6, 3])
      expect(tok.encode('the cat', { addBOS: true })).toEqual([2, 5, 6])
      expect(tok.encode('the cat', { addEOS: true })).toEqual([5, 6, 3])
    })

    it('texto vacío → [] y con marcas de secuencia → [BOS, EOS] = [2, 3]', () => {
      const tok = fittedWord()
      expect(tok.encode('')).toEqual([])
      expect(tok.encode('', { addBOS: true, addEOS: true })).toEqual([2, 3])
    })

    it('lanza si no se ha llamado a fit()', () => {
      expect(() => new Tokenizer().encode('a')).toThrow(/fit\(\)/)
    })
  })

  // ── encodeBatch ────────────────────────────────────────────────────────────
  describe('encodeBatch', () => {
    it('sin padTo deja las secuencias a su longitud natural', () => {
      // "the cat" → [5,6] (2); "sat" → [4] (1)
      expect(fittedWord().encodeBatch(['the cat', 'sat'])).toEqual([[5, 6], [4]])
    })

    it('padTo 4 rellena por la derecha con PAD (0)', () => {
      // [5,6] + 2 ceros; [4] + 3 ceros
      expect(fittedWord().encodeBatch(['the cat', 'sat'], { padTo: 4 })).toEqual([
        [5, 6, 0, 0],
        [4, 0, 0, 0],
      ])
    })

    it('padTo trunca por la derecha: "the cat sat" → [5,6,4] → [5,6]', () => {
      expect(fittedWord().encodeBatch(['the cat sat'], { padTo: 2 })).toEqual([[5, 6]])
    })

    it('padTo cuenta BOS y EOS: [2,5,6,3] + 2 ceros = longitud 6', () => {
      expect(
        fittedWord().encodeBatch(['the cat'], { padTo: 6, addBOS: true, addEOS: true })
      ).toEqual([[2, 5, 6, 3, 0, 0]])
    })
  })

  // ── decode ─────────────────────────────────────────────────────────────────
  describe('decode', () => {
    it('por defecto quita BOS y EOS: [2,5,6,3] → "the cat"', () => {
      expect(fittedWord().decode([2, 5, 6, 3])).toBe('the cat')
    })

    it('stripSpecial = false conserva las marcas: "<BOS> the cat <EOS>"', () => {
      expect(fittedWord().decode([2, 5, 6, 3], false)).toBe('<BOS> the cat <EOS>')
    })

    it('PAD se quita pero UNK no: [0,5,1] → "the <UNK>"', () => {
      expect(fittedWord().decode([0, 5, 1])).toBe('the <UNK>')
    })

    it('id desconocido → "<UNK>" (decisión Q3): [99] → "<UNK>", [5,99] → "the <UNK>"', () => {
      const tok = fittedWord()
      expect(tok.decode([99])).toBe('<UNK>')
      expect(tok.decode([5, 99])).toBe('the <UNK>')
    })

    it('modo word une con espacios; modo char une sin separador', () => {
      expect(fittedWord().decode([5, 4])).toBe('the sat')
      // char: a=4, b=5 → [5,4] = "ba"
      expect(new Tokenizer({ mode: 'char' }).fit(['ba']).decode([5, 4])).toBe('ba')
    })

    it('ida y vuelta: decode(encode("the dog sat")) = "the dog sat"', () => {
      const tok = fittedWord()
      expect(tok.decode(tok.encode('the dog sat'))).toBe('the dog sat')
    })

    it('lanza si no se ha llamado a fit()', () => {
      expect(() => new Tokenizer().decode([1])).toThrow(/fit\(\)/)
    })
  })

  // ── oneHot ─────────────────────────────────────────────────────────────────
  describe('oneHot', () => {
    it('cada fila tiene longitud vocabSize (8) con un 1 en la posición del ID', () => {
      // id 5 → posición 5; id 6 → posición 6
      expect(fittedWord().oneHot([5, 6])).toEqual([
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
      ])
    })

    it('IDs fuera de rango (8, −1) dan vector de ceros', () => {
      expect(fittedWord().oneHot([8, -1])).toEqual([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
      ])
    })

    it('secuencia vacía → []', () => {
      expect(fittedWord().oneHot([])).toEqual([])
    })

    it('lanza si no se ha llamado a fit()', () => {
      expect(() => new Tokenizer().oneHot([0])).toThrow(/fit\(\)/)
    })
  })

  // ── persistencia ───────────────────────────────────────────────────────────
  describe('toJSON / fromJSON', () => {
    it('toJSON devuelve el snapshot esperado a mano', () => {
      expect(fittedWord().toJSON()).toEqual({
        mode: 'word',
        lowercase: true,
        maxVocab: 0,
        token2id: {
          '<PAD>': 0,
          '<UNK>': 1,
          '<BOS>': 2,
          '<EOS>': 3,
          sat: 4,
          the: 5,
          cat: 6,
          dog: 7,
        },
      })
    })

    it('ida y vuelta por JSON: mismo vocabulario, mismos IDs y mismo snapshot', () => {
      const original = fittedWord()
      const restored = Tokenizer.fromJSON(JSON.parse(JSON.stringify(original.toJSON())))
      expect(restored.getVocabulary()).toEqual(VOCAB)
      expect(restored.encode('the cat', { addBOS: true, addEOS: true })).toEqual([2, 5, 6, 3])
      expect(restored.toJSON()).toEqual(original.toJSON())
    })

    it('restaura lowercase: un tokenizer con lowercase false sigue sin bajar a minúsculas', () => {
      // Corpus "Ab" en modo char con lowercase false: tokens "A" y "b".
      const original = new Tokenizer({ mode: 'char', lowercase: false }).fit(['Ab'])
      const restored = Tokenizer.fromJSON(JSON.parse(JSON.stringify(original.toJSON())))
      expect(restored.tokenize('A')).toEqual(['A'])
      expect(restored.encode('A')).toEqual(original.encode('A'))
      // Si lowercase se perdiera, 'A' → 'a' → UNK (id 1).
      expect(restored.encode('A')).not.toEqual([1])
    })

    it('restaura mode: el modo char sobrevive la vuelta (decode une sin espacios)', () => {
      // char: A=4, b=5 → decode [4,5] = "Ab"; en modo word sería "A b".
      const original = new Tokenizer({ mode: 'char', lowercase: false }).fit(['Ab'])
      const restored = Tokenizer.fromJSON(JSON.parse(JSON.stringify(original.toJSON())))
      expect(restored.decode([4, 5])).toBe('Ab')
    })

    it('restaura maxVocab: al reentrenar, el tope de 6 sigue aplicado (2 tokens + 4 especiales)', () => {
      const restored = Tokenizer.fromJSON(
        JSON.parse(JSON.stringify(fittedWord({ maxVocab: 6 }).toJSON()))
      )
      // Corpus nuevo, todas freq 1 → alfabético: here, new, words → se quedan here y new.
      restored.fit(['new words here'])
      expect(restored.vocabSize).toBe(6)
      expect(restored.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'here', 'new'])
    })

    it('toJSON lanza si no se ha llamado a fit()', () => {
      expect(() => new Tokenizer().toJSON()).toThrow(/fit\(\)/)
    })
  })
})
