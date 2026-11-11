import { describe, it, expect } from 'vitest'
import { Tokenizer } from '../../src/Tokenizer'

// ─── T049: correcciones de Tokenizer ──────────────────────────────────────────
// Corpus: "the cat sat" / "the dog sat". Frecuencias: sat 2, the 2, cat 1, dog 1.
// Orden: sat, the, cat, dog (frecuencia desc., luego alfabético).
// Con 4 especiales (PAD=0 UNK=1 BOS=2 EOS=3): sat=4 the=5 cat=6 dog=7.
// Con 1 especial extra <CLS>=4: sat=5 the=6 cat=7 dog=8.
const CORPUS = ['the cat sat', 'the dog sat']

describe('Tokenizer.fixes (T049)', () => {
  describe('golden de casos válidos (capturado con la base, antes del cambio)', () => {
    it('sin tope: vocabulario, encode y decode', () => {
      const t = new Tokenizer().fit(CORPUS)
      expect(t.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'sat', 'the', 'cat', 'dog'])
      // BOS=2, the=5, cat=6, "ab" y "zzz" desconocidos → UNK=1 dos veces, EOS=3
      expect(t.encode('the cat ab zzz', { addBOS: true, addEOS: true })).toEqual([2, 5, 6, 1, 1, 3])
      expect(t.decode(t.encode('the cat'), false)).toBe('the cat')
    })

    it('maxVocab = 6 (2 tokens + 4 especiales)', () => {
      const t = new Tokenizer({ maxVocab: 6 }).fit(CORPUS)
      expect(t.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'sat', 'the'])
      expect(t.encode('the cat')).toEqual([5, 1])
    })

    it('maxVocab = 4 (justo los especiales) sigue siendo válido', () => {
      const t = new Tokenizer({ maxVocab: 4 }).fit(CORPUS)
      expect(t.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>'])
    })

    it('especial propio con tope: maxVocab 7 = 5 especiales + 2 tokens', () => {
      const t = new Tokenizer({ specialTokens: ['<CLS>'], maxVocab: 7 }).fit(CORPUS)
      expect(t.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CLS>', 'sat', 'the'])
    })
  })

  describe('(a) maxVocab menor que el nº de especiales', () => {
    it('lanza con el mensaje exacto para maxVocab = 2 (4 especiales)', () => {
      expect(() => new Tokenizer({ maxVocab: 2 })).toThrow(
        'Tokenizer: maxVocab must be >= number of special tokens (4)'
      )
    })

    it('lanza con maxVocab = 3 (justo por debajo de 4)', () => {
      expect(() => new Tokenizer({ maxVocab: 3 })).toThrow(/\(4\)/)
    })

    it('cuenta los especiales propios: 4 + 1 = 5, así que maxVocab 4 lanza', () => {
      expect(() => new Tokenizer({ specialTokens: ['<CLS>'], maxVocab: 4 })).toThrow(/\(5\)/)
    })

    it('maxVocab 0 sigue significando sin tope', () => {
      const t = new Tokenizer({ maxVocab: 0 }).fit(CORPUS)
      expect(t.vocabSize).toBe(8)
    })
  })

  describe('(b) specialTokens en toJSON / fromJSON', () => {
    it('toJSON incluye specialTokens (vacío por defecto, [<CLS>] si se dieron)', () => {
      expect(new Tokenizer().fit(CORPUS).toJSON().specialTokens).toEqual([])
      expect(new Tokenizer({ specialTokens: ['<CLS>'] }).fit(CORPUS).toJSON().specialTokens).toEqual(['<CLS>'])
    })

    it('ida y vuelta: al reajustar, mismo vocabulario e IDs con especiales propios', () => {
      const original = new Tokenizer({ specialTokens: ['<CLS>', '<SEP>'] }).fit(CORPUS)
      const restored = Tokenizer.fromJSON(JSON.parse(JSON.stringify(original.toJSON())))
      restored.fit(CORPUS)
      // 4 + 2 especiales (ids 0..5), luego sat=6 the=7 cat=8 dog=9
      expect(restored.getVocabulary()).toEqual(original.getVocabulary())
      expect(restored.tokenToId('<SEP>')).toBe(5)
      expect(restored.tokenToId('sat')).toBe(6)
      expect(restored.encode('the dog')).toEqual(original.encode('the dog'))
      expect(restored.encode('the dog')).toEqual([7, 9])
    })

    it('un JSON antiguo sin specialTokens carga con los especiales por defecto', () => {
      const old = {
        mode: 'word' as const,
        lowercase: true,
        maxVocab: 0,
        token2id: { '<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, sat: 4, the: 5 },
      }
      const t = Tokenizer.fromJSON(old)
      expect(t.encode('the sat')).toEqual([5, 4])
      t.fit(CORPUS)
      expect(t.getVocabulary()).toEqual(['<PAD>', '<UNK>', '<BOS>', '<EOS>', 'sat', 'the', 'cat', 'dog'])
    })
  })
})
