// ─── Tokenizer ────────────────────────────────────────────────────────────────
//
// Converts raw text into sequences of integer token IDs that neural networks
// can process, and maps those IDs back to human-readable text.
//
// Three tokenization strategies (set via `mode`):
//   'char'       — each character is a token  ("hello" → ["h","e","l","l","o"])
//   'word'       — splits on whitespace + punctuation  ("hi there!" → ["hi","there","!"])
//   'whitespace' — splits only on whitespace, no punctuation handling
//
// Special tokens reserved at the start of every vocabulary:
//   <PAD>  (id 0) — padding for fixed-length batches
//   <UNK>  (id 1) — unknown tokens not seen during fit()
//   <BOS>  (id 2) — beginning of sequence marker
//   <EOS>  (id 3) — end of sequence marker
//
// Typical workflow:
//   const tok = new Tokenizer({ mode: 'word', lowercase: true })
//   tok.fit(["the cat sat on the mat", "the cat ate the rat"])
//   const ids  = tok.encode("the cat", { addBOS: true, addEOS: true })
//   const text = tok.decode(ids)
//   const batch = tok.encodeBatch(["hello", "world"], { padTo: 10 })
//
// Works naturally with Word2Vec (pass encoded sequences as training pairs)
// and with NetworkTransformer (embed token ids via EmbeddingMatrix).
//
// ─────────────────────────────────────────────────────────────────────────────

export type TokenizerMode = 'char' | 'word' | 'whitespace'

export interface TokenizerOptions {
  /** Tokenization strategy. Default: 'word' */
  mode?: TokenizerMode
  /** Normalize text to lowercase before processing. Default: true */
  lowercase?: boolean
  /** Maximum vocabulary size (most frequent tokens kept). 0 = unlimited. Default: 0 */
  maxVocab?: number
  /** Additional special tokens to reserve (appended after PAD/UNK/BOS/EOS). */
  specialTokens?: string[]
}

export interface EncodeOptions {
  /** Prepend <BOS> token. Default: false */
  addBOS?: boolean
  /** Append <EOS> token. Default: false */
  addEOS?: boolean
}

export interface EncodeBatchOptions extends EncodeOptions {
  /**
   * Pad or truncate all sequences to this length.
   * Sequences shorter than padTo are right-padded with <PAD> (id 0).
   * Sequences longer than padTo are truncated on the right.
   * If omitted, sequences are left at their natural length.
   */
  padTo?: number
}

// ─── Serialization snapshot ───────────────────────────────────────────────────
export interface TokenizerSnapshot {
  mode: TokenizerMode
  lowercase: boolean
  maxVocab: number
  token2id: Record<string, number>
}

// ─── Tokenizer ────────────────────────────────────────────────────────────────
export class Tokenizer {
  // ── Built-in special tokens ────────────────────────────────────────────────
  static readonly PAD = '<PAD>'
  static readonly UNK = '<UNK>'
  static readonly BOS = '<BOS>'
  static readonly EOS = '<EOS>'

  private readonly _mode: TokenizerMode
  private readonly _lowercase: boolean
  private readonly _maxVocab: number
  private readonly _extraSpecial: string[]

  private _token2id: Map<string, number> = new Map()
  private _id2token: Map<number, string>  = new Map()
  private _fitted = false

  constructor(options: TokenizerOptions = {}) {
    this._mode         = options.mode         ?? 'word'
    this._lowercase    = options.lowercase    ?? true
    this._maxVocab     = options.maxVocab     ?? 0
    this._extraSpecial = options.specialTokens ?? []
  }

  // ── Fit ───────────────────────────────────────────────────────────────────
  /**
   * Build vocabulary from an array of text strings.
   * Calling fit() again resets and rebuilds the vocabulary from scratch.
   *
   * @param texts - corpus to build the vocabulary from
   * @returns this (chainable)
   */
  fit(texts: string[]): this {
    // Reset
    this._token2id = new Map()
    this._id2token = new Map()

    // Register special tokens first so they always occupy the lowest IDs
    const specials = [
      Tokenizer.PAD,
      Tokenizer.UNK,
      Tokenizer.BOS,
      Tokenizer.EOS,
      ...this._extraSpecial,
    ]
    for (const s of specials) this._register(s)

    // Count token frequencies across the whole corpus
    const freq = new Map<string, number>()
    for (const text of texts) {
      for (const token of this.tokenize(text)) {
        freq.set(token, (freq.get(token) ?? 0) + 1)
      }
    }

    // Sort by frequency descending, then alphabetically for determinism
    let entries = [...freq.entries()].sort(
      ([a, fa], [b, fb]) => fb - fa || a.localeCompare(b)
    )

    // Apply vocab cap (excludes special tokens from the count)
    if (this._maxVocab > 0) {
      entries = entries.slice(0, this._maxVocab - specials.length)
    }

    for (const [token] of entries) this._register(token)

    this._fitted = true
    return this
  }

  // ── Tokenize ──────────────────────────────────────────────────────────────
  /**
   * Split raw text into an array of string tokens (no ID conversion yet).
   * Useful for inspecting what the tokenizer produces before encoding.
   */
  tokenize(text: string): string[] {
    const t = this._lowercase ? text.toLowerCase() : text

    switch (this._mode) {
      case 'char':
        return t.split('')

      case 'whitespace':
        return t.split(/\s+/).filter(Boolean)

      case 'word':
      default:
        // Split on word boundaries: keep letter/digit runs and punctuation marks
        // as separate tokens. E.g. "don't" → ["don", "'", "t"]
        return t.match(/[a-z0-9àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ]+|[^\w\s]/gi) ?? []
    }
  }

  // ── Encode ────────────────────────────────────────────────────────────────
  /**
   * Convert a text string to a sequence of token IDs.
   * Unknown tokens map to <UNK> (id 1).
   *
   * @param text    - input text
   * @param options - addBOS / addEOS flags
   */
  encode(text: string, options: EncodeOptions = {}): number[] {
    this._assertFitted()

    const ids: number[] = []
    if (options.addBOS) ids.push(this._token2id.get(Tokenizer.BOS)!)

    for (const token of this.tokenize(text)) {
      ids.push(this._token2id.get(token) ?? this._token2id.get(Tokenizer.UNK)!)
    }

    if (options.addEOS) ids.push(this._token2id.get(Tokenizer.EOS)!)
    return ids
  }

  // ── Encode batch ──────────────────────────────────────────────────────────
  /**
   * Encode an array of texts, optionally padding/truncating to a fixed length.
   *
   * @param texts   - array of input texts
   * @param options - addBOS / addEOS / padTo
   */
  encodeBatch(texts: string[], options: EncodeBatchOptions = {}): number[][] {
    const sequences = texts.map(t => this.encode(t, options))

    if (options.padTo !== undefined) {
      const len = options.padTo
      const padId = this._token2id.get(Tokenizer.PAD)!
      return sequences.map(seq => {
        if (seq.length >= len) return seq.slice(0, len)
        return [...seq, ...Array(len - seq.length).fill(padId)]
      })
    }

    return sequences
  }

  // ── Decode ────────────────────────────────────────────────────────────────
  /**
   * Convert a sequence of token IDs back to a human-readable string.
   *
   * @param ids          - array of token IDs
   * @param stripSpecial - remove PAD/BOS/EOS tokens from output. Default: true
   */
  decode(ids: number[], stripSpecial = true): string {
    this._assertFitted()

    const specials = new Set([Tokenizer.PAD, Tokenizer.BOS, Tokenizer.EOS])
    const tokens: string[] = []

    for (const id of ids) {
      const token = this._id2token.get(id) ?? Tokenizer.UNK
      if (stripSpecial && specials.has(token)) continue
      tokens.push(token)
    }

    return this._mode === 'char' ? tokens.join('') : tokens.join(' ')
  }

  // ── One-hot encoding ──────────────────────────────────────────────────────
  /**
   * Convert a sequence of token IDs to one-hot vectors.
   * Each vector has length `vocabSize` with a single 1 at the token's position.
   * Useful when feeding tokens directly into a Network without an embedding layer.
   *
   * @param ids - array of token IDs (e.g. from encode())
   * @returns   - 2D array of shape [seqLen, vocabSize]
   */
  oneHot(ids: number[]): number[][] {
    this._assertFitted()
    const V = this.vocabSize
    return ids.map(id => {
      const vec = new Array(V).fill(0)
      if (id >= 0 && id < V) vec[id] = 1
      return vec
    })
  }

  // ── Vocabulary helpers ────────────────────────────────────────────────────
  /** Number of tokens in the vocabulary (including special tokens). */
  get vocabSize(): number {
    return this._token2id.size
  }

  /** True if fit() has been called at least once. */
  get isFitted(): boolean {
    return this._fitted
  }

  /** Get the integer ID for a token string, or undefined if not in vocabulary. */
  tokenToId(token: string): number | undefined {
    return this._token2id.get(token)
  }

  /** Get the token string for an integer ID, or undefined if out of range. */
  idToToken(id: number): string | undefined {
    return this._id2token.get(id)
  }

  /**
   * Return the full vocabulary as an array ordered by ID.
   * Index i of the returned array is the token with ID i.
   */
  getVocabulary(): string[] {
    return Array.from({ length: this.vocabSize }, (_, i) => this._id2token.get(i)!)
  }

  // ── Persistence ───────────────────────────────────────────────────────────
  /**
   * Serialize the fitted tokenizer to a plain JSON-compatible object.
   * Store it with JSON.stringify(); reload with Tokenizer.fromJSON().
   */
  toJSON(): TokenizerSnapshot {
    this._assertFitted()
    return {
      mode:      this._mode,
      lowercase: this._lowercase,
      maxVocab:  this._maxVocab,
      token2id:  Object.fromEntries(this._token2id),
    }
  }

  /**
   * Restore a Tokenizer from a snapshot produced by toJSON().
   */
  static fromJSON(snapshot: TokenizerSnapshot): Tokenizer {
    const tok = new Tokenizer({
      mode:      snapshot.mode,
      lowercase: snapshot.lowercase,
      maxVocab:  snapshot.maxVocab,
    })
    for (const [token, id] of Object.entries(snapshot.token2id)) {
      tok._token2id.set(token, id)
      tok._id2token.set(id, token)
    }
    tok._fitted = true
    return tok
  }

  // ── Private ───────────────────────────────────────────────────────────────
  private _register(token: string): void {
    if (this._token2id.has(token)) return
    const id = this._token2id.size
    this._token2id.set(token, id)
    this._id2token.set(id, token)
  }

  private _assertFitted(): void {
    if (!this._fitted) {
      throw new Error('Tokenizer: call fit() before encoding or decoding.')
    }
  }
}
