// ─── WORD2VEC ─────────────────────────────────────────────────────────────────
//
// Learns dense vector representations of words (embeddings) from raw text.
// The core intuition: words appearing in similar contexts will end up with
// similar vectors — no labels needed, just distributional co-occurrence.
//
// Two architectures are supported:
//
//   Skip-gram  — given a center word, predict its surrounding context words.
//                Works well with rare words and small datasets.
//
//   CBOW       — given the surrounding context, predict the center word.
//   (Continuous Bag of Words)
//                Faster to train; smoother on frequent words.
//
// ─── NETWORK ARCHITECTURE ────────────────────────────────────────────────────
//
//   W1  [vocabSize × embeddingDim]   Embedding matrix. Each row is a word vector.
//   W2  [embeddingDim × vocabSize]   Output weight matrix.
//
//   Skip-gram forward:
//     h      = W1[centerIdx]               (lookup — no matrix multiply)
//     scores = h · W2                      [1 × vocabSize]
//     probs  = softmax(scores)             normalize to a probability distribution
//     loss   = -log(probs[targetIdx])      cross-entropy for each context word
//
//   CBOW forward:
//     h      = mean(W1[contextIdx_0], ..., W1[contextIdx_k])   average embeddings
//     scores = h · W2
//     probs  = softmax(scores)
//     loss   = -log(probs[centerIdx])
//
// ─── GRADIENTS ────────────────────────────────────────────────────────────────
//
//   err_j = probs_j - 1{j == target}      softmax + cross-entropy gradient
//   ∂L/∂W2 = h^T · err                    outer product  [embeddingDim × vocabSize]
//   ∂L/∂h  = W2 · err                     [embeddingDim]
//   ∂L/∂W1[centerIdx] += ∂L/∂h            accumulate over context words
//
// ─── ANALOGY ARITHMETIC ──────────────────────────────────────────────────────
//
//   The famous result:  king - man + woman ≈ queen
//   This emerges naturally because the embedding space encodes semantic
//   relationships as geometric offsets — no explicit supervision for analogies.
//
// ─────────────────────────────────────────────────────────────────────────────

export type Word2VecModel = 'skipgram' | 'cbow';

export interface Word2VecOptions {
  /** Size of the sliding context window on each side of the center word. Default 2. */
  windowSize?: number;
  /** Training architecture. Default 'skipgram'. */
  model?: Word2VecModel;
  /** Ignore words with corpus frequency below this threshold. Default 1. */
  minCount?: number;
}

export class Word2Vec {
  /** Learned word vectors, shape [vocabSize][embeddingDim]. */
  embeddings: number[][];
  /** Maps each vocabulary word to its integer index. */
  vocab: Map<string, number>;
  vocabSize: number;
  embeddingDim: number;

  private _indexToWord: string[];
  private _W2: number[][];            // output weight matrix [embeddingDim × vocabSize]
  private _windowSize: number;
  private _model: Word2VecModel;
  private _minCount: number;
  private _trained = false;

  constructor(embeddingDim = 50, options: Word2VecOptions = {}) {
    this.embeddingDim  = embeddingDim;
    this._windowSize   = options.windowSize ?? 2;
    this._model        = options.model      ?? 'skipgram';
    this._minCount     = options.minCount   ?? 1;

    this.embeddings    = [];
    this._W2           = [];
    this.vocab         = new Map();
    this._indexToWord  = [];
    this.vocabSize     = 0;
  }

  // ── buildVocab ─────────────────────────────────────────────────────────────
  // Scans the corpus, counts word frequencies, discards rare words (< minCount),
  // and assigns each remaining word a unique integer index.
  buildVocab(sentences: string[][]): void {
    const freq = new Map<string, number>();
    for (const sentence of sentences) {
      for (const word of sentence) {
        freq.set(word, (freq.get(word) ?? 0) + 1);
      }
    }

    this.vocab         = new Map();
    this._indexToWord  = [];

    for (const [word, count] of freq) {
      if (count >= this._minCount) {
        const idx = this._indexToWord.length;
        this.vocab.set(word, idx);
        this._indexToWord.push(word);
      }
    }

    this.vocabSize = this._indexToWord.length;
    if (this.vocabSize === 0) {
      throw new Error('Word2Vec.buildVocab: vocabulary is empty after applying minCount filter');
    }

    // ── Xavier initialization ───────────────────────────────────────────────
    // Variance = 1 / fan_in keeps the gradient signal from vanishing or exploding
    // during the first steps of training.
    //   std = sqrt(1 / embeddingDim)
    const scale1 = Math.sqrt(1 / this.embeddingDim);
    const scale2 = Math.sqrt(1 / this.vocabSize);

    this.embeddings = Array.from({ length: this.vocabSize }, () =>
      Array.from({ length: this.embeddingDim }, () => (Math.random() * 2 - 1) * scale1)
    );

    // W2 is transposed relative to embeddings: rows are embedding dims, cols are words.
    this._W2 = Array.from({ length: this.embeddingDim }, () =>
      Array.from({ length: this.vocabSize }, () => (Math.random() * 2 - 1) * scale2)
    );

    this._trained = false;
  }

  // ── tokenize ───────────────────────────────────────────────────────────────
  // Simple tokenizer: lowercase, strip punctuation, split on whitespace.
  // Returns an array of tokens suitable for buildVocab / train.
  static tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s'-]/g, ' ')  // keep apostrophes and hyphens inside words
      .split(/\s+/)
      .filter(t => t.length > 0);
  }

  // ── train ──────────────────────────────────────────────────────────────────
  // Runs SGD over all (center, context) pairs in the corpus for `epochs` passes.
  // Returns the average cross-entropy loss per epoch.
  //
  // Note: uses full-vocabulary softmax (not negative sampling) for educational
  // clarity. This is O(vocabSize) per step — for large vocabularies you would
  // normally switch to negative sampling or hierarchical softmax.
  train(sentences: string[][], lr = 0.025, epochs = 5): number[] {
    if (this.vocabSize === 0) this.buildVocab(sentences);

    const lossHistory: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let nPairs    = 0;

      for (const sentence of sentences) {
        // Filter out out-of-vocabulary tokens
        const indices = sentence
          .map(w => this.vocab.get(w))
          .filter((idx): idx is number => idx !== undefined);

        for (let t = 0; t < indices.length; t++) {
          const centerIdx = indices[t];

          // Collect context indices within the window (excluding center)
          const contextIndices: number[] = [];
          for (let offset = -this._windowSize; offset <= this._windowSize; offset++) {
            if (offset === 0) continue;
            const pos = t + offset;
            if (pos >= 0 && pos < indices.length) {
              contextIndices.push(indices[pos]);
            }
          }
          if (contextIndices.length === 0) continue;

          if (this._model === 'skipgram') {
            // Skip-gram: one center → predict each context word independently
            for (const contextIdx of contextIndices) {
              totalLoss += this._skipgramStep(centerIdx, contextIdx, lr);
              nPairs++;
            }
          } else {
            // CBOW: all context words → predict center word
            totalLoss += this._cbowStep(centerIdx, contextIndices, lr);
            nPairs++;
          }
        }
      }

      lossHistory.push(nPairs > 0 ? totalLoss / nPairs : 0);
    }

    this._trained = true;
    return lossHistory;
  }

  // ── getEmbedding ───────────────────────────────────────────────────────────
  // Returns the learned embedding vector for a word. Throws if unknown.
  getEmbedding(word: string): number[] {
    const idx = this.vocab.get(word);
    if (idx === undefined) throw new Error(`Word2Vec: unknown word "${word}"`);
    return this.embeddings[idx];
  }

  // ── similarity ─────────────────────────────────────────────────────────────
  // Cosine similarity between two words.
  //   cos(v1, v2) = (v1 · v2) / (‖v1‖ · ‖v2‖)
  // Returns a value in [-1, 1]. Higher → more similar context usage.
  similarity(word1: string, word2: string): number {
    const v1 = this.getEmbedding(word1);
    const v2 = this.getEmbedding(word2);
    return this._cosine(v1, v2);
  }

  // ── mostSimilar ────────────────────────────────────────────────────────────
  // Returns the topK words (excluding `word` itself) sorted by cosine similarity.
  mostSimilar(word: string, topK = 10): { word: string; score: number }[] {
    const v = this.getEmbedding(word);
    return this._nearestByVector(v, topK, new Set([word]));
  }

  // ── analogy ───────────────────────────────────────────────────────────────
  // Vector arithmetic analogy: positive1 - negative + positive2 ≈ result
  //
  //   getAnalogy('king', 'man', 'woman') finds the word closest to
  //   vec('king') - vec('man') + vec('woman') ≈ vec('queen')
  //
  // The result is excluded from the input words so they don't pollute the top-K.
  analogy(
    positive1: string,
    negative: string,
    positive2: string,
    topK = 5
  ): { word: string; score: number }[] {
    const vPos1 = this.getEmbedding(positive1);
    const vNeg  = this.getEmbedding(negative);
    const vPos2 = this.getEmbedding(positive2);

    // target = positive1 - negative + positive2
    const target = vPos1.map((v, i) => v - vNeg[i] + vPos2[i]);

    const exclude = new Set([positive1, negative, positive2]);
    return this._nearestByVector(target, topK, exclude);
  }

  // ── Private: skip-gram step ───────────────────────────────────────────────
  // Forward + backward for one (center, target) pair.
  // Returns the cross-entropy loss for this pair.
  private _skipgramStep(centerIdx: number, targetIdx: number, lr: number): number {
    // Forward: h = W1[centerIdx]  (embedding lookup)
    const h = this.embeddings[centerIdx];

    // scores_j = h · W2[:,j]  for all j in vocab
    const scores = this._hiddenToScores(h);

    // probs = softmax(scores)  — numerically stable (subtract max)
    const probs = _softmax(scores);

    // Loss = -log(probs[target])
    const loss = -Math.log(probs[targetIdx] + 1e-12);

    // ── Backward ──────────────────────────────────────────────────────────
    // Gradient of softmax + cross-entropy:
    //   err_j = probs_j - 1{j == target}
    const err = probs.map((p, j) => (j === targetIdx ? p - 1 : p));

    // ∂L/∂W2[d,j] = h[d] · err[j]
    // ∂L/∂h[d]    = Σ_j W2[d,j] · err[j]
    const dh = new Array(this.embeddingDim).fill(0);
    for (let d = 0; d < this.embeddingDim; d++) {
      for (let j = 0; j < this.vocabSize; j++) {
        // Update W2 in place
        this._W2[d][j] -= lr * h[d] * err[j];
        // Accumulate gradient for the embedding
        dh[d] += this._W2[d][j] * err[j];   // note: after W2 update — minor approx, ok for SGD
      }
    }

    // Update embedding for center word: W1[centerIdx] -= lr * dh
    for (let d = 0; d < this.embeddingDim; d++) {
      this.embeddings[centerIdx][d] -= lr * dh[d];
    }

    return loss;
  }

  // ── Private: CBOW step ────────────────────────────────────────────────────
  // Forward + backward for one (contextIndices → centerIdx) pair.
  // h is the mean of all context embeddings. The gradient is distributed
  // equally back to each context word's embedding row.
  private _cbowStep(centerIdx: number, contextIndices: number[], lr: number): number {
    const k = contextIndices.length;

    // Forward: h = (1/k) · Σ W1[contextIdx]
    const h = new Array(this.embeddingDim).fill(0);
    for (const ci of contextIndices) {
      for (let d = 0; d < this.embeddingDim; d++) {
        h[d] += this.embeddings[ci][d];
      }
    }
    for (let d = 0; d < this.embeddingDim; d++) h[d] /= k;

    // scores and softmax
    const scores = this._hiddenToScores(h);
    const probs  = _softmax(scores);

    const loss = -Math.log(probs[centerIdx] + 1e-12);

    // err_j = probs_j - 1{j == center}
    const err = probs.map((p, j) => (j === centerIdx ? p - 1 : p));

    // ∂L/∂h = Σ_j W2[d,j] · err[j]
    const dh = new Array(this.embeddingDim).fill(0);
    for (let d = 0; d < this.embeddingDim; d++) {
      for (let j = 0; j < this.vocabSize; j++) {
        this._W2[d][j] -= lr * h[d] * err[j];
        dh[d] += this._W2[d][j] * err[j];
      }
    }

    // Distribute gradient equally to each context embedding
    for (const ci of contextIndices) {
      for (let d = 0; d < this.embeddingDim; d++) {
        this.embeddings[ci][d] -= lr * dh[d] / k;
      }
    }

    return loss;
  }

  // Computes scores = h · W2  →  [vocabSize]
  private _hiddenToScores(h: number[]): number[] {
    const scores = new Array(this.vocabSize).fill(0);
    for (let d = 0; d < this.embeddingDim; d++) {
      for (let j = 0; j < this.vocabSize; j++) {
        scores[j] += h[d] * this._W2[d][j];
      }
    }
    return scores;
  }

  // Returns topK words (from all embeddings) sorted by cosine similarity to v,
  // skipping any word in the exclude set.
  private _nearestByVector(
    v: number[],
    topK: number,
    exclude: Set<string>
  ): { word: string; score: number }[] {
    const results: { word: string; score: number }[] = [];

    for (let i = 0; i < this.vocabSize; i++) {
      const w = this._indexToWord[i];
      if (exclude.has(w)) continue;
      results.push({ word: w, score: this._cosine(v, this.embeddings[i]) });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  // Cosine similarity: (v1 · v2) / (‖v1‖ · ‖v2‖)
  private _cosine(v1: number[], v2: number[]): number {
    let dot = 0, n1 = 0, n2 = 0;
    for (let i = 0; i < v1.length; i++) {
      dot += v1[i] * v2[i];
      n1  += v1[i] * v1[i];
      n2  += v2[i] * v2[i];
    }
    const denom = Math.sqrt(n1) * Math.sqrt(n2);
    return denom < 1e-12 ? 0 : dot / denom;
  }
}

// ─── SOFTMAX (numerically stable) ─────────────────────────────────────────────
// Subtract max before exp to prevent overflow. The constant cancels in the ratio.
//   softmax(x)_i = exp(x_i - max) / Σ_j exp(x_j - max)
function _softmax(scores: number[]): number[] {
  const max = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}
