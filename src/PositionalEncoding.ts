// ─── POSITIONAL ENCODING ──────────────────────────────────────────────────────
//
// Transformers process all tokens simultaneously — unlike RNNs, they have no
// built-in sense of order. Without positional information, "The cat sat on the
// mat" and "The mat sat on the cat" would produce identical representations.
//
// The fix: before feeding embeddings into the Transformer, add a position-
// dependent signal so each token knows where it sits in the sequence.
//
// Two strategies exist:
//   1. Sinusoidal (Vaswani et al., 2017) — deterministic, computed once,
//      generalizes to sequence lengths never seen during training.
//   2. Learned — a trainable embedding table [maxSeqLen × dModel] updated via
//      backprop. More expressive but bound to the training length.
//
// ─────────────────────────────────────────────────────────────────────────────

// ─── SINUSOIDAL POSITIONAL ENCODING ──────────────────────────────────────────
//
// Formula (Vaswani et al., "Attention Is All You Need", 2017):
//
//   PE[pos][2i]   = sin( pos / 10000^(2i / dModel) )
//   PE[pos][2i+1] = cos( pos / 10000^(2i / dModel) )
//
// where:
//   pos  — position of the token in the sequence (0, 1, 2, ...)
//   i    — dimension index in the embedding (0, 1, ..., dModel/2 - 1)
//   2i   — even dimension  → uses sine
//   2i+1 — odd  dimension  → uses cosine
//
// Why sines and cosines?
// ──────────────────────
// These are the only periodic functions that satisfy the key composition property:
//
//   PE[pos + k] = f( PE[pos] )   for any fixed offset k
//
// Concretely, using the angle-sum identities:
//
//   sin(pos + k) = sin(pos)·cos(k) + cos(pos)·sin(k)
//   cos(pos + k) = cos(pos)·cos(k) − sin(pos)·sin(k)
//
// This means PE[pos+k] is a fixed linear transformation of PE[pos].
// The Transformer's attention weights can therefore represent "distance k apart"
// as a dot-product operation over the position vectors — it doesn't need to
// learn absolute positions, it can learn relative offsets directly.
//
// Why different frequencies per dimension?
// ─────────────────────────────────────────
// The term 10000^(2i/dModel) creates a geometric progression of wavelengths,
// ranging from 2π (very high frequency, changes every token) at i=0 to
// 2π·10000 (very low frequency, changes slowly over long sequences) at i=dModel/2.
//
// This is analogous to binary encoding: low-indexed bits flip fast (high freq),
// high-indexed bits flip slowly (low freq). Together they uniquely identify
// every position over a wide range without ambiguity.
//
// Why does this generalize to unseen lengths?
// ───────────────────────────────────────────
// The encoding is purely mathematical — it has no learned parameters.
// A model trained on sequences of length 512 can still process length 1024
// because PE[pos] is well-defined for any pos. Learned encodings can't do this.
//
export class PositionalEncoding {
  // Compute the full PE vector for one token at position `pos`.
  // Returns an array of length `dModel`.
  //
  // Each pair of dimensions (2i, 2i+1) shares the same frequency 1/10000^(2i/dModel)
  // but is 90° out of phase (sin vs cos), which ensures no two positions produce
  // the identical vector.
  static encode(pos: number, dModel: number): number[] {
    const pe = new Array<number>(dModel);
    for (let i = 0; i < Math.floor(dModel / 2); i++) {
      // Denominator grows geometrically: slow-varying at large i, fast at small i.
      const freq = Math.pow(10000, (2 * i) / dModel);
      pe[2 * i]     = Math.sin(pos / freq);   // even dimension → sine
      pe[2 * i + 1] = Math.cos(pos / freq);   // odd  dimension → cosine
    }
    // Handle odd dModel: the last dimension gets only a sine term.
    if (dModel % 2 !== 0) {
      const i    = Math.floor(dModel / 2);
      const freq = Math.pow(10000, (2 * i) / dModel);
      pe[dModel - 1] = Math.sin(pos / freq);
    }
    return pe;
  }

  // Build the full positional encoding matrix for a sequence of `seqLen` tokens.
  // Returns shape [seqLen][dModel].
  //
  // In practice this matrix is computed once and cached — it doesn't change
  // across examples, batches, or epochs.
  static encodeSequence(seqLen: number, dModel: number): number[][] {
    return Array.from({ length: seqLen }, (_, pos) =>
      PositionalEncoding.encode(pos, dModel),
    );
  }

  // Add positional encoding to an existing embedding matrix (in-place on a copy).
  //
  // `embeddings` shape: [seqLen][dModel].
  // `seqLen` is optional; defaults to embeddings.length.
  //
  // The sum e = token_embedding + PE is what actually enters the first
  // Transformer layer. Summing (rather than concatenating) keeps the model
  // dimension fixed and lets the network distribute its capacity freely —
  // it can choose how much of each dimension to allocate to content vs. position.
  static apply(embeddings: number[][], seqLen?: number): number[][] {
    const len    = seqLen ?? embeddings.length;
    const dModel = embeddings[0].length;
    const pe     = PositionalEncoding.encodeSequence(len, dModel);
    return embeddings.map((emb, pos) =>
      emb.map((val, d) => val + pe[pos][d]),
    );
  }
}

// ─── LEARNED POSITIONAL ENCODING ─────────────────────────────────────────────
//
// Instead of fixed sinusoids, maintain a trainable weight matrix
//   W_pos: [maxSeqLen × dModel]
// where row `pos` is the learned encoding for position `pos`.
//
// Pros vs. sinusoidal:
//   + The model can shape position signals to match its task's distance needs.
//   + May outperform sinusoidal on fixed-length benchmarks (e.g., BERT uses this).
//
// Cons vs. sinusoidal:
//   − Requires max sequence length to be fixed at construction time.
//   − Cannot generalize to positions not seen during training.
//   − Adds maxSeqLen × dModel parameters (e.g. 512 × 512 = 262 144 extra params).
//
// Initialization:
//   Small random values (Xavier-like, ±√(1/dModel)) so that at the start of
//   training the position signal is small but non-zero, breaking symmetry.
//
export class LearnedPositionalEncoding {
  // [maxSeqLen][dModel] — updated by gradient descent during training.
  weights: number[][];

  constructor(readonly maxSeqLen: number, readonly dModel: number) {
    const limit = Math.sqrt(1 / dModel);
    this.weights = Array.from({ length: maxSeqLen }, () =>
      Array.from({ length: dModel }, () => (Math.random() * 2 - 1) * limit),
    );
  }

  // Return the learned encoding for one position.
  // Returns a copy so callers cannot accidentally mutate the weight table.
  getEncoding(pos: number): number[] {
    if (pos >= this.maxSeqLen) {
      throw new Error(
        `Position ${pos} exceeds maxSeqLen=${this.maxSeqLen}. ` +
        `Learned encodings cannot generalize beyond their training length.`,
      );
    }
    return [...this.weights[pos]];
  }

  // Add learned positional encodings to `embeddings` (returns a new matrix).
  // Shape: [seqLen][dModel] → [seqLen][dModel].
  apply(embeddings: number[][], seqLen?: number): number[][] {
    const len = seqLen ?? embeddings.length;
    if (len > this.maxSeqLen) {
      throw new Error(
        `Sequence length ${len} exceeds maxSeqLen=${this.maxSeqLen}.`,
      );
    }
    return embeddings.map((emb, pos) =>
      emb.map((val, d) => val + this.weights[pos][d]),
    );
  }

  // Apply gradient update to position encoding weights.
  //
  // `dWeights` has the same shape as `weights`: [maxSeqLen][dModel].
  // Each entry is dL/dW_pos[pos][d] — the loss gradient w.r.t. that weight.
  //
  // Simple SGD is used here (matching EmbeddingMatrix in MatMul.ts):
  // position embeddings are updated every step for all positions in the batch,
  // so the sparse-update problem of token embeddings doesn't apply.
  update(dWeights: number[][], lr: number): void {
    for (let pos = 0; pos < this.maxSeqLen; pos++) {
      for (let d = 0; d < this.dModel; d++) {
        this.weights[pos][d] += lr * dWeights[pos][d];
      }
    }
  }
}
