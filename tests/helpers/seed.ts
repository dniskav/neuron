// Deterministic Math.random replacement for tests (mulberry32).
const originalRandom: () => number = Math.random

/** Returns a PRNG producing floats in [0, 1) from a 32-bit seed. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Replaces Math.random with a seeded generator. */
export function seedRandom(seed: number = 0x5eed): void {
  Math.random = mulberry32(seed)
}

/** Restores the original Math.random. */
export function restoreRandom(): void {
  Math.random = originalRandom
}

/** Runs fn with a seeded Math.random, restoring the previous one afterwards. */
export function withSeed<T>(seed: number, fn: () => T): T {
  const previous = Math.random
  seedRandom(seed)
  try {
    return fn()
  } finally {
    Math.random = previous
  }
}
