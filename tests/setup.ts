import { afterEach, beforeEach } from 'vitest'
import { restoreRandom, seedRandom } from './helpers/seed'

// Deterministic Math.random for every test (see specs/002-gradient-oracle).
beforeEach(() => seedRandom())
afterEach(restoreRandom)
