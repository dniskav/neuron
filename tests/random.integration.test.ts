import { afterEach, describe, expect, it } from 'vitest'
import { NetworkN } from '../src/NetworkN'
import { seededRandom, setRandom } from '../src/random'
import { mulberry32, withSeed } from './helpers/seed'

const STRUCTURE = [3, 4, 2]

// Expected getWeights() of NetworkN(structure) with a mulberry32 stream of this seed.
// Built from the NeuronN initialization formula, not from a copied output: for each
// neuron, nIn weights (r * 2 - 1) * sqrt(1 / nIn) in order, then a bias of 0.
// Layer order and neuron order match NetworkN.getWeights().
function expectedWeights(seed: number, structure: number[]): number[] {
  const draw = mulberry32(seed)
  const out: number[] = []
  for (let i = 1; i < structure.length; i++) {
    const nIn = structure[i - 1]
    const limit = Math.sqrt(1 / nIn)
    for (let k = 0; k < structure[i]; k++) {
      for (let j = 0; j < nIn; j++) out.push((draw() * 2 - 1) * limit)
      out.push(0)
    }
  }
  return out
}

describe('random() injection reaches the models', () => {
  afterEach(() => setRandom(null))

  it('two networks built from the same injected seed have the same weights', () => {
    setRandom(seededRandom(7))
    const a = new NetworkN(STRUCTURE).getWeights()
    setRandom(seededRandom(7))
    const b = new NetworkN(STRUCTURE).getWeights()
    expect(b).toEqual(a)
  })

  it('an injected generator gives the NeuronN initialization formula over mulberry32', () => {
    setRandom(seededRandom(7))
    expect(new NetworkN(STRUCTURE).getWeights()).toEqual(expectedWeights(7, STRUCTURE))
  })

  it('without an injected generator, the models draw from Math.random in the same order', () => {
    setRandom(null)
    const w = withSeed(7, () => new NetworkN(STRUCTURE).getWeights())
    expect(w).toEqual(expectedWeights(7, STRUCTURE))
  })
})
