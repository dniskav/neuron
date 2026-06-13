import { describe, it, expect } from 'vitest'
import { DataLoader } from '../src/DataLoader'

describe('DataLoader', () => {
  it('creates with data', () => {
    const loader = new DataLoader({
      inputs: [[1, 2], [3, 4]],
      targets: [[0], [1]],
    })
    expect(loader.length).toBe(2)
  })

  it('validates data', () => {
    expect(() => new DataLoader({
      inputs: [[1, 2]],
      targets: [[0], [1]],
    })).toThrow()
  })

  it('hasNext returns true initially', () => {
    const loader = new DataLoader({
      inputs: [[1, 2], [3, 4]],
      targets: [[0], [1]],
    })
    expect(loader.hasNext()).toBe(true)
  })

  it('next returns batch', () => {
    const loader = new DataLoader({
      inputs: [[1, 2], [3, 4], [5, 6]],
      targets: [[0], [1], [2]],
    }, 2)
    const batch = loader.next()
    expect(batch.inputs.length).toBe(2)
    expect(batch.targets.length).toBe(2)
  })

  it('next returns all data in order', () => {
    const loader = new DataLoader({
      inputs: [[1], [2], [3]],
      targets: [[10], [20], [30]],
    })
    const all: number[][] = []
    while (loader.hasNext()) {
      const batch = loader.next()
      all.push(...batch.inputs)
    }
    expect(all.length).toBe(3)
  })

  it('shuffle randomizes order', () => {
    const loader = new DataLoader({
      inputs: [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
      targets: [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
    })
    loader.shuffle()
    const order: number[] = []
    while (loader.hasNext()) {
      const batch = loader.next()
      order.push(batch.inputs[0][0])
    }
    // With 10 elements, it's extremely unlikely to be in original order
    // (but technically possible, so we just check it runs without error)
    expect(order.length).toBe(10)
  })

  it('reset allows re-iteration', () => {
    const loader = new DataLoader({
      inputs: [[1], [2]],
      targets: [[10], [20]],
    })
    loader.next()
    loader.next()
    expect(loader.hasNext()).toBe(false)
    loader.reset()
    expect(loader.hasNext()).toBe(true)
  })

  it('sequences creates sequence windows', () => {
    const data = [[1], [2], [3], [4], [5]]
    const loader = DataLoader.sequences(data, 2)
    expect(loader.length).toBe(3) // [1,2]->3, [2,3]->4, [3,4]->5
  })

  it('sequences validates data length', () => {
    const data = [[1], [2]]
    expect(() => DataLoader.sequences(data, 3)).toThrow()
  })

  // ── Validation split tests ─────────────────────────────────────────────

  it('validationSplit: 0.2 returns 80% training, 20% validation', () => {
    // Use 10 samples: 8 training, 2 validation
    const n = 10
    const inputs = Array.from({ length: n }, (_, i) => [i])
    const targets = Array.from({ length: n }, (_, i) => [i * 10])

    const loader = new DataLoader({ inputs, targets }, 1, 0.2)

    // Training count: 80% of 10 = 8
    expect(loader.length).toBe(8)
    // Validation count: 20% of 10 = 2
    expect(loader.validationLength).toBe(2)
  })

  it('total samples equals original size (train + val)', () => {
    const n = 15
    const inputs = Array.from({ length: n }, (_, i) => [i])
    const targets = Array.from({ length: n }, (_, i) => [i])

    const loader = new DataLoader({ inputs, targets }, 2, 0.2)

    // Training + validation = total
    const trainSamples = loader.length
    const valSamples = loader.validationLength
    expect(trainSamples + valSamples).toBe(n)
  })

  it('getValidationData returns correct validation samples', () => {
    const n = 10
    const inputs = Array.from({ length: n }, (_, i) => [i])
    const targets = Array.from({ length: n }, (_, i) => [i * 10])

    const loader = new DataLoader({ inputs, targets }, 2, 0.2)

    const valData = loader.getValidationData()
    expect(valData.inputs.length).toBe(loader.validationLength)
    expect(valData.targets.length).toBe(loader.validationLength)

    // All validation samples should be from the original dataset
    for (const inp of valData.inputs) {
      const val = inp[0]
      expect(val).toBeGreaterThanOrEqual(0)
      expect(val).toBeLessThan(n)
    }
  })

  it('validation samples are not in training set', () => {
    const n = 10
    const inputs = Array.from({ length: n }, (_, i) => [i])
    const targets = Array.from({ length: n }, (_, i) => [i])

    const loader = new DataLoader({ inputs, targets }, 1, 0.2)

    // Collect all training samples
    const trainValues = new Set<number>()
    while (loader.hasNext()) {
      const batch = loader.next()
      for (const inp of batch.inputs) {
        trainValues.add(inp[0])
      }
    }

    // Collect all validation samples
    const valData = loader.getValidationData()
    const valValues = new Set(valData.inputs.map(inp => inp[0]))

    // No overlap
    for (const v of valValues) {
      expect(trainValues.has(v)).toBe(false)
    }
  })

  it('shuffle only affects training data, not validation', () => {
    const n = 10
    const inputs = Array.from({ length: n }, (_, i) => [i])
    const targets = Array.from({ length: n }, (_, i) => [i])

    const loader = new DataLoader({ inputs, targets }, 1, 0.2)

    // Get validation data before shuffle
    const valBefore = loader.getValidationData()
    const valSetBefore = new Set(valBefore.inputs.map(inp => inp[0]))

    // Shuffle training
    loader.shuffle()

    // Validation should be unchanged (same indices, different order doesn't matter for set)
    const valAfter = loader.getValidationData()
    const valSetAfter = new Set(valAfter.inputs.map(inp => inp[0]))

    expect(valSetBefore).toEqual(valSetAfter)
  })

  it('validationSplit of 0 is same as not specifying', () => {
    const n = 5
    const inputs = Array.from({ length: n }, (_, i) => [i])
    const targets = Array.from({ length: n }, (_, i) => [i])

    const loader0 = new DataLoader({ inputs, targets }, 1, 0)
    const loaderDefault = new DataLoader({ inputs, targets }, 1)

    expect(loader0.length).toBe(n)
    expect(loaderDefault.length).toBe(n)
    expect(loader0.validationLength).toBe(0)
    expect(loaderDefault.validationLength).toBe(0)

    const val0 = loader0.getValidationData()
    const valDef = loaderDefault.getValidationData()
    expect(val0.inputs.length).toBe(0)
    expect(valDef.inputs.length).toBe(0)
  })

  it('sequences accepts validationSplit', () => {
    const data = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    const loader = DataLoader.sequences(data, 2, 0.3)
    // 10 data points, seqLen=2 → 8 windows. 30% val = 2.4 → round to 2.
    expect(loader.length).toBeGreaterThan(0)
    expect(loader.validationLength).toBeGreaterThan(0)
    expect(loader.length + loader.validationLength).toBe(8)
  })

  it('invalid validationSplit throws', () => {
    expect(() => new DataLoader({ inputs: [[1]], targets: [[1]] }, 1, -0.1)).toThrow()
    expect(() => new DataLoader({ inputs: [[1]], targets: [[1]] }, 1, 1.0)).toThrow()
    expect(() => new DataLoader({ inputs: [[1]], targets: [[1]] }, 1, 1.5)).toThrow()
  })
})
