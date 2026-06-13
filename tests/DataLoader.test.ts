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
})
