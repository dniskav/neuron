import { describe, it, expect } from 'vitest'
import { ModelSaver } from '../src/ModelSaver'
import { NetworkN } from '../src/NetworkN'

describe('ModelSaver', () => {
  it('toJSON returns valid JSON', () => {
    const net = new NetworkN([2, 4, 1])
    const json = ModelSaver.toJSON(net)
    const data = JSON.parse(json)
    expect(data.weights).toBeDefined()
    expect(Array.isArray(data.weights)).toBe(true)
    expect(data.timestamp).toBeDefined()
  })

  it('fromJSON restores weights', () => {
    const net = new NetworkN([2, 4, 1])
    const origWeights = net.getWeights()
    const json = ModelSaver.toJSON(net)

    // Modify weights
    net.setWeights(origWeights.map(v => v + 1))

    // Restore from JSON
    ModelSaver.fromJSON(net, json)
    expect(net.getWeights()).toEqual(origWeights)
  })

  it('fromJSON validates data', () => {
    const net = new NetworkN([2, 4, 1])
    expect(() => ModelSaver.fromJSON(net, '{"invalid": true}')).toThrow()
  })

  it('saveToFile and loadFromFile work with callbacks', () => {
    const net = new NetworkN([2, 4, 1])
    const origWeights = net.getWeights()

    // Mock file storage
    let stored = ''
    const writeFn = (_path: string, data: string) => { stored = data }
    const readFn = (_path: string) => stored

    ModelSaver.saveToFile(net, 'model.json', writeFn)
    net.setWeights(origWeights.map(v => v + 1))
    ModelSaver.loadFromFile(net, 'model.json', readFn)

    expect(net.getWeights()).toEqual(origWeights)
  })
})
