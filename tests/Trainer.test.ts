import { describe, it, expect } from 'vitest'
import { Trainer } from '../src/Trainer'
import { NetworkN } from '../src/NetworkN'

describe('Trainer', () => {
  it('creates with default options', () => {
    const net = new NetworkN([2, 4, 1])
    const trainer = new Trainer(net)
    expect(trainer.epochs).toBe(1000)
    expect(trainer.lrInitial).toBe(0.1)
    expect(trainer.lrDecay).toBe(1.0)
  })

  it('creates with custom options', () => {
    const net = new NetworkN([2, 4, 1])
    const trainer = new Trainer(net, { epochs: 100, lr: 0.5, lrDecay: 0.99 })
    expect(trainer.epochs).toBe(100)
    expect(trainer.lrInitial).toBe(0.5)
    expect(trainer.lrDecay).toBe(0.99)
  })

  it('train returns history', () => {
    const net = new NetworkN([2, 4, 1])
    const trainer = new Trainer(net, { epochs: 10 })
    const history = trainer.train({
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      targets: [[0], [1], [1], [0]],
    })
    expect(history.length).toBe(10)
    expect(history.every(v => isFinite(v))).toBe(true)
  })

  it('getHistory returns copy of history', () => {
    const net = new NetworkN([2, 4, 1])
    const trainer = new Trainer(net, { epochs: 5 })
    trainer.train({
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      targets: [[0], [1], [1], [0]],
    })
    const history = trainer.getHistory()
    expect(history.length).toBe(5)
  })

  it('validates dataset', () => {
    const net = new NetworkN([2, 4, 1])
    const trainer = new Trainer(net, { epochs: 5 })
    expect(() => trainer.train({
      inputs: [[0, 0]],
      targets: [[0], [1]],
    })).toThrow()
  })

  it('loss decreases over training', () => {
    const net = new NetworkN([2, 8, 1])
    const trainer = new Trainer(net, { epochs: 500, lr: 0.3 })
    const history = trainer.train({
      inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
      targets: [[0], [1], [1], [0]],
    })
    // Loss should generally decrease
    const firstAvg = (history[0] + history[1] + history[2]) / 3
    const lastAvg = (history[497] + history[498] + history[499]) / 3
    expect(lastAvg).toBeLessThan(firstAvg)
  }, 30000)
})
