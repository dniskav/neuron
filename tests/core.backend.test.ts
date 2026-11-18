import { describe, it, expect, afterEach } from 'vitest'
import { defaultBackend, type Backend } from '../src/core/tensor'
import { getBackend, setBackend } from '../src/core/backend'
import { checkBackend } from '../src/core/checkBackend'

afterEach(() => {
  setBackend(null)
})

describe('checkBackend', () => {
  it('accepts defaultBackend with no failures', () => {
    const report = checkBackend(defaultBackend)
    expect(report.ok).toBe(true)
    expect(report.failures).toEqual([])
  })

  it('accepts a partial backend that only replaces matTVec with a correct version', () => {
    const report = checkBackend({ matTVec: defaultBackend.matTVec })
    expect(report.ok).toBe(true)
    expect(report.failures).toEqual([])
  })

  it('does not change the active backend', () => {
    checkBackend({ matVec: defaultBackend.matVec })
    expect(getBackend()).toBe(defaultBackend)
  })

  it('fails matVec when it adds 1e-6 too much', () => {
    const bad: Partial<Backend> = {
      matVec(W, rows, cols, x, b, out) {
        defaultBackend.matVec(W, rows, cols, x, b, out)
        for (let k = 0; k < rows; k++) out[k] += 1e-6
        return out
      },
    }
    const report = checkBackend(bad)
    expect(report.ok).toBe(false)
    const hit = report.failures.filter((f) => f.op === 'matVec')
    expect(hit.length).toBeGreaterThan(0)
    expect(report.failures.every((f) => f.op === 'matVec')).toBe(true)
    expect(hit[0].maxRelError).toBeGreaterThan(1e-12)
    expect(Number.isFinite(hit[0].maxRelError)).toBe(true)
  })

  it('fails matVec when it writes one element past out', () => {
    const bad: Partial<Backend> = {
      matVec(W, rows, cols, x, b, out) {
        defaultBackend.matVec(W, rows, cols, x, b, out)
        out[rows] = 1 // one element past the logical size, inside the padding
        return out
      },
    }
    const report = checkBackend(bad)
    expect(report.ok).toBe(false)
    const hit = report.failures.filter((f) => f.op === 'matVec')
    expect(hit.length).toBeGreaterThan(0)
    expect(hit[0].maxRelError).toBe(Infinity)
    expect(report.failures.every((f) => f.op === 'matVec')).toBe(true)
  })

  it('fails addOuter when it writes one element past G', () => {
    const bad: Partial<Backend> = {
      addOuter(G, rows, cols, d, x, s) {
        defaultBackend.addOuter(G, rows, cols, d, x, s)
        G[rows * cols] = 7
        return G
      },
    }
    const report = checkBackend(bad)
    expect(report.ok).toBe(false)
    expect(report.failures.some((f) => f.op === 'addOuter' && f.maxRelError === Infinity)).toBe(true)
  })

  it('fails when an input is modified', () => {
    const bad: Partial<Backend> = {
      matVec(W, rows, cols, x, b, out) {
        defaultBackend.matVec(W, rows, cols, x, b, out)
        W[0] += 1
        return out
      },
    }
    const report = checkBackend(bad)
    expect(report.ok).toBe(false)
    expect(report.failures.some((f) => f.op === 'matVec' && f.maxRelError === Infinity)).toBe(true)
  })

  it('fails when the operation returns something other than the buffer it was given', () => {
    const bad: Partial<Backend> = {
      matTVec(W, rows, cols, d, out) {
        defaultBackend.matTVec(W, rows, cols, d, out)
        return new Float64Array(out)
      },
    }
    const report = checkBackend(bad)
    expect(report.ok).toBe(false)
    expect(report.failures.every((f) => f.op === 'matTVec')).toBe(true)
  })

  it('reports an operation that throws as a failure', () => {
    const bad: Partial<Backend> = {
      matMulBatchT() {
        throw new Error('boom')
      },
    }
    const report = checkBackend(bad)
    expect(report.ok).toBe(false)
    expect(report.failures.some((f) => f.op === 'matMulBatchT' && f.maxRelError === Infinity)).toBe(true)
  })

  it('accepts an error of 1e-13 with the default tolerance', () => {
    const offByTiny: Partial<Backend> = {
      matVec(W, rows, cols, x, b, out) {
        defaultBackend.matVec(W, rows, cols, x, b, out)
        for (let k = 0; k < rows; k++) out[k] += 1e-13
        return out
      },
    }
    expect(checkBackend(offByTiny).ok).toBe(true)
  })

  it('respects an explicit tolerance and rejects the same 1e-13 error when it is stricter', () => {
    const offByTiny: Partial<Backend> = {
      matVec(W, rows, cols, x, b, out) {
        defaultBackend.matVec(W, rows, cols, x, b, out)
        for (let k = 0; k < rows; k++) out[k] += 1e-13
        return out
      },
    }
    const report = checkBackend(offByTiny, { tolerance: 1e-15 })
    expect(report.ok).toBe(false)
    expect(report.failures.some((f) => f.op === 'matVec')).toBe(true)
  })

  it('is deterministic for a fixed seed', () => {
    const bad: Partial<Backend> = {
      matVec(W, rows, cols, x, b, out) {
        defaultBackend.matVec(W, rows, cols, x, b, out)
        out[0] += 1e-6
        return out
      },
    }
    expect(checkBackend(bad, { seed: 3 })).toEqual(checkBackend(bad, { seed: 3 }))
  })
})
