import { describe, it, expect } from 'vitest'
import { DecisionTree } from '../src/DecisionTree'

// Minimal view of the private tree, only to pin hand-derived structure.
type Node = { isLeaf: boolean; value?: number; featureIndex?: number; threshold?: number; left?: Node; right?: Node }
const rootOf = (t: DecisionTree): Node => (t as unknown as { _root: Node })._root
const depthOf = (n: Node): number => (n.isLeaf ? 0 : 1 + Math.max(depthOf(n.left!), depthOf(n.right!)))
const accuracy = (t: DecisionTree, X: number[][], y: number[]): number =>
  t.predictBatch(X).filter((p, i) => p === y[i]).length / y.length

const B = [[0, 0], [0, 1], [1, 0], [1, 1]]

describe('DecisionTree', () => {
  describe('constructor and errors', () => {
    it('rejects non-positive maxDepth', () => {
      expect(() => new DecisionTree({ maxDepth: 0 })).toThrow('maxDepth must be positive')
      expect(() => new DecisionTree({ maxDepth: -3 })).toThrow('maxDepth must be positive')
    })

    it('rejects minSamplesSplit < 2', () => {
      expect(() => new DecisionTree({ minSamplesSplit: 1 })).toThrow('minSamplesSplit must be at least 2')
    })

    it('fit rejects empty X and length mismatch', () => {
      const t = new DecisionTree()
      expect(() => t.fit([], [])).toThrow('X is empty')
      expect(() => t.fit([[0], [1]], [0])).toThrow('X has 2 rows but y has 1 labels')
    })

    it('predict and predictBatch throw before fit', () => {
      const t = new DecisionTree()
      expect(() => t.predict([0])).toThrow('has not been fitted')
      expect(() => t.predictBatch([[0]])).toThrow('has not been fitted')
    })
  })

  describe('classification trees deduced by hand', () => {
    it('AND: root splits at 0.5 and the tree is perfect', () => {
      // Parent Gini = 1 - (0.75^2 + 0.25^2) = 0.375.
      // Split x0 <= 0.5: left {0,0} G=0, right {0,1} G=0.5 -> 0.5*0 + 0.5*0.5 = 0.25 < 0.375.
      // (x1 gives the same 0.25, so either feature is a valid root; threshold = (0+1)/2.)
      const t = new DecisionTree()
      t.fit(B, [0, 0, 0, 1])
      const root = rootOf(t)
      expect(root.isLeaf).toBe(false)
      expect([0, 1]).toContain(root.featureIndex)
      expect(root.threshold).toBe(0.5)
      expect(t.predictBatch(B)).toEqual([0, 0, 0, 1])
      // Off-grid points follow the thresholds at 0.5.
      expect(t.predict([0.9, 0.9])).toBe(1)
      expect(t.predict([0.1, 0.9])).toBe(0)
      expect(t.predict([0.9, 0.1])).toBe(0)
    })

    it('OR: perfect, root threshold 0.5', () => {
      // Parent Gini = 0.375; x0 <= 0.5: left {0,1} 0.5, right {1,1} 0 -> 0.25.
      const t = new DecisionTree()
      t.fit(B, [0, 1, 1, 1])
      expect(rootOf(t).threshold).toBe(0.5)
      expect(t.predictBatch(B)).toEqual([0, 1, 1, 1])
    })

    it('3 classes in 1-D: tie between thresholds 1.5 and 3.5 keeps the first', () => {
      // y = [0,0,1,1,2,2]; parent G = 1 - 3*(1/3)^2 = 2/3.
      // x <= 1.5: left {0,0} G=0, right {1,1,2,2} G=0.5 -> (4/6)*0.5 = 1/3.
      // x <= 3.5: left {0,0,1,1} G=0.5, right {2,2} G=0 -> 1/3 (tie).
      // x <= 2.5: 4/9 ; x <= 0.5: (5/6)*0.64 = 0.533. So 1.5 wins by being first.
      const X = [[0], [1], [2], [3], [4], [5]]
      const y = [0, 0, 1, 1, 2, 2]
      const t = new DecisionTree()
      t.fit(X, y)
      expect(rootOf(t).threshold).toBe(1.5)
      expect(t.predictBatch(X)).toEqual(y)
      expect(t.predict([0.4])).toBe(0)
      expect(t.predict([2.6])).toBe(1)
      expect(t.predict([4.2])).toBe(2)
    })

    it('predict uses <= : a point exactly on the threshold goes left', () => {
      // X = 1..6 as 1-D, y = [0,0,0,1,1,1]: threshold (3+4)/2 = 3.5; Gini 0.5 -> 0.
      const t = new DecisionTree()
      t.fit([[1], [2], [3], [4], [5], [6]], [0, 0, 0, 1, 1, 1])
      expect(rootOf(t).threshold).toBe(3.5)
      expect(t.predict([3.5])).toBe(0) // x <= 3.5 -> left
      expect(t.predict([3.5000001])).toBe(1)
    })

    it('pure node and unsplittable node become leaves', () => {
      const pure = new DecisionTree()
      pure.fit([[1], [2], [3]], [7, 7, 7])
      expect(rootOf(pure).isLeaf).toBe(true)
      expect(pure.predict([100])).toBe(7)

      // Same x, different labels: no candidate threshold; majority tie keeps the first class seen.
      const same = new DecisionTree()
      same.fit([[1], [1]], [0, 1])
      expect(rootOf(same).isLeaf).toBe(true)
      expect(same.predict([1])).toBe(0)
    })
  })

  describe('XOR (zero-gain splits allowed)', () => {
    const yXor = [0, 1, 1, 0]

    it('maxDepth 2 reaches accuracy 1.0', () => {
      // Root: parent G = 0.5; every split gives 0.5*0.5 + 0.5*0.5 = 0.5 (zero gain), accepted by <=.
      // Each child is {0,1} (G=0.5); the other feature splits it to G=0 (gain 0.5).
      const t = new DecisionTree({ maxDepth: 2 })
      t.fit(B, yXor)
      expect(accuracy(t, B, yXor)).toBe(1)
      expect(depthOf(rootOf(t))).toBe(2)
    })

    it('default depth also reaches accuracy 1.0', () => {
      const t = new DecisionTree()
      t.fit(B, yXor)
      expect(accuracy(t, B, yXor)).toBe(1)
    })

    it('maxDepth 1 is a stump with accuracy 0.5', () => {
      // Each leaf holds one 0 and one 1, so exactly 1 of its 2 points is right: 2/4 = 0.5.
      const t = new DecisionTree({ maxDepth: 1 })
      t.fit(B, yXor)
      expect(depthOf(rootOf(t))).toBe(1)
      expect(accuracy(t, B, yXor)).toBe(0.5)
    })
  })

  describe('maxDepth and minSamplesSplit', () => {
    it('maxDepth bounds the depth', () => {
      // 4 classes at x = 0..3: a full tree has depth 2 at least; with maxDepth 1 only one split.
      const X = [[0], [1], [2], [3]]
      const y = [0, 1, 2, 3]
      const d1 = new DecisionTree({ maxDepth: 1 })
      d1.fit(X, y)
      expect(depthOf(rootOf(d1))).toBe(1)
      const full = new DecisionTree()
      full.fit(X, y)
      expect(depthOf(rootOf(full))).toBeGreaterThanOrEqual(2)
      expect(full.predictBatch(X)).toEqual(y)
    })

    it('minSamplesSplit larger than n gives a majority leaf', () => {
      // y = [0,0,0,1] -> majority 0. With minSamplesSplit 5 > 4 samples: no split.
      const X = [[0], [1], [2], [3]]
      const y = [0, 0, 0, 1]
      const big = new DecisionTree({ minSamplesSplit: 5 })
      big.fit(X, y)
      expect(rootOf(big).isLeaf).toBe(true)
      expect(big.predictBatch(X)).toEqual([0, 0, 0, 0])
      // With the default (2) the 1 is isolated: x <= 2.5 -> {0,0,0}, x > 2.5 -> {1}.
      const small = new DecisionTree()
      small.fit(X, y)
      expect(small.predict([3])).toBe(1)
    })
  })

  describe('regression', () => {
    it('leaves are the mean of their samples (maxDepth 1)', () => {
      // x = [1,2,3,10,11,12], y = x. Best split between 3 and 10: threshold 6.5.
      // Leaves: mean(1,2,3) = 2 ; mean(10,11,12) = 11. Weighted MSE = (2/3 + 2/3)/2 = 2/3.
      const t = new DecisionTree({ task: 'regression', maxDepth: 1 })
      t.fit([[1], [2], [3], [10], [11], [12]], [1, 2, 3, 10, 11, 12])
      expect(rootOf(t).threshold).toBe(6.5)
      expect(t.predict([5])).toBe(2)
      expect(t.predict([8])).toBe(11)
      expect(t.predict([6.5])).toBe(2) // on the threshold -> left
    })

    it('mean differs from median when skewed', () => {
      // y = [1,2,6 | 100,101,108]. Split at 6.5: MSE left = (4+1+9)/3 = 14/3, right = (9+4+25)/3 = 38/3,
      // weighted 52/6 = 8.67; splitting after 2 values is far worse (~1182).
      // Leaves: mean(1,2,6) = 3 (median 2) ; mean(100,101,108) = 103 (median 101).
      const t = new DecisionTree({ task: 'regression', maxDepth: 1 })
      t.fit([[1], [2], [3], [10], [11], [12]], [1, 2, 6, 100, 101, 108])
      expect(rootOf(t).threshold).toBe(6.5)
      expect(t.predict([2])).toBeCloseTo(3, 12)
      expect(t.predict([11])).toBeCloseTo(103, 12)
    })

    it('unlimited depth interpolates the training targets', () => {
      const X = [[1], [2], [3], [10], [11], [12]]
      const y = [1, 2, 3, 10, 11, 12]
      const t = new DecisionTree({ task: 'regression' })
      t.fit(X, y)
      expect(t.predictBatch(X)).toEqual(y)
    })
  })
})
