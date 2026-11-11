import { describe, it, expect } from 'vitest';
import { Trainer } from '../src/Trainer';
import { NetworkN } from '../src/NetworkN';
import type { TrainableNetwork } from '../src/Trainer';
import { f1Score, precision, recall } from '../src/Metrics';

describe('Trainer', () => {
  // ── Basic tests (existing) ──────────────────────────────────────────────

  it('creates with default options', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net);
    expect(trainer.epochs).toBe(1000);
    expect(trainer.lrInitial).toBe(0.1);
    expect(trainer.lrDecay).toBe(1.0);
    expect(trainer.weightDecay).toBe(0);
  });

  it('creates with custom options', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, {
      epochs: 100,
      lr: 0.5,
      lrDecay: 0.99,
      weightDecay: 0.001,
    });
    expect(trainer.epochs).toBe(100);
    expect(trainer.lrInitial).toBe(0.5);
    expect(trainer.lrDecay).toBe(0.99);
    expect(trainer.weightDecay).toBe(0.001);
  });

  it('train returns history', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { epochs: 10 });
    const history = trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    });
    expect(history.length).toBe(10);
    expect(history.every((v) => isFinite(v))).toBe(true);
  });

  it('getHistory returns copy of history', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { epochs: 5 });
    trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    });
    const history = trainer.getHistory();
    expect(history.length).toBe(5);
  });

  it('validates dataset', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { epochs: 5 });
    expect(() =>
      trainer.train({
        inputs: [[0, 0]],
        targets: [[0], [1]],
      })
    ).toThrow();
  });

  it('loss decreases over training', () => {
    const net = new NetworkN([2, 8, 1]);
    const trainer = new Trainer(net, { epochs: 500, lr: 0.3 });
    const history = trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    });
    // Loss should generally decrease
    const firstAvg = (history[0] + history[1] + history[2]) / 3;
    const lastAvg =
      (history[497] + history[498] + history[499]) / 3;
    expect(lastAvg).toBeLessThan(firstAvg);
  }, 30000);

  // ── P2.1: Weight Decay ──────────────────────────────────────────────────

  it('weightDecay produces smaller final weights than without', () => {
    const netDecay = new NetworkN([2, 4, 1]);
    const netNoDecay = new NetworkN([2, 4, 1]);

    // Copy initial weights so both start identical
    const initWeights = netDecay.getWeights();
    netNoDecay.setWeights([...initWeights]);

    const data = {
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    };

    const trainerDecay = new Trainer(netDecay, {
      epochs: 200,
      lr: 0.3,
      weightDecay: 0.01,
    });
    trainerDecay.train(data);

    const trainerNoDecay = new Trainer(netNoDecay, {
      epochs: 200,
      lr: 0.3,
      weightDecay: 0,
    });
    trainerNoDecay.train(data);

    // Weight decay should produce smaller L2 norm
    const normDecay = Math.sqrt(
      netDecay.getWeights().reduce((s, w) => s + w * w, 0)
    );
    const normNoDecay = Math.sqrt(
      netNoDecay.getWeights().reduce((s, w) => s + w * w, 0)
    );

    expect(normDecay).toBeLessThan(normNoDecay);
  }, 30000);

  it('weightDecay with 0 does not affect training', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, {
      epochs: 20,
      lr: 0.1,
      weightDecay: 0,
    });
    const history = trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    });
    expect(history.length).toBe(20);
    expect(history.every((v) => isFinite(v))).toBe(true);
  });

  // ── P2.2: Early Stopping ────────────────────────────────────────────────

  it('early stopping stops training when validation loss plateaus', () => {
    const net = new NetworkN([2, 8, 1]);
    const trainer = new Trainer(net, {
      epochs: 2000,
      lr: 0.1,
      earlyStopping: { patience: 50, minDelta: 0.0001 },
    });

    // Provide validation data that is the same as training data
    const data = {
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    };
    trainer.setValidationData(data);

    const history = trainer.train(data);

    // Should stop early (well before 2000 epochs)
    expect(history.length).toBeLessThan(2000);
    expect(trainer.getStopReason()).toBe('earlyStopping');
  }, 30000);

  it('getBestLoss returns the best validation loss', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, {
      epochs: 500,
      lr: 0.1,
      earlyStopping: { patience: 200, minDelta: 0.0001 },
    });

    const data = {
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    };
    trainer.setValidationData(data);
    trainer.train(data);

    const bestLoss = trainer.getBestLoss();
    expect(bestLoss).toBeGreaterThan(0);
    expect(bestLoss).toBeLessThan(1);
  });

  it('getStopReason returns maxEpochs when no early stopping', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { epochs: 5 });
    trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
      ],
      targets: [[0], [1]],
    });
    expect(trainer.getStopReason()).toBe('maxEpochs');
  });

  it('getBestLoss returns -1 when no validation data', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { epochs: 5 });
    trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
      ],
      targets: [[0], [1]],
    });
    expect(trainer.getBestLoss()).toBe(-1);
  });

  it('earlyStopping configured but no validation data throws (T011, Q2)', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, {
      epochs: 20,
      earlyStopping: { patience: 5, minDelta: 0.001 },
    });

    // No setValidationData() called — the option would be silently ignored
    expect(() =>
      trainer.train({
        inputs: [
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ],
        targets: [[0], [1], [1], [0]],
      })
    ).toThrow(/earlyStopping/);
  });

  it('setValidationData validates input lengths', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net);
    expect(() =>
      trainer.setValidationData({
        inputs: [[0, 0]],
        targets: [[0], [1]],
      })
    ).toThrow();
  });

  // ── P2.3: Classification Metrics ───────────────────────────────────────

  it('computeMetrics with one-hot targets produces valid metrics', () => {
    const net = new NetworkN([2, 8, 2]);
    const trainer = new Trainer(net, {
      epochs: 100,
      lr: 0.3,
      computeMetrics: true,
    });

    trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
      ],
    });

    const metrics = trainer.getMetrics();
    expect(metrics.length).toBe(100);

    for (const m of metrics) {
      expect(m.accuracy).toBeGreaterThanOrEqual(0);
      expect(m.accuracy).toBeLessThanOrEqual(1);
      expect(m.precision).toBeGreaterThanOrEqual(0);
      expect(m.precision).toBeLessThanOrEqual(1);
      expect(m.recall).toBeGreaterThanOrEqual(0);
      expect(m.recall).toBeLessThanOrEqual(1);
      expect(m.f1).toBeGreaterThanOrEqual(0);
      expect(m.f1).toBeLessThanOrEqual(1);
    }
  });

  it('metrics improve for a solvable classification problem', () => {
    const net = new NetworkN([2, 8, 2]);
    const trainer = new Trainer(net, {
      epochs: 1000,
      lr: 0.5,
      computeMetrics: true,
    });

    trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
      ],
    });

    const metrics = trainer.getMetrics();
    // Accuracy should be higher at the end than at the beginning
    const firstAcc = metrics.slice(0, 5).reduce((s, m) => s + m.accuracy, 0) / 5;
    const lastAcc =
      metrics.slice(-5).reduce((s, m) => s + m.accuracy, 0) / 5;
    expect(lastAcc).toBeGreaterThan(firstAcc);
  }, 30000);

  it('computeMetrics=false does not compute metrics', () => {
    const net = new NetworkN([2, 4, 2]);
    const trainer = new Trainer(net, {
      epochs: 20,
      computeMetrics: false,
    });

    trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
      ],
      targets: [
        [1, 0],
        [0, 1],
      ],
    });

    expect(trainer.getMetrics()).toEqual([]);
  });

  it('metrics work with single-element binary targets', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, {
      epochs: 200,
      lr: 0.3,
      computeMetrics: true,
    });

    trainer.train({
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [[0], [1], [1], [0]],
    });

    const metrics = trainer.getMetrics();
    expect(metrics.length).toBe(200);
    // Final accuracy should be reasonable
    const finalMetrics = metrics[metrics.length - 1];
    expect(finalMetrics.accuracy).toBeGreaterThanOrEqual(0);
    expect(finalMetrics.f1).toBeGreaterThanOrEqual(0);
  });

  it('getMetrics returns a copy (defensive)', () => {
    const net = new NetworkN([2, 4, 2]);
    const trainer = new Trainer(net, {
      epochs: 5,
      computeMetrics: true,
    });

    trainer.train({
      inputs: [
        [0, 0],
        [1, 1],
      ],
      targets: [
        [1, 0],
        [0, 1],
      ],
    });

    const metrics = trainer.getMetrics();
    metrics.pop(); // Mutate the copy
    expect(trainer.getMetrics().length).toBe(5); // Original unchanged
  });

  // ── P2.4: clipValue removed (T011, Q1) — see tests/api/Trainer.hardening.test.ts

  // ── Integration: All features together ──────────────────────────────────

  it('weightDecay + earlyStopping + metrics work together', () => {
    const net = new NetworkN([2, 8, 2]);
    const trainer = new Trainer(net, {
      epochs: 2000,
      lr: 0.1,
      weightDecay: 0.0001,
      earlyStopping: { patience: 100, minDelta: 0.001 },
      computeMetrics: true,
    });

    const data = {
      inputs: [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ],
      targets: [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
      ],
    };

    trainer.setValidationData(data);
    const history = trainer.train(data);

    // Should have stopped early
    expect(trainer.getStopReason()).toBe('earlyStopping');
    expect(history.length).toBeLessThan(2000);

    // Metrics should be available
    const metrics = trainer.getMetrics();
    expect(metrics.length).toBeGreaterThan(0);
    expect(metrics.length).toBe(history.length); // one per epoch

    // Best loss should be tracked
    expect(trainer.getBestLoss()).toBeGreaterThan(0);
  }, 30000);

  // ── Edge cases ──────────────────────────────────────────────────────────

  it('works with a network that does NOT implement getWeights/setWeights', () => {
    // Minimal TrainableNetwork without getWeights/setWeights
    class MinimalNet implements TrainableNetwork {
      train(_inputs: number[], _targets: number[], _lr: number): number {
        return 0.5;
      }
    }

    const trainer = new Trainer(new MinimalNet(), { epochs: 5 });
    const history = trainer.train({
      inputs: [[0], [1]],
      targets: [[0], [1]],
    });
    expect(history.length).toBe(5);
    expect(trainer.getStopReason()).toBe('maxEpochs');
    expect(trainer.getMetrics()).toEqual([]);
  });

  it('empty dataset throws', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { epochs: 5 });
    // T011 (Q2): an empty dataset used to give NaN losses; now it throws
    expect(() => trainer.train({ inputs: [], targets: [] })).toThrow(/empty/);
  });

  it('verbose option can be set', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { verbose: true, epochs: 5 });
    expect(trainer.verbose).toBe(true);
  });
});

// ── T019: f1 is the mean of per-class F1 (as Metrics.f1Score), not 2PR/(P+R) ──

// Fake model with fixed outputs: sample i (input [i]) always gets class pred[i].
// `train` does nothing, so the metrics after one epoch are those of this table.
function fixedModel(pred: number[], nClasses: number, single = false): TrainableNetwork & {
  predict(x: number[]): number[];
} {
  return {
    train: () => 0,
    predict: (x: number[]) => {
      const c = pred[x[0]];
      if (single) return [c];
      const out = new Array(nClasses).fill(0);
      out[c] = 1;
      return out;
    },
  };
}

function oneHot(labels: number[], k: number): number[][] {
  return labels.map((c) => {
    const v = new Array(k).fill(0);
    v[c] = 1;
    return v;
  });
}

function trainerMetricsFor(yTrue: number[], yPred: number[], k: number) {
  const t = new Trainer(fixedModel(yPred, k), { epochs: 1, lr: 0.1, computeMetrics: true });
  t.train({ inputs: yTrue.map((_, i) => [i]), targets: oneHot(yTrue, k) });
  return t.getMetrics()[0];
}

describe('Trainer f1 (T019)', () => {
  it('3 classes, unbalanced: f1 is the mean of per-class F1 (hand-derived)', () => {
    // Confusion matrix C[true][pred], 8 samples (class sizes 3, 1, 4):
    //        pred0 pred1 pred2
    //  true0   1     2     0
    //  true1   0     1     0
    //  true2   0     0     4
    // class 0: tp=1, predicted=1, actual=3 -> P=1,   R=1/3 -> F1 = 2*1*(1/3)/(4/3) = 1/2
    // class 1: tp=1, predicted=3, actual=1 -> P=1/3, R=1   -> F1 = 1/2
    // class 2: tp=4, predicted=4, actual=4 -> P=1,   R=1   -> F1 = 1
    // macro P = (1+1/3+1)/3 = 7/9, macro R = (1/3+1+1)/3 = 7/9
    // mean of F1 = (1/2+1/2+1)/3 = 2/3   (new)
    // harmonic of macros = 2*(7/9)(7/9)/(14/9) = 7/9   (old, wrong)
    const yTrue = [0, 0, 0, 1, 2, 2, 2, 2];
    const yPred = [0, 1, 1, 1, 2, 2, 2, 2];
    const m = trainerMetricsFor(yTrue, yPred, 3);
    expect(m.f1).toBeCloseTo(2 / 3, 12);
    expect(m.precision).toBeCloseTo(7 / 9, 12);
    expect(m.recall).toBeCloseTo(7 / 9, 12);
    expect(m.accuracy).toBeCloseTo(6 / 8, 12);
  });

  it('a class with P + R = 0 has F1 0 and the result is finite', () => {
    // C[true][pred], 5 samples:
    //  true0: 2 -> pred0 ; true1: 1 -> pred1 ; true2: 2 -> pred1 (class 2 never right, never predicted)
    // class 0: P=1, R=1 -> F1=1
    // class 1: tp=1, predicted=3, actual=1 -> P=1/3, R=1 -> F1 = 2*(1/3)/(4/3) = 1/2
    // class 2: tp=0, predicted=0, actual=2 -> P=0, R=0 -> F1 = 0 (guard, not 0/0)
    // mean F1 = (1 + 1/2 + 0)/3 = 1/2   (old formula: 8/15)
    const yTrue = [0, 0, 1, 2, 2];
    const yPred = [0, 0, 1, 1, 1];
    const m = trainerMetricsFor(yTrue, yPred, 3);
    expect(Number.isFinite(m.f1)).toBe(true);
    expect(m.f1).toBeCloseTo(1 / 2, 12);
  });

  it('matches Metrics macro f1, precision and recall on the same predictions', () => {
    const yTrue = [0, 0, 0, 1, 2, 2, 2, 2, 1, 0];
    const yPred = [0, 1, 1, 1, 2, 0, 2, 2, 2, 0];
    const m = trainerMetricsFor(yTrue, yPred, 3);
    expect(Math.abs(m.f1 - f1Score(yTrue, yPred))).toBeLessThan(1e-12);
    expect(Math.abs(m.precision - precision(yTrue, yPred))).toBeLessThan(1e-12);
    expect(Math.abs(m.recall - recall(yTrue, yPred))).toBeLessThan(1e-12);
  });

  it('matches Metrics.f1Score on the predictions of a trained NetworkN', () => {
    const net = new NetworkN([2, 6, 3]);
    const inputs = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.2, 0.9]];
    const labels = [0, 1, 1, 2, 2, 0];
    const t = new Trainer(net, { epochs: 30, lr: 0.3, computeMetrics: true });
    t.train({ inputs, targets: oneHot(labels, 3) });
    const preds = inputs.map((x) => {
      const o = net.predict(x);
      return o.indexOf(Math.max(...o));
    });
    const last = t.getMetrics()[29];
    expect(Math.abs(last.f1 - f1Score(labels, preds))).toBeLessThan(1e-12);
  });

  it('golden: binary single output where both formulas agree stays the same', () => {
    // C = [[3,1],[1,3]]: both classes have P = R = 3/4, so F1 = 3/4 either way.
    const yTrue = [0, 0, 0, 0, 1, 1, 1, 1];
    const yPred = [0, 0, 0, 1, 1, 1, 1, 0];
    const t = new Trainer(fixedModel(yPred, 2, true), { epochs: 1, lr: 0.1, computeMetrics: true });
    t.train({ inputs: yTrue.map((_, i) => [i]), targets: yTrue.map((c) => [c]) });
    const m = t.getMetrics()[0];
    expect(m.f1).toBeCloseTo(0.75, 12);
    expect(m.accuracy).toBeCloseTo(0.75, 12);
  });

  // Known difference: Trainer uses K = target width (2 for single output), so a
  // class with no samples and no predictions inside that K counts as F1 0 in the
  // mean; Metrics uses K = max label + 1 over the data, so a trailing empty class
  // is not counted there. Interior empty classes count as 0 in both.
  it('a trailing class with no samples counts as F1 0 in Trainer (K = target width)', () => {
    const yTrue = [0, 0, 1, 1];
    const yPred = [0, 0, 1, 1];
    const m = trainerMetricsFor(yTrue, yPred, 3);
    expect(m.f1).toBeCloseTo(2 / 3, 12);
    expect(f1Score(yTrue, yPred)).toBeCloseTo(1, 12);
  });
});
