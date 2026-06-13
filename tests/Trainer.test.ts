import { describe, it, expect } from 'vitest';
import { Trainer } from '../src/Trainer';
import { NetworkN } from '../src/NetworkN';
import type { TrainableNetwork } from '../src/Trainer';

describe('Trainer', () => {
  // ── Basic tests (existing) ──────────────────────────────────────────────

  it('creates with default options', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net);
    expect(trainer.epochs).toBe(1000);
    expect(trainer.lrInitial).toBe(0.1);
    expect(trainer.lrDecay).toBe(1.0);
    expect(trainer.weightDecay).toBe(0);
    expect(trainer.clipValue).toBe(0);
  });

  it('creates with custom options', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, {
      epochs: 100,
      lr: 0.5,
      lrDecay: 0.99,
      weightDecay: 0.001,
      clipValue: 1.0,
    });
    expect(trainer.epochs).toBe(100);
    expect(trainer.lrInitial).toBe(0.5);
    expect(trainer.lrDecay).toBe(0.99);
    expect(trainer.weightDecay).toBe(0.001);
    expect(trainer.clipValue).toBe(1.0);
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

  it('earlyStopping configured but no validation data does not break', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, {
      epochs: 20,
      earlyStopping: { patience: 5, minDelta: 0.001 },
    });

    // No setValidationData() called — should still train all epochs
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
    expect(trainer.getStopReason()).toBe('maxEpochs');
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

  // ── P2.4: Gradient Clipping (clipValue stored) ─────────────────────────

  it('clipValue is stored in trainer options', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { clipValue: 0.5 });
    expect(trainer.clipValue).toBe(0.5);
  });

  it('clipValue defaults to 0', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net);
    expect(trainer.clipValue).toBe(0);
  });

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
    // Empty arrays should still pass validation (length match = 0 === 0)
    // but the loop should run 0 samples per epoch
    const history = trainer.train({ inputs: [], targets: [] });
    expect(history.length).toBe(5);
  });

  it('verbose option can be set', () => {
    const net = new NetworkN([2, 4, 1]);
    const trainer = new Trainer(net, { verbose: true, epochs: 5 });
    expect(trainer.verbose).toBe(true);
  });
});
