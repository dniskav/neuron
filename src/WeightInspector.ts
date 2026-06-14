// ─── WEIGHT INSPECTOR ─────────────────────────────────────────────────────────
//
// Diagnostic utilities for inspecting the internal state of a NetworkN.
// Helps detect common training pathologies:
//
//   Dead weights (|w| < threshold):
//     Weights very close to zero contribute little signal. A high fraction of
//     dead weights often indicates a learning-rate or initialisation problem.
//
//   Dead ReLUs:
//     Neurons whose output is always 0 because their pre-activation sum is
//     always negative. Once a ReLU neuron dies it can never recover (gradient
//     = 0). Detected by checking activations collected from a forward pass.
//
//   Weight explosion:
//     |max| >> 1 or std >> 1 often signals an unstable training loop. Use
//     gradient clipping or a smaller learning rate.
//
//   High standard deviation:
//     Large spread in weights can cause saturation in sigmoid/tanh activations,
//     leading to vanishing gradients. Xavier / He initialisation avoids this.
//
// Usage:
//
//   WeightInspector.print(net);
//   // →  Layer 0: mean= 0.012  std= 0.341  min=-0.982  max= 0.876  dead=  3/72
//   // →  Layer 1: mean=-0.003  std= 0.289  min=-0.764  max= 0.741  dead=  0/9
//   // →  Global:  mean= 0.009  std= 0.330  min=-0.982  max= 0.876  dead=  3/81
//

import { NetworkN } from "./NetworkN";

// ─── Interfaces ───────────────────────────────────────────────────────────────

export interface WeightStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  // Number of weights with |w| < deadThreshold
  deadCount: number;
  totalParams: number;
}

// ─── WeightInspector ─────────────────────────────────────────────────────────

export class WeightInspector {
  // ── Per-layer statistics ─────────────────────────────────────────────────
  // Returns one WeightStats per layer in network.layers order.
  static inspect(network: NetworkN, deadThreshold = 1e-3): WeightStats[] {
    return network.layers.map((layer) => {
      const weights: number[] = [];
      for (const neuron of layer.neurons) {
        weights.push(...neuron.weights, neuron.bias);
      }
      return _computeStats(weights, deadThreshold);
    });
  }

  // ── Global statistics ────────────────────────────────────────────────────
  // Aggregates all weights across the entire network.
  static inspectAll(network: NetworkN, deadThreshold = 1e-3): WeightStats {
    const allWeights: number[] = [];
    for (const layer of network.layers) {
      for (const neuron of layer.neurons) {
        allWeights.push(...neuron.weights, neuron.bias);
      }
    }
    return _computeStats(allWeights, deadThreshold);
  }

  // ── Formatted table ──────────────────────────────────────────────────────
  // Prints a compact diagnostic table to the console.
  static print(network: NetworkN, deadThreshold = 1e-3): void {
    const perLayer = WeightInspector.inspect(network, deadThreshold);
    const global   = WeightInspector.inspectAll(network, deadThreshold);

    const header = [
      'Layer'.padEnd(8),
      'mean'.padStart(9),
      'std'.padStart(9),
      'min'.padStart(9),
      'max'.padStart(9),
      'dead'.padStart(11),
      'params'.padStart(8),
    ].join('  ');

    console.log('');
    console.log('Weight Inspector:');
    console.log('─'.repeat(header.length));
    console.log(header);
    console.log('─'.repeat(header.length));

    perLayer.forEach((s, i) => {
      console.log(_formatRow(`Layer ${i}`, s));
    });

    console.log('─'.repeat(header.length));
    console.log(_formatRow('Global', global));
    console.log('');
  }

  // ── Dead ReLU detection ──────────────────────────────────────────────────
  //
  // Given a matrix of activations collected over a forward pass (rows = samples,
  // cols = neurons), counts neurons that output exactly 0 for every sample.
  //
  // How to collect activations:
  //   Run net.predict() for each validation sample and record the output of
  //   each hidden layer. Pass those as `activations` here.
  //
  // threshold: activations below this are counted as "dead" (default: 1e-6).
  //
  static countDeadReLUs(activations: number[][], threshold = 1e-6): number {
    if (activations.length === 0) return 0;
    const numNeurons = activations[0].length;
    let dead = 0;
    for (let j = 0; j < numNeurons; j++) {
      const allDead = activations.every((row) => Math.abs(row[j]) < threshold);
      if (allDead) dead++;
    }
    return dead;
  }
}

// ─── Private Helpers ──────────────────────────────────────────────────────────

function _computeStats(weights: number[], deadThreshold: number): WeightStats {
  const n = weights.length;
  if (n === 0) {
    return { mean: 0, std: 0, min: 0, max: 0, deadCount: 0, totalParams: 0 };
  }

  let sum = 0, sumSq = 0, min = Infinity, max = -Infinity, deadCount = 0;
  for (const w of weights) {
    sum   += w;
    sumSq += w * w;
    if (w < min) min = w;
    if (w > max) max = w;
    if (Math.abs(w) < deadThreshold) deadCount++;
  }

  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  return { mean, std, min, max, deadCount, totalParams: n };
}

function _fmt(n: number): string {
  return (n >= 0 ? ' ' : '') + n.toFixed(4);
}

function _formatRow(label: string, s: WeightStats): string {
  const deadStr = `${s.deadCount}/${s.totalParams}`;
  return [
    label.padEnd(8),
    _fmt(s.mean).padStart(9),
    _fmt(s.std).padStart(9),
    _fmt(s.min).padStart(9),
    _fmt(s.max).padStart(9),
    deadStr.padStart(11),
    String(s.totalParams).padStart(8),
  ].join('  ');
}
