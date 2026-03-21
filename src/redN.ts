import { Capa } from "./capa";

// ─── N-LAYER NETWORK ──────────────────────────────────────────────────────────
// Generalization of Red for arbitrary depth.
//
//   new RedN([3, 16, 8, 1])
//   → 3 inputs → layer 16 → layer 8 → 1 output
//
export class RedN {
  capas: Capa[];

  constructor(readonly estructura: number[]) {
    this.capas = [];
    for (let i = 1; i < estructura.length; i++) {
      this.capas.push(new Capa(estructura[i], estructura[i - 1]));
    }
  }

  predecir(entradas: number[]): number[] {
    return this.capas.reduce((acc, capa) => capa.predecir(acc), entradas);
  }

  // Generalized backpropagation across L layers.
  // Returns the mean squared error for the example.
  entrenar(entradas: number[], correctos: number[], tasa: number): number {
    // Forward pass — store activations at every layer
    const act: number[][] = [entradas];
    for (const capa of this.capas) act.push(capa.predecir(act[act.length - 1]));

    const pred = act[act.length - 1];

    // Output layer deltas
    let deltas = pred.map((p, i) => (correctos[i] - p) * p * (1 - p));

    // Backprop from last layer to first
    for (let l = this.capas.length - 1; l >= 0; l--) {
      const capa    = this.capas[l];
      const entCapa = act[l];

      // Error reaching each neuron in the previous layer (computed before updating weights)
      const deltasAnt = entCapa.map((sal, j) => {
        const errProp = capa.neuronas.reduce((s, n, k) => s + deltas[k] * n.pesos[j], 0);
        return errProp * sal * (1 - sal);
      });

      capa.neuronas.forEach((n, k) => {
        n.pesos = n.pesos.map((p, j) => p + tasa * deltas[k] * entCapa[j]);
        n.sesgo += tasa * deltas[k];
      });

      deltas = deltasAnt;
    }

    return pred.reduce((s, p, i) => s + (correctos[i] - p) ** 2, 0) / pred.length;
  }

  // Backprop with externally provided output-layer deltas.
  // Useful for custom loss functions (e.g. physics-based gradients).
  entrenarConDeltas(entradas: number[], deltasOutput: number[], tasa: number): void {
    const act: number[][] = [entradas];
    for (const capa of this.capas) act.push(capa.predecir(act[act.length - 1]));

    let deltas = deltasOutput;
    for (let l = this.capas.length - 1; l >= 0; l--) {
      const capa    = this.capas[l];
      const entCapa = act[l];
      const deltasAnt = entCapa.map((sal, j) => {
        const errProp = capa.neuronas.reduce((s, n, k) => s + deltas[k] * n.pesos[j], 0);
        return errProp * sal * (1 - sal);
      });
      capa.neuronas.forEach((n, k) => {
        n.pesos = n.pesos.map((p, j) => p + tasa * deltas[k] * entCapa[j]);
        n.sesgo += tasa * deltas[k];
      });
      deltas = deltasAnt;
    }
  }
}
