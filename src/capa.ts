import { NeuronaN } from "./neuronaN";

// ─── LAYER ────────────────────────────────────────────────────────────────────
// A group of neurons that share the same inputs.
// Each neuron produces its own output independently.
// The layer output is an array of each neuron's output,
// which becomes the input for the next layer.
export class Capa {
  neuronas: NeuronaN[];

  constructor(nNeuronas: number, nEntradas: number) {
    this.neuronas = Array.from({ length: nNeuronas }, () => new NeuronaN(nEntradas));
  }

  predecir(entradas: number[]): number[] {
    return this.neuronas.map(n => n.predecir(entradas));
  }
}
