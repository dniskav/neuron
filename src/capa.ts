import { NeuronaN } from "./neuronaN";

// ─── CAPA ─────────────────────────────────────────────────────────────────────
// Grupo de neuronas que comparten las mismas entradas.
// Cada neurona produce su propia salida de forma independiente.
// La salida de la capa es el array de salidas de cada neurona,
// que se convierte en la entrada de la siguiente capa.
export class Capa {
  neuronas: NeuronaN[];

  constructor(nNeuronas: number, nEntradas: number) {
    this.neuronas = Array.from({ length: nNeuronas }, () => new NeuronaN(nEntradas));
  }

  predecir(entradas: number[]): number[] {
    return this.neuronas.map(n => n.predecir(entradas));
  }
}
