// ─── ACTIVATION FUNCTION ─────────────────────────────────────────────────────
// Squashes any number into a value between 0 and 1
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── NEURON (single input) ────────────────────────────────────────────────────
// Educational version: one input, one weight.
// See NeuronaN for the N-input version with Xavier initialization.
export class Neurona {
  peso: number;
  sesgo: number;

  constructor() {
    this.peso = Math.random() * 0.1;
    this.sesgo = Math.random() * 0.1;
  }

  predecir(entrada: number): number {
    return sigmoid(entrada * this.peso + this.sesgo);
  }

  entrenar(entrada: number, correcto: number, tasa: number): void {
    const prediccion = this.predecir(entrada);
    const error = correcto - prediccion;
    this.peso  += tasa * error * entrada;
    this.sesgo += tasa * error;
  }
}
