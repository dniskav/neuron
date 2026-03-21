// ─── FUNCIÓN DE ACTIVACIÓN ───────────────────────────────────────────────────
// Convierte cualquier número a un valor entre 0 y 1
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── NEURONA (1 entrada) ──────────────────────────────────────────────────────
// Versión didáctica: una sola entrada, un solo peso.
// Ver NeuronaN para la versión con N entradas e inicialización Xavier.
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
