// ─── FUNCIÓN DE ACTIVACIÓN ───────────────────────────────────────────────────
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── NEURONA N ENTRADAS ───────────────────────────────────────────────────────
// Versión generalizada de Neurona: acepta cualquier número de entradas.
// Usa inicialización Xavier simplificada: pesos en [-√(1/n), +√(1/n)].
// Esto garantiza que el gradiente fluya bien desde el inicio del entrenamiento.
export class NeuronaN {
  pesos: number[];
  sesgo: number;

  constructor(nEntradas: number) {
    const limit = Math.sqrt(1 / nEntradas);
    this.pesos = Array.from({ length: nEntradas }, () => (Math.random() * 2 - 1) * limit);
    this.sesgo = 0;
  }

  predecir(entradas: number[]): number {
    const suma = entradas.reduce((acc, e, i) => acc + e * this.pesos[i], this.sesgo);
    return sigmoid(suma);
  }

  entrenar(entradas: number[], correcto: number, tasa: number): void {
    const prediccion = this.predecir(entradas);
    const error = correcto - prediccion;
    this.pesos = this.pesos.map((p, i) => p + tasa * error * entradas[i]);
    this.sesgo += tasa * error;
  }
}
