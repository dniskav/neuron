import { Capa } from "./capa";

// ─── RED N CAPAS ──────────────────────────────────────────────────────────────
// Generalización de Red para profundidad arbitraria.
//
//   new RedN([3, 16, 8, 1])
//   → 3 entradas → capa 16 → capa 8 → 1 salida
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

  // Backpropagation generalizada para L capas.
  // Devuelve el error cuadrático medio del ejemplo.
  entrenar(entradas: number[], correctos: number[], tasa: number): number {
    const act: number[][] = [entradas];
    for (const capa of this.capas) act.push(capa.predecir(act[act.length - 1]));

    const pred = act[act.length - 1];
    let deltas = pred.map((p, i) => (correctos[i] - p) * p * (1 - p));

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

    return pred.reduce((s, p, i) => s + (correctos[i] - p) ** 2, 0) / pred.length;
  }

  // Backprop con deltas externos en la capa de salida.
  // Útil para losses personalizadas (p. ej. gradiente físico).
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
