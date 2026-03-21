import { Capa } from "./capa";

// ─── RED (2 capas) ────────────────────────────────────────────────────────────
// Red neuronal con una capa oculta y una capa de salida.
// Usa backpropagation para ajustar los pesos.
// Para redes de profundidad arbitraria ver RedN.
export class Red {
  capaOculta: Capa;
  capaSalida: Capa;

  constructor(nEntradas: number, nNeuronasOcultas: number, nSalidas: number) {
    this.capaOculta = new Capa(nNeuronasOcultas, nEntradas);
    this.capaSalida = new Capa(nSalidas, nNeuronasOcultas);
  }

  predecir(entradas: number[]): number {
    const salidasOcultas = this.capaOculta.predecir(entradas);
    return this.capaSalida.predecir(salidasOcultas)[0];
  }

  entrenar(entradas: number[], correcto: number, tasa: number): number {
    const salidasOcultas = this.capaOculta.predecir(entradas);
    const prediccion     = this.capaSalida.predecir(salidasOcultas)[0];

    const errorSalida  = correcto - prediccion;
    const deltaSalida  = errorSalida * prediccion * (1 - prediccion);

    const neuronaSalida = this.capaSalida.neuronas[0];
    neuronaSalida.pesos = neuronaSalida.pesos.map(
      (p, i) => p + tasa * deltaSalida * salidasOcultas[i]
    );
    neuronaSalida.sesgo += tasa * deltaSalida;

    this.capaOculta.neuronas.forEach((neurona, i) => {
      const salidaOculta = salidasOcultas[i];
      const errorOculto  = deltaSalida * neuronaSalida.pesos[i];
      const deltaOculto  = errorOculto * salidaOculta * (1 - salidaOculta);
      neurona.pesos = neurona.pesos.map((p, j) => p + tasa * deltaOculto * entradas[j]);
      neurona.sesgo += tasa * deltaOculto;
    });

    return errorSalida * errorSalida;
  }
}
