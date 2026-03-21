import { Capa } from "./capa";

// ─── NETWORK (2 layers) ───────────────────────────────────────────────────────
// Neural network with one hidden layer and one output layer.
// Uses backpropagation to adjust weights.
// For arbitrary depth see RedN.
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

  // Trains on a single example. Returns the squared error.
  entrenar(entradas: number[], correcto: number, tasa: number): number {
    const salidasOcultas = this.capaOculta.predecir(entradas);
    const prediccion     = this.capaSalida.predecir(salidasOcultas)[0];

    const errorSalida = correcto - prediccion;
    const deltaSalida = errorSalida * prediccion * (1 - prediccion);

    const neuronaSalida = this.capaSalida.neuronas[0];
    neuronaSalida.pesos = neuronaSalida.pesos.map(
      (p, i) => p + tasa * deltaSalida * salidasOcultas[i]
    );
    neuronaSalida.sesgo += tasa * deltaSalida;

    // Backpropagate error to hidden layer
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
