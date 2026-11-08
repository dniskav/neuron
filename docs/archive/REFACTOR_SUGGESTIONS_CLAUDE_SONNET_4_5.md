# Refactor Suggestions — Claude Sonnet 4.5

Model: `anthropic/claude-sonnet-4.5` via OpenRouter  
Date: 2026-04-01  
Scope: revisión completa de `src/` (17 archivos, ~1700 líneas)

---

## Índice de severidad

| Severidad | Criterio |
|---|---|
| **Alta** | Rompe la abstracción, produce gradientes incorrectos, o genera bugs silenciosos |
| **Media** | Inconsistencia de API, duplicación no trivial, o riesgo de error en expansión futura |
| **Baja** | Deuda técnica menor, cosmética, o trade-off documentado aceptable |

---

## 1. `sigmoid` y `tanh` privados duplicados en `LSTMLayer.ts` — *Media*

**Archivos:** `src/LSTMLayer.ts:25-32`, `src/Neuron.ts:3-5`, `src/activations.ts:1-35`

`LSTMLayer.ts` define sus propias funciones privadas `sigmoid` y `tanh` con cuerpos idénticos a los de `activations.ts`. `Neuron.ts` también tiene su propio `sigmoid`, pero eso es un trade-off deliberado (la clase es intencionalmente sin dependencias). `LSTMLayer.ts` ya usa el sistema de módulos y no tiene esa justificación.

```ts
// LSTMLayer.ts:25-27 — duplicado innecesario
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
```

**Propuesta:** Importar `sigmoid` y `tanh` desde `activations.ts` en `LSTMLayer.ts`. La copia en `Neuron.ts` es un trade-off documentado y puede quedarse.

---

## 2. `defaultOptimizer` declarado cuatro veces — *Baja*

**Archivos:** `src/NeuronN.ts:4`, `src/Layer.ts:5`, `src/NetworkN.ts:5`, `src/NetworkLSTM.ts:6`

La misma constante está copy-pasteada en cuatro archivos:

```ts
const defaultOptimizer: OptimizerFactory = () => new SGD();
```

**Propuesta:** Exportar `defaultOptimizer` desde `src/optimizers.ts` e importarlo donde se necesite.

---

## 3. Bug de gradiente en `LayerNorm.backwardOne` — **Alta**

**Archivo:** `src/LayerNorm.ts:73-86`

`γ` se actualiza en el bucle *antes* de ser usado para calcular `D` (el gradiente que se propaga hacia atrás). El propio comentario del código lo reconoce como aproximación:

```ts
// LayerNorm.ts:73-86
for (let i = 0; i < N; i++) {
  this.gamma[i] += lr * dOut[i] * x_norm[i]  // ← gamma actualizado aquí
  this.beta[i]  += lr * dOut[i]
}
// D usa el gamma YA actualizado — esto es matemáticamente incorrecto
const D = dOut.map((d, i) => d * this.gamma[i])
```

`D` debería usar el `γ` *previo* a la actualización para que el gradiente sea correcto.

**Propuesta:**
```ts
const gammaOld = [...this.gamma];
for (let i = 0; i < N; i++) {
  this.gamma[i] += lr * dOut[i] * x_norm[i];
  this.beta[i]  += lr * dOut[i];
}
const D = dOut.map((d, i) => d * gammaOld[i]);
```

---

## 4. `Network.train` bypasea `_update()` y la abstracción de optimizadores — **Alta**

**Archivo:** `src/Network.ts:30-41`

`Network.train` muta `neuron.weights` y `neuron.bias` directamente con SGD hardcodeado, ignorando el método `_update()` de `NeuronN` y el optimizador que pudiera tener asignado. Además hardcodea la derivada de sigmoid en vez de usar `activation.dfn(out)`:

```ts
// Network.ts:30-33
outputNeuron.weights = outputNeuron.weights.map(
  (w, i) => w + lr * outputDelta * hiddenOut[i]  // SGD hardcodeado
);
```

Si alguien pasa Adam o Momentum a la `Layer` del constructor, `Network.train` lo ignora silenciosamente.

**Propuesta:** Usar `neuron._update(weightGrads, biasGrad, lr)` y `neuron.activation.dfn(out)`, igual que `NetworkN`.

---

## 5. `Network.predict` devuelve `number`; todos los demás devuelven `number[]` — **Alta**

**Archivos:** `src/Network.ts:16`, `src/NetworkN.ts:41`, `src/NetworkLSTM.ts:81`, `src/NetworkTransformer.ts:89`

La asimetría en el tipo de retorno impide escribir código polimórfico sobre distintos tipos de red. `Network.train` también acepta `target: number` mientras que `NetworkN.train` acepta `targets: number[]`.

**Propuesta:** Cambiar `Network.predict` para devolver `number[]` (con un solo elemento) para mantener consistencia. Introducir una interfaz `INetwork` que todos los modelos implementen:

```ts
interface INetwork {
  predict(inputs: number[]): number[];
  train(inputs: number[], targets: number[], lr: number): void;
}
```

---

## 6. Loop de backprop duplicado entre `NetworkN` y `NetworkLSTM` — *Media*

**Archivos:** `src/NetworkN.ts:89-101`, `src/NetworkLSTM.ts:119-142`

El patrón de iteración inversa por capas, acumulación de `prevDeltas` y la expresión `layer.neurons.reduce((s, n, k) => s + deltas[k] * n.weights[j], 0)` son virtualmente idénticos en ambos archivos. La única diferencia es que `NetworkLSTM` acumula gradientes antes de aplicarlos, mientras que `NetworkN` los aplica directamente.

**Propuesta:** Extraer una función pura:

```ts
function backpropLayers(
  layers: Layer[],
  acts: number[][],
  deltas: number[]
): { prevDeltas: number[]; dW: number[][][]; db: number[][] }
```

Cada caller decide cuándo y cómo aplicar los gradientes devueltos.

---

## 7. Forward pass duplicado en `NetworkN.train` y `NetworkN.trainWithDeltas` — *Baja*

**Archivo:** `src/NetworkN.ts:49-50` y `src/NetworkN.ts:85-86`

```ts
// Aparece idéntico en ambos métodos:
const act: number[][] = [inputs];
for (const layer of this.layers) act.push(layer.predict(act[act.length - 1]));
```

`NetworkTransformer` ya resuelve este patrón correctamente con `_forward()`.

**Propuesta:** Extraer `_forwardAll(inputs: number[]): number[][]` en `NetworkN` y llamarlo desde `train` y `trainWithDeltas`.

---

## 8. Proyección de salida duplicada en `NetworkTransformer` — *Baja*

**Archivo:** `src/NetworkTransformer.ts:92-96` y `src/NetworkTransformer.ts:111-115`

El bucle de proyección de salida (logits por token) es idéntico en `predict` y en `train`.

**Propuesta:** Extraer `_project(h: number[][]): number[][]`. `predict` llama `this._project(h).flat()` y `train` llama `this._project(h)`.

---

## 9. Patrón bias+Adam repetido en `TransformerBlock` y `NetworkTransformer` — *Media*

**Archivos:** `src/TransformerBlock.ts:43-48,70-71,154-155,179-180`, `src/NetworkTransformer.ts:57,83,163-164`

Ambas clases almacenan arrays `number[]` de biases con arrays paralelos `Adam[]` de optimizadores y los actualizan en bucles manuales. El patrón es idéntico al de `WeightMatrix` pero para vectores 1D, re-implementado desde cero en cada clase.

**Propuesta:** Extraer una clase `BiasVector` (o `ParameterVector`):

```ts
class BiasVector {
  values: number[];
  private opts: Adam[];

  constructor(size: number) { /* inicializar */ }
  update(grad: number[], lr: number): void { /* bucle Adam */ }
}
```

---

## 10. `matMul` existe pero no se usa en las proyecciones del Transformer — *Media*

**Archivos:** `src/AttentionHead.ts:58-65`, `src/TransformerBlock.ts:91-98`, `src/NetworkTransformer.ts:112-115`

El patrón `row.reduce((s, w, m) => s + w * x[m], bias)` — un producto matriz-vector — aparece al menos 8 veces en los archivos del Transformer. `matMul` ya existe en `MatMul.ts:13` pero no se usa en ninguna de estas proyecciones.

**Propuesta:** Añadir una función helper a `MatMul.ts`:

```ts
export function linearProject(
  X: number[][],
  W: WeightMatrix,
  b?: number[]
): number[][]
```

Y usar `matMul` o `linearProject` en todos los sitios donde hoy se usa el reduce inline.

---

## 11. Aserciones non-null sin guard en el Transformer — *Media*

**Archivos:** `src/AttentionHead.ts:101`, `src/MultiHeadAttention.ts:74`, `src/TransformerBlock.ts:124-126`

Múltiples `!` non-null assertions sobre cachés que solo son válidas tras llamar a `predict()`. Si `backward()` se llama antes, el runtime lanza un error difícil de diagnosticar.

```ts
const { X, Q, K, V, attn } = this.cache!  // AttentionHead.ts:101
const concat = this._concat!               // MultiHeadAttention.ts:74
```

**Propuesta:** Reemplazar con guards explícitos:

```ts
if (!this.cache) throw new Error('AttentionHead.backward() called before predict()');
```

---

## 12. Inicialización Xavier inconsistente — *Media*

**Archivos:** `src/NeuronN.ts:27`, `src/LSTMLayer.ts:44`, `src/MatMul.ts:74,112`

Tres fórmulas distintas se usan en nombre de "Xavier":

| Ubicación | Fórmula | Notas |
|---|---|---|
| `NeuronN.ts:27` | `√(1/n)` | Xavier fan-in |
| `LSTMLayer.ts:44` | `√(2/n)` | He/Kaiming — diseñado para ReLU, no sigmoid/tanh |
| `MatMul.ts:74` | `√(2/(rows+cols))` | Xavier fan-in+out correcto |
| `MatMul.ts:112` | `√(1/d_model)` | Xavier fan-in |

Las puertas LSTM usan activaciones sigmoid/tanh pero se inicializan con la fórmula He (diseñada para ReLU). Es subóptimo.

**Propuesta:** Exportar funciones de init desde un fichero compartido (`src/init.ts` o dentro de `activations.ts`):

```ts
export const xavierInit = (fanIn: number, fanOut = fanIn) =>
  (Math.random() * 2 - 1) * Math.sqrt(2 / (fanIn + fanOut));

export const heInit = (fanIn: number) =>
  (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn);
```

Y usar cada una donde corresponda por la activación de la capa.

---

## 13. Serialización ausente en `NetworkN`, `Network` y `NetworkTransformer` — *Media*

**Archivos:** `src/Network.ts`, `src/NetworkN.ts`, `src/NetworkTransformer.ts`

`NetworkLSTM` y `LSTMLayer` tienen `getWeights()`/`setWeights()`, pero los otros tres modelos no. Un modelo `NetworkN` entrenado no se puede guardar ni restaurar.

Adicionalmente, el tipo de `setWeights` en `LSTMLayer.ts:236` usa `ReturnType<LSTMLayer["getWeights"]>` — una auto-referencia que oculta cambios de forma en el tipo serializado.

**Propuesta:**
- Añadir `getWeights()`/`setWeights()` a `NetworkN`, `Network` y `NetworkTransformer`.
- Definir interfaces explícitas para los formatos de serialización:
  ```ts
  export interface LSTMWeights { /* ... */ }
  export interface NetworkNWeights { /* ... */ }
  ```
- Considerar una interfaz común:
  ```ts
  interface ISerializable<T> {
    getWeights(): T;
    setWeights(data: T): void;
  }
  ```

---

## 14. `mseDelta` y `crossEntropyDelta` son funciones idénticas — *Baja*

**Archivo:** `src/losses.ts:36-44`

Ambas devuelven `actual - predicted`. El comentario explica por qué (la derivada de sigmoid cancela en CE), pero exponer dos nombres para la misma función sin hacer la relación explícita en el código es confuso.

**Propuesta:**

```ts
export function mseDelta(predicted: number, actual: number): number {
  return actual - predicted;
}

// La derivada sigmoid se cancela con la CE, resultando en la misma expresión
export const crossEntropyDelta = mseDelta;
```

---

## 15. `WeightMatrix` y `EmbeddingMatrix` tienen optimizador hardcodeado — *Baja*

**Archivo:** `src/MatMul.ts:71-125`

`WeightMatrix` usa siempre `Adam` y `EmbeddingMatrix` usa siempre SGD inline. Al contrario que `NeuronN`, `Layer`, `NetworkN` y `NetworkLSTM`, ninguna de las dos acepta un `OptimizerFactory`. Hay buenas razones para los defaults actuales (documentadas), pero la falta de override cierra la puerta a experimentar.

**Propuesta:** Añadir parámetro opcional con el default actual:

```ts
constructor(rows: number, cols: number, optimizerFactory: OptimizerFactory = () => new Adam())
```

Esto mantiene compatibilidad hacia atrás.

---

## 16. `weights` y `bias` son `public` en `NeuronN` pero debería haber un contrato claro — *Baja*

**Archivo:** `src/NeuronN.ts:14-16`

`weights` y `bias` son mutados directamente por `Network.ts` (líneas 30-33, 40-41) y `NetworkLSTM.ts` (líneas 180-181), bypasseando `_update()`. Tener campos públicos mutables junto a un método `_update` da señales contradictorias sobre la API prevista.

**Propuesta:** Si los campos deben seguir siendo públicos para deserialización, documentarlo con un comentario explícito. Si no, hacerlos privados y añadir `getWeights()`/`setWeights()` a `NeuronN`. Como mínimo, hacer que `Network.ts` use `_update()` (ver punto 4).

---

## Resumen

| # | Descripción | Severidad |
|---|---|---|
| 3 | Bug de gradiente en `LayerNorm.backwardOne` (usa `γ` post-update) | **Alta** |
| 4 | `Network.train` bypasea `_update()` y hardcodea SGD | **Alta** |
| 5 | `Network.predict` devuelve `number`; inconsistente con el resto | **Alta** |
| 6 | Loop de backprop duplicado en `NetworkN` y `NetworkLSTM` | Media |
| 9 | Patrón bias+Adam repetido sin abstracción en Transformer | Media |
| 10 | `matMul` existente no se usa en proyecciones del Transformer | Media |
| 11 | Non-null assertions sin guard en caché del Transformer | Media |
| 12 | Fórmulas de Xavier inconsistentes (He usado donde corresponde sigmoid) | Media |
| 13 | Serialización ausente en `NetworkN`, `Network`, `NetworkTransformer` | Media |
| 1 | `sigmoid`/`tanh` duplicados en `LSTMLayer.ts` | Baja-Media |
| 2 | `defaultOptimizer` declarado 4 veces | Baja |
| 7 | Forward pass duplicado en `NetworkN` | Baja |
| 8 | Proyección de salida duplicada en `NetworkTransformer` | Baja |
| 14 | `mseDelta` y `crossEntropyDelta` son idénticas sin alias explícito | Baja |
| 15 | Optimizador hardcodeado en `WeightMatrix`/`EmbeddingMatrix` | Baja |
| 16 | `weights`/`bias` públicos en `NeuronN` sin contrato claro | Baja |
