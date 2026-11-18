# Contratos de 010 (firmas exactas)

Fuente de verdad para los agentes. Si una tarea necesita apartarse de esta página, informa `blocked`. No se improvisa.

## 1. `src/random.ts`

```ts
export type RandomFn = () => number            // [0, 1)
export function random(): number               // usa el generador activo; por defecto llama a Math.random() EN CADA LLAMADA
export function setRandom(fn: RandomFn | null): void   // null = volver a Math.random
export function getRandom(): RandomFn | null   // null si se usa Math.random
export function seededRandom(seed: number): RandomFn   // mulberry32, idéntico al de tests/helpers/seed.ts
export function withRandom<T>(fn: RandomFn, body: () => T): T  // restaura el anterior también si body lanza
```

Regla de sustitución: `Math.random()` → `random()` y nada más. No se reordenan llamadas ni se cachean valores. `KMeans` y `TSNE` conservan su `options.seed` y, sin semilla, pasan a usar `random()`.

## 2. `src/core/tensor.ts` (backend por defecto)

Matrices en `Float64Array`, fila a fila (row-major). `W` de una capa: `rows = nOut`, `cols = nIn`, `W[k*nIn + j]` es el peso de la neurona `k` para la entrada `j`. Ninguna función reserva memoria salvo que lo diga. Todas escriben en `out` y lo devuelven.

```ts
export type Vec = Float64Array
export interface Backend {
  // out[k] = b[k] + Σ_j W[k,j]·x[j]                    (b puede ser null)
  matVec(W: Vec, rows: number, cols: number, x: Vec, b: Vec | null, out: Vec): Vec
  // out[j] = Σ_k W[k,j]·d[k]                           (Wᵀ·d, para propagar hacia atrás)
  matTVec(W: Vec, rows: number, cols: number, d: Vec, out: Vec): Vec
  // G[k,j] += s·d[k]·x[j]                              (acumula el gradiente de una muestra)
  addOuter(G: Vec, rows: number, cols: number, d: Vec, x: Vec, s: number): Vec
  // Lote de n filas: X es n×cols, OUT es n×rows. OUT[i,k] = b[k] + Σ_j W[k,j]·X[i,j]
  matMulBatch(W: Vec, rows: number, cols: number, X: Vec, n: number, b: Vec | null, OUT: Vec): Vec
  // D es n×rows, OUT es n×cols. OUT[i,j] = Σ_k W[k,j]·D[i,k]
  matMulBatchT(W: Vec, rows: number, cols: number, D: Vec, n: number, OUT: Vec): Vec
  // G[k,j] += s·Σ_i D[i,k]·X[i,j]
  addOuterBatch(G: Vec, rows: number, cols: number, D: Vec, X: Vec, n: number, s: number): Vec
}
export const defaultBackend: Backend
```

Orden de suma de referencia: `j` creciente en `matVec` (empezando por `b[k]`, como `NeuronN.predict`, que empieza en el sesgo y suma `x[j]·w[j]` en orden), `k` creciente en `matTVec` (como `reduce` sobre neuronas en `NetworkN._backward`). Con ese orden, `NetworkFast` reproduce `NetworkN` bit a bit en la mayoría de los casos; el contrato exige solo ≤ 1e-12 relativo.

## 3. `src/core/backend.ts`

```ts
export function setBackend(b: Partial<Backend> | null): void  // mezcla con defaultBackend; null restaura
export function getBackend(): Backend
export interface BackendReport { ok: boolean; failures: { op: keyof Backend; case: string; maxRelError: number }[] }
export function checkBackend(b: Partial<Backend>, opts?: { tolerance?: number; seed?: number }): BackendReport
// tolerance por defecto 1e-12; casos: tamaños 1×1, 3×5, 17×9, 64×128, lote n = 1, 4, 33; entradas con
// ceros, negativos y magnitudes 1e-8..1e8; comprueba también que solo escribe en `out`/`G`.
```

## 4. `src/core/optim.ts`

```ts
export interface VecOptimizer {
  readonly size: number
  // params[i] = paso(params[i], grads[i], lr) para TODO i. grads es -dL/dw (convención de ascenso).
  step(params: { [i: number]: number; length: number }, grads: { [i: number]: number; length: number }, lr: number): void
}
export class VecSGD implements VecOptimizer { constructor(size: number) }
export class VecMomentum implements VecOptimizer { constructor(size: number, beta?: number) }      // v: Float64Array
export class VecAdam implements VecOptimizer { constructor(size: number, beta1?: number, beta2?: number, epsilon?: number) } // m, v: Float64Array; t: number COMPARTIDO
export class VecClip implements VecOptimizer { constructor(inner: VecOptimizer, clipValue: number) }
export class ScalarArrayOptimizer implements VecOptimizer { constructor(factory: OptimizerFactory, size: number) } // llama factory() size veces, en orden
export function vecFromFactory(factory: OptimizerFactory, size: number): VecOptimizer
```

`vecFromFactory` llama a `factory()` **una vez** para inspeccionar la instancia:
- `instanceof SGD` → `VecSGD`
- `instanceof Momentum` → `VecMomentum(size, inst.beta)`
- `instanceof Adam` → `VecAdam(size, inst.beta1, inst.beta2, inst.epsilon)`
- `instanceof ClipOptimizer` y su `inner` es integrado → `VecClip(vec(inner), inst.clipValue)`
- otro caso → `ScalarArrayOptimizer(factory, size)`, que crea `size` instancias NUEVAS y descarta la de inspección.

Reglas de exactitud: las fórmulas se copian literalmente de `optimizers.ts`, con el mismo orden de operaciones, `Math.pow(beta, t)` incluido. `VecAdam` usa un `t` compartido: solo es válido si `step` actualiza siempre todos los elementos. Las clases con actualizaciones parciales (`EmbeddingMatrix`) siguen con el camino escalar.

Lo único observable de llamar a `factory()` una vez de más: si la fábrica tiene efectos secundarios, como contar instancias, el conteo cambia. Hay que documentarlo en el comentario de `vecFromFactory`.

## 5. `NetworkFast` (`src/NetworkFast.ts`)

```ts
export class NetworkFast {
  readonly structure: number[]
  constructor(structure: number[], options?: NetworkNOptions & { backend?: Partial<Backend> })
  predict(inputs: number[], training?: boolean): number[]
  predictBatch(X: number[][]): number[][]                 // inferencia, sin dropout
  train(inputs: number[], targets: number[], lr: number): number          // MSE media, paso (2/n)(t − p)
  trainBatch(X: number[][], T: number[][], lr: number): number            // media del lote; un solo paso del optimizador
  trainWithDeltas(inputs: number[], outputDeltas: number[], lr: number, reuseMasks?: boolean): void
  inputGradient(inputs: number[], outputDeltas: number[], reuseMasks?: boolean): number[]
  getWeights(): number[]
  setWeights(weights: number[]): void
  toConfig(): NetworkNConfig            // T015
  static fromConfig(c: NetworkNConfig): NetworkFast
}
```

Estado interno por capa `l`: `W[l]: Float64Array (nOut·nIn)`, `b[l]: Float64Array (nOut)`, `opt[l]: VecOptimizer` de tamaño `nOut·(nIn+1)`. El buffer de parámetros va intercalado en el orden de `getWeights` (`[w_k0..w_k(nIn−1), b_k]` por neurona) **o** se guardan `W` y `b` aparte con un optimizador para cada uno. La tarea decide, pero `getWeights` debe devolver el orden de `NetworkN`. Si van aparte, el estado del optimizador de cada elemento es independiente, así que el resultado es igual.

Validación y mensajes: los de `NetworkN`, con `NetworkFast.` en lugar de `NetworkN.`. `trainBatch` exige `X.length === T.length ≥ 1` y valida cada fila antes de tocar nada.

## 6. Formato 2 (`ModelSaver`)

```ts
type ActivationSpec = 'sigmoid' | 'relu' | 'tanh' | 'linear' | 'leakyRelu' | 'elu' | { name: 'leakyRelu' | 'elu'; alpha: number }
type OptimizerSpec = { name: 'sgd' } | { name: 'momentum'; beta: number } | { name: 'adam'; beta1: number; beta2: number; epsilon: number }
                   | { name: 'clip'; clipValue: number; inner: OptimizerSpec } | { name: 'custom' }
interface NetworkNConfig { structure: number[]; activations: ActivationSpec[]; optimizer: OptimizerSpec; residual: boolean[]; dropoutRate: number }
interface SavedModel { format: 2; kind: string; config: unknown; length: number; weights: (number | null)[]; timestamp: number }

ModelSaver.save(model: { kind?: string; toConfig(): unknown; getWeights(): number[] }, opts?: { allowCustomOptimizer?: boolean }): string
ModelSaver.load(json: string): unknown          // instancia nueva
ModelSaver.register(kind: string, entry: { fromConfig(config: unknown): { setWeights(w: number[]): void; getWeights(): number[] } }): void
```

Cada clase con formato 2 expone `static readonly kind = '<Nombre>'` y un getter de instancia `kind` que lo devuelve. El nombre de la clase no se lee de `constructor.name`, porque los empaquetadores lo minimizan. Las activaciones integradas ganan `readonly name` (y `alpha` las de fábrica). Para guardar el optimizador, `NetworkN` y `NetworkFast` conservan la fábrica que recibieron y la inspeccionan igual que `vecFromFactory`.

## 7. `src/infer.ts` → `@dniskav/neuron/infer`

```ts
export interface InferenceModel { kind: string; structure: number[]; predict(x: number[]): number[]; predictBatch(X: number[][]): number[][] }
export function loadForInference(json: string, opts?: { backend?: Partial<Backend> }): InferenceModel
```

Solo importa `core/tensor`, `core/backend`, `activations` y `Validation`. Nunca importa `optimizers`, `random`, `NetworkN`, `NetworkFast` ni `ModelSaver`: tiene su propio lector del formato 2, que comparte con `ModelSaver` solo el validador puro (`src/core/format2.ts`).
