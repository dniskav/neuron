# Spec 010: núcleo rápido, lotes, semilla, exportación completa e inferencia

**Fecha**: 2026-11-18 | **Estado**: aprobada (Q1–Q9 resueltas, ver HANDOFF) | **Versión objetivo**: 0.5.0
**Origen**: conversación del 2026-11-18 sobre la madurez de la librería. Medición de partida en Node 26 (Apple Silicon), `NetworkN` con Adam:

| Red | Parámetros | Bytes por parámetro | Entrenar 1 ejemplo | Predecir 1 ejemplo |
|---|---|---|---|---|
| `[256,256,256,10]` SGD | 134 154 | ~41 | 5,9 ms | 0,46 ms |
| `[256,256,256,10]` Adam | 134 154 | ~90 | 7,4 ms | 0,15 ms |
| `[512,512,512,10]` Adam | 530 442 | ~88 | 27,7 ms | 0,59 ms |

Unos 20 millones de actualizaciones de peso por segundo al entrenar. El techo práctico para entrenar está hoy entre 100 k y 1 M de parámetros. La causa es estructural: un objeto optimizador por cada peso escalar, arrays de arrays y un ejemplo por llamada.

## 1. Objetivo

Subir el techo práctico a 1–2 M de parámetros y hacer que un modelo entrenado se pueda guardar, cargar y servir sin conocer su código. Todo ello **sin dependencias de runtime** (Constitución I) y **sin romper la API pública de 0.4** (los cambios son aditivos, salvo los observables listados en la sección 7).

## 2. Decisiones de partida (a confirmar en Q1–Q7)

1. **Las clases educativas no se reescriben.** `NeuronN`, `Layer` y `NetworkN` siguen siendo legibles, escalar a escalar. La velocidad llega por una clase nueva, `NetworkFast` (Q1), con la misma API y el mismo orden de pesos que `NetworkN`. Así los pesos se intercambian entre las dos y la clase lenta sirve de **oráculo exacto** de la rápida.
2. **El ahorro de memoria sí llega a las clases existentes**, sin cambiar su API: el estado de los optimizadores integrados (`SGD`, `Momentum`, `Adam`, `ClipOptimizer` sobre uno de ellos) pasa de un objeto por peso a buffers por neurona o por matriz. Un optimizador propio del usuario sigue funcionando por el camino escalar.
3. **La semilla es inyectable y opcional.** Por defecto todo sigue usando `Math.random` en el momento de la llamada, así que los tests que siembran `Math.random` no cambian.
4. **Formato de guardado 2** con clase, configuración y pesos. El formato 1 se sigue leyendo.
5. **Entrada solo de inferencia** `@dniskav/neuron/infer`, sin código de entrenamiento ni optimizadores.

## 3. Requisitos funcionales

### FR-1 Generador aleatorio inyectable
- Módulo `src/random.ts` con `random()`, `setRandom(fn | null)`, `seededRandom(seed)` (mulberry32) y `withRandom(fn, body)`.
- Ningún archivo de `src/` llama a `Math.random` salvo `src/random.ts`. Las semillas propias de `TSNE` y `KMeans` (`options.seed`) siguen funcionando igual.
- Con `setRandom(null)` (por defecto) el comportamiento es idéntico al de 0.4, número a número.

### FR-2 Núcleo numérico (`src/core/`)
- `src/core/tensor.ts`: operaciones sobre `Float64Array` planas en orden fila (row-major): `matVec`, `matTVec`, `addOuter`, `matMulBatch`, `matMulBatchT`, `addOuterBatch` y utilidades de copia. Contrato exacto en `contracts.md`.
- `src/core/optim.ts`: optimizadores vectoriales `VecSGD`, `VecMomentum`, `VecAdam` y `VecClip`, más `vecFromFactory(factory, size)`. Este último reconoce las instancias integradas, copia sus hiperparámetros y, si no las reconoce, cae en `ScalarArrayOptimizer`, que llama a la fábrica una vez por elemento como hoy.
- **Equivalencia exacta:** para la misma secuencia de gradientes, el optimizador vectorial produce **exactamente** (`Object.is`) los mismos pesos que N instancias escalares.

### FR-3 Memoria de los optimizadores en las clases existentes
- `NeuronN`, `WeightMatrix` y `BiasVector` guardan el estado del optimizador en buffers (FR-2) en lugar de un objeto por escalar.
- La API pública no cambia. Los pesos resultantes son idénticos (`Object.is`) a los de 0.4 con la misma semilla y las mismas llamadas.
- En una fase opcional se aplica lo mismo a `LSTMLayer`, `GRULayer`, `RNN`, `Conv1D`, `Conv2D`, `CausalConv1D` y `Seq2Seq`, siempre que actualicen todos sus elementos en cada paso.

### FR-4 `NetworkFast`
- Constructor `new NetworkFast(structure, options: NetworkNOptions)` con la misma validación y los mismos mensajes que `NetworkN` (cambiando el nombre de la clase).
- Métodos con el mismo contrato que `NetworkN`: `predict(x, training?)`, `train(x, t, lr)`, `trainWithDeltas(x, deltas, lr, reuseMasks?)`, `inputGradient(x, deltas, reuseMasks?)`, `getWeights()`, `setWeights(w)` y el campo `structure`.
- **Inicialización idéntica:** con la misma semilla, `new NetworkFast(s, o).getWeights()` es idéntico a `new NetworkN(s, o).getWeights()`. Consume `random()` en el mismo orden: capa a capa, neurona a neurona, peso a peso, y sesgo 0.
- **Entrenamiento equivalente:** tras K llamadas idénticas, los pesos coinciden con los de `NetworkN` con error relativo ≤ 1e-12. No se exige igualdad exacta porque el orden de las sumas puede cambiar.
- Los pesos viven en `Float64Array` por capa, en el mismo orden que el plano de `getWeights()`: para cada neurona, sus pesos y después su sesgo.

### FR-5 Entrenamiento por lotes
- `NetworkFast.trainBatch(X: number[][], T: number[][], lr): number`: un único paso del optimizador con la **media** de los gradientes por ejemplo. Devuelve la MSE media del lote. Con un lote de 1 es idéntico a `train`.
- `NetworkFast.predictBatch(X): number[][]`.
- Con dropout, cada ejemplo del lote sortea su propia máscara, en orden de ejemplo.
- `Trainer` acepta la opción `batchSize` (entero ≥ 1, por defecto 1). Si el modelo tiene `trainBatch`, lo usa; si no, y `batchSize > 1`, lanza un error claro (Constitución V). Con `batchSize = 1` el comportamiento es idéntico al actual.

### FR-6 Guardado completo (formato 2)
- `ModelSaver.save(model): string` produce `{ format: 2, kind, config, length, weights, timestamp }`.
- `ModelSaver.load(json): Model` devuelve una instancia nueva lista para predecir y para seguir entrenando, con el estado del optimizador reiniciado (Q4).
- `ModelSaver.toJSON` y `fromJSON` (formato 1) no cambian. `fromJSON` también acepta un JSON de formato 2 si la clase y la estructura coinciden con las del modelo.
- Clases con `kind` en esta versión: `NetworkN`, `NetworkFast`, `Network`, `NetworkLSTM`, `NetworkTransformer` y `NetworkTransformerRL`.
- **Activaciones por nombre:** cada activación integrada lleva `name` (y `alpha` las de fábrica). Guardar una activación propia sin nombre lanza un error que lo explica.
- **Optimizador por nombre** con sus hiperparámetros. Uno propio lanza error salvo que se pase `{ allowCustomOptimizer: true }`, en cuyo caso al cargar se usa SGD y se avisa en `load`.
- **`residual` como función** se guarda evaluada, como un array de booleanos por capa.
- Validación completa antes de crear nada, con mensajes que empiezan por `ModelSaver.load:`.

### FR-7 Entrada de inferencia `@dniskav/neuron/infer`
- `loadForInference(json)` acepta formato 2 de `NetworkN`, `NetworkFast` y `Network` y devuelve `{ kind, structure, predict(x), predictBatch(X) }`.
- Las predicciones coinciden con `model.predict` con error relativo ≤ 1e-12.
- El bundle `dist/infer.*` no contiene optimizadores, backprop, `Trainer` ni `random`. Tamaño ≤ 12 KB sin minificar.
- La inferencia de Transformers queda fuera de esta versión (Q5).

### FR-8 Benchmarks reproducibles
- `bench/run.mjs` (Node puro, sobre `dist/`) mide memoria por parámetro, ms por ejemplo al entrenar y al predecir, y ms por ejemplo con `trainBatch(32)`. Imprime JSON.
- La línea base de 0.4 se guarda en `specs/010-fast-core/bench-baseline.json` antes de cambiar `src/`.

### FR-9 Piezas intercambiables ("Lego")
Petición del usuario del 2026-11-18: poder reemplazar una parte de la librería por una propia, incluso de pago, y conservar el resto.
- **Backend numérico inyectable:** `setBackend(b: Partial<Backend>)` y `getBackend()` en `src/core/backend.ts`. `Backend` es el contrato de `tensor.ts` (`matVec`, `matTVec`, `addOuter`, `matMulBatch`, `matMulBatchT`, `addOuterBatch`). `NetworkFast` y la entrada de inferencia llaman siempre a `getBackend()`, nunca a `tensor.ts` directamente. Una función que falte en el `Partial` usa la implementación por defecto. `setBackend(null)` restaura la de por defecto.
- **Kit de conformidad** exportado: `checkBackend(b, { tolerance })` ejecuta los casos de equivalencia contra el backend por defecto y devuelve un informe `{ ok, failures[] }`. Sirve para que quien escriba su backend compruebe que da los mismos números.
- **Puntos de extensión documentados:** una sección "Extender la librería" en el README que enumera los contratos que ya existen (`Activation`, `Optimizer`/`OptimizerFactory`, deltas propios con `trainWithDeltas`/`inputGradient`, `TrainableNetwork` para `Trainer`, `Serializable` para `ModelSaver`, `setRandom`, `setBackend`). Cada uno con un ejemplo ejecutable verificado por `docs.readme`.
- **Registro de clases para el formato 2:** `ModelSaver.register(kind, { fromConfig })` permite que una clase propia que implemente `toConfig()`/`getWeights()`/`setWeights()` se guarde y cargue con `ModelSaver.save`/`load`.

### FR-10 Embeddings propios
Petición del usuario del 2026-11-18: usar embeddings calculados fuera (por otro modelo, propio o de pago) dentro de `neuron`.
Hoy ya se puede: como entrada de `NetworkN`, como secuencia de `NetworkTransformerRL` (entrada continua) o escribiendo en `NetworkTransformer.tokenEmb` con `setWeights`. Lo que falta:
- `NetworkTransformerOptions.freezeEmbeddings?: boolean | { token?: boolean; position?: boolean }` (por defecto `false`, idéntico a 0.4). Con el valor activado, `train` no modifica la tabla o tablas congeladas. Se comprueba con `getWeights` antes y después y con el oráculo de los demás parámetros.
- `NetworkTransformer.setTokenEmbeddings(rows: number[][], opts?: { freeze?: boolean })`: valida `rows.length === vocabSize`, que cada fila tenga `dModel` columnas y que todo sea finito, y escribe. Si el tamaño de los embeddings no es `dModel`, lanza un error que sugiere usar `NetworkTransformerRL` (que proyecta cualquier `inputDim`) o reducir antes con `PCA`.
- Un ejemplo ejecutable en el README: "Usar tus propios embeddings", con los tres caminos y cuándo conviene cada uno.
- Fuera de alcance: leer formatos de archivo de otros modelos (`.bin`, `.safetensors`, `.npy`). La librería recibe `number[][]` y no hace E/S.

## 4. Requisitos no funcionales (objetivos medibles, Q6)

Sobre `[256,256,256,10]`, `relu, relu, sigmoid`, en la misma máquina que la línea base. "`NetworkN`" en los cocientes es el de **0.4** (`bench-baseline.json`, `master`): FR-3 también acelera el `NetworkN` de la rama (con Adam pasó de ~6 ms a ~1,9 ms por ejemplo), así que compararse con él mediría otra cosa. Aclaración del orquestador del 2026-11-18, al verificar T012; no cambia ningún umbral.

| Id | Medida | Objetivo |
|---|---|---|
| NFR-1 | `NetworkFast.train`, Adam, ms por ejemplo | ≤ 1/5 de `NetworkN` |
| NFR-2 | `NetworkFast.trainBatch(32)`, Adam, ms por ejemplo | ≤ 1/8 de `NetworkN.train` |
| NFR-3 | `NetworkFast`, Adam, bytes por parámetro (heapUsed + arrayBuffers) | ≤ 36 (2026-11-18: el usuario delega y el orquestador sube de 32 a 36, porque el lote necesita el buffer de gradiente: peso, gradiente, m y v son 32 bytes) |
| NFR-4 | `NetworkN`, Adam, bytes por parámetro (FR-3; heapUsed + arrayBuffers) | ≤ 50 (0.4: 136,7, medido en T001b) |
| NFR-5 | `NetworkFast.predict`, ms por ejemplo | ≤ 0,8 de `NetworkN.predict` de 0.4 (2026-11-18: el objetivo inicial ≤ 1/3 no era realista en JS puro; T026 midió 1,22; el usuario aprueba T031 y el nuevo objetivo) |
| NFR-6 | La suite completa | Sin regresión de tiempo > 20 % |

Si la línea base de T001 sale muy distinta de la tabla de arriba, el orquestador propone nuevos objetivos al usuario antes de lanzar el Lote 3.

## 5. Fuera de alcance

- GPU, WASM, workers, SIMD.
- Un autograd vectorizado que sustituya el backprop escrito a mano. Se estudiará en la feature 011 a partir de un informe (T024 opcional).
- Versiones `Fast` de LSTM y Transformer (011).
- Guardar el estado de los optimizadores.

## 6. Criterios de aceptación

1. Toda la suite de 0.4 pasa **sin modificar ningún test existente**, salvo los listados en la sección 7.
2. Oráculos nuevos: la equivalencia exacta de los optimizadores vectoriales, la equivalencia de `NetworkFast` con `NetworkN`, las diferencias finitas de `NetworkFast` (coseno > 0.999 y `ratio` en [0.99, 1.01]) y la del gradiente del lote.
3. Los objetivos NFR-1 a NFR-5 se cumplen, medidos por el `verifier`, no por el autor.
4. `dependencies` sigue vacío. `npx tsc --noEmit -p .`, `npm run typecheck`, `npm run build` y `npm run coverage` pasan (con los umbrales actuales).
5. 20 ejecuciones seguidas de la suite y 3 con Node 20.19, sin fallos.
6. `AGENTS.md` (mapa de exportaciones), `README` y `CHANGELOG` actualizados; el test `docs.agents.exports` pasa.

## 7. Cambios observables

- `NeuronN.weights` se actualiza en sitio: tras `_update`, el array es el mismo objeto con valores nuevos. En 0.4 se reemplazaba por uno nuevo, así que quien guardara una referencia veía los pesos viejos (Q3).
- Las activaciones integradas ganan el campo `name`. Es aditivo, pero un `toEqual` contra el objeto literal `{ fn, dfn }` dejaría de pasar.

## 8. Preguntas abiertas (resolver antes de lanzar las tareas que dependen de ellas)

- **Q1 (T010–T013):** ¿la clase rápida se llama `NetworkFast` y convive con `NetworkN`? Recomendado: sí. La alternativa es reescribir `NetworkN` por dentro, que rompe `layers[].neurons[]`, del que dependen `VAE`, `GAN`, `Autoencoder`, `WeightInspector` y el juego.
- **Q2 (Lote 0):** rama `feat/0.5-core` desde `origin/master` (0.4.0) y versión 0.5.0. La Constitución dice "el trabajo se integra en `fix/v0.4`". Propuesta de enmienda: "en la rama de integración de la feature en curso". Requiere aprobación del usuario.
- **Q3 (T009):** ¿se acepta que `NeuronN.weights` se escriba en sitio? Recomendado: sí; es más correcto y evita copiar arrays.
- **Q4 (T014):** ¿el formato 2 no guarda el estado del optimizador? Recomendado: no guardarlo; el archivo es tres veces más pequeño y casi nadie reanuda Adam a mitad.
- **Q5 (T018):** ¿la inferencia cubre solo los modelos densos en 0.5? Recomendado: sí; el Transformer en 011.
- **Q6 (T022):** ¿se aceptan los objetivos NFR-1 a NFR-5?
- **Q8 (T005, T025):** ¿el backend es global (`setBackend`) o también por instancia (`new NetworkFast(s, { backend })`)? Recomendado: los dos, con la opción por instancia por encima de la global; cuesta poco y permite comparar dos backends en el mismo proceso.
- **Q9 (T027):** ¿basta con congelar y cargar la tabla de `NetworkTransformer`, o también hace falta una capa `Embedding` suelta para `NetworkFast` (ids → vectores aprendibles)? Recomendado: solo congelar y cargar en 0.5; la capa suelta en 011.
- **Q7 (T024):** ¿se hace ahora el informe exploratorio del autograd vectorizado, o se deja para 011?
