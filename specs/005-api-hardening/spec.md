# Feature Specification: api-hardening

**Feature Branch**: `fix/v0.4` (se trabaja directamente en ella; no se ejecutó `create-new-feature.sh` porque crearía una rama)
**Created**: 2026-11-08
**Status**: Draft
**Input**: `specs/ORCHESTRATION.md` sección "005 api-hardening", `specs/HANDOFF.md` (hallazgos N4, N5, N7, N10, "Notas", "Cambios observables" de 003 y 004, limitación de 004, "Lección CI"), `specs/000-audit/AUDIT.md` (I2, I3, I4, I5, I6, I7), Constitución V, VI, VII, IV. Condición previa: 001–003 fusionadas en `master`, 004 hecha en `fix/v0.4` (PR #7 pendiente de CI).

## Estado verificado hoy (leído en `src/` y `tests/`)

Convención: validar con `validateArray(arr, n, 'Clase.método')` de `src/Validation.ts` (ya exportada desde `src/index.ts`, no hace falta tocar el índice): comprueba que es un array, la longitud y que cada valor es un número finito.

- **I2, `setWeights` sin longitud.** Los `setWeights`/`setWeightsFlat` planos de `NetworkN` (`src/NetworkN.ts:169`), `Network` (`:65`), `WeightMatrix`/`BiasVector`/`EmbeddingMatrix` (`src/MatMul.ts:109,137,180`), `AttentionHead`, `MultiHeadAttention`, `LayerNorm`, `BatchNorm`, `Conv1D`, `Conv2D`, `TransformerBlock`, `NetworkTransformer`, `NetworkTransformerRL`, `LSTMLayer`/`GRULayer`/`NetworkLSTM` (`...Flat`) copian con `weights[idx++]` sin comprobar nada: un array corto deja `undefined`/`NaN` en los pesos, uno largo se recorta en silencio. `ModelSaver.fromJSON` (`src/ModelSaver.ts:36-42`) solo comprueba que `weights` sea un array; no compara con el modelo, no detecta un `null` (`JSON.stringify(NaN)` da `null`) y `toJSON` no guarda ni longitud ni estructura. Además `NetworkN.setWeights` resetea las máscaras de dropout ANTES de escribir, así que un fallo a mitad deja el modelo medio escrito.
- **I6, referencias vivas.** Solo las tres versiones estructuradas devuelven los arrays internos: `LSTMLayer.getWeights()` (`src/LSTMLayer.ts:281`), `GRULayer.getWeights()` (`src/GRU.ts:260`) y, por delegación, `NetworkLSTM.getWeights().lstm` (`src/NetworkLSTM.ts:167-169`; la parte `dense` ya es copia). Sus `setWeights(data)` hacen lo contrario: ASIGNAN las referencias de `data`, así que `b.setWeights(a.getWeights())` deja a `a` y `b` compartiendo los mismos arrays y entrenar uno mueve al otro. Las variantes planas (`*Flat`, `NetworkN`, `Conv`, MatMul, etc.) devuelven arrays nuevos: no hay defecto. `NetworkN.getWeights` ya es una copia (`w.push(...)`).
- **I5 / N4, `Network` con varias salidas.** `Network.train(inputs, target: number, lr)` (`src/Network.ts:24-50`) usa `outputLayer.predict(...)[0]` y solo actualiza `neurons[0]` de la capa de salida; el constructor admite `nOutputs > 1` y `predict` las devuelve todas. Con `nOutputs > 1` las demás salidas no cambian nunca. Además `target: number` hace que `Network` NO cumpla `TrainableNetwork` (`train(inputs, targets: number[], lr)` de `Trainer`): `new Trainer(new Network(...))` no compila sin cast y en ejecución `validateNumber(array)` lanza. `it.todo('AUDIT I5')` en `tests/oracle/Network.oracle.test.ts:69`.
- **I3, `Trainer`** (`src/Trainer.ts`): (a) `clipValue` se guarda (`:49,80,104`) y nadie lo lee: no hace nada (la cabecera admite "actual clipping must be configured at network level"); (b) `earlyStopping` sin `setValidationData` no hace nada y no avisa (`canValidate` es `false` en silencio, `:146-147`); `weightDecay`, `earlyStopping` y `computeMetrics` con una red sin `getWeights/setWeights/predict` solo emiten `console.warn` (`:153-175`) y siguen sin la función; (c) con `earlyStopping` el entrenamiento se detiene dejando los pesos de la última época, no los mejores (`_bestLoss` se guarda, los pesos no); (d) un dataset vacío da `epochLoss /= 0` = `NaN` en el historial sin error (`:203`; el test `'empty dataset throws'` de `tests/Trainer.test.ts` afirma lo contrario de su nombre: espera 5 entradas de historial); (e) `epochs`, `lr`, `lrDecay`, `weightDecay` y `patience` no se validan; (f) el decaimiento de pesos se aplica también a los sesgos (comportamiento a documentar, no se cambia).
- **I4, `padding: 'same'` con kernel par.** `Conv1D` rellena `floor((k−1)/2)` por cada lado (`src/Conv1D.ts:94`): con `k` par y stride 1 la salida mide `L−1`, pero `getOutputLength()` (`:179`) anuncia `ceil(L/stride)` = `L`. `Conv2D` rellena `floor(k/2)` por lado (`src/Conv2D.ts:105`): con `k` par la salida mide `H+1` (4×4 → 5×5) y `outputShape()` coincide con esa salida errónea. Con `k` impar ambos son correctos y NO deben cambiar. Las dos `backward` recortan el relleno con el mismo desplazamiento. Ninguna `backward` valida la forma de `dOut`. Los oráculos de 002 cubren pesos, no `dInput` (N5).
- **I7, `Value.backward()`** (`src/Tape.ts:166-185`): `buildTopo` es recursivo; una cadena de 20 000 nodos desborda la pila (`RangeError`). El orden de visita (hijos antes que el padre, en el orden de iteración del `Set` `_prev`) determina el orden de suma de los gradientes: una versión iterativa debe conservarlo para no cambiar los resultados ni en el último bit.
- **N7 y la familia de regresiones** (`src/LogisticRegression.ts`, `src/LinearRegression.ts`): `LogisticRegression.predict` (`:136`) y `SoftmaxRegression.predictProba` (`:260`) validan contra `_nFeatures`, privado y fijado solo por `train()`: con `weights` asignados a mano lanzan `expected array of length 0`. `LinearRegression.predict` (`:224`) y `getCoefficients` (`:244`) tienen el MISMO defecto (el HANDOFF solo menciona las dos primeras): `x.length !== _nFeatures`. Hoy hay 2 `it.fails // AUDIT N7` en `tests/oracle/LogisticRegression.oracle.test.ts:286,293`.
- **Entrenamiento incremental**: `LogisticRegression.train` YA es incremental (solo reinicia si `weights.length !== nFeatures`, `:89`). `SoftmaxRegression.train` reinicia SIEMPRE con `Math.random` (`:214`, y recalcula `_nClasses = max(y)+1`, `:206`, que con un lote sin la clase mayor encogería el modelo). `LinearRegression.fitGD` reinicia SIEMPRE a cero (`:186`). Los oráculos de 002 dependen de ello: `LinearRegression.oracle` usa `fitGD(X, y, lr, 1)` como "paso" confiando en el reinicio a cero; `Softmax` fija los pesos iniciales con un stub de `Math.random` y su cabecera dice "llamadas sucesivas no acumulan". Tampoco se valida que las filas de `X` tengan todas `nFeatures` ni que las etiquetas de `SoftmaxRegression` sean enteros ≥ 0 (con una etiqueta fuera de rango `probs[trueClass]` es `undefined` y la pérdida es `NaN`).
- **N10 en `NetworkN.train`/`Network.train`:** `NetworkN.train` (`:106-118`) usa `g = t − p` (gradiente de `L = ½Σ(t−p)²`) y devuelve `Σ(t−p)²/n` (MSE medio): con `n` salidas el paso es `n/2` veces el gradiente de la pérdida devuelta (para `n = 1`, el paso es la MITAD: el gradiente de `(t−p)²` sería el doble). `Network.train` hace lo mismo con una salida (delta `(t−p)·dfn`, devuelve `(t−p)²`, factor 2). `Trainer` mide la pérdida de validación como MSE medio (`_computeLoss`), de modo que el historial de `train` y esa pérdida usan la misma escala, y la pérdida es la que devuelve `train`. `NetworkN.train` solo la usan los tests y el código de usuario (`Autoencoder`, `VAE`, `GAN` usan `trainWithDeltas`).
- **Dropout en `inputGradient`** (`src/NetworkN.ts:147-153`): `inputGradient` llama a `_forwardAll(inputs, true)` y SORTEA una máscara nueva. Quien lo llama ya hizo un forward antes: `Autoencoder.train` hace `decoder.predict(z, true)` (máscara 1, de la que salen `x̂` y los deltas), luego `inputGradient` (máscara 2) y luego `trainWithDeltas` (máscara 3, `:125-129`); `VAE.forward` usa `decoder.predict(z)` SIN dropout y después `inputGradient`/`trainWithDeltas` sortean máscaras; `GAN.trainStep` genera `xFake2` y puntúa con `predict` (sin dropout) y luego `D.inputGradient` sortea una máscara. El gradiente, los deltas y la actualización pertenecen a funciones distintas. Con `dropoutRate = 0` no pasa nada (por eso 004 no lo vio).
- **`trainWithDeltas` no valida longitudes** (`src/NetworkN.ts:125-129`): `inputGradient` sí (`:149-150`); `trainWithDeltas` no valida ni `inputs` ni `outputDeltas`, y un `outputDeltas` corto deja `undefined` en los deltas.
- **`LSTMLayer.backprop` ante longitud distinta** (`src/LSTMLayer.ts:187`): `if (T === 0 || dh_seq.length !== T) return {ceros}` sin error, sin vaciar la trayectoria y sin avisar. Mismo patrón en `GRULayer.backprop` (`src/GRU.ts:148`, `return` vacío) y `NetworkLSTM.train` (`src/NetworkLSTM.ts:99`, `return` vacío). `dcFinal` de longitud errónea se ignora (`:197`). No validan que cada `dh_seq[t]` mida `hSize`.
- **N5, oráculos de `dx`:** `Conv1D`, `Conv2D` (con `backward` que ya devuelve `dInput`), `Dropout`, `Flatten` y `MaxPool2D` no tienen oráculo de `dx` (HANDOFF "Pendiente de la revisión T038 (d)" y N5).
- **No hay en `src/` ningún `getWeights` que dependa de `Serializable` más allá de `ModelSaver`**; `Serializable` (`src/ModelSaver.ts`) es `getWeights(): number[]` + `setWeights(number[])`. `NetworkLSTM`/`LSTMLayer`/`GRULayer.getWeights()` devuelven un objeto y NO son `Serializable`; sus `getWeightsFlat()` sí lo serían.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Los pesos no se cargan a medias (I2, I6) (Priority: P1)

Cargar pesos de longitud o forma equivocada lanza un error claro y NO modifica el modelo; obtener pesos devuelve copias.

**Why this priority**: es el hallazgo I2/I6 de la auditoría y lo usa `ModelSaver`; hoy un archivo de otro modelo "se carga" y produce `NaN`.

**Independent Test**: `npx vitest run tests/api` para los `setWeights`/`getWeights`; `tests/ModelSaver.test.ts` y `tests/api/ModelSaver.hardening.test.ts` para el guardado.

**Acceptance Scenarios**:
1. **Given** cada clase con `setWeights`/`setWeightsFlat` plano, **When** se pasa un array corto, largo, con `NaN`/`Infinity`, con `undefined` o que no es un array, **Then** lanza `Error` con mensaje que contiene el nombre `Clase.setWeights`, la longitud esperada y la recibida, y `getWeights()` antes/después es idéntico (comparación exacta).
2. **Given** pesos válidos, **When** se hace `set(get())`, **Then** el resultado es idéntico a hoy (`toEqual`) y las pruebas existentes pasan sin cambios.
3. **Given** `LSTMLayer`, `GRULayer` y `NetworkLSTM`, **When** se muta lo que devuelve `getWeights()` o se entrena después de obtenerlo, **Then** el modelo no cambia por la mutación y la copia obtenida no cambia al entrenar; y **When** `b.setWeights(a.getWeights())` y se entrena `b`, **Then** `a` no cambia.
4. **Given** `ModelSaver.toJSON`, **Then** el JSON lleva `weights`, `timestamp` (como hoy) y metadatos aditivos (`format`, `length`, y `structure` si el modelo la expone). **Given** `fromJSON`, **Then** rechaza (con modelo intacto): longitud distinta de la del modelo, `length` que no coincide con `weights.length`, `structure` distinta, valores no numéricos o `null`, y acepta el JSON antiguo (sin metadatos) si la longitud coincide.

### User Story 2 - `Network` entrena todas sus salidas (I5, N4) (Priority: P1)

`Network.train` acepta el objetivo como `number[]` (o `number` si hay una sola salida) y actualiza todas las salidas.

**Acceptance Scenarios**:
1. **Given** `Network(3, 4, 2)` y un objetivo `[0.2, 0.8]`, **Then** el oráculo (coseno > 0.999 contra diferencias finitas de `½Σ(y_k − t_k)²` sobre TODOS los pesos, incluido cada vector de salida) pasa y "aprende" (pérdida < 0.5 × inicial) para ambas salidas.
2. **Given** `Network(3, 4, 1)`, **Then** los pesos tras un paso, con `train(x, 0.9, lr)` y `train(x, [0.9], lr)`, son idénticos entre sí y a los del código anterior (comparados con una réplica del algoritmo antiguo en el propio test, `toEqual`).
3. **Given** `Network(3, 4, 2)` con un `number` escalar, **Then** lanza un error que dice que hacen falta 2 valores.
4. **Given** `new Trainer(new Network(2, 3, 2))`, **Then** compila (typecheck) y entrena.

### User Story 3 - `Trainer` no promete lo que no hace (I3) (Priority: P1)

Ninguna opción de `Trainer` se ignora en silencio; los datos inválidos lanzan.

**Acceptance Scenarios**: según las decisiones Q1–Q3: (a) `clipValue` deja de existir (o hace algo medible); (b) cada combinación de opción + red/datos que hoy se ignora lanza un error con el nombre de la opción; (c) con `restoreBestWeights: true` los pesos finales son los de la mejor época (`getWeights()` final `toEqual` los de la época con menor pérdida de validación, mutante: restaurar los últimos falla); (d) `train({inputs: [], targets: []})` lanza; (e) `epochs`, `lr`, `lrDecay`, `weightDecay`, `patience` inválidos lanzan antes de entrenar; (f) con opciones válidas el historial es idéntico al actual (golden con `toEqual`).

### User Story 4 - `padding: 'same'` con kernel par devuelve la forma anunciada (I4) (Priority: P1)

`Conv1D`/`Conv2D` con `same`: la salida mide `ceil(L/stride)` (por eje) para cualquier tamaño de kernel y `getOutputLength()`/`outputShape()` coinciden con `forward`. Con kernel impar los números no cambian.

**Acceptance Scenarios**:
1. **Given** `k ∈ 1..6`, `stride ∈ 1..3` y varias longitudes, **Then** `forward(...).length` (y por eje en Conv2D, incluidos kernels rectangulares como `[4, 3]`) coincide con lo anunciado y con `ceil(L/stride)`.
2. **Given** `k` impar, **Then** salida y `dInput` son idénticas a una réplica del código antiguo (`toEqual`).
3. **Given** `k` par, **Then** el relleno es asimétrico (propuesta Q4), la salida se calcula a mano en un caso pequeño, y `dInput` coincide con diferencias finitas por componente (`|a − b| < 1e-6`) en `valid`, `same` impar, `same` par y `stride 2`.
4. **Given** un `dOut` con forma errónea, **Then** `backward` lanza antes de modificar nada.

### User Story 5 - `Value.backward()` no desborda la pila (I7) (Priority: P2)

Una cadena de 20 000 (y 100 000) nodos hace `backward()` sin `RangeError`, con el mismo gradiente que la versión recursiva en grafos pequeños y medianos (igualdad exacta, no solo aproximada), incluidos nodos compartidos (diamante) y un nodo usado dos veces (`a.add(a)`).

### User Story 6 - Los modelos de regresión aceptan pesos a mano y se pueden seguir entrenando (N7, incremental) (Priority: P2)

1. **Given** `weights`/`bias` asignados a mano sin `train()`, **Then** `LogisticRegression.predict`, `SoftmaxRegression.predictProba` y `LinearRegression.predict`/`getCoefficients` funcionan y validan contra `weights.length`; los dos `it.fails // AUDIT N7` pasan a `it` normales.
2. **Given** pesos existentes con forma compatible, **When** se llama dos veces a `train`/`fitGD`, **Then** el resultado es EXACTAMENTE el de una sola llamada con el doble de épocas (los tres modelos son deterministas desde un estado fijo: sin barajar).

### User Story 7 - La máscara de dropout es la misma en la pérdida, el gradiente y la actualización (Priority: P2)

Con `dropoutRate > 0` en el decoder (`Autoencoder`, `VAE`) o en el discriminador (`GAN`), el paso sigue `−dL/dw` de la pérdida calculada con la MISMA máscara (oráculo con máscara fija mediante `seedRandom`), y con `dropoutRate = 0` las trayectorias son idénticas a las de hoy.

### User Story 8 - Las recurrentes fallan en vez de devolver ceros (Priority: P2)

`LSTMLayer.backprop`, `GRULayer.backprop` y `NetworkLSTM.train` lanzan ante longitud de secuencia distinta, `dh_seq[t]`/`targets[t]` de tamaño erróneo o `dcFinal` de longitud errónea, sin vaciar la trayectoria; con entradas válidas no cambia nada (los oráculos de LSTM, GRU, NetworkLSTM y Seq2Seq siguen en verde).

### User Story 9 - La escala del paso de `NetworkN`/`Network` coincide con la pérdida que devuelven (N10) (Priority: P3, decisión Q8)

Según Q8: con la opción A, el paso de `NetworkN.train`/`Network.train` es el gradiente de la MSE media que devuelven (`ratio ∈ [0.99, 1.01]`).

### Edge Cases

- **Atomicidad:** cada `setWeights` valida ANTES de escribir; un error nunca deja un modelo a medias (se prueba comparando `getWeights()` antes/después).
- **Composición:** `TransformerBlock`/`NetworkTransformer`/`NetworkTransformerRL`/`NetworkLSTM` delegan con `slice`; validan la longitud total en el nivel exterior (si solo validaran los hijos, un array largo se recortaría en silencio).
- **JSON y `NaN`:** `JSON.stringify(NaN) === 'null'`; `fromJSON` lo detecta y lo dice.
- **`NetworkN.setWeights` y dropout:** hoy resetea las máscaras antes de escribir; el nuevo orden es validar, resetear, escribir.
- **`Network` con `nOutputs = 1` y objetivo `number[]` de un elemento:** válido y equivalente al escalar.
- **`Value`:** `a.add(a)` (mismo hijo dos veces) debe seguir acumulando el doble; `_prev` es un `Set` (la duplicación se pierde en el `Set`, no en `_backward`).
- **`Value.backward()` repetido** acumula `grad` sin ponerlo a cero (comportamiento actual, tipo micrograd): NO se cambia aquí; se anota para 007.
- **`Conv` `same` con `k > L`:** `Conv1D` con `k > inputLength` y `same` es válido (la guardia solo cubre `valid`); el relleno asimétrico debe funcionar también ahí.
- **`SoftmaxRegression` continuado con una clase nueva:** si una etiqueta es ≥ `weights.length` y el modelo ya existe, lanza (no amplía el modelo en silencio).
- **`Math.max(...y)` con muchos datos** (> ~100 000 elementos) desborda la pila de argumentos: la tarea de regresiones lo calcula con un bucle (cambio interno, mismo resultado).
- **Restaurar mejores pesos y optimizador:** `Trainer` restaura solo los pesos; el estado de Adam/momentum (en cada neurona/optimizador) no se restaura (se documenta).
- **Lección CI:** ninguna prueba nueva depende de un umbral de entrenamiento caótico; las de "aprende" (`Network` con 2 salidas, regresiones, `Trainer` con `restoreBestWeights`) se validan con perturbaciones de `lr` y semilla y quedan por debajo de `0.7 ×` el criterio.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (I2): Todo `setWeights`/`setWeightsFlat` plano de `src/` MUST llamar a `validateArray(weights, esperada, 'Clase.setWeights')` antes de escribir cualquier valor. La longitud esperada se calcula de la estructura (no de `weights`). El mensaje contiene el nombre de la clase y método, la longitud esperada y la recibida.
- **FR-002** (I2): `ModelSaver.toJSON` MUST añadir `format: 1`, `length` y, solo si el modelo expone `structure` (array de números), `structure`; `ModelSaver.fromJSON` MUST validar sin modificar el modelo: `weights` array de números finitos, longitud igual a `model.getWeights().length` (también para JSON antiguo), `length` coherente con `weights.length` si existe, `structure` igual a la del modelo si ambas existen. `Serializable` gana `structure?` opcional (aditivo). (Q11)
- **FR-003** (I6): `LSTMLayer.getWeights`, `GRULayer.getWeights` MUST devolver copias profundas; sus `setWeights(data)` MUST validar la forma (matrices `hSize × (inputSize + hSize)`, sesgos `hSize`) y COPIAR en lugar de asignar referencias; `NetworkLSTM.setWeights` igual (parte `dense` ya copia pero sin validar). (Q12)
- **FR-004** (I5): `Network.train(inputs, target: number | number[], lr)` MUST aceptar `number[]` de longitud `nOutputs` (o `number` solo si `nOutputs === 1`), actualizar TODAS las neuronas de salida, calcular los deltas ocultos con los pesos de salida ORIGINALES sumando sobre todas las salidas, y devolver el error cuadrático medio sobre las salidas (igual que antes si hay una). Con `nOutputs = 1` la trayectoria es idéntica a la actual. (Q5)
- **FR-005** (I3): `Trainer` MUST NOT ignorar opciones en silencio y MUST validar sus entradas según Q1–Q3; con opciones válidas y sin las nuevas el historial es idéntico al actual.
- **FR-006** (I4): `Conv1D`/`Conv2D` con `same` MUST rellenar `total = k − 1` por eje: `left = floor(total/2)`, `right = total − left`, de modo que con stride 1 la salida mide `L` y en general `ceil(L/stride)`; `getOutputLength()`/`outputShape()` MUST coincidir con `forward`; el recorte del relleno en `backward` MUST usar `left`. `backward` MUST validar la forma de `dOut`. Con `k` impar el relleno es el actual (`left = right`). (Q4)
- **FR-007** (I7): `Value.backward()` MUST ser iterativa, con el MISMO orden topológico (post-orden, hijos en el orden de iteración de `_prev`) que la versión recursiva.
- **FR-008** (N7): `LogisticRegression.predict`, `SoftmaxRegression.predictProba` y `LinearRegression.predict`/`getCoefficients` MUST derivar el número de características de `weights` (`weights.length`, `weights[0].length`, `weights.length − 1`) y validar que la forma sea coherente (filas del mismo ancho y `biases.length === weights.length` en Softmax); `train()` sin pesos previos no cambia. (Q6)
- **FR-009** (incremental): `SoftmaxRegression.train` y `LinearRegression.fitGD` MUST continuar desde los pesos existentes si su forma coincide con los datos (como ya hace `LogisticRegression.train`) e inicializar en caso contrario; `train` MUST validar filas de `X` (longitud y finitud) y etiquetas (`Softmax`: enteros ≥ 0 y < K si el modelo existe; `Logistic`: en [0, 1]). (Q7)
- **FR-010** (dropout): `NetworkN.inputGradient` y `trainWithDeltas` MUST poder reutilizar las máscaras del último forward de entrenamiento (parámetro opcional `reuseMasks`, por defecto `false`: comportamiento actual), y `Autoencoder`, `VAE` y `GAN` MUST usarlo de modo que pérdida, gradiente y actualización compartan máscara; sin consumir `Math.random` adicional al reutilizar. Con `dropoutRate = 0` las trayectorias de todos los modelos son idénticas. (Q9)
- **FR-011** (validación): `NetworkN.trainWithDeltas` MUST validar `inputs` y `outputDeltas` con `validateArray`, antes de consumir `Math.random` o modificar pesos.
- **FR-012** (recurrentes): `LSTMLayer.backprop`, `GRULayer.backprop` y `NetworkLSTM.train` MUST lanzar ante longitud de secuencia distinta de la trayectoria, tamaños de fila erróneos o `dcFinal` de longitud errónea; con trayectoria vacía y secuencia vacía devuelven (sin error) lo mismo que hoy. Un error no vacía la trayectoria. (Q10)
- **FR-013** (N10, Q8): según la decisión, `NetworkN.train` y `Network.train` MUST dar un paso igual al gradiente de la pérdida que devuelven (opción A), o el cambio MUST quedar documentado (opción C).
- **FR-014** (N5 parcial): `Conv1D` y `Conv2D` MUST tener oráculo de `dInput` por componente (`1e-6`) en `valid`, `same` impar, `same` par y stride 2. `Dropout`, `Flatten` y `MaxPool2D` se aplazan a 006.
- **FR-015** (cierre de N7): tras la feature NO queda ningún `it.fails` ni `it.todo` en `tests/`, ni `AUDIT N7|I5|I2|I3|I4|I6|I7` como defecto abierto.
- **FR-016** (IV, V): Cada test de validación MUST fallar antes del cambio y pasar después (`git stash push -- <src de la tarea>`, correr el test nuevo, `git stash pop`), con mensaje de error claro (`toThrow(/.../)` sobre el texto), y NO cambiar el comportamiento válido: para cada cambio de código con trayectoria numérica, un test "golden" (réplica del algoritmo antiguo en el test, o constantes capturadas en la base antes de tocar `src/`, tolerancia 1e-9) lo demuestra.
- **FR-017**: Ninguna tarea modifica `package.json`, `package-lock.json`, `src/index.ts` ni el changelog; cada tarea toca solo sus `[files]`.
- **FR-018**: La suite MUST quedar en verde y estable: 20 ejecuciones seguidas, 5 con `--sequence.shuffle`, la suite completa bajo Node 20.19, y el CI de GitHub (ubuntu, Node 20) en verde antes de cerrar la feature.

### Reparto (propuesta): qué entra en 005 y qué se aplaza

| Elemento | 005 | Razón (código) |
|---|---|---|
| I2 `setWeights` + `ModelSaver` | sí | `weights[idx++]` sin comprobación en ~20 clases |
| I6 copias `getWeights` (LSTM/GRU/NetworkLSTM) | sí | única fuente de referencias vivas; `setWeights` además hace alias |
| I5 / N4 `Network` varias salidas | sí | `[0]` fijo en `train`; además impide usar `Network` con `Trainer` |
| I3 `Trainer` | sí | `clipValue` muerto, opciones ignoradas, sin mejores pesos |
| I4 `padding: 'same'` | sí, con oráculos de `dx` de Conv1D/Conv2D | el desplazamiento del recorte en `backward` cambia: sin oráculo de `dInput` el arreglo no es verificable |
| I7 `Value.backward` | sí | una función, criterio medible |
| N7 | sí (+ `LinearRegression`, mismo defecto) | retira los 2 `it.fails` |
| Entrenamiento incremental Softmax/fitGD | sí (Q7) | `Logistic` ya lo es; sin esto N7 queda cojo (pesos a mano + `train` los pisa) |
| `trainWithDeltas` valida longitudes | sí | 1 línea, mismo contrato que `inputGradient` |
| Dropout y `inputGradient` | sí (Q9) | afecta a AE/VAE/GAN con `dropoutRate > 0` |
| `LSTMLayer.backprop` ceros en silencio | sí (+ GRU, NetworkLSTM) | mismo patrón |
| `NetworkN.train`/`Network.train` ½Σ vs MSE | sí, decisión Q8 | cambia el `lr` efectivo; última tarea de la cadena `NetworkN` |
| N5: Conv1D/Conv2D `dInput` | sí (dentro de I4) | verifica el arreglo |
| N5: Dropout, Flatten, MaxPool2D `dx` | NO → 006 | no cambian en 005; es cobertura pura |
| Validación de entradas del resto de modelos (filas de `X` en KMeans, PCA, etc.) | NO → 006 | fuera de los hallazgos I2–I7; se revisa módulo a módulo |
| Pérdida: `weightDecay` sobre sesgos | NO | documentado, no es un defecto de contrato |
| `Value.backward` repetido acumula | NO → 007 (documentar) | diseño tipo micrograd |

### Cambios observables para el changelog (feature 007/008)

1. `setWeights`/`setWeightsFlat` de todas las clases LANZAN con longitud o forma incorrecta (antes dejaban `NaN`/recortaban). `ModelSaver.fromJSON` rechaza JSON de otro modelo; `toJSON` añade `format`, `length`, `structure` (aditivo; los archivos antiguos siguen cargando si la longitud coincide).
2. `LSTMLayer`/`GRULayer`/`NetworkLSTM.getWeights()` devuelven copias; `setWeights(data)` copia en vez de compartir referencias.
3. `Network.train` acepta `number[]` (aditivo) y entrena todas las salidas (antes solo la primera); un escalar con `nOutputs > 1` lanza. `Network` ya cumple `TrainableNetwork`.
4. `Trainer`: se ELIMINA `clipValue` (si Q1 = eliminar; usar `ClippedOptimizerFactory`); opciones sin efecto y datos vacíos LANZAN en vez de ignorarse (Q2); opción nueva `restoreBestWeights` (Q3).
5. `Conv1D`/`Conv2D` con `same` y kernel par: la salida pasa a medir `ceil(L/stride)` (antes `L−1` / `H+1`), con relleno asimétrico (más a la derecha/abajo); `getOutputLength`/`outputShape` coinciden con `forward`. Kernel impar: sin cambios. `backward` valida la forma de `dOut`.
6. `Value.backward()` ya no desborda la pila en grafos profundos; mismos gradientes.
7. `LogisticRegression.predict`, `SoftmaxRegression.predictProba`, `LinearRegression.predict`/`getCoefficients` funcionan con `weights` asignados a mano. `SoftmaxRegression.train` y `LinearRegression.fitGD` CONTINÚAN desde los pesos existentes si la forma coincide (antes reiniciaban; para reiniciar, crear una instancia nueva); `train` valida filas de `X` y etiquetas.
8. `NetworkN.trainWithDeltas` valida longitudes. `NetworkN.inputGradient`/`trainWithDeltas` ganan el parámetro opcional `reuseMasks`; con dropout activo, `Autoencoder`/`VAE`/`GAN` pasan a usar una sola máscara por paso (las trayectorias con `dropoutRate > 0` cambian).
9. `LSTMLayer.backprop`, `GRULayer.backprop` y `NetworkLSTM.train` lanzan ante secuencias de longitud distinta (antes no hacían nada).
10. (Q8, si A) `NetworkN.train` y `Network.train` dan un paso `2/n` veces el anterior (para `n = 1`, el doble): para conservar la dinámica anterior, dividir el `lr` entre `2/n`.

## Success Criteria *(mandatory)*

- **SC-001**: `rg "it\.fails|it\.todo" tests` no devuelve nada; `rg "AUDIT (N7|I5|I2|I3|I4|I6|I7)" tests` no devuelve nada como defecto abierto.
- **SC-002**: Por tarea, los tests nuevos FALLAN sin el cambio de `src/` (stash) y PASAN con él; los mutantes listados en `tasks.md` (que QUITAN o ALTERAN una validación o un término) hacen fallar el archivo de tests correspondiente; revertidos, `git diff --quiet -- src`.
- **SC-003**: Las trayectorias válidas no cambian: las pruebas golden (réplica antigua o constantes de la base) pasan con `toEqual`/1e-9, y toda la carpeta `tests/oracle` pasa sin tocar los umbrales; la única excepción es lo declarado en Q4, Q7, Q8 y Q9.
- **SC-004**: `npx tsc --noEmit -p .`, `npm run typecheck`, `npm run build` y `npx vitest run` en verde; 20 ejecuciones seguidas, 5 con `--sequence.shuffle` y la suite completa bajo Node 20.19 (`~/.nvm/versions/node/v20.19.0`) sin fallos; CI de GitHub en verde antes de dar la feature por cerrada.
- **SC-005**: La revisión `math-reviewer` no deja bloqueantes.
- **SC-006**: `git diff --quiet -- package.json package-lock.json src/index.ts` en cada tarea.

## Assumptions

- Los worktrees parten de `master` sin `node_modules`: cada tarea se ejecuta tras `git rebase fix/v0.4` y `npm ci`; `spec.md`, `plan.md` y `tasks.md` están commiteados antes de lanzar agentes (lo hace el orquestador).
- Los tests nuevos van en `tests/api/` (el `include` de `vitest.config.mts` es `tests/**/*.test.ts`); un archivo por tarea para no solapar. Los scripts temporales de medición usan nombres ÚNICOS por tarea en el scratchpad compartido (p. ej. `t007_<tema>_<n>.mjs`).
- `validateArray` se importa de `./Validation`; no se añade ninguna utilidad nueva ni se toca `src/index.ts`.
- El CI de GitHub corre en ubuntu con Node 20; los agentes validan con Node 20.19 además de la versión local.

## Preguntas abiertas

Cada pregunta indica qué tareas bloquea. **Públicas** = cambian la API pública o el comportamiento observable; **técnicas** = decisiones de implementación.

### Públicas

- **Q1 (`Trainer.clipValue`; bloquea T011).** Hoy se guarda y no hace nada (Constitución V: implementar o eliminar). Opciones: (a) **eliminar** la opción y el campo `readonly clipValue` (**propuesto**: `Trainer` solo ve `train()`, no puede cortar el gradiente; ya existe `ClippedOptimizerFactory`, que lo hace de verdad en el optimizador; el cambio es de tipos y lo detecta el compilador); (b) implementarlo como recorte del cambio de pesos por muestra leyendo `getWeights()` antes/después (cuesta 2 copias por muestra, solo existe para redes con `getWeights`, y para Adam recorta el paso, no el gradiente: semántica distinta); (c) mantener el campo y lanzar si se pasa distinto de 0. Se borran/ajustan los tests de `clipValue` de `tests/Trainer.test.ts`. ¿Eliminar?
- **Q2 (`Trainer`: opciones sin efecto y datos vacíos; bloquea T011).** **Propongo lanzar `Error`** (Constitución V) en: `earlyStopping` sin `setValidationData` al llamar a `train`; `weightDecay`/`earlyStopping` con una red sin `getWeights/setWeights/predict`; `computeMetrics` con una red sin `predict`; dataset vacío; `epochs`/`lr`/`lrDecay`/`weightDecay`/`patience` inválidos. Se queda como `console.warn` el caso `computeMetrics` con objetivos que no parecen de clasificación (es una heurística sobre los datos). Alternativa: dejar todo como aviso (más compatible, pero sigue siendo "ignorar"). Cambian 2 tests existentes (`'earlyStopping configured but no validation data does not break'`, `'empty dataset throws'`). ¿Lanzar?
- **Q3 (mejores pesos; bloquea T011).** Opción aditiva `restoreBestWeights`. **Propongo defecto `false`** (comportamiento actual, como Keras; el historial y los pesos de quien no la use no cambian) y `true` solo si se pide. Requiere `earlyStopping` + datos de validación (si no, lanza por Q2). Alternativa: defecto `true` (más seguro para el usuario, pero cambia los pesos finales de quien ya usa `earlyStopping`). ¿`false`?
- **Q4 (`same` con kernel par; bloquea T008, T009).** Opciones: (a) relleno **asimétrico** estilo TensorFlow (`total = k−1`, `left = floor(total/2)`, el resto a la derecha/abajo): cumple el contrato "salida = `ceil(L/stride)`" ya anunciado por `getOutputLength` y la documentación de `Conv2D`; el kernel impar no cambia (**propuesto**); (b) lanzar `Error` para `same` con kernel par (más simple, pierde la función). Cambio observable: la forma de salida pasa de `L−1`/`H+1` a `L`/`H`. ¿(a)?
- **Q5 (`Network` varias salidas; bloquea T007).** (a) **entrenar todas** con `target: number | number[]` (**propuesto**: la clase ya las construye y `predict` las devuelve; hace que `Network` cumpla `TrainableNetwork`; una salida es idéntica a hoy); (b) lanzar si `nOutputs > 1` (más simple, pero deja un constructor con un parámetro engañoso). Detalle: el valor de retorno pasa a ser el error cuadrático MEDIO sobre las salidas (igual con una). ¿(a)?
- **Q6 (N7 y `LinearRegression`; bloquea T012).** Validar contra `weights.length` en vez del campo privado en `LogisticRegression.predict`, `SoftmaxRegression.predictProba` y también `LinearRegression.predict`/`getCoefficients` (**propuesto**: mismo defecto, no listado en el HANDOFF). Alternativa: hacer privados `weights`/`bias` (cambio de API mayor; los oráculos de 002 los leen y escriben). ¿Conforme con ampliar a `LinearRegression`?
- **Q7 (entrenamiento incremental; bloquea T014).** **Propongo que `SoftmaxRegression.train` y `LinearRegression.fitGD` continúen desde los pesos existentes si la forma coincide** (como `LogisticRegression.train` ya hace) y que para reiniciar se cree una instancia nueva (no se añade `reset()`). Efectos: (1) quien llamaba dos veces a `train`/`fitGD` esperando un reinicio ahora acumula; (2) con pesos puestos a mano (N7) `train` ya no los pisa; (3) los oráculos de 002 de `LinearRegression` y `Softmax` se simplifican (ya no necesitan el stub de `Math.random`); (4) `Softmax` con modelo existente y etiqueta ≥ K lanza. Alternativa: documentar el reinicio y no cambiar nada (N7 queda funcional, pero "pesos a mano + `train`" los destruye). También se añade validación de filas de `X` y etiquetas (Constitución V). ¿Continuar?
- **Q8 (escala de `NetworkN.train`/`Network.train`, N10; bloquea T016).** Hoy entrenan `½Σ(t−p)²` y devuelven `Σ(t−p)²/n`: el paso es `n/2` veces el gradiente de la pérdida devuelta (con `n = 1`, la mitad). Opción A: delta `×2/n` (`Network`: `×2/nOutputs`) de modo que el paso es el gradiente de lo que devuelve (**propuesto**: misma regla que 003 en `NetworkTransformer`/`TCN` y que 004 en `Autoencoder`; además `Trainer` compara el historial de entrenamiento con una pérdida de validación que también es MSE media, así que quedan en la misma escala); coste: **cambia el `lr` efectivo del componente más usado** (×2 con una salida; ×`2/n` en general), con riesgo para tests y apps (`Integration`, `Trainer`, `NetworkN` usan `lr` elegidos con la escala antigua). Opción B: devolver `½Σ`; mantiene la dinámica, pero el historial de `Trainer` deja de coincidir con su pérdida de validación (`MSE` medio), peor. Opción C: solo documentar el factor. La tarea mide primero cuántos tests existentes se mueven. ¿A, B o C?
- **Q9 (dropout y `inputGradient`; bloquea T015).** Hoy pérdida, gradiente y actualización usan máscaras distintas. **Propongo** el parámetro opcional `reuseMasks` (por defecto `false`: nada cambia para quien no lo use) en `inputGradient`/`trainWithDeltas`, y que `Autoencoder`, `VAE` y `GAN` hagan su forward de entrenamiento con dropout activo (`predict(..., true)`) y reutilicen la máscara. Efecto: con `dropoutRate > 0` en el decoder/discriminador las trayectorias cambian (VAE y GAN no aplicaban dropout en el forward). Alternativas: (b) que `inputGradient` sea siempre determinista (sin dropout, el gradiente de `predict`) y que VAE/GAN dejen de aplicar dropout en el paso; (c) solo documentar. La decisión de fondo es si el dropout de la red es parte del modelo entrenado en VAE/GAN. ¿Propuesta (reutilizar máscara)?
- **Q10 (recurrentes lanzan; bloquea T013).** Convertir el `return` silencioso en `Error` en `LSTMLayer.backprop` (pedido), y por el mismo patrón en `GRULayer.backprop` y `NetworkLSTM.train` (**propuesto**). `NetworkLSTM.train([], lr)` tras ningún `predict` sigue siendo un no-op sin error. ¿Conforme con ampliar a GRU y NetworkLSTM?

### Técnicas

- **Q11 (formato de `ModelSaver`; bloquea T006).** Propongo `{ format: 1, length, structure?, weights, timestamp }`: `structure` solo si el modelo expone una propiedad `structure` de números (`NetworkN` la tiene; `Network` no, así que su cobertura es solo por longitud); NO se guarda `constructor.name` (los empaquetadores minimizan los nombres de clase). Los JSON sin metadatos se aceptan si la longitud coincide. ¿Conforme?
- **Q12 (alias en `LSTMLayer`/`GRULayer.setWeights`; bloquea T004).** Se considera parte de I6 (la misma causa: referencias compartidas) y se corrige con copia profunda + validación de forma. ¿Conforme?
- **Q13 (alcance de N5; resuelto en este documento).** Entran en 005 solo los oráculos de `dInput` de `Conv1D`/`Conv2D`, porque I4 cambia el desplazamiento del recorte del relleno y hay que verificarlo; `Dropout`, `Flatten` y `MaxPool2D` van a 006 (no los toca ninguna tarea de 005). Se pregunta solo si se prefiere meter los tres en 005.
- **Q14 (sin cambio, solo aviso).** `Value.backward()` acumula `grad` en llamadas repetidas y no pone a cero los nodos intermedios; no se toca en 005 (documentar en 007). `Trainer` aplica el decaimiento de pesos también a los sesgos; se documenta en la cabecera de `Trainer`.

## Criterio de escalado a Opus (no se lanza nada)

Esta feature es de Haiku (mecánica) y Sonnet (diseño). El único candidato razonable es T015 (dropout y máscaras compartidas en `NetworkN` + `Autoencoder` + `VAE` + `GAN`), porque exige reordenar el forward de entrenamiento de tres modelos sin cambiar las trayectorias con `dropoutRate = 0`. Se pide permiso ANTES, tras dos intentos fallidos de Sonnet con informe de cosenos por bloque. El resto no es motivo de Opus.
