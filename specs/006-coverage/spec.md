# Feature Specification: coverage

**Feature Branch**: `fix/v0.4` (se trabaja directamente en ella; no se ejecutó `create-new-feature.sh` porque crearía una rama)
**Created**: 2026-11-09
**Status**: Draft
**Input**: `specs/ORCHESTRATION.md` sección "006 coverage", `specs/HANDOFF.md` ("Siguiente acción" 2, hallazgo N5, "Pendiente de la revisión T038 (d)", "Pendiente de tratar", "Hallazgo sin tratar", "Pendientes de 005 aplazados"), `specs/000-audit/AUDIT.md` (C3, "DecisionTree da 0.5 de acierto en XOR"), Constitución II, III, IV, V, VI. Condición previa: 001–005 fusionadas en `master` (PR #4–#8); `fix/v0.4` recreada desde `master`.

## Estado verificado hoy (medido en `fix/v0.4`, 1316 tests, 81 archivos)

Medido con `npx vitest run --coverage --coverage.include='src/**'` (la configuración actual NO incluye `src/**`: el informe solo lista los archivos que algún test importa, así que los módulos sin test no aparecen; ver FR-001).

**Módulos de `src/` con 0 % de cobertura (15), sin ningún test directo:** `DataAugmentation`, `DatasetLoader`, `DecisionTree`, `EarlyStopping`, `Flatten`, `HopfieldNetwork`, `KMeans`, `LossPlotter`, `MaxPool2D`, `Metrics`, `NaiveBayes` (clase `GaussianNaiveBayes`), `PCA`, `TSNE`, `Tokenizer`, `WeightInspector`. De los 29 de la auditoría, 14 siguen a 0 % (todos los de arriba salvo `Metrics`, que la auditoría cuenta aparte), y los otros 15 (`Autoencoder`, `ContrastiveLearning`, `Conv2D`, `GAN`, `LinearRegression`, `LogisticRegression`, `Perceptron`, `PositionalEncoding`, `RNN`, `SOM`, `Seq2Seq`, `TCN`, `Tape`, `VAE`, `Word2Vec`) ya tienen archivo en `tests/oracle/` desde 002 (ver siguiente punto).

**Módulos con solo oráculo de gradiente o cobertura parcial (líneas / ramas):** `losses` 10 % / 100 %, `PositionalEncoding` 34 % / 30 %, `Autoencoder` 76 % / 57 %, `Word2Vec` 77 % / 69 %, `ContrastiveLearning` 88 % / 63 %, `Perceptron` 87 % / 79 %, `LinearRegression` 93 % / 71 %, `LogisticRegression` 96 % / 84 % (incluye `SoftmaxRegression`), `SOM` 94 % / 85 %, `RNN` 99 % / 70 %, `TCN` 97 % / 73 %, `Seq2Seq` 99 % / 77 %, `GAN` 100 % / 75 %, `Conv2D` 96 % / 87 %, `LRScheduler` 100 % / 80 %, `Validation` 86 % / 88 %, `VAE` 98 % / 92 %. Estos módulos tienen oráculo y "aprende" (002–004) pero no tests de comportamiento con valores a mano. Cobertura global actual: 95.3 % líneas, 87 % ramas (sin contar los 15 módulos de 0 %, que no entran en el denominador).

**Hallazgos de N5 y del T038(d):** los oráculos de `dInput` de `Conv1D` y `Conv2D` YA existen (se añadieron en 005 por la Q13: `tests/oracle/Conv1D.oracle.test.ts:121`, `tests/oracle/Conv2D.oracle.test.ts:143`, y las pruebas `same`/par de `tests/api/Conv*.padding.test.ts`), así que N5 queda CERRADO para las dos convoluciones. Siguen sin oráculo de `dx`: `Dropout.backward` (`src/Dropout.ts`), `Flatten.backward` (`src/Flatten.ts:45`) y `MaxPool2D.backward` (`src/MaxPool2D.ts:99`).

**Sondeo de entradas inválidas (script en el scratchpad, `t006_probe`, ejecutado hoy contra `src/index.ts`).** Lo que hoy NO lanza o lanza un error poco claro:

- `Metrics`: `accuracy([0,1,1],[0,1])` da 0.667; `mae`/`rmse` con longitudes distintas o vacías dan `NaN`; `confusionMatrix` con una etiqueta negativa lanza `TypeError: Cannot read properties of undefined`; con 200 000 etiquetas lanza `RangeError: Maximum call stack size exceeded` (`Math.max(...yTrue, ...yPred)`, `src/Metrics.ts:55`); `auc` con una sola clase da 0 sin aviso; `r2Score` con `yTrue` constante devuelve 1 aunque la predicción sea mala (`src/Metrics.ts:182`, "trivial case").
- `EarlyStopping`: acepta `patience: -1` y `update(NaN, …)` (devuelve `false` siempre); `src/EarlyStopping.ts` no tiene ningún `throw`.
- `LRScheduler`: `cosineAnnealing(…, maxEpochs = 0)` da `NaN`; `stepDecay(…, epochsDrop = 0)` da un valor sin sentido (`src/LRScheduler.ts` no tiene `throw`).
- `KMeans.fit` con filas de distinto ancho, `PCA.fit` con filas irregulares, `GaussianNaiveBayes.fit` con filas irregulares, `KMeans.predict` con un vector de otra dimensión y `DecisionTree.predict` con un vector vacío: no lanzan. `DecisionTree` acepta `NaN` en `X` y `maxDepth: 1.5`.
- `SOM.train(X, 0)` no hace nada; `Word2Vec.train(…, lr = NaN)` devuelve pérdida `NaN`; `RNN.forward` con filas de ancho incorrecto devuelve `NaN`; `RNN.backward` y `TCN.train` con longitudes distintas lanzan `TypeError` sin mensaje útil.
- `PositionalEncoding.encode(-1, 4)` y `encode(1, 3)` (`dModel` impar) devuelven números sin error.
- `DataLoader` acepta `batchSize: 0`; `ContrastiveLearning.trainStep([], lr)` devuelve `NaN`; `DataAugmentation.shuffle` con `X` e `y` de distinta longitud devuelve `y: [1, NaN]`; `DataAugmentation.standardize([])` devuelve vacío sin error; `MaxPool2D.forward` con entrada menor que `poolSize` devuelve `[]`; `Flatten.forward([])` lanza `TypeError`; `Tokenizer.decode` con un id inexistente devuelve `"<UNK>"` (probablemente por diseño).
- `LinearRegression.fitNormal` no valida `X` ni `y`: filas irregulares no lanzan y un `y` con `NaN` da pesos `NaN` (`src/LinearRegression.ts`); solo lanza con `X` vacío y con `XᵀX` singular.
- `WeightMatrix.getWeights()` y `EmbeddingMatrix.getWeights()` con una fila de 200 000 elementos lanzan `RangeError: Maximum call stack size exceeded`, porque hacen `w.push(...row)` (`src/MatMul.ts:106,180`). El mismo patrón está en `AttentionHead.ts:190-192`, `MultiHeadAttention.ts:123-127`, `LSTMLayer.ts:343-350`, `GRU.ts:256-261`, `TransformerBlock.ts:213-219`, `NetworkTransformer.ts:196-200`, `NetworkTransformerRL.ts:180-183` y `NetworkLSTM.ts:251`; en los dos transformers el argumento es `block.getWeights()` entero, de modo que el límite (~120 000 argumentos) se alcanza con un modelo moderado. Es el hueco aplazado de 005 y es más ancho de lo que dice el HANDOFF.
- `DecisionTree` con XOR: confirmado `predictBatch([[0,0],[0,1],[1,0],[1,1]])` → `[0,0,0,0]` (acierto 0.5). Causa en el código: `_bestSplit` exige `weightedImpurity < parentImpurity` (`src/DecisionTree.ts`); en XOR cualquier umbral deja ambos hijos con gini 0.5, igual que el padre, así que no hay división "estrictamente mejor" y el nodo raíz queda como hoja.
- La afirmación C3 de la auditoría ("29 de 56 módulos no aparecen en ningún test") ya no es cierta para 15 de ellos (tienen oráculo); sigue siéndolo para los 15 de 0 % de arriba.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cada módulo tiene un archivo de test con valores calculados a mano (Priority: P1)

Los 15 módulos de 0 % tienen `tests/<Módulo>.test.ts` que fija su comportamiento con valores esperados deducidos a mano (en un comentario junto al `expect`), no copiados de la salida del código.

**Why this priority**: es el objetivo de la feature (AUDIT C3) y condición de la Constitución IV (un test debe fallar si se rompe el código).

**Independent Test**: `npx vitest run tests/<Módulo>.test.ts` por módulo; `npx vitest run --coverage` muestra cada uno de los 15 con líneas ≥ 90 %.

**Acceptance Scenarios**:
1. **Given** un módulo de la lista, **When** se ejecuta su archivo de test, **Then** pasa, no usa `Math.random` sin semilla (`seedRandom` de `tests/helpers/seed.ts` cuando el módulo es estocástico) y cada `expect` numérico lleva su derivación en un comentario o usa una constante calculable de cabeza (p. ej. `accuracy` de 3 de 4 aciertos = 0.75).
2. **Given** el archivo de test, **When** se aplica un mutante (cambiar un operador, un índice o quitar una validación del módulo), **Then** el archivo FALLA; el informe del agente dice qué mutante usó. Un test que no detecta ningún mutante se rechaza.
3. **Given** los tests de aprendizaje de un módulo (`TSNE`, `KMeans`, `Word2Vec`, …), **Then** el margen entre el resultado y el umbral es ≥ 30 % y se valida con otra semilla y con `lr` ×0.7 y ×1.3 (lección CI).

### User Story 2 - Los módulos con solo oráculo ganan tests de comportamiento (Priority: P1)

`losses`, `PositionalEncoding`, `Autoencoder`, `Word2Vec`, `ContrastiveLearning`, `Perceptron`, `LRScheduler`, `SOM`, `LinearRegression`, `LogisticRegression`/`SoftmaxRegression`, `RNN`, `TCN`, `Seq2Seq`, `GAN`, `VAE` y `Conv2D` tienen `tests/<Módulo>.test.ts` con valores a mano para la parte determinista (formas, fórmulas cerradas, errores) y cubren las ramas de la lista de arriba.

**Independent Test**: `npx vitest run tests/<Módulo>.test.ts` y ramas del módulo ≥ 85 % en el informe de cobertura.

**Acceptance Scenarios**: como US1 (1) y (2). No se duplican los oráculos ni los "aprende" de 002–004; se citan.

### User Story 3 - `DecisionTree` con XOR está decidido y documentado (Priority: P1)

Hay un test que fija el comportamiento de `DecisionTree` en XOR y una decisión escrita (Q1) sobre si es una limitación documentada o un defecto.

**Acceptance Scenarios**:
1. **Given** XOR de cuatro puntos, **When** se ajusta con el árbol, **Then** el test afirma la decisión de Q1: (a) limitación documentada → acierto exactamente 0.5, con una cabecera en `src/DecisionTree.ts` que lo explica y un caso de control donde el mismo árbol SÍ separa (AND, o XOR con una característica extra `x1·x2`); (b) defecto corregido → acierto 1.0 con `maxDepth ≥ 2`, y los tests de árboles ya existentes por golden.
2. **Given** la opción elegida, **Then** el mutante opuesto (cambiar `<` por `<=` en `_bestSplit`) hace fallar el test.

### User Story 4 - `Dropout`, `Flatten` y `MaxPool2D` tienen oráculo de `dx` (Priority: P1)

**Acceptance Scenarios**:
1. **Given** una entrada fija, **When** se compara `backward(dOut)` con la derivada por diferencias finitas de `L = Σ c_i·forward(x)_i` respecto de cada componente de `x`, **Then** `|a − b| < 1e-6` por componente (no solo coseno) para `Flatten` (permutación lineal), `Dropout` con máscara fija (`seedRandom`, después `reuseMask` en la pasada numérica) y `MaxPool2D` en puntos sin empates (la derivada con empates no está definida; se prueba aparte que se enruta al primer máximo o a todos según el código, leído de `src/MaxPool2D.ts`).
2. **Given** `MaxPool2D` con una ventana que se solapa (`stride < poolSize`), **Then** los gradientes se SUMAN en la posición compartida (valor calculado a mano).
3. Mutantes: `Dropout.backward` sin la máscara, `Flatten.backward` con otro orden de índices, `MaxPool2D.backward` enrutando a un índice distinto del máximo: los tres deben hacer fallar su archivo.

### User Story 5 - Las entradas inválidas fallan con un error claro en el resto de modelos (Priority: P2, Constitución V)

Cada caso sondeado de la lista de arriba lanza un `Error` con el nombre `Clase.método` y el motivo, o queda documentado (Q3) como comportamiento aceptado. Hay un helper compartido de validación de datos (FR-009) que sustituye la validación duplicada de `LogisticRegression`/`LinearRegression`.

**Acceptance Scenarios**:
1. **Given** cada caso de la lista (longitudes distintas, filas irregulares, `NaN`/`Infinity`, hiperparámetros ≤ 0 o no enteros, dimensión de predicción distinta de la de entrenamiento), **Then** lanza antes de modificar el modelo (se compara el estado con `getWeights()`/campos antes y después).
2. **Given** entradas válidas, **Then** el resultado es idéntico al de hoy (golden con `toEqual` capturado antes de tocar `src/`).

### User Story 6 - `getWeights` no desborda con matrices grandes (Priority: P2)

`getWeights()` de `WeightMatrix`, `EmbeddingMatrix`, `AttentionHead`, `MultiHeadAttention`, `TransformerBlock`, `LSTMLayer`, `GRULayer`, `NetworkTransformer`, `NetworkTransformerRL` y `NetworkLSTM` devuelve el mismo array que hoy con tamaños pequeños y no lanza `RangeError` con ≥ 200 000 elementos.

**Acceptance Scenarios**: `new WeightMatrix(1, 200000).getWeights().length === 200000`; igualdad exacta (`toEqual`) con una réplica del bucle antiguo para tamaños pequeños; `set(get())` es identidad. Mutante: volver a `push(...row)` hace fallar el test.

### Edge Cases

- **Mutantes en `src/` de módulos sin test:** el agente aplica el mutante en su worktree, comprueba que el test falla y restaura con `git checkout -- src`. Si hay varios agentes en paralelo, cada uno trabaja en su worktree (regla de ORCHESTRATION); el mutante nunca se aplica en el directorio principal.
- **`Math.random` en tests:** `Tokenizer`, `DatasetLoader`, `HopfieldNetwork`, `Metrics`, `EarlyStopping`, `LossPlotter` son deterministas (sin semilla); `KMeans` (inicialización), `TSNE`, `SOM`, `Word2Vec`, `Autoencoder`, `GAN`, `VAE`, `ContrastiveLearning`, `DataAugmentation.addNoise/jitter/shuffle` usan `Math.random`: semilla de `tests/helpers/seed.ts`.
- **Salida por consola:** `LossPlotter.print`, `WeightInspector.print`, `printConfusionMatrix` y `classificationReport` escriben con `console.log`; el test espía `console.log` (`vi.spyOn`) y comprueba el contenido; ningún test ensucia la salida.
- **`HopfieldNetwork.store` avisa por `console.warn`** (`src/HopfieldNetwork.ts:85`, "surfaced for awareness"); el test lo espía.
- **Tolerancias:** 1e-9 para fórmulas cerradas en doble precisión; 1e-6 para diferencias finitas; los tests de "aprende" con margen ≥ 30 %.
- **`DecisionTree` y empates en gini:** el recorrido de umbrales es determinista (valores ordenados, punto medio), por lo que los árboles esperados se pueden deducir a mano.
- **`TSNE`** es estocástico (inicialización) y sin solución cerrada: se prueban propiedades (forma, KL finita y decreciente en las iteraciones, determinismo con semilla, separación de dos clústeres lejanos con margen ≥ 30 %), no coordenadas.
- **El umbral de cobertura** se fija al final (T045) con el valor medido menos un margen, no antes: un umbral inventado rompería el CI.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `vitest.config.ts` MUST declarar `coverage: { provider: 'v8', include: ['src/**'], exclude: ['src/index.ts'] }` para que el informe cuente los módulos sin tests (hoy no aparecen). Después de las tareas de test se añade un umbral (T045) igual al valor medido menos un margen de 2 puntos.
- **FR-002**: Cada uno de los 15 módulos de 0 % MUST tener un archivo `tests/<Módulo>.test.ts` (`NaiveBayes` → `tests/NaiveBayes.test.ts`) con líneas ≥ 90 % y ramas ≥ 80 % del módulo.
- **FR-003**: Cada módulo de la lista de cobertura parcial MUST llegar a ramas ≥ 85 % con un archivo `tests/<Módulo>.test.ts` (o ampliando el existente de su nombre, `LRScheduler`), sin duplicar los oráculos.
- **FR-004**: Los valores esperados MUST derivarse a mano y aparecer en un comentario junto al `expect`; no se copia la salida del código. Cada archivo MUST detectar al menos tres mutantes distintos, listados en el informe del agente (operador, índice o validación quitada).
- **FR-005** (US3): `DecisionTree` con XOR MUST quedar decidido según Q1 y fijado por un test; la cabecera de `src/DecisionTree.ts` MUST describirlo.
- **FR-006** (US4, N5): `Dropout`, `Flatten` y `MaxPool2D` MUST tener un oráculo de `dx` por componente (`tests/oracle/<Módulo>.dx.oracle.test.ts`). `tests/oracle/coverage.test.ts` NO exige hoy oráculos para estas capas sin parámetros: no se toca (Q4).
- **FR-007** (US5): los casos sondeados en "Estado verificado" MUST lanzar `Error` con `Clase.método` y el motivo antes de modificar el modelo, salvo los marcados como decisión en Q2/Q3.
- **FR-008** (US6): los `getWeights` que usan `push(...fila)` o `push(...array)` con arrays que escalan con el modelo MUST copiar con un bucle (o `concat`), con el mismo orden de elementos; `Network.ts` y `NetworkN.ts` (`push(...n.weights, n.bias)`, un vector por neurona) quedan fuera (Q5).
- **FR-009** (helper): `src/Validation.ts` MUST ofrecer un helper de datos reutilizable (`validateDataset`, nombre por confirmar en el plan) que comprueba `X` no vacía, filas del mismo ancho, finitud y, si se pasa, `y` de la misma longitud y finita; y lo MUST usar `LogisticRegression`, `LinearRegression` (incluido `fitNormal`) y `SoftmaxRegression` en lugar de su validación propia, con los mismos mensajes que hoy (los tests de 005 los fijan).
- **FR-010** (Constitución IV): la suite MUST pasar 20 ejecuciones seguidas, 5 con `--sequence.shuffle`, y bajo Node 20.19, y el CI de GitHub es el árbitro.
- **FR-011**: un meta-test (T044) MUST fallar si un módulo de `src/` (excepto `index.ts` y los de la lista de excepciones con motivo) no tiene `tests/<Módulo>*.test.ts` o una entrada en `tests/oracle/`.
- **FR-012**: ninguna tarea toca `package.json`, `package-lock.json`, `src/index.ts`, el changelog ni `tasks.md`. Una tarea que cambie `src/` lo hace solo en los archivos declarados.

### Key Entities

- **Módulo sin cobertura:** archivo de `src/` con 0 % en `--coverage.include='src/**'`.
- **Módulo con cobertura parcial:** archivo con ramas < 85 % o sin test de comportamiento.
- **Caso inválido sondeado:** una línea de la lista "Estado verificado" que hoy no lanza.

## Success Criteria *(mandatory)*

- **SC-001**: `npx vitest run --coverage` (con FR-001) muestra los 15 módulos de 0 % con líneas ≥ 90 % y la cobertura global de líneas ≥ 95 % y de ramas ≥ 90 % (hoy 95.3 % / 87 %, sin los 15).
- **SC-002**: `grep -c` de `it.fails` y `it.todo` en `tests/` sigue en 0.
- **SC-003**: 20 ejecuciones seguidas verdes; 5 con shuffle; Node 20.19 verde; CI de GitHub verde.
- **SC-004**: cada archivo de test nuevo detecta al menos 3 mutantes (informe de cada agente); el revisor Sonnet repite los mutantes de una muestra (Metrics, DecisionTree, PCA, TSNE).
- **SC-005**: `typecheck`, `build` y suite completa pasan; `git diff --quiet -- package.json package-lock.json src/index.ts`.

## Preguntas abiertas (decisiones del usuario o del orquestador)

### Públicas

- **Q1 (`DecisionTree` y XOR; bloquea T003).** Hoy el árbol no divide si ninguna división mejora estrictamente el gini del padre (`weightedImpurity < parentImpurity`), y en XOR ninguna lo hace (acierto 0.5, medido). Opciones: (a) **limitación documentada** (**propuesto**): es el comportamiento de un árbol voraz con "ganancia > 0"; se fija con un test y una cabecera en `src/DecisionTree.ts`, sin cambiar números; (b) **permitir divisiones de ganancia cero** cuando el nodo es impuro (`<=`): XOR se resuelve con profundidad 2 (scikit-learn lo hace así, no exige ganancia estricta), pero cambia árboles existentes de quien use la clase (más hojas, posible sobreajuste) y exige un test de que los datasets ya cubiertos no cambian. La propuesta (a) es la menos invasiva; (b) es más correcta frente a la referencia. ¿Cuál?
- **Q2 (`Metrics`: qué hacer con entradas degeneradas; bloquea T036).** Propongo LANZAR en longitudes distintas, vectores vacíos, etiquetas negativas o no enteras y `auc` con una sola clase (hoy devuelven `NaN`/0.667/0 sin avisar), y calcular `max`/`min` con bucles (sin `Math.max(...)`). Para `r2Score` con `yTrue` constante, hoy devuelve 1 aunque `ssRes > 0` (comentario "trivial case"): propongo devolver 1 solo si `ssRes === 0` y 0 si no (convención de scikit-learn), o lanzar. ¿Conforme? Cambio observable: ver Q3.
- **Q3 (cambios incompatibles de 006 para el changelog).** Todo lo que pasa de "devuelve un valor" a "lanza" (Q2 y FR-007) es incompatible en 0.x (Constitución VII). Se anota para 007. `Tokenizer.decode` con un id desconocido → `"<UNK>"` se propone MANTENER (probablemente por diseño, como un vocabulario con token desconocido); `LossPlotter`/`WeightInspector` no validan nada y no se tocan. ¿Conforme?
- **Q4 (`tests/oracle/coverage.test.ts`).** El meta-test de 002 solo cubre modelos con parámetros. Propongo NO ampliarlo a capas sin parámetros (`Dropout`, `Flatten`, `MaxPool2D`) y que sus oráculos de `dx` vivan en `tests/oracle/` por convención. Alternativa: ampliar `COVERED`. Es una decisión de test, no de API.
- **Q5 (alcance de la corrección de `push(...)`).** Entran los `getWeights` cuyo argumento escala con una matriz o con un bloque entero (lista en FR-008). Quedan fuera `Network`/`NetworkN`/`Conv1D`/`Conv2D`/`WeightInspector` (`push(...n.weights, n.bias)`, `push(...this.biases)`): un solo vector de neurona o de canales, que no llega a 100 000. Si se prefiere un barrido total, es una tarea mecánica más. ¿Barrido total?

### Técnicas

- **Q6 (nombre y firma del helper de datos; bloquea T033).** Propuesta: `validateDataset(X, y, name, { labels?: 'integer' | 'unit' | 'finite', nFeatures? })` en `src/Validation.ts`, que devuelve `nFeatures`; mensajes idénticos a los actuales de `LogisticRegression`/`LinearRegression`/`SoftmaxRegression` (los tests de 005 los fijan). No se exporta desde `src/index.ts` (exclusivo del orquestador; `validateArray` ya se exporta).
- **Q7 (decisiones aplazadas a 007).** `Trainer.setValidationData` sin `earlyStopping` se ignora en silencio, `NetworkN.getWeights()`/`setWeights()` borran las máscaras de dropout guardadas, y la mejora opcional de `validateArray` (incluir las dos longitudes en el error de NaN/undefined): se documentan en 007 (`AGENTS.md`/changelog); no tocan datos ni algoritmos. Motivo: son decisiones de contrato ya tomadas por el orquestador tras T018 ("aplazados a 007/006") y su único cambio pendiente es de texto. El comentario "estilo TensorFlow" de `same` ya está corregido (`src/Conv1D.ts:233`). Quedan para 008 `NetworkLSTM` con `[]` en `neuron_app` y los `lr`.
- **Q8 (`ContrastiveLearning` frágil, N9).** Los tests de 004 usan semilla fija 9; 006 NO cambia el algoritmo (opción abierta documentada). Su archivo de tests nuevo (T022) cubre solo lo determinista (estáticos, `cosineSimilarity`, formas, errores) y `computeLoss` con valores a mano; no añade tests de "aprende".

## Decisiones del usuario (2026-11-09)

- **Q1:** (b) permitir divisiones de ganancia cero; XOR debe dar 1.0 con `maxDepth ≥ 2`. Cambio observable para el changelog (007).
- **Q2:** lanzar ante entradas degeneradas de `Metrics`; `r2Score` con `yTrue` constante devuelve 1 solo si `ssRes === 0`, 0 si no. Incompatible: changelog (007).
- **Q5:** barrido total de `push(...)` (T048). FR-008 ya no excluye `Network`, `NetworkN`, `Conv1D`, `Conv2D` ni `WeightInspector`.
- Q3, Q4, Q6, Q7, Q8: se aplican las propuestas del agente (mantener `Tokenizer.decode` → `<UNK>`, no ampliar `coverage.test.ts`, helper `validateDataset`, aplazados a 007, `ContrastiveLearning` sin cambio de algoritmo).
