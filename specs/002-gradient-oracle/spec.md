# Feature Specification: gradient-oracle

**Feature Branch**: `fix/v0.4` (se trabaja directamente en ella; no se crea rama por feature. No se ejecutó `create-new-feature.sh` porque crearía una rama)
**Created**: 2026-11-08
**Status**: Draft
**Input**: `specs/ORCHESTRATION.md` sección "002 gradient-oracle", hallazgos C1, C2, C3 e I1 de `specs/000-audit/AUDIT.md`, Constitución II, III y IV

## Estado verificado hoy (commit `fba4576`)

- `tests/GradientCheck.test.ts` es el único sitio con `signedPerturbationCheck` y `multiStepLossCheck` (definidos en el propio archivo, sin otros usos en `src/` ni `tests/`).
  - `signedPerturbationCheck` solo perturba un peso con `setWeight` y recalcula la pérdida: nunca llama a `train`/`backward`, así que no prueba el código de entrenamiento. Aprueba con `improved/checked > 0.5` aunque el modelo entrene al revés.
  - `multiStepLossCheck` acepta `finalLoss < initialLoss * 1.5`, es decir, tolera subir un 50 %. Varios tests de ese archivo solo exigen `loss < loss0 * 2`.
- `tests/Integration.test.ts` tiene dos tests llamados "gradients match finite differences" (`Network`, `NetworkN`) que solo comprueban que la predicción se acerca al objetivo; no usan diferencias finitas.
- Ningún archivo de `tests/` llama a `Math.random` ni fija una semilla (grep de `seed`/`mulberry` vacío). Los pesos iniciales vienen de `Math.random` en los constructores de `Neuron`, `NeuronN`, `WeightMatrix`/`EmbeddingMatrix` (`src/MatMul.ts:81,158`), `LSTMLayer`, `GRULayer`, `Conv1D`, `Conv2D`, `RNN`, `TCN`, etc.; la máscara de `Dropout` (`src/Dropout.ts:36`), el muestreo de `GAN`/`VAE`, `DataLoader` y otros también la usan.
- **Causa de la inestabilidad** medida en 12 ejecuciones de `npx vitest run`: 1 fallo (`GradientCheck > NetworkTransformer > multi-step training decreases loss`, `expected false to be true`). La auditoría midió 5 de 25, incluyendo el dropout de `NetworkN`. Es la combinación de pesos aleatorios sin semilla con umbrales que dependen de la inicialización: el `NetworkTransformer` entrena al revés (C1), así que pasa o falla según el azar.
- Optimizadores: `SGD` hace `w + lr·g`. `WeightMatrix` (`MatMul.ts:74`) y `BiasVector` (`MatMul.ts:121`) fijan un `Adam` por escalar sin opción de cambiarlo, y `Adam.step` normaliza (`lr·m̂/(√v̂+ε)`): en el primer paso la actualización es `≈ lr·sign(g)`, no `lr·g`. `LayerNorm`, `BatchNorm` y `EmbeddingMatrix` usan SGD directo.
- Acceso a pesos:
  - Con `getWeights`/`setWeights` planos: `Network`, `NetworkN`, `AttentionHead`, `MultiHeadAttention`, `TransformerBlock`, `NetworkTransformer`, `NetworkTransformerRL`, `Conv1D`, `Conv2D`, `BatchNorm`, `LayerNorm`, `WeightMatrix`, `BiasVector`, `EmbeddingMatrix`.
  - Con `getWeightsFlat`/`setWeightsFlat`: `LSTMLayer`, `GRULayer`, `NetworkLSTM`.
  - Solo campos públicos: `RNN` (`Wxh`, `Whh`, `Why`, `bh`, `by`), `CausalConv1D` (`kernels`, `biases`), `Seq2Seq` (`W_out`, `b_out`, `encoder`, `decoder`), `Neuron`, `NeuronN`, `Perceptron`, `LogisticRegression`, `SoftmaxRegression`, `GAN`/`VAE`/`Autoencoder`/`ContrastiveLearning` (a través de sus `NetworkN` públicos).
  - `TCN` guarda la proyección final en campos `private` (`_outputW`, `_outputB`).
- `ContrastiveLearning.trainStep` calcula su gradiente por diferencias finitas internas (`src/ContrastiveLearning.ts`, comentario del paso de entrenamiento), no por backprop.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Un oráculo que detecta una dirección de entrenamiento equivocada (Priority: P1)

Una utilidad compara la **actualización real** de pesos tras un paso de entrenamiento (`Δw = w_después − w_antes`, obtenida llamando a `train`/`backward`/`trainStep` del modelo) con `−∇L` calculado por diferencias finitas centrales, y exige coseno mayor que 0.999 (Constitución II). Un modelo que entrena al revés da coseno cercano a `−1`.

**Why this priority**: sin ella, las features 003 y 004 no tienen forma objetiva de saber si arreglaron algo.

**Independent Test**: sobre un modelo de juguete con el gradiente correcto, el oráculo da coseno `≈ 1`; sobre el mismo modelo con el signo invertido a propósito, da `≈ −1` y el test del oráculo lo reporta.

**Acceptance Scenarios**:

1. **Given** `Network`, `NetworkN`, `LSTMLayer`, `GRULayer`, `Conv1D`, `Conv2D`, `AttentionHead` y `Autoencoder`, **When** se ejecuta su prueba de oráculo, **Then** pasa con coseno > 0.999 (la auditoría midió 1.0000 en ellos).
2. **Given** `NetworkTransformer`, `NetworkTransformerRL`, `RNN`, `TCN`, `Seq2Seq` (C1), `GAN`, `VAE` (C2) y `NetworkN` con `residual` (I1), **When** se ejecuta su prueba, **Then** está marcada `it.fails` y CI sigue en verde; al corregir el modelo, el `it.fails` empieza a fallar y avisa de que hay que quitarlo.
3. **Given** un `it.fails` cuyo fallo es una excepción ajena (import roto, `TypeError`), **When** corre, **Then** pasaría en falso. Se evita con un test "humo" no marcado en el mismo archivo (el modelo se construye, hace forward y entrena con resultados finitos) y con errores tipados del oráculo (ver Q8).

---

### User Story 2 - "Entrenar significa aprender" (Priority: P1)

Una utilidad `expectLearns` entrena con semilla fija en una tarea pequeña y exige que una pérdida, medida **fuera** del `train` sobre datos fijos, baje a una fracción del valor inicial (Constitución III).

**Acceptance Scenarios**:

1. **Given** cada modelo entrenable, **When** se entrena en su tarea mínima, **Then** `finalLoss < ratio · initialLoss` y la pérdida es finita. Los modelos con C1/C2 llevan esta prueba en `it.fails` hasta 003/004.
2. **Given** un modelo que no entrena, que diverge o da `NaN`, **When** corre `expectLearns`, **Then** falla.

---

### User Story 3 - Suite determinista (Priority: P1)

`Math.random` se sustituye por un generador con semilla fija antes de cada test y se restaura después.

**Acceptance Scenarios**:

1. **Given** el repositorio tras la feature, **When** se ejecuta `npx vitest run` 20 veces seguidas, **Then** las 20 terminan con código 0.
2. **Given** dos construcciones de un modelo con la misma semilla, **When** se comparan sus pesos, **Then** son idénticos.

---

### User Story 4 - Cobertura completa de modelos y capas entrenables (Priority: P2)

Un meta-test comprueba que cada clase exportada de `src/index.ts` con método de entrenamiento está en el listado de cobertura de esta feature (con su archivo de oráculo) o en una lista de exclusiones justificada (p. ej. modelos sin gradiente, que se cubren en la feature 006).

### Edge Cases

- Capas ReLU/maxpool con puntos no diferenciables: las entradas del test evitan el cero y empates de máximo.
- Estado interno (trayectoria de `LSTMLayer`, caches de `LayerNorm`, `Dropout`): `loss()` y `step()` reinician el estado y la semilla antes de cada llamada.
- Modelos con muestreo interno (`VAE` reparametrización, `GAN` ruido `z`, `Dropout`): el oráculo acepta un `resetRng` que se invoca antes de cada `loss()` y `step()` para que todas vean el mismo ruido.
- Modelos de actualización por muestra (`LogisticRegression.train` con `epochs`, `Trainer`, `GAN.trainStep` con lote): el oráculo exige que un `step` sea una única actualización; se usa un lote de 1 muestra y 1 época.
- El gradiente interno puede estar escalado (`NetworkLSTM` divide por `T`): el coseno es invariante a la escala; el oráculo devuelve también `ratio = ‖Δw‖ / (lr·‖∇‖)` para informar.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: MUST existir `tests/helpers/seed.ts` con un PRNG determinista (mulberry32) que reemplaza `Math.random` y se restaura.
- **FR-002**: MUST existir `tests/helpers/gradientOracle.ts` que, dado `getParams`, `setParams`, `loss` y `step(lr)`, calcule `cosine`, `ratio` y `n` comparando `Δw` real con `−∇L` por diferencias finitas centrales (ε por defecto `1e-5`), con muestreo determinista si hay más de 400 parámetros. Debe fallar si `‖Δw‖ = 0`, si hay `NaN`, o si menos de 3 parámetros tienen gradiente numérico no nulo.
- **FR-003**: El oráculo MUST funcionar con modelos cuyo optimizador es `Adam` fijo. Propuesta (Q1): durante la prueba, sustituir `Adam.prototype.step` por `w + lr·g` y restaurarlo después.
- **FR-004**: MUST existir `tests/helpers/learns.ts` con `expectLearns`.
- **FR-005**: MUST existir un `setupFiles` global (`tests/setup.ts` + `vitest.config.mts`) que siembre `Math.random` antes de cada test (Q2).
- **FR-006**: Cada modelo/capa entrenable de la tabla siguiente MUST tener `tests/oracle/<Nombre>.oracle.test.ts` con: test humo, oráculo y `expectLearns`. Los marcados "fails" usan `it.fails` y un comentario con el ID de auditoría.
- **FR-007**: `tests/GradientCheck.test.ts`, `signedPerturbationCheck` y `multiStepLossCheck` MUST eliminarse. Sus casos válidos (dropout estable en `NetworkN`) se trasladan a los archivos de oráculo.
- **FR-008**: Un test nuevo MUST fallar al introducir un error a propósito en el código que cubre (Constitución IV). Se verifica en el oráculo con modelos de juguete, y por tarea con una mutación temporal en `src/` que se revierte (el gate comprueba `git diff --quiet -- src`).
- **FR-009**: Ninguna tarea modifica `src/`, `package.json`, `package-lock.json`, `src/index.ts` ni el changelog.
- **FR-010**: La suite MUST pasar 20 ejecuciones seguidas, también con `--sequence.shuffle`.

### Cobertura requerida y resultado esperado hoy (según AUDIT.md)

| Modelo o capa | Archivo de test | Gradiente (oráculo) | Aprende |
|---|---|---|---|
| `Neuron`, `NeuronN`, `Layer` | `tests/oracle/{Neuron,NeuronN,Layer}.oracle.test.ts` | pasa | pasa |
| `Network` | `tests/oracle/Network.oracle.test.ts` | pasa; con varias salidas solo entrena la primera (I5, de 005) | pasa |
| `NetworkN` (sin residual, con dropout 0) | `tests/oracle/NetworkN.oracle.test.ts` | pasa | pasa |
| `NetworkN` con `residual` | mismo archivo | **it.fails** (I1, cos 0.86) | pasa o `it.fails`; confirmar al medir |
| `LSTMLayer`, `GRULayer`, `NetworkLSTM` | `tests/oracle/*.oracle.test.ts` | pasa | pasa |
| `Conv1D`, `Conv2D`, `BatchNorm`, `LayerNorm` | idem | pasa | pasa |
| `AttentionHead`, `MultiHeadAttention`, `TransformerBlock` | idem | pasa | pasa |
| `Autoencoder` | idem | pasa (cos 0.998 a 1.000) | pasa |
| `NetworkTransformer`, `NetworkTransformerRL`, `RNN` | idem | **it.fails** (C1) | **it.fails** |
| `TCN` (con `CausalConv1D`), `Seq2Seq` | idem | **it.fails** (C1; `Seq2Seq` además aproxima el encoder) | **it.fails** |
| `GAN`, `VAE` | idem | **it.fails** (C2) | **it.fails** |
| `ContrastiveLearning` | idem | no auditado; gradiente por diferencias finitas internas | por medir |
| `LogisticRegression`, `SoftmaxRegression`, `LinearRegression` (`fitGD`) | idem | pasa (auditoría: resultados correctos) | pasa |
| `Perceptron` | idem | **excepción** (regla del perceptrón, sin pérdida diferenciable) | pasa |
| `Word2Vec`, `SOM` | idem | por medir / excepción SOM | pasa |
| `Value` (`Tape.ts`) | idem | pasa (autograd contra diferencias finitas) | pasa |
| `WeightMatrix`, `BiasVector`, `EmbeddingMatrix`, `LearnedPositionalEncoding` | idem | pasa | pasa |

Sin gradiente entrenable (se cubren en la feature 006): `KMeans`, `PCA`, `TSNE`, `GaussianNaiveBayes`, `DecisionTree`, `HopfieldNetwork`, `Tokenizer`, `Metrics`, `Dropout`, `Flatten`, `MaxPool2D`, `PositionalEncoding` fijo, `DataLoader`, `DataAugmentation`, `DatasetLoader`, `EarlyStopping`, `LRScheduler`, `LossPlotter`, `WeightInspector`, `Trainer`, `ModelSaver`, `Validation`. Su gradiente hacia atrás se ejercita indirectamente en los modelos que los usan.

### Key Entities

- **Oráculo de gradiente**: función de test que devuelve `{cosine, ratio, n}`; `expectGradientMatches` la envuelve con el umbral de la constitución.
- **Prueba de aprendizaje**: `expectLearns(lossFn, stepFn, opts)`.
- **`it.fails`**: marcador de un defecto conocido; lo retira la feature que lo arregla (003: C1; 004: C2 e I1).

## Success Criteria *(mandatory)*

- **SC-001**: `for i in $(seq 1 20); do npx vitest run || exit 1; done` termina con código 0.
- **SC-002**: Cada fila de la tabla tiene su archivo, y el meta-test de cobertura (T037) pasa.
- **SC-003**: Con una mutación de signo en `NetworkN` (`src/NetworkN.ts`) o `LSTMLayer`, los tests de su archivo fallan; revertida, pasan.
- **SC-004**: Toda prueba `it.fails` lleva un id de auditoría (`// AUDIT <id>`) y una medición que la respalda, y falla por el motivo declarado; cualquier otro modelo con coseno < 0.999 se reporta al orquestador, no se marca `it.fails` en silencio. Ids al cierre de 002: C1 (NetworkTransformer, NetworkTransformerRL, RNN, TCN, Seq2Seq), C2 (GAN, VAE), I1 (NetworkN con residual), N1 (dX de AttentionHead, MultiHeadAttention, TransformerBlock), N6 (Autoencoder), N7 (LogisticRegression, SoftmaxRegression) y Q3 (encoder de Seq2Seq). *(Actualizado el 2026-11-08: la redacción original limitaba los ids a C1, C2 e I1; la ejecución descubrió defectos adicionales, todos medidos y documentados en HANDOFF.md.)*
- **SC-005**: `npx tsc --noEmit -p .`, `npm run typecheck`, `npm run build` y `npx vitest run` en verde.

## Assumptions

- Orden de ejecución de vitest: cada archivo de test corre en su worker; el `setupFiles` se evalúa por archivo, por lo que la semilla no se comparte entre archivos.
- Los `train` de los modelos pueden llamarse con una muestra y un `lr` pequeño y producen una sola actualización equivalente a un paso de gradiente (se verifica por modelo; si no, se informa como ambigüedad).
- `tests/**/*.test.ts` ya incluye `tests/oracle/` y `tests/helpers/*.test.ts` (`vitest.config.mts`).
- La feature 006 crea `tests/<Modulo>.test.ts`; los nombres `tests/oracle/*.oracle.test.ts` no colisionan.

## Preguntas abiertas

- **Q1 (Adam fijo)**: `WeightMatrix`/`BiasVector` fuerzan `Adam`, así que `Δw ≈ lr·sign(g)` en el primer paso y el coseno con `−∇L` no llega a 0.999 aunque el gradiente sea correcto. Opciones: (a) el oráculo sustituye `Adam.prototype.step` por `w + lr·g` durante la prueba (no toca `src/`, pero no prueba `Adam`, que ya cubre `tests/Optimizers.test.ts`); (b) añadir un parámetro de optimizador a `WeightMatrix` (cambio de `src/`, fuera del alcance de 002). **Propongo (a).**
- **Q2 (semilla global)**: ¿sembrar en un `setupFiles` global (afecta a los 27 archivos existentes, que podrían descubrir tests sensibles a la semilla) o solo en los archivos nuevos? **Propongo global** con semilla fija y arreglar por cada archivo existente que falle; es lo que pide la constitución (IV).
- **Q3 (umbral y excepciones)**: coseno > 0.999 es el de la constitución. `Seq2Seq` aproxima el encoder (C1) y `TCN` tiene un coseno de -0.8945 que no es -1 exacto: se dejan en `it.fails` con 0.999 y 003 decidirá si documenta una excepción con umbral propio. `Perceptron` y `SOM` no tienen pérdida diferenciable: solo "aprende". ¿Conforme?
- **Q4 (pérdidas de referencia de `GAN`/`VAE`/`ContrastiveLearning`)**: no hay una pérdida canónica en el código para el oráculo. **Propongo:** `GAN`, pérdida del generador `−log D(G(z))` con `D` congelado y `z` fijo (descrita en `src/GAN.ts`, comentario de `trainStep`); `VAE`, ELBO = MSE de reconstrucción + KL con ε de reparametrización fijo (`resetRng`); `ContrastiveLearning`, `computeLoss(pairs)`. La tarea de cada modelo confirma que coinciden con lo que implementa `trainStep`/`train` y, si no, lo registra como ambigüedad.
- **Q5 (campos privados)**: `TCN` guarda `_outputW`/`_outputB` como `private`; los tests accederían con `(tcn as any)`. Alternativa: añadir `getWeights`/`setWeights` a `TCN` (cambio de `src/`, lo haría 003/005). **Propongo `as any` en tests** y avisar en 003.
- **Q6 (qué hacer con `GradientCheck.test.ts`)**: **Propongo** eliminarlo por completo y trasladar a los archivos de oráculo el único caso útil (dropout estable de `NetworkN`). El test "intentionally wrong perturbation" no prueba nada del código.
- **Q7 (`Integration.test.ts`)**: los dos tests llamados "gradients match finite differences" no lo hacen. **Propongo** renombrarlos (sin cambiar su lógica) para que digan "moves prediction toward target" (Constitución VI).
- **Q8 (falsos verdes de `it.fails`)**: un `it.fails` pasa con cualquier excepción. **Propongo:** test humo no marcado en cada archivo, y que el oráculo y `expectLearns` lancen errores de una clase `OracleError` que los tests de `it.fails` no comprueban pero que el humo cubre.
- **Q9 (una pérdida por modelo)**: la pérdida para diferencias finitas debe coincidir con la que optimiza `train`. Cuando `train` devuelve su pérdida, el test la reutiliza como cruce; si `train` no la devuelve (`NetworkLSTM`, `LSTMLayer`, `AttentionHead`) se define en el test. ¿Se acepta que el factor de escala (`½`, `1/T`, `1/n`) no importe, porque solo se compara la dirección (coseno), y se informe `ratio`?
- **Q10 (`NetworkN` con dropout)**: para el oráculo se usa `dropoutRate: 0`. El caso con dropout solo comprueba que no explota con semilla fija. ¿Basta?
