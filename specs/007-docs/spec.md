# Feature Specification: docs

**Feature Branch**: `fix/v0.4` (se trabaja en ella; no se ejecutó `create-new-feature.sh` porque crearía una rama)
**Created**: 2026-11-11
**Status**: Draft
**Input**: `specs/ORCHESTRATION.md` sección «007 `docs`», `specs/HANDOFF.md` (todas las secciones «Cambios observables…», «Notas para 007…», «Pendientes de 005 aplazados a 007/006»), notas por lote y «Decisiones del usuario (2026-11-11)» de `specs/006-coverage/tasks.md`, Constitución VI y VII. Condición previa: 001–006 fusionadas en `master` (PR #4–#10; `fix/v0.4` = `d0228b7`).

## Estado verificado hoy (leído en `fix/v0.4`, `d0228b7`)

- **`README.md` (751 líneas):** inglés salvo cuatro encabezados y comentarios en español (`Word2Vec`, `t-SNE`, `PositionalEncoding`, `ContrastiveLearning`, líneas 410–481). Incluye un diagrama mermaid (líneas 8–54), tablas «What's inside» (56–188), 28 ejemplos de uso (196–652), «How it works» (654–667), «Roadmap» (681–707) y una sección «Changelog» (711–747) con las versiones 0.2.5 a 0.3.2, también en inglés. El `description` de `package.json`, `AGENTS.md` y todos los comentarios de `src/` están en inglés; `specs/` está en español.
- **Los ejemplos no son ejecutables hoy:** `specs/000-audit/evidence/extract.mjs` (uso: `node extract.mjs <raíz> <scratch>`, con `<scratch>/dist/index.mjs` y `<scratch>/readme/` existentes y `esbuild` resoluble desde `<raíz>`) extrae los bloques ```` ```ts ````/```` ```js ````, sustituye `from '@dniskav/neuron'` por `<scratch>/dist/index.mjs` y ejecuta cada bloque en un proceso Node aparte (30 s de tiempo máximo). Solo comprueba que no lance (código de salida 0), no los valores impresos, y su propio código de salida es siempre 0: hay que leer la línea final `bloques=N ok=M fail=K`. Muchos bloques del README usan variables sin definir (`X_train`, `points`, `images`, `embeddings128D`, `trainEpoch`, `pattern1`…) y fallarían con `ReferenceError`; otros son pesados (`Autoencoder` de 784 entradas, `GAN` con 10 000 pasos). Ningún ejemplo cubre `Trainer`, `ModelSaver`, `DataLoader`, `Tokenizer` ni `DatasetLoader`.
- **Exports sin mencionar** (comprobado con un script que lee `src/index.ts`, 126 nombres entre valores y tipos): el README omite 32 (todos tipos y los cuatro `validate*`); `AGENTS.md` omite 105 de 126 (casi todo lo posterior a `NetworkTransformerRL`).
- **`AGENTS.md` (committed) está desfasado:** aviso «Active repair» (línea 5), tabla de 15 clases, «All activations are sigmoid» (hay `relu`, `tanh`, `linear`, `leakyRelu`, `elu`), «No convolutional layers», «No data loaders», «No loss functions» (existen `Conv1D`, `Conv2D`, `DataLoader`, `losses`, `Trainer`, `Value`).
- **Cambios en la copia de trabajo del usuario:** `AGENTS.md` tiene 41 líneas borradas SIN commitear (tabla de clases, «What this library does NOT do», build, contexto de uso). Este worktree contiene la versión committed; no se toca esa decisión (pregunta 2).
- **Comentarios de `src/` revisados uno a uno** (solo siguen siendo ciertos los marcados «vigente»):
  - `src/TSNE.ts:43` (paso 5 de la cabecera): `Y(t) = Y(t-1) + lr · grad + …`; el código resta (`src/TSNE.ts:183-188`). **Vigente.** `TSNEOptions.lr` (`:72`) dice «Default 200» sin avisar de que diverge con n pequeño. **Vigente.**
  - `src/Trainer.ts:150-158` `setValidationData`: sin comentario; los datos solo se usan con `earlyStopping` (`:199`, `:243`, `:325`); sin él se ignoran en silencio. **Vigente** (documentar).
  - `src/NetworkN.ts:174-177` y `:190-193`: `getWeights()` y `setWeights()` llaman a `resetMask()` en cada `Dropout`; un `getWeights()` entre `predict(x, true)` y `trainWithDeltas/inputGradient(…, true)` hace lanzar «needs the dropout masks» (`:225-233`). El comentario solo dice «to avoid stale state». **Vigente** (documentar la consecuencia).
  - `src/Autoencoder.ts:65-69`: el comentario dice «we warn so the user is aware» pero no hay `console.warn` (el `throw` está comentado). **Vigente** (el comentario miente; la decisión de fondo es la pregunta 3).
  - `src/ContrastiveLearning.ts:61` `Augmenter.dropoutFeatures`: pone a 0 con probabilidad `rate` y no reescala (no es dropout invertido). El comentario no lo dice. **Vigente** como nota (decisión de fondo: pregunta 4).
  - `src/TCN.ts`: documentar el rango estable de `lr` (en el caso medido de T028, `lr` 1.0 sube la pérdida y ≥ 2 diverge; ver inventario). **Vigente.**
  - **Ya corregidos, no se tocan:** «estilo TensorFlow» de `same` en `Conv1D.ts:14-19, 232-234` y `Conv2D.ts:16-25` (T022 de 005; ahora dicen que no es idéntico a TensorFlow), «Returns: MSE loss» en `NetworkTransformer.ts:102` (003), `LayerNorm.backwardOne` con `−dL/dx` en `LayerNorm.ts:63-69` (003), el comentario del residual de `TCN` (T058), el `tanh` de `RNN` (T057/T059) y la cabecera de `WeightInspector` «activations below this» (T054).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Quien actualiza a 0.4.0 encuentra todos los cambios incompatibles y cómo migrarlos (Priority: P1)

`CHANGELOG.md` en la raíz tiene la entrada «0.4.0 (sin publicar)» con todos los cambios incompatibles de 002–006 agrupados por área, cada uno con «antes → ahora» y una guía de migración concreta, y las versiones anteriores migradas desde el README.

**Why this priority**: Constitución VII lo exige para aceptar cambios incompatibles en 0.x; la 008 actualiza `neuron_app` a partir de este archivo.

**Independent Test**: `specs/007-docs/breaking-changes.md` (inventario) y `CHANGELOG.md` se comparan fila por fila: cada fila del inventario aparece en el changelog y cada entrada del changelog cita su origen en el inventario. Para las migraciones con receta numérica (escala de `NetworkN.train`/`Network.train`, `Autoencoder`) hay evidencia reproducible frente a la fuente de 0.3.2 (`723fa58`).

**Acceptance Scenarios**:
1. **Given** el inventario, **When** se cuenta, **Then** cada fila tiene área, clase/método, antes, ahora, migración y feature de origen, y su origen se puede rastrear a una línea de `HANDOFF.md` o de `specs/006-coverage/tasks.md` (o a un `git diff` citado).
2. **Given** `CHANGELOG.md`, **When** se busca `DecisionTree` `maxDepth: Infinity`, `PositionalEncoding` con `dModel` impar o `GAN` con una sigmoide propia equivalente, **Then** NO aparecen como incompatibles (las revirtieron T064 y T065).
3. **Given** `CHANGELOG.md`, **Then** la entrada 0.4.0 no lleva fecha y `package.json` sigue en `0.3.2`.

### User Story 2 - El README tiene un solo idioma y todos sus ejemplos se ejecutan (Priority: P1)

El README está en un solo idioma y cada bloque ```` ```ts ```` es autosuficiente (datos definidos, tamaños que acaban en segundos) y se ejecuta con éxito contra `dist/` recién construido.

**Independent Test**: `node specs/007-docs/tools/readme-gate.mjs` (T002) construye el directorio de trabajo, llama a `extract.mjs` y falla si `fail > 0`; se repite 3 veces para detectar ejemplos con azar.

**Acceptance Scenarios**:
1. **Given** el README ensamblado, **When** corre el gate con `dist/` recién construido, **Then** `bloques=N ok=N fail=0`, tres veces seguidas.
2. **Given** los comentarios de resultado de un ejemplo (`// ~10`, `// 1`), **Then** coinciden con la salida real (el agente la compara y la anota en su informe) o expresan un rango; ningún ejemplo afirma una cifra exacta que dependa de un azar sin semilla.
3. **Given** el README, **Then** no queda texto en el otro idioma (comprobable con `grep` de los cuatro encabezados actuales y de caracteres con tilde o `¿`/`¡` fuera de código).

### User Story 3 - El README dice lo que el código hace hoy (Priority: P1, Constitución VI)

Lo que 002–006 cambió está reflejado: firmas, escala del paso (`lr`), comportamiento de `VAE` (`klWeight`, salida lineal del encoder), `GAN` (D con salida sigmoide; el generador aprende a través de D), `Autoencoder` (paso de la MSE media), `Trainer` (sin `clipValue`; `restoreBestWeights`; `setValidationData` solo con `earlyStopping`), `ModelSaver` (formato `{format:1,…}`), `Conv` `same` con kernel par, validación («Clase.método: …» y lanza antes de tocar estado), `lr` estable de `TSNE` y `TCN`, y los límites documentados (`ContrastiveLearning` con semillas muertas con inicialización cruda; `PCA` con ejes nulos con rango deficiente mientras T066 siga abierta).

**Independent Test**: cada afirmación nueva del README se traza a una fila del inventario o a un archivo de `src/` con su línea; un revisor distinto del autor la comprueba por muestreo (T017).

**Acceptance Scenarios**:
1. **Given** la tabla «What's inside», **When** se compara con `src/index.ts`, **Then** los 126 nombres aparecen (valores y tipos), cada uno una vez como mínimo.
2. **Given** los ejemplos de `Autoencoder`, `VAE`, `GAN`, `NetworkN`, `Trainer` y `ModelSaver`, **Then** usan firmas y `lr` coherentes con la fuente actual (leídos del código, no de la memoria).

### User Story 4 - `AGENTS.md` describe la librería real (Priority: P2; BLOQUEADA hasta la pregunta 2)

`AGENTS.md` lista la jerarquía completa según `src/index.ts`, las restricciones de diseño reales (activaciones `sigmoid/relu/tanh/linear/leakyRelu/elu`, convoluciones, cargadores de datos, pérdidas, autograd, optimizadores), el contrato de validación (`«Clase.método: …»`, lanza antes de tocar estado, `lr = 0` es no-op válido) y no lleva el aviso «Active repair».

**Independent Test**: `tests/docs.agents.exports.test.ts` (T016) comprueba que cada export de `src/index.ts` aparece en `AGENTS.md`.

### User Story 5 - Los comentarios de `src/` coinciden con el código (Priority: P2)

Los cinco comentarios vigentes de la sección anterior quedan corregidos o ampliados, sin tocar una sola línea ejecutable.

**Independent Test**: el diff de `src/` contra la base solo contiene líneas de comentario y la suite completa pasa sin cambios.

### User Story 6 - Las decisiones de diseño pendientes quedan escritas, no implementadas (Priority: P3)

La spec recoge, con opciones y recomendación, si `Autoencoder` restaura el aviso `latentSize >= inputSize` y si `ContrastiveLearning.dropoutFeatures` pasa a dropout invertido. Ninguna es tarea de 007 (cambian comportamiento).

### User Story 7 - La documentación no se desincroniza (Priority: P2)

Un meta-check de cada export de `src/index.ts` en el README (y en `AGENTS.md`) corre en `npm test`, y el gate de ejemplos corre en el cierre.

**Independent Test**: borrar un nombre del README hace fallar `tests/docs.readme.exports.test.ts`; romper un ejemplo hace fallar `readme-gate.mjs`.

### Edge Cases

- Un ejemplo que depende de `Math.random` (p. ej. `TCN`, `GAN`): fijar semilla propia en el bloque o usar solo aserciones con margen; no afirmar cifras exactas.
- Ejemplos pesados: reducir tamaños (p. ej. `Autoencoder(8, [6], 2, [6])` en lugar de 784) con una nota «escala a tu problema»; deben acabar muy por debajo de los 30 s del extractor.
- Un ejemplo ilustrativo no ejecutable (esbozo de código con `…`): se marca con otra etiqueta de bloque (```` ```text ````) para que el extractor lo ignore, y cada uso se justifica en el informe.
- Versiones anteriores a 0.2.5: no hay datos en el repositorio; no se inventan (el changelog empieza en 0.2.5 con lo que ya dice el README).
- Los tipos (`Activation`, `TrainerOptions`…) no tienen ejemplo: basta una tabla «Types» en el README.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Se crea `specs/007-docs/breaking-changes.md` con una tabla única: área, clase/método, antes, ahora, migración, feature de origen y una columna «origen» (línea de `HANDOFF.md`, tarea de `tasks.md` o `git diff`). Cada afirmación del apéndice A de esta spec se confirma, se corrige o se marca «no verificada» con el motivo. Es la fuente única de la que beben `CHANGELOG.md` y README.
- **FR-002**: El inventario excluye (o marca como «revertido») `DecisionTree` con `maxDepth: Infinity`, `PositionalEncoding` con `dModel` impar y `GAN` con sigmoide propia equivalente (T064, T065), y describe el estado neto actual de cada área, no la historia de lotes.
- **FR-003**: Se crea `CHANGELOG.md` en la raíz, en el idioma del README, con la entrada `0.4.0` sin fecha (la fecha y la versión final las pone la 008) y las secciones: «Breaking changes» agrupados por área (entrenamiento y gradientes; escala del paso y `lr`; validación que ahora lanza; formato de serialización y empaquetado; API nueva aditiva), «Migration guide» con receta concreta por cambio y «Known limitations»; más `0.3.2` a `0.2.5` migradas del README sin pérdida.
- **FR-004**: La guía de migración incluye, como mínimo, estas recetas (cada una a verificar contra el inventario): `NetworkN.train` y `Network.train` (paso `2/n` veces el anterior; usar `lr·n/2`; con una salida el `lr` efectivo se duplica), `Autoencoder` (multiplicar `lr` por `d/2`), pesos guardados con el entrenamiento roto (`RNN`, `TCN`, `Seq2Seq`, `NetworkTransformer`, `NetworkTransformerRL`, `VAE`, `GAN`, `Autoencoder`, `ContrastiveLearning`, `Word2Vec`, `NetworkN` con `residual`) no son comparables: reentrenar; `ModelSaver` formato `{format:1,length,structure?,weights,timestamp}` (el JSON antiguo carga si la longitud coincide; `format` distinto de 1 lanza); `setWeights` ya no recorta ni acepta NaN (un modelo divergido a NaN no se restaura con `setWeights(getWeights())`); `Trainer` sin `clipValue` (usar `ClippedOptimizerFactory`); `Conv1D/Conv2D` `same` con kernel par (la salida pasa de `L−1` a `ceil(L/stride)`); `VAE` (`klWeight`, salida lineal del encoder); `GAN` (D con salida sigmoide); `SoftmaxRegression.train` y `LinearRegression.fitGD` incrementales (instancia nueva para reiniciar); `fitNormal` y `Metrics` lanzan (capturar o limpiar datos).
- **FR-005**: Las recetas con fórmula numérica (`NetworkN`/`Network`, `Autoencoder`) se comprueban contra la fuente de 0.3.2 (`723fa58`, en un worktree aparte) y la evidencia queda en `specs/007-docs/migration-evidence.md`; si la receta no reproduce el comportamiento previo, se corrige el changelog, no el criterio.
- **FR-006**: El README se redacta en UN solo idioma (pregunta 1) y se compone en partes disjuntas bajo `specs/007-docs/readme-parts/` que un único paso (T013) concatena en `README.md` con `specs/007-docs/tools/assemble-readme.mjs`. Ninguna otra tarea escribe `README.md`.
- **FR-007**: Cada bloque ```` ```ts ```` del README es autosuficiente y se ejecuta con éxito contra `dist/` recién construido (`npm run build`); tardan ≪ 30 s y no usan red ni archivos externos. Los bloques con azar usan semilla propia o no afirman cifras exactas.
- **FR-008**: Los comentarios de resultado de cada ejemplo coinciden con la salida real o declaran un rango; el agente anota en su informe la comparación (valor impreso frente a comentario) de cada ejemplo con cifras.
- **FR-009**: El README corrige lo que ya no es cierto tras 002–006 (US3), incluidos: ejemplos con `lr` coherente con la escala actual, `VAE` con `klWeight`, `GAN` con D sigmoide, `Trainer`/`ModelSaver`/`DataLoader`/`Tokenizer`/`DatasetLoader` con ejemplo nuevo, tabla de tipos exportados, `lr` estable de `TSNE` y `TCN` con el caso medido, límites de `ContrastiveLearning` y `PCA`, y el comentario de `ContrastiveLearning` del README (verificar si «projection head: 64 → 32» coincide con el constructor `(inputSize, encoderHidden, projectionDim, options)` de `src/ContrastiveLearning.ts:143`).
- **FR-010**: La sección «Changelog» del README se sustituye por un enlace a `CHANGELOG.md`; «Build» menciona `npm run typecheck` y `npm run coverage`; se añade una sección corta «Upgrading to 0.4» (o equivalente en el idioma elegido) que enlaza la guía de migración.
- **FR-011**: `AGENTS.md` se actualiza según US4 (BLOQUEADA): jerarquía completa de los 126 exports, restricciones reales, contrato de validación y sin el aviso «Active repair». Ninguna tarea toca `AGENTS.md` hasta que el usuario responda la pregunta 2.
- **FR-012**: Se corrigen los comentarios de `src/` listados como «vigentes» (TSNE paso 5 y `lr`, `Trainer.setValidationData`, `NetworkN.getWeights/setWeights`, `Autoencoder` aviso, `Augmenter.dropoutFeatures`, rango de `lr` de `TCN`) sin tocar código ejecutable. Criterio: el diff de `src/` contra la base solo tiene líneas de comentario y la suite pasa.
- **FR-013**: Un meta-check automatizado (`tests/docs.readme.exports.test.ts`, y `tests/docs.agents.exports.test.ts` cuando se desbloquee) lee `src/index.ts` y falla si algún nombre exportado (valor o tipo) no aparece como palabra completa en `README.md` (o en `AGENTS.md`).
- **FR-014**: El gate de ejemplos corre 3 veces contra `dist/` construido en el cierre; `npm run typecheck`, `npm run coverage` y `npm run build` terminan limpios y `npx vitest run` pasa 20 veces seguidas, 5 con `--sequence.shuffle` y 3 bajo Node 20.19.
- **FR-015**: `package.json` mantiene `"version": "0.3.2"` y, junto con `package-lock.json` y `src/index.ts`, no cambia. `CHANGELOG.md` solo lo edita la tarea que lo crea (excepción expresa a «el changelog es exclusivo del orquestador»; en 007 es el entregable).
- **FR-016**: Las decisiones pendientes (US6) y el límite de `PCA` (T066) quedan documentadas como «Known limitations» o preguntas, nunca como tareas de comportamiento. Fuera de alcance: subir versión, publicar, tocar `neuron_app`, cambiar comportamiento de `src`, T066.

### Key Entities

- **Inventario** (`breaking-changes.md`): tabla de cambios incompatibles con trazabilidad.
- **Partes del README** (`readme-parts/NN-*.md`): fragmentos disjuntos que se concatenan en orden numérico.
- **Gate de ejemplos** (`tools/readme-gate.mjs`): envoltorio de `extract.mjs` con directorio temporal, `dist` enlazado y 3 repeticiones.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `node specs/007-docs/tools/readme-gate.mjs` da `fail=0` en 3 ejecuciones consecutivas, con N ≥ 30 bloques.
- **SC-002**: 0 nombres de `src/index.ts` ausentes en `README.md` (126/126) y, desbloqueada la pregunta 2, en `AGENTS.md`.
- **SC-003**: cada fila del inventario tiene origen rastreable (0 filas sin origen); cada receta numérica de FR-005 tiene evidencia con diferencia máxima ≤ 1e-9 frente a 0.3.2 o queda corregida.
- **SC-004**: 0 líneas ejecutables cambiadas en `src/` (el diff solo tiene comentarios); `git diff --quiet d0228b7 -- package.json package-lock.json src/index.ts` sin cambios y `grep '"version"' package.json` = `0.3.2`.
- **SC-005**: `npm run typecheck`, `npm run build` y `npm run coverage` limpios; suite verde 20 + 5 shuffle + 3 en Node 20.19; CI de GitHub verde (`ci` ×2, CodeQL, GitGuardian).
- **SC-006**: el README no tiene texto en el idioma descartado (0 coincidencias de los cuatro encabezados y comentarios actuales).

## Assumptions

- Los cambios de comentarios de `src/` (T003, T004) los hace un agente con permiso sobre `src/` (`api-dev`), no `docs-chore`, y solo con el criterio «solo líneas de comentario».
- `esbuild` está disponible por dependencia transitiva de `tsup` tras `npm ci`; si el extractor no lo resuelve, el gate lo avisa y se informa (no se añade dependencia: Constitución I).
- Las cifras de `lr` estable de `TSNE` y `TCN` proceden de los casos medidos en T020 y T028; T004 las reproduce antes de escribirlas.

## Preguntas abiertas al usuario (máx. 6)

1. **Idioma del README.** (a) Inglés; (b) español. **Recomendación: (a) inglés.** Evidencia: 24 de 28 secciones de uso, todas las tablas, el changelog, el `description` de `package.json`, `AGENTS.md`, `SECURITY.md` y todos los comentarios de `src/` ya están en inglés (el commit `e9a21be` tradujo los comentarios a inglés); los cuatro encabezados en español (líneas 410, 435, 448, 463) parecen un resto de la sección de la app. `specs/` y los informes siguen en español. Si el público principal es hispanohablante, (b) cuesta una traducción completa de tablas, changelog y 28 ejemplos.
2. **`AGENTS.md` con 41 líneas borradas sin commitear** (tabla de clases, «What this library does NOT do», build, contexto de uso). (a) El usuario las commitea o las descarta y avisa; (b) la tarea parte de la versión de `HEAD` y reescribe esas secciones al día; (c) se respeta el borrado y T012 solo añade lo nuevo. **Recomendación: (b) tras confirmar que el borrado era para acortar el archivo**; «What this library does NOT do» hoy es FALSA (hay `Conv1D`, `Conv2D`, `DataLoader`, `losses`), así que la sección conviene reescribirla, no borrarla. T012 y T016 quedan BLOQUEADAS hasta la respuesta.
3. **`Autoencoder` y el aviso `latentSize >= inputSize`** (`src/Autoencoder.ts:65-69`, hoy comentado; el comentario promete un aviso que no existe). (a) Dejarlo sin aviso y corregir el comentario («no error, no warning»); (b) restaurar un `console.warn` una vez; (c) lanzar. **Recomendación: (a)**: un autoencoder sobrecompleto es válido (denoising), la Constitución V trata las entradas inválidas y esta es válida, y `HopfieldNetwork` ya avisa solo para un caso de capacidad medible. T003 aplica (a) en el comentario; (b)/(c) serían una tarea de comportamiento aparte.
4. **`ContrastiveLearning.dropoutFeatures` sin reescalar** (`src/ContrastiveLearning.ts:61`). (a) Mantener (es enmascarado tipo BERT, no dropout invertido) y documentarlo; (b) pasar a dropout invertido (`v / (1 − rate)`). **Recomendación: (a)**: (b) cambia las trayectorias de entrenamiento y la limitación de semillas muertas ya medida (T012), y es un cambio de comportamiento fuera del alcance de 007. T003 documenta (a).
5. **`PCA` con ejes de norma ≈ 0 en rango deficiente (T066, propuesta no lanzada).** (a) 007 documenta el límite en «Known limitations» y T066 se decide aparte; (b) se lanza T066 antes de cerrar 007 y el README no menciona el límite. **Recomendación: (a)**: T066 cambia `src/PCA.ts`, fuera de alcance; el límite está acotado (`transform` y la reconstrucción son correctas).
6. **Meta-checks como tests permanentes** (`tests/docs.*.test.ts`, entran en `npm test` y en CI). (a) Sí; (b) solo scripts en `specs/007-docs/tools/`, corridos en el cierre. **Recomendación: (a)**: añadir un export sin documentarlo rompería el CI, que es justo el desfase que 007 arregla. Coste: cada export nuevo exige una mención en el README (y en `AGENTS.md`).

---

## Apéndice A — Inventario preliminar de cambios incompatibles (fuente; lo verifica T001)

Claves de origen: `H·003(n)` = `HANDOFF.md`, «Cambios observables de 003», ítem (n); `H·004(n)`, igual para 004; `H·005-I2` = «Cambio observable de 005 (I2)»; `H·005-p` = «Estado de 005 (parcial…)»; `H·005-ad` = «Cambios observables adicionales de 005 (T013–T016)»; `H·005-L7` = nota del Lote 7 en «Dónde estamos»; `H·005-apl` = «Pendientes de 005 aplazados…»; `H·006-L9-12` = «Cambios observables adicionales de 006…, Lotes 9–12»; `H·006` = «Cambios observables de 006 para el changelog»; `T0xx` = nota o tarea de `specs/006-coverage/tasks.md`; `H·N6/N10…` = «Hallazgos nuevos». «002»: solo pruebas (a verificar con `git diff --stat 723fa58..957d618 -- src`).

**Entrenamiento y gradientes (la pérdida baja donde antes subía o el paso era otro)**
- `RNN`, `TCN`, `Seq2Seq`, `NetworkTransformer`, `NetworkTransformerRL`: de subir la pérdida a bajarla; `lr` de las apps y pesos guardados con el entrenamiento roto no son comparables (H·003(1)).
- `NetworkTransformer.train`: gradiente de la CE media (`dLogits/count`): el paso de SGD directo (embeddings, LayerNorm, sesgos) baja por el factor `count`; con máscara (Sudoku) puede exigir reajustar `lr` (H·003(2)).
- `TCN`: las capas conv ya no se mueven T veces más rápido que la proyección (H·003(3)).
- `Seq2Seq`: el encoder recibe el gradiente exacto (BPTT completo, escala `lr·T_in/T_out`) (H·003(4)); `LSTMLayer.backprop(dh_seq, lr, dcFinal?)` devuelve `{dh0, dc0}` en vez de `void` (aditivo).
- `AttentionHead`/`MultiHeadAttention`/`TransformerBlock`/`LayerNorm`: el `dX` propagado ya no depende de `lr` (H·003(5)); `NetworkTransformerRL`: un solo paso de Adam por llamada sobre `inputProj` con el gradiente sumado (H·003(7)).
- `Autoencoder`: el encoder deja de moverse ~4× lo correcto con sigmoide (H·003(6), H·N6); gradiente de la MSE media: el paso baja `d/2` veces, multiplicar `lr` por `d/2` (H·004(4), H·N10); el encoder usa `decoder.inputGradient`.
- `NetworkN` con `residual`: gradiente distinto y entrenamiento distinto; `trainWithDeltas`/`inputGradient` LANZAN si la última capa tiene residual (alcanza a `Autoencoder`, `VAE`, `GAN`) (H·004(1)).
- `BatchNorm.backward`: derivada exacta (con momentum 0.1 el `dX` baja ~10×; momentum 0 da 0; momentum 1 no cambia) (H·004(3)).
- `Word2Vec` skip-gram/CBOW: gradiente exacto del par; skip-gram ya no diverge con `lr` 0.1 (H·004(5)).
- `ContrastiveLearning`: cabeza de proyección lineal (`project()` puede dar negativos), paso simultáneo, `cosineSimilarity = dot/(‖a‖‖b‖+1e-8)`, paso de FD interno 1e-6 (H·004(6)); limitación: semillas muertas con inicialización cruda (H·«Limitación documentada tras 004»).
- `GAN.trainStep`: el generador recibe el gradiente real de −log D(G(z)) a través de D congelado; D debe tener salida sigmoide (por valor, T065) (H·004(7), T041, T065).
- `VAE`: encoder LINEAL (`mu`, `logVar` fuera de (0,1); σ ya no ≥ 1), el encoder aprende de la reconstrucción, nuevo `klWeight` (defecto 1; ≈ 0.003 aprende en datos pequeños en [0,1]), `forward` devuelve `eps`, `options.activations` ya no se aplica a la salida del encoder y exige `decoderHidden.length + 1` entradas (H·004(8), T041).
- `MaxPool2D.backward` con ventanas solapadas (stride < poolSize) correcto (antes sumaba a posiciones que no eran el máximo de esa ventana) (T055, H·006).
- `PCA`: `explainedVarianceRatio` divide por la varianza total (ya no suma 1 con menos componentes), más precisión (~1e-15), ejes de autovalor 0 ortonormales (T051, H·006); límite conocido: rango deficiente puede dar ejes de norma ≈ 0 (T062, T066 abierta).
- `DecisionTree` acepta divisiones de ganancia cero (XOR = 1.0 con `maxDepth ≥ 2`) (T003, H·006).
- `tanh` estable: `RNN`, `GRU`, `LSTMLayer`, `activations.tanh` ya no dan NaN para x ≳ 355 (T057, T059).
- `f1Score` macro = media de F1 por clase (convención sklearn); `r2Score` con `yTrue` constante da 1 solo si SS_res = 0, si no 0 (T036, H·006-L9-12).

**Escala del paso y `lr`**
- `NetworkN.train` y `Network.train`: paso `2/n` veces el anterior (n = nº de salidas); con una salida el `lr` efectivo se duplica; para conservar la dinámica anterior usar `lr·n/2`; `trainWithDeltas`/`inputGradient` no cambian de escala (H·005-ad).
- `Autoencoder`: ver arriba (`lr·d/2`). `NetworkTransformer.train`: ver arriba (`count`). `TCN`: `lr` estable acotado (en el caso medido de T028, `lr` 1.0 sube la pérdida y ≥ 2 diverge; los tests usan 0.21–0.39). `TSNE`: `lr` 200 (defecto) diverge con n = 30; los tests usan 5–10 (T020). A verificar en T001/T004: ambos son «medido en un caso», no una ley.
- Dropout: con `dropoutRate > 0`, `Autoencoder`/`VAE`/`GAN` usan UNA máscara por paso (trayectorias cambian; con 0 son idénticas); `VAE.train` aplica dropout en el forward de entrenamiento; API nueva `NetworkN.inputGradient(…, reuseMasks=false)` y `trainWithDeltas(…, lr, reuseMasks=false)`, `Dropout.reuseMask/hasMask` (H·005-ad).

**Validación que ahora lanza («Clase.método: …», antes de tocar estado)**
- `setWeights` de `WeightMatrix`, `BiasVector`, `EmbeddingMatrix`, `LayerNorm`, `BatchNorm`, `AttentionHead`, `MultiHeadAttention`, `TransformerBlock`, `NetworkTransformer`, `NetworkTransformerRL`: longitud incorrecta, NaN/Infinity/undefined o no-array (antes recortaban o dejaban undefined/NaN); un modelo divergido a NaN no se restaura con `setWeights(getWeights())` (H·005-I2).
- `LSTMLayer.backprop`, `GRULayer.backprop`, `NetworkLSTM.train`: longitud de secuencia incorrecta o NaN/Infinity (antes ceros/no-op); un error no vacía la trayectoria. `NetworkLSTM` con `denseStructure` vacío lanza en el constructor (H·005-ad, H·005-L7).
- `Trainer`: se elimina `clipValue`; opciones inválidas (`epochs`, `lr`, `lrDecay`, `weightDecay`, `patience`, `minDelta`) y dataset vacío lanzan; `earlyStopping` sin datos de validación (o vacíos) lanza; nueva `restoreBestWeights` (H·005-p, H·005-L7). `Conv1D`/`Conv2D` validan `stride` y `valid` con k > H (H·005-L7). `Conv` `same` con kernel par: relleno asimétrico, la salida pasa de `L−1` a `ceil(L/stride)` (H·005-p).
- `Logistic`/`Softmax`/`Linear` `train`/`fitGD`: datos validados (filas de longitud constante y finitas, `y` finito, etiquetas enteras ≥ 0 en Softmax, `y` en [0,1] en Logistic); ahora INCREMENTALES (continúan desde los pesos si la forma coincide; para reiniciar, instancia nueva); `predict`/`predictProba`/`getCoefficients` funcionan con pesos a mano (H·005-ad, H·005-p). `LinearRegression.fitNormal` lanza con filas irregulares o no finitos (antes pesos NaN) (T037).
- `Network.train(inputs, number | number[], lr)` entrena todas las salidas y `Network` cumple `TrainableNetwork` (H·005-p).
- `Metrics`: lanza con longitudes distintas, vacíos (incluido `accuracy([], [])`, antes 0), etiquetas negativas/no enteras, `positiveClass`/`numClasses` inválidos, etiqueta ≥ `numClasses` y `auc` con una sola clase (H·006-L9-12, T036, decisión 1 de 2026-11-11).
- `SOM`: lanza con X vacía en `quantizationError`, NaN/Infinity o ancho erróneo en `train`, `epochs` no entero o < 1 y tamaños no enteros en el constructor (antes truncaba) (T056, T038). `KMeans.predict`/`PCA.transform` rechazan dimensión errónea y NaN/Infinity; `KMeans.fit`/`PCA.fit` rechazan X irregular o no finita (T038).
- `RNN`/`TCN`/`CausalConv1D`/`Seq2Seq`: lanzan con ancho/targets/`lr` NaN o negativo, NaN/Infinity en filas y contexto, secuencia vacía (`RNN.backward([])`, `Seq2Seq.encode([])`, `Seq2Seq.trainStep` vacío), constructores con tamaños NaN/no enteros (T040, T060). `lr = 0` es no-op válido en todos los modelos (decisión 2 de 2026-11-11).
- `Autoencoder`/`VAE`/`GAN`/`ContrastiveLearning`: lanzan con lotes vacíos, ancho/no finitos, `lr` NaN o negativo (antes NaN y pesos corrompidos); `Autoencoder.trainBatch` valida todas las filas antes de entrenar (T041, T060).
- `GaussianNaiveBayes`/`DecisionTree`: `fit` valida X (irregular, NaN, etiquetas no finitas, ancho 0); `DecisionTree` exige `maxDepth` entero ≥ 1 o `Infinity` y `minSamplesSplit` entero ≥ 2; `predict` valida dimensión y finitud (T039, T064).
- `EarlyStopping`: `patience` entero ≥ 1 (antes 0 se aceptaba), `minDelta` finito ≥ 0, `update(NaN)` lanza; `LRScheduler`: `stepDecay` exige `epochsDrop > 0`, `exponentialDecay` `decayRate` finito, `cosineAnnealing` `maxEpochs > 0`; `DataLoader`: `batchSize` entero ≥ 1; `PositionalEncoding`: `pos`/`seqLen` enteros ≥ 0 y `dModel` entero ≥ 1 (impar admitido, T064); `Word2Vec.train`: `epochs` entero ≥ 1 y `lr` finito ≥ 0 (T042).
- `DataAugmentation`: `shuffle` y `split` lanzan con longitudes distintas; `normalize`/`standardize` con X vacía; `MaxPool2D.forward` con entrada menor que `poolSize`; `Flatten.forward` con entrada vacía o irregular (T043).
- `Tokenizer`: lanza con `maxVocab` menor que el nº de especiales; el JSON incluye `specialTokens` (aditivo) (T049). `DatasetLoader`: lanza con celdas numéricas vacías y filas irregulares (T050).
- `HopfieldNetwork`: lanza con entradas no bipolares/NaN en `store`/`recall`/`energy` y NaN en `binarize`/`unbinarize`; `store` avisa por consola (una vez por red) al superar `max(1, floor(0.138·N))` patrones (T052).
- `LossPlotter`: lanza con NaN/Infinity y `width`/`height` no enteros o ≤ 0; etiqueta media correcta con serie constante (T053). `WeightInspector`: lanza con `deadThreshold` NaN o negativo, filas irregulares y activaciones no finitas (T054).

**Formato de serialización y empaquetado**
- `ModelSaver`: formato `{format:1, length, structure?, weights, timestamp}` con validación al cargar; el JSON antiguo carga si la longitud coincide; `format` ≠ 1 lanza (H·005-p).
- `getWeights` de LSTM/GRU/`NetworkLSTM` devuelve copias profundas y `setWeights` no deja alias; `getWeights`/`getWeightsFlat` y varias utilidades sin desborde de pila con vectores > ~125 000 elementos (T034, T035, T048) (H·005-p, H·006-L9-12).
- `Value.backward` iterativo (100 000 nodos sin desbordar; resultados idénticos) (I7).
- Empaquetado (001, a verificar con `git diff 723fa58 -- package.json`): `exports` con `types` primero, `engines.node >= 20`, `sideEffects: false`, `LICENSE`, `repository`, `homepage`, `bugs`.

**API nueva y aditiva:** `NetworkN.inputGradient`, `Trainer` `restoreBestWeights`, `VAE` `klWeight`, `Dropout.reuseMask/hasMask`, `LSTMLayer.backprop` devuelve `{dh0, dc0}`, `Tokenizer` snapshot con `specialTokens`, `Network` con varias salidas.

**Revertidos por T064/T065 (NO son incompatibles):** `DecisionTree` `maxDepth: Infinity` válido, `PositionalEncoding` `dModel` impar, `GAN` sigmoide por valor.
