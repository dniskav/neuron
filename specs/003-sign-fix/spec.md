# Feature Specification: sign-fix

**Feature Branch**: `fix/v0.4` (se trabaja directamente en ella; no se ejecutó `create-new-feature.sh` porque crearía una rama)
**Created**: 2026-11-08
**Status**: Draft
**Input**: `specs/ORCHESTRATION.md` sección "003 sign-fix", hallazgos C1, N1, N3 y N6 de `specs/000-audit/AUDIT.md` y `specs/HANDOFF.md` ("Hallazgos nuevos" y "Notas para 003 por modelo"), Constitución II, III, IV, VI y VII. Condición previa: la feature 002 (oráculo de gradiente, `tests/helpers/gradientOracle.ts`, `tests/oracle/*.oracle.test.ts`).

## Estado verificado hoy (leído en `src/` y `tests/oracle/`)

Convención del repositorio: los optimizadores hacen `w + lr·g`, así que las capas reciben `g = −dL/dw` (Constitución II). Los `backward` de bloque (`TransformerBlock`, `LayerNorm.backwardOne`, `LSTMLayer.backprop`, `CausalConv1D.backward`) ya siguen esa convención y devuelven `−dL/dx`. Los modelos con defecto C1 construyen el delta de salida como `+dL/dout` y lo entregan tal cual.

- **`NetworkTransformer.train`** (`src/NetworkTransformer.ts:127-138`): `dLogits = prob − target` (= `+dL/dz`); sin negar. Escala: `dLogits` no se divide por `count`, pero la pérdida devuelta sí (`loss /= count`, línea 139). Es el ratio ≈ 3 de HANDOFF. El comentario de la línea 101 dice "MSE loss" y es cross-entropy. `dH` (líneas 145-149) se calcula con `outputProj.W` antes de actualizar: correcto.
- **`NetworkTransformerRL.train`** (`src/NetworkTransformerRL.ts:127`): `dPred = 2·(pred − target)/n` (= `+dL/dpred`). Escala coherente con la pérdida (MSE medio sobre `n` acciones). Además `inputProj.update` se llama una vez por posición (líneas 155-162) y `WeightMatrix` usa `Adam` por escalar: con el optimizador real son `seqLen` pasos de Adam con gradientes parciales en lugar de uno con la suma (no es un error de signo; con `withPlainSGD` el oráculo no lo ve). Ver Q5.
- **`RNN.backward`** (`src/RNN.ts:185-190`): `dOutputs = 2·diff/outputSize` sin negar. Escala ya correcta (`lr/T`, pérdida media sobre `T·outputSize`). `dh` usa `Why` antes de actualizar: correcto.
- **`TCN.train`** (`src/TCN.ts:316-348`): `dOut = 2·diff/outputSize` sin negar (signo). Escala: la proyección usa `lr/T` (línea 342), pero `CausalConv1D.backward(dConvOut, lr)` (línea 364) recibe `dOut` sumado sobre `T` y `lr` entero, así que las capas conv dan un paso `T` veces mayor que el de la pérdida media. `CausalConv1D.backward` en sí es correcto (HANDOFF; `dPadded` usa los kernels antes del update).
- **`Seq2Seq.trainStep`** (`src/Seq2Seq.ts:174-225`): `dProjOut = 2·diff/outputSize` sin negar (signo). Encoder: `dContext = dhSeq[0]` (línea 221) es el gradiente respecto a la salida del decodificador en el paso 0, no respecto al estado inicial `(h, c)` del decodificador, que es lo que entrega el encoder. `LSTMLayer.backprop` (`src/LSTMLayer.ts:176`) devuelve `void`: descarta `dh_next` y `dc_next` finales (líneas 193-194, 234-246), justo lo que hace falta. Con el signo arreglado, HANDOFF midió coseno global 0.996. Escala: `W_out` y decodificador usan `lr/T_out` (correcto para pérdida media sobre `T_out`); el encoder usaría `lr/T_in` dentro de `LSTMLayer.backprop`.
- **`AttentionHead.backward`** (`src/AttentionHead.ts:161-172`): `Wq/Wk/Wv.update` se ejecuta antes de calcular `dX`, que lee `this.Wq.W` ya movidos (N1). Confirmado en el código.
- **N1 es MÁS ESTRECHO de lo que dice HANDOFF** (discrepancia que se somete a Q4): `MultiHeadAttention.backward` calcula `dConcat` con `Wo.W` (línea 80) antes de `Wo.update` (línea 92): correcto; solo hereda el error de `heads[h].backward`. `TransformerBlock.backward` calcula `dFf1Out` con `ff2.W` (línea 135) antes de `ff2.update` (150) y `dH1_fromFf` con `ff1.W` (160) antes de `ff1.update` (174): correcto; solo hereda de `attn.backward`. `LayerNorm.backwardOne` captura `gammaOld` antes de actualizar y calcula `D` con él (líneas 72-82): correcto. Por tanto el único arreglo de `src/` para N1 está en `AttentionHead`; MHA, bloque y LayerNorm necesitan tests de regresión de `dX` con `lr > 0` (hoy solo existe `AttentionHead.dx.oracle.test.ts`) y subir `lr` en sus oráculos, hoy 1e-3 y por tanto ciegos a N1 (HANDOFF, T038).
- **`LayerNorm.backwardOne`** (N3): el comentario (líneas 58-60) dice "computes dL/dx" y la función devuelve `−dL/dx` porque recibe `dOut = −dL/dy`. Solo es un defecto de comentario.
- **`Autoencoder.train`** (N6, `src/Autoencoder.ts:190-193`): `outputDeltas` sí lleva `dfn` de la activación de salida del decodificador, pero `dLdz` (gradiente respecto a `z`) se pasa a `encoder.trainWithDeltas(x, dLdz, lr)`, que espera deltas de la capa de salida (con la derivada de su activación ya aplicada; el mismo contrato que `outputDeltas` en la línea 149). Falta multiplicar por `encoder.layers[last].neurons[k].activation.dfn(z[k])`. Confirmado: el defecto es de ámbito de 003, sin cuestiones de signo (convención `(x − x̂)` ya es `−dL/dx̂` salvo factor 2).
- Estado de los oráculos (`tests/oracle/`): `it.fails // AUDIT C1` en `NetworkTransformer`, `NetworkTransformerRL`, `RNN`, `TCN`, `Seq2Seq` (global y por bloques, más "learns"); `it.fails // AUDIT N1` en `AttentionHead.dx.oracle.test.ts:118`; `it.fails // AUDIT N6` en `Autoencoder.oracle.test.ts:61`. Los cinco primeros tienen además un test "control" que niega Δw a mano y se pone en rojo al corregir (`NetworkTransformer.oracle:116`, `NetworkTransformerRL.oracle:104`, `RNN.oracle:103`, `TCN.oracle:173`, `Seq2Seq.oracle:150`).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Los modelos de secuencia entrenan en la dirección correcta (Priority: P1)

`NetworkTransformer`, `NetworkTransformerRL`, `RNN`, `TCN` y `Seq2Seq` siguen `−dL/dw` (C1), y su oráculo y su prueba "aprende" pasan como tests normales.

**Why this priority**: hoy estos cinco modelos suben la pérdida; las apps que los usan (Sudoku, texto, laberinto) se ajustaron con entrenamiento roto (ORCHESTRATION, 008).

**Independent Test**: `npx vitest run tests/oracle/<Modelo>.oracle.test.ts` pasa sin `it.fails`, con coseno > 0.999 y `expectLearns` en verde.

**Acceptance Scenarios**:

1. **Given** el oráculo de cada modelo con `lr` de la tabla de abajo, **When** se ejecuta, **Then** coseno > 0.999 como test normal (sin `it.fails`).
2. **Given** el modelo corregido, **When** se invierte temporalmente el signo en `src/`, **Then** el oráculo y "aprende" fallan (mutante, Constitución IV); revertido, pasan.
3. **Given** `TCN`, **When** se corrige el signo y la escala, **Then** coseno > 0.999 de la proyección y de las capas conv a la vez; si queda por debajo, se investiga una tercera causa (ver Q6) y no se baja el umbral sin documentar.
4. **Given** `Seq2Seq`, **When** se aplica BPTT completo (Q2), **Then** el oráculo global y el del encoder pasan con coseno > 0.999, sin excepción de umbral.

---

### User Story 2 - N1: `dX` independiente del `lr` (Priority: P1)

`AttentionHead.backward` calcula todos los gradientes con los pesos previos al paso y actualiza al final; `dX` no depende de `lr`.

**Acceptance Scenarios**:

1. **Given** `AttentionHead`, **When** se ejecuta `AttentionHead.dx.oracle.test.ts`, **Then** el defecto ("dX con lr > 0 igual que con lr = 0") pasa como test normal, sin `it.fails`.
2. **Given** `MultiHeadAttention`, `TransformerBlock` y `LayerNorm`, **When** se ejecuta un test equivalente de `dX` con `lr` mayor (0.05, donde N1 daba error hasta 85 %), **Then** pasa; los oráculos de pesos de los tres suben de `lr` 1e-3 a 0.05 y siguen con coseno > 0.999.

---

### User Story 3 - N6: el encoder del `Autoencoder` recibe el gradiente correcto (Priority: P1)

`dL/dz` incluye la derivada de la activación de salida del encoder. El oráculo global pasa con coseno > 0.999 (hoy 0.998 con [4]/[4] y 0.835 sin ocultas) también con una estructura sin capas ocultas.

---

### User Story 4 - Comentarios veraces (Priority: P2)

N3 (`LayerNorm.backwardOne` devuelve `−dL/dx`) y "Returns: MSE loss" de `NetworkTransformer.train` (es cross-entropy) se corrigen (Constitución VI). Los controles de signo y los `it.fails` residuales se eliminan y las cabeceras de los archivos de oráculo dejan de mencionar defectos ya corregidos.

### Edge Cases

- **Adam fijo** en `WeightMatrix`/`BiasVector`: el oráculo usa `withPlainSGD` (Q1 de 002). El comportamiento con `Adam` real no queda cubierto por el coseno; `expectLearns` sí usa el optimizador real.
- **Escala**: el coseno es invariante a la escala (limitación Q9 de 002). La convención de escala se fija por decisión (Q1, Q3) y se comprueba con el `ratio` informativo, no con el coseno.
- **Puntos no diferenciables** (ReLU en `TCN`/FFN): las entradas del oráculo evitan el cero; si el nuevo `lr` mayor cruza un quiebre, se reduce el `lr` del test, no el umbral.
- **`NetworkTransformer` con `mask`**: `count` es el número de posiciones enmascaradas; si `count = 0` no se divide (se mantiene el `if (count > 0)` actual) y la actualización es nula.
- **Teacher forcing en `Seq2Seq`**: la entrada del decodificador son objetivos constantes; no se propaga gradiente a través de ellos.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `NetworkTransformer.train`, `NetworkTransformerRL.train`, `RNN.backward`, `TCN.train` y `Seq2Seq.trainStep` MUST entregar a los optimizadores `−dL/dw` (signo C1). Las pérdidas devueltas no cambian de definición salvo lo dicho en FR-003.
- **FR-002**: `AttentionHead.backward` MUST calcular `dQ`, `dK`, `dV`, `dWq`, `dWk`, `dWv` y `dX` con los pesos previos al paso, y actualizar `Wq/Wk/Wv` al final (N1). Ninguna otra clase de `src/` requiere cambio por N1 (Q4); `MultiHeadAttention`, `TransformerBlock` y `LayerNorm` reciben tests de regresión de `dX` con `lr > 0`.
- **FR-003**: Convención de escala de `NetworkTransformer.train` (propuesta, Q1): el gradiente es el de la pérdida devuelta, la cross-entropy **media** sobre las posiciones enmascaradas; `dLogits` se divide por `count`. El comentario de retorno dice "cross-entropy loss, mean over masked positions".
- **FR-004**: `TCN.train` MUST usar la misma escala en la proyección y en las capas conv, la del gradiente de la pérdida media sobre `T·outputSize` (Q3): `dOut = −2·diff/(T·outputSize)` y `lr` sin dividir (o el equivalente que dé `ratio ≈ 1`).
- **FR-005**: `Seq2Seq.trainStep` MUST calcular el gradiente del encoder por BPTT completo (Q2): `LSTMLayer.backprop` devuelve `{ dh0, dc0 }` (gradiente respecto al estado inicial) y admite un `dc` final opcional; el decodificador pasa `dh0/dc0` al último paso del encoder, escalado para que el factor total sea `1/T_out`. El cambio de tipo de retorno de `LSTMLayer.backprop` (`void` → objeto) es aditivo y se anota en el changelog.
- **FR-006**: `Autoencoder.train` MUST multiplicar `dLdz` por `dfn` de la activación de salida del encoder antes de `encoder.trainWithDeltas` (N6).
- **FR-007**: N3 y el comentario "MSE loss" de `NetworkTransformer.train` MUST corregirse. `LayerNorm.backwardOne` solo cambia en comentarios.
- **FR-008**: Por modelo corregido, MUST quitarse el `it.fails` y borrarse el control de signo; MUST seguir existiendo el test humo y `expectLearns` como tests normales.
- **FR-009**: Un test nuevo o convertido MUST fallar con un error a propósito (mutante: revertir el signo o la actualización tardía), según Constitución IV.
- **FR-010**: Ninguna tarea modifica `package.json`, `package-lock.json`, `src/index.ts` ni el changelog. Cada tarea declara sus `[files]` y solo toca esos.
- **FR-011**: La suite completa MUST quedar en verde y estable: 20 ejecuciones seguidas y 5 con `--sequence.shuffle`.

### Tabla de reparto

| Defecto | Modelo | 003 | Archivos de `src/` |
|---|---|---|---|
| C1 | `NetworkTransformer`, `NetworkTransformerRL`, `RNN` | sí | uno por modelo |
| C1 + escala | `TCN` | sí | `src/TCN.ts` |
| C1 + encoder aproximado (Q3 de 002) | `Seq2Seq` | sí | `src/Seq2Seq.ts`, `src/LSTMLayer.ts` |
| N1 | `AttentionHead` (heredado por MHA, bloque, `NetworkTransformer*`) | sí | `src/AttentionHead.ts` |
| N3 y comentario "MSE loss" | `LayerNorm`, `NetworkTransformer` | sí | solo comentarios |
| N6 | `Autoencoder` | sí | `src/Autoencoder.ts` |
| C2 | `GAN`, `VAE` (más N8: derivada de la sigmoide de salida y factor 2 del MSE) | **004** | |
| I1 | `NetworkN` con `residual` | **004** | |
| N2 | `BatchNorm` (momentum, aproximación stop-gradient) | **004** | |
| N9 | `ContrastiveLearning` | **004** (o 006) | |
| Observación Word2Vec (`dh` con `W2` ya actualizado) | `Word2Vec` | **004** | |
| N4/I5, N7 | `Network` varias salidas, `LogisticRegression`/`SoftmaxRegression` | 005 | |
| N5, `dx` de `Dropout`/`Flatten`/`MaxPool2D`, `dInput` de Conv | | 006 | |

Motivo del corte: C2 y N8 exigen rediseñar el flujo de gradiente a través de un módulo congelado o de una reparametrización (decisiones de diseño, candidatas a Opus según ORCHESTRATION); I1 cambia la semántica de `residual`; las demás no son de signo.

### Cambios observables para el changelog (feature 007/008)

1. `NetworkTransformer.train`, `NetworkTransformerRL.train`, `RNN.backward`, `TCN.train` y `Seq2Seq.trainStep` pasan de subir la pérdida a bajarla. Los `lr` elegidos en las apps con el entrenamiento roto deben reajustarse (008); los pesos guardados por versiones anteriores no son comparables.
2. `NetworkTransformer.train`: el gradiente pasa de la suma sobre posiciones a la media (Q1). Los pasos de embeddings y `LayerNorm` (SGD directo) se reducen en un factor `count`; Adam apenas lo nota. El valor devuelto (CE media) no cambia.
3. `TCN.train`: el paso de las capas conv se reduce en un factor `T` respecto a hoy; el de la proyección no cambia de magnitud.
4. `Seq2Seq.trainStep`: el encoder recibe el gradiente exacto (antes, aproximado y de signo equivocado). `LSTMLayer.backprop` devuelve `{ dh0, dc0 }` en lugar de `void` (compatible hacia atrás).
5. `AttentionHead.backward` (y por herencia `MultiHeadAttention`, `TransformerBlock`, `NetworkTransformer`, `NetworkTransformerRL`): el `dX` devuelto cambia (error `O(lr)` antes). La dinámica de entrenamiento cambia ligeramente.
6. `Autoencoder.train`: el paso del encoder se reduce (≈ 4x con sigmoide); más estable con tasas altas.
7. Comentarios de `LayerNorm.backwardOne` y `NetworkTransformer.train` (sin efecto de ejecución).

## Success Criteria *(mandatory)*

- **SC-001**: Por modelo (`NetworkTransformer`, `NetworkTransformerRL`, `RNN`, `TCN`, `Seq2Seq`, `AttentionHead`, `Autoencoder`) su archivo de oráculo pasa sin ningún `it.fails` y sin control de signo; coseno > 0.999 y `expectLearns` en verde.
- **SC-002**: `rg "it\.fails" tests/oracle` solo devuelve defectos que siguen abiertos fuera de 003 (C2, I1, N7) y cada uno lleva `// AUDIT <id>`. Ningún `AUDIT C1`, `AUDIT N1`, `AUDIT N6` ni `AUDIT Q3` permanece (`Q3` queda cerrado si se acepta Q2; si no, se documenta la excepción con umbral).
- **SC-003**: Un mutante temporal (invertir el signo del delta de cada modelo, o mover `update` antes de `dX` en `AttentionHead`) hace fallar el archivo de oráculo de ese modelo; revertido, pasa (`git diff --quiet -- src`).
- **SC-004**: `ratio = ‖Δw‖/(lr·‖∇‖)` de `NetworkTransformer` y `TCN` ≈ 1 (0.9 a 1.1) con `withPlainSGD`, si el oráculo lo informa; no es criterio de coseno.
- **SC-005**: `npx tsc --noEmit -p .`, `npm run typecheck`, `npm run build` y `npx vitest run` en verde; la suite completa pasa 20 ejecuciones seguidas y 5 con `--sequence.shuffle`.
- **SC-006**: Tras el cierre, ningún test de `tests/oracle/` en `it.fails` pertenece a los modelos de 003.

## Assumptions

- Los worktrees parten de `master` sin `node_modules`: cada tarea se ejecuta tras `git rebase fix/v0.4` y `npm ci`. `spec.md`, `plan.md` y `tasks.md` están commiteados antes de lanzar agentes (lo hace el orquestador).
- Los `lr` actuales de los oráculos de `NetworkTransformer*`, `RNN`, `TCN` y `Seq2Seq` (medidos en 002) siguen siendo válidos tras el cambio de signo; si un `lr` mayor cruza un quiebre de ReLU se reduce el `lr` del test.
- Las pruebas existentes `tests/NetworkTransformer.test.ts` y `tests/NetworkTransformerRL.test.ts` pasan hoy con entrenamiento al revés; podrían depender del signo o de umbrales. Si el cambio las rompe, la tarea del modelo las ajusta (están en sus `[files]`) y lo anota en su informe.
- `LSTMLayer.backprop` solo lo llaman `NetworkLSTM`, `Seq2Seq` y los oráculos (verificar con `rg "backprop\(" src tests` antes de tocarlo).

## Preguntas abiertas

- **Q1 (escala de `NetworkTransformer.train`)**: hoy `dLogits` es la suma sobre posiciones y la pérdida devuelta es la media (ratio ≈ 3). Opciones: (a) dividir `dLogits` por `count` (gradiente de la pérdida devuelta; convención de `RNN`/`TCN`/`Seq2Seq`, que usan `lr/T`), (b) devolver la suma de las CE. **Propongo (a)**: coincide con "la pérdida devuelta es la que se minimiza" y con el resto de la librería. Coste: el paso de embeddings y LayerNorm (SGD directo) baja un factor `count` (p. ej. ~50 celdas vacías en Sudoku); se reajusta en 008. ¿Conforme?
- **Q2 (`Seq2Seq`: BPTT completo o excepción)**: ORCHESTRATION permite documentar una excepción con umbral. **Propongo BPTT completo** (exacto, sin excepción): el trabajo es pequeño (`LSTMLayer.backprop` ya acumula `dh_next/dc_next`; solo hay que devolverlos y aceptar un `dc` final) y evita una excepción a la Constitución II. Requiere tocar `src/LSTMLayer.ts` (aditivo). Si el usuario prefiere no tocar `LSTMLayer`, la alternativa es documentar el umbral medido (coseno 0.996 tras corregir el signo). ¿Conforme con el cambio aditivo?
- **Q3 (escala de `TCN`)**: **propongo** dividir `dOut` por `T` (gradiente de la pérdida media) y usar `lr` sin dividir en proyección y capas; equivale a bajar el paso de las capas conv `T` veces. Alternativa: pasar `lr/T` a `layers[l].backward`. Es la misma cosa matemáticamente; el cambio en `dOut` es más legible. ¿Conforme?
- **Q4 (alcance de N1)**: HANDOFF dice que `MultiHeadAttention` (Wo), el FFN y `LayerNorm.backwardOne` también calculan `dX` tras actualizar. El código lo contradice: los tres leen los pesos antes del `update` (ver "Estado verificado"). **Propongo** corregir solo `AttentionHead`, añadir tests de regresión de `dX` con `lr = 0.05` en `MultiHeadAttention`, `TransformerBlock` y `LayerNorm`, y reconciliar HANDOFF (lo hace el orquestador). Si alguno de esos tests falla, la tarea se escala y se corrige la clase afectada. ¿Conforme?
- **Q5 (`NetworkTransformerRL`: `inputProj` actualizado por posición)**: con el `Adam` por escalar de `WeightMatrix`, `train` hace `seqLen` pasos de Adam por llamada con gradientes parciales; el oráculo con `withPlainSGD` no lo ve. **Propongo** acumular `dInputProj` sobre posiciones y llamar a `update` una vez, dentro de la tarea del modelo. Es un cambio de comportamiento adicional (magnitud del paso de `inputProj`), no de signo. ¿Se incluye en 003 o pasa a 004?
- **Q6 (`TCN`: causas del coseno)**: con solo el signo corregido el coseno queda en +0.77, y la diferencia de escala entre capas explica la dirección pero no está medido que sea la única causa (también cabría una máscara ReLU o una rama residual con `convOut` previo/posterior a la suma). La tarea debe medir el coseno por bloque (proyección, cada capa conv) tras cada corrección. Si tras signo y escala queda < 0.999, es el candidato más probable a escalar a Opus.
- **Q7 (reparto 003/004)**: ¿se confirma la tabla de reparto (GAN, VAE, NetworkN residual, BatchNorm N2, ContrastiveLearning N9 y Word2Vec a 004)?
- **Q8 (candidato a Opus, no lanzado)**: en 003 no propongo Opus de entrada. Candidatas si Sonnet falla dos veces: T005 (`Seq2Seq`, BPTT completo con estado inicial del decodificador y escala `T_in/T_out`) y T004 (`TCN`, si queda coseno < 0.999 tras signo y escala). Para 004: `GAN` y `VAE`. Se pediría permiso al usuario con la evidencia de los dos intentos fallidos.
