# Feature Specification: models-fix

**Feature Branch**: `fix/v0.4` (se trabaja directamente en ella; no se ejecutó `create-new-feature.sh` porque crearía una rama)
**Created**: 2026-11-08
**Status**: Draft
**Input**: `specs/ORCHESTRATION.md` sección "004 models-fix", `specs/HANDOFF.md` (hallazgos N2, N6, N8, N9, N10, "Notas para 003 por modelo", "Cambios observables de 003", "Lección CI"), `specs/000-audit/AUDIT.md` (C2, I1), Constitución II, III, IV, V, VI y VII. Condición previa: las features 002 (oráculo) y 003 (`sign-fix`, ya fusionada en `master` y `fix/v0.4`).

## Estado verificado hoy (leído en `src/` y `tests/oracle/`)

Convención del repositorio: los optimizadores hacen `w + lr·g`, así que las capas reciben `g = −dL/dw`. En `NetworkN`, `trainWithDeltas(inputs, outputDeltas, lr)` espera deltas de la capa de salida YA multiplicados por `dfn` de su activación (`src/NetworkN.ts:121-124`, `_backpropLayers` 182-203). `dfn` recibe el valor POST-activación.

- **`GAN.trainStep`** (`src/GAN.ts:92-147`): el discriminador se actualiza primero con `[1 − dReal]` (real) y `[0 − dFake]` (falso), que es exacto para salida sigmoide (el oráculo de 002 lo confirma, coseno 0.99999996). El generador recibe `gDelta = xFake2.map(() => gError / n)` con `gError = 1 − D(G(z2))` (líneas 130-138): un número positivo uniforme que no pasa por el discriminador (C2) y tampoco incluye `dfn` de la salida del generador. Orden: D se actualiza ANTES de que se calcule el paso de G, y G usa `z2` nuevo y el D ya actualizado. El comentario de las líneas 117-129 ("We cannot backprop through the discriminator directly… approximation") es falso tras la corrección (VI). No existe en `NetworkN` ningún método que devuelva el gradiente respecto a la entrada; `Autoencoder.train` lo reconstruye a mano (`src/Autoencoder.ts:178-201`).
- **`VAE.train`** (`src/VAE.ts:112-147`): (1) el decoder recibe `(x − x̂)/n` (líneas 131-133): falta el factor 2 de `d(MSE)/dx̂` (la pérdida devuelta es `Σ(x−x̂)²/n`) y falta `dfn` de la salida; (2) el encoder recibe solo `[−μ, −0.5(exp(lv) − 1)]` (líneas 140-144): solo el gradiente de KL, sin `dfn` de la salida del encoder y sin el gradiente de la reconstrucción a través de `z = μ + σ·ε` (C2); (3) `reparametrize` no devuelve `ε` (se puede recuperar como `(z − μ)/σ`, o añadirlo a `forward` de forma aditiva). El encoder se construye con `options` compartidas con el decoder (`new NetworkN(encoderStructure, options ?? {})`, línea 57), por lo que sin `options` sus salidas `μ` y `logVar` pasan por **sigmoide** (rango (0,1)): `σ = exp(0.5·logVar) ∈ (1, 1.65)`, es decir, el ruido nunca baja de 1 y `μ` nunca puede ser negativo. `activations.ts` ya exporta `linear` (`dfn = 1`).
- **`NetworkN` con `residual`** (I1, `src/NetworkN.ts:182-203`): `_forwardAll` guarda en `act[l+1]` el valor YA sumado (`f(z) + entrada`). En el backward: (a) `prevAct.dfn(out)` (línea 193) evalúa `dfn` sobre ese valor sumado en vez de sobre `f(z)`; (b) el salto suma `deltas[j]` (delta de la capa `l`, ya con `dfn` de `l`) a `prevDeltas` (línea 196), pero el salto debe llevar el gradiente respecto a la SALIDA de la capa `l` (antes de multiplicar por `dfn`), no el delta. Además, `dropout` se evalúa sobre el mismo valor (efecto análogo, no medido). `train` calcula `outAct.dfn(p)` con `p` sumado si la última capa tiene residual (no cubierto por el oráculo: su estructura `[3,3,3,1]` no tiene residual en la salida). Medido en 002: coseno 0.887.
- **`BatchNorm`** (N2, `src/BatchNorm.ts:41-79`): `forward` actualiza `runningMean/runningVar` con la propia `x` y normaliza con las estadísticas ya actualizadas; `backward` devuelve `dOut·γ/std`, tratando las estadísticas como constantes. `backward` contiene además un bucle vacío (líneas 70-74, código muerto con un comentario que contradice a `trainParams`). Nadie en `src/` usa `BatchNorm` salvo el export. Aquí `momentum` es el factor de RETENCIÓN (`m·viejo + (1−m)·nuevo`), al revés que en PyTorch. Derivación (a verificar con diferencias finitas): con `u = x − media_vieja`, `diff = m·u`, `v' = m·v + (1−m)·m²·u²`, `s = √(v'+ε)`, `d(xNorm)/dx = m/s − (1−m)·m³·u²/s³`, y `dy/dx = γ·d(xNorm)/dx`. Con `m = 1` se reduce a `γ/s` (hoy exacto). `tests/oracle/BatchNorm.momentum.oracle.test.ts` fija la excepción (coseno > 0.9, razón de normas 8-13) y los controles con `momentum = 1`.
- **`ContrastiveLearning`** (N9, `src/ContrastiveLearning.ts`): (a) la cabeza de proyección es `activations: [relu, relu]` (línea 172): con ReLU final y Xavier hay semillas con todas las proyecciones a cero (pérdida `log(2N−1)`, gradiente 0, sin recuperación); la cabeza estándar de SimCLR termina en capa lineal; (b) `trainStep` (241-298) recorre los parámetros y aplica `w += lr·(−grad)` EN SITIO dentro del bucle: el gradiente del parámetro `k` se calcula con los parámetros `< k` ya movidos (Gauss-Seidel), así que no es un paso de SGD sobre el gradiente en el punto inicial; además restaura con `+= eps; −= 2eps; += eps` (deriva de redondeo); (c) `cosineSimilarity` tiene curvatura ~1/‖z‖: con normas pequeñas el gradiente cambia muy rápido y el oráculo solo da > 0.999 para ciertas semillas o `lr` muy bajo (HANDOFF). El test de 002 (`ContrastiveLearning.oracle.test.ts`) evita (a) y (c) con semilla 3 y con `semilla 9 × 3`, temperatura 0.1 y 400 pasos, y su cabecera lo admite. No tiene `it.fails`.
- **`Autoencoder.train`** (N10, `src/Autoencoder.ts:151-221`): `outputDeltas = (x − x̂)·dfn(x̂)` sin `2/d`; `loss = mse(x̂, x)` (media sobre `d`). El paso es `d/2` veces el de la pérdida devuelta (ratio 1.50 con `d = 3`). Con `d = 2` el factor sería 1 y no se vería: el oráculo debe usar `d ≠ 2`. La convención de `NetworkN.train` es la misma (delta `(t−p)·dfn(p)`, devuelve MSE medio; su oráculo usa `L = 0.5·Σ` y comprueba `ratio ≈ 1` contra ese `L`).
- **`Word2Vec`** (`src/Word2Vec.ts:273-281, 317-323`): `_skipgramStep` y `_cbowStep` actualizan `W2[d][j]` en sitio y acumulan `dh[d] += W2[d][j]·err[j]` con la `W2` ya movida (comentario "minor approx"): el paso del embedding queda `dh_real − lr·h[d]·‖err‖²`. Error de segundo orden en `lr`, pero se puede eliminar sin coste calculando `dh` antes de tocar `W2`. El oráculo de 002 pasa (coseno ~1 con `lr` pequeño) y su cabecera registra que skip-gram diverge con `lr = 0.1` (pérdida 0.6 → 18.4) y converge con 0.03; la causa no está medida. Como el término espurio REDUCE el paso, es poco probable que explique la divergencia (hipótesis: inestabilidad de SGD bilineal; a medir).
- Estado de los oráculos: `it.fails` en `GAN.oracle.test.ts` (líneas 70 "oracle" y 115 "learns", `// AUDIT C2`), `VAE.oracle.test.ts` (129 global, 172 "reconstruye", 188 "mu varía"), `NetworkN.oracle.test.ts:77` (I1) más el test de banda 0.8-0.95 (línea 81) que fija el defecto; `LogisticRegression.oracle.test.ts:286,293` (N7, de 005). El oráculo de `VAE` tiene además `KL term alone: encoder step follows -dKL/dw` (línea 157), que dejará de ser cierto cuando el encoder reciba también el gradiente de la reconstrucción.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - `NetworkN` con `residual` entrena en la dirección correcta (I1) (Priority: P1)

El paso de `NetworkN.train` con `residual` (en cualquier capa, también la de salida y con la forma `(i) => boolean`) sigue `−dL/dw` con coseno > 0.999 y `ratio ≈ 1`.

**Why this priority**: la corrección de GAN y VAE añade un método a `NetworkN` (US2); ambas dependen de que `NetworkN` sea correcto.

**Independent Test**: `npx vitest run tests/oracle/NetworkN.oracle.test.ts` sin `it.fails` ni el test de banda 0.8-0.95.

**Acceptance Scenarios**:
1. **Given** `[3,3,3,1]` con `residual: true`, **When** se ejecuta el oráculo, **Then** coseno > 0.999 y `ratio` en [0.99, 1.01] (test normal).
2. **Given** residual en la capa de salida (`[3,3,3,3]`) y con `residual: (i) => i === 1`, **Then** igual.
3. **Given** el código corregido, **When** se reintroduce cualquiera de las tres causas (dfn sobre el valor sumado, salto que suma el delta, salto omitido), **Then** el oráculo falla (mutantes que QUITAN un término).

### User Story 2 - Gradiente respecto a la entrada de `NetworkN` sin tocar pesos (Priority: P1)

Un método aditivo de `NetworkN` devuelve `−dL/dx` (convención del repositorio) dados los deltas de la capa de salida, sin modificar ningún peso. Lo usan GAN y VAE.

**Acceptance Scenarios**:
1. **Given** una red con y sin residual, **When** se llama al método, **Then** coincide con diferencias finitas (`|a − b| < 1e-6` por componente, no solo coseno) y `getWeights()` es idéntico antes y después.

### User Story 3 - El generador del GAN aprende a través del discriminador congelado (C2) (Priority: P1)

El paso de G sigue `−dL/dw` de `L_G = −log D(G(z))` con D congelado; el generador se acerca a la distribución objetivo.

**Acceptance Scenarios**:
1. **Given** el oráculo de 002 (`dFrozen`, salida de dimensión 3), **When** corre, **Then** coseno > 0.999 y `ratio` en [0.9, 1.1] (el factor `(1−D)` es un escalar global que el coseno no ve: Q9 de 002).
2. **Given** `trainStep`, **When** se compara con una réplica del paso de D hecha con la API pública (mismas `z`, mismo RNG), **Then** los pesos de D tras `trainStep` son idénticos (`toEqual`): el paso de G no los modifica.
3. **Given** el test "learns" (hoy `it.fails`), **When** se ejecuta con 3 semillas, **Then** la brecha a la media objetivo baja de la mitad (como test normal) y se mantiene bajo `0.7 × criterio` en todas las variantes de `lr` ±30 % y de semilla (Lección CI).
4. **Given** salida de dimensión 1, **Then** NO se usa como único caso (trampa de 002): el oráculo conserva dimensión ≥ 3.

### User Story 4 - El VAE entrena el encoder con la reconstrucción (C2 + N8) (Priority: P1)

El encoder recibe `−dMSE/dμ` y `−dMSE/dlogVar` a través de `z = μ + σ·ε` además de KL, y los deltas de decoder y encoder llevan `dfn` de su salida y el factor 2.

**Acceptance Scenarios**:
1. **Given** el oráculo de 002, **When** corre, **Then** el coseno global y el de los bloques `encoder (all)`, `μ`, `logVar` y `decoder` superan 0.999 como tests normales y `ratio` en [0.9, 1.1].
2. **Given** los tests de aprendizaje (hoy 2 `it.fails`), **When** corren, **Then** la reconstrucción baja a < 0.5 × la del predictor de la media y `μ` depende de la entrada según el criterio de Q5.
3. **Given** el código corregido, **When** se quita cualquiera de los términos (factor 2, `dfn` del decoder, `dMSE/dz` hacia `μ`, el término `0.5·σ·ε` hacia `logVar`, `dfn` de la salida del encoder), **Then** el oráculo falla.

### User Story 5 - N2: `BatchNorm.backward` con momentum < 1 (Priority: P2)

`backward` devuelve el gradiente exacto de `forward` (incluida la dependencia de las estadísticas); el test de excepción desaparece (Q6).

**Acceptance Scenarios**: para `m ∈ {0, 0.1, 0.5, 0.9, 1}` el resultado coincide con diferencias finitas por componente (`1e-6`), incluido un caso donde el término de la varianza supere el 5 % del total; el control `momentum = 1` se conserva (ahora es un caso más).

### User Story 6 - N9: `ContrastiveLearning` robusto a la inicialización (Priority: P2)

Cabeza de proyección con salida lineal y paso simultáneo; el oráculo y "aprende" pasan para semillas de inicialización crudas, sin inicializaciones elegidas a mano.

**Acceptance Scenarios**:
1. **Given** las semillas 1..20 sin escalar ni elegir, **When** corre `computeLoss` inicial, **Then** ninguna deja la proyección entera a cero (pérdida ≠ `log(2N−1)` por esa causa) y el oráculo da coseno > 0.999 (si alguna semilla falla por (c), ver Q7).
2. **Given** `trainStep` con `lr` grande (donde la actualización en sitio es visible), **Then** el paso coincide con el gradiente por diferencias finitas calculado en el punto inicial (coseno > 0.999, `ratio` en [0.9, 1.1]).
3. **Given** "learns", **Then** `final/inicial < 0.35` (criterio 0.5, margen ≥ 30 %) en todas las variantes (20 semillas de init, `lr` ×0.75/×1.25, ruido 1e-9).

### User Story 7 - N10 y Word2Vec (Priority: P2)

N10: el paso del `Autoencoder` es el gradiente de la pérdida que devuelve (`ratio ≈ 1`; Q8). Word2Vec: `dh` se calcula con `W2` previa al paso.

**Acceptance Scenarios**: (N10) con `d = 3` y `d = 5`, `ratio` en [0.9, 1.1] con SGD plano y el oráculo sigue > 0.999. (Word2Vec) un paso suelto de `_skipgramStep` y de `_cbowStep` con `lr = 0.5` coincide EXACTAMENTE (`1e-9`) con `−lr·∇` de la pérdida en el punto previo, tanto en embeddings como en `W2`.

### Edge Cases

- **Salida de dimensión 1 en GAN**: el coseno sale +1 por casualidad con un delta uniforme; el oráculo conserva dimensión 3 y añade dimensión 2.
- **Orden D/G en `trainStep`**: D se actualiza antes que G y G usa el D actualizado; el oráculo congela D en `dFrozen` (diferencia `O(lr)`, `lr = 1e-3`). Se acepta: es el orden estándar (Goodfellow, algoritmo 1). Si el coseno < 0.999 solo por esto, se mide y se pide al orquestador (no se baja el umbral).
- **Salida del discriminador no sigmoide**: `trainStep` ya supone sigmoide (`[1 − dReal]`). Se documenta; no se valida (Q10).
- **`logVar` lineal sin límite**: `exp(logVar)` puede desbordar en KL y en `σ`. Si el entrenamiento diverge con las semillas de prueba se acota (p. ej. `[-10, 10]`) y se documenta; no se acota de entrada.
- **Colapso posterior del VAE**: ver Q4. Con `L = MSE_medio + KL_suma` y datos en [0,1] la ganancia máxima de MSE por nat de información es ≈ 2·Var/n ≈ 0.05-0.17, menor que el coste 1 de KL: el óptimo de ese objetivo es `μ` constante. Es un argumento teórico (cota de tasa-distorsión), sin medir: la primera acción de la tarea es medirlo.
- **Residual en la capa de salida con `trainWithDeltas`**: el salto necesita `dL/dsalida` y `trainWithDeltas` solo recibe el delta ya multiplicado por `dfn` (Q2).
- **`d = 2` en N10** esconde el factor `d/2`: el oráculo usa `d = 3` y `d = 5`.
- **Dropout × residual** en `NetworkN`: `dropout.backward` y `dfn` se evalúan sobre el valor escalado; la reescritura de `_backpropLayers` debe guardar la salida de `f` antes de residual y dropout. No hay oráculo de dropout (Q2).
- **Puntos no diferenciables** (ReLU en `ContrastiveLearning`): las entradas del oráculo evitan el cero; se reduce el `lr` del test, no el umbral.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001** (I1): `NetworkN._forwardAll`/`_backpropLayers`/`train` MUST evaluar `dfn` sobre la salida de la capa antes del salto y antes de dropout, y propagar el salto con el gradiente respecto a la salida de la capa. Cubre cualquier capa con residual, incluida la última en `train`.
- **FR-002** (Q1, Q2): `NetworkN` MUST ofrecer un método público aditivo (propuesta: `inputGradient(inputs, outputDeltas): number[]`) con el mismo contrato de deltas que `trainWithDeltas`, que devuelve `−dL/dx` y NO modifica pesos. Si la última capa tiene residual, `trainWithDeltas` e `inputGradient` MUST lanzar un error explícito (Q2).
- **FR-003** (C2 GAN): `GAN.trainStep` MUST calcular `gDelta[i] = inputGradient_D(xFake2, [dfn_D(D)/D])[i] · dfn_G(xFake2[i])` (para sigmoide, `(1 − D)·∂s/∂x̂`) y llamar a `generator.trainWithDeltas(z2, gDelta, lr)`; el paso de G MUST NOT modificar D. Se corrigen los comentarios de cabecera y de las líneas 117-129 (VI).
- **FR-004** (C2 + N8 VAE): `VAE.train` MUST usar `ε` del forward, calcular el delta del decoder como `2·(x − x̂)/n · dfn(x̂)`, obtener `−dMSE/dz` con `inputGradient` del decoder ANTES de actualizarlo, y entregar al encoder los deltas `[(−dMSE/dμ − dKL/dμ)·dfn(μ_out), (−dMSE/dlogVar − dKL/dlogVar)·dfn(lv_out)]` con `−dMSE/dμ = gz` y `−dMSE/dlogVar = gz · 0.5·σ·ε`. Las salidas `μ` y `logVar` del encoder MUST ser no acotadas (capa lineal; Q3). Si Q4 se acepta, el peso de KL es una opción con valor por defecto 1.
- **FR-005** (N2): `BatchNorm.backward` MUST devolver la derivada exacta de `forward` respecto a `x` (fórmula de la sección anterior), guardando en `forward` lo necesario (`x − media_vieja`); se elimina el bucle muerto; se documenta que `momentum` es retención (Q6).
- **FR-006** (N9): la cabeza de proyección MUST terminar en activación lineal; `trainStep` MUST calcular todo el gradiente por diferencias finitas en el punto inicial y aplicarlo después, restaurando cada parámetro a su valor exacto guardado; la limitación (c) se documenta en el código y en el spec (Q7).
- **FR-007** (N10): `Autoencoder.train` MUST multiplicar `outputDeltas` por `2/d` (Q8, si se acepta).
- **FR-008** (Word2Vec): `_skipgramStep` y `_cbowStep` MUST calcular `dh` antes de actualizar `W2`. Se mide si skip-gram sigue divergiendo con `lr = 0.1`; si sigue, es una limitación documentada (comentario en `src` y en el test), no un defecto de 004.
- **FR-009**: Cada corrección MUST quitar su `it.fails` y todo test de excepción o control que solo valga mientras el defecto exista (banda 0.8-0.95 de `NetworkN`, `momentum` N2, `KL term alone` de VAE si deja de ser cierto), y conservar el humo y "learns" como tests normales. Tras 004 solo permanecen los `it.fails` de N7.
- **FR-010**: Un test nuevo o convertido MUST fallar con un mutante que QUITE un término (no solo uno que invierta el signo): la tarea los lista y comprueba. Las constantes globales invisibles al coseno (`(1−D)`, factor 2, `2/d`) se cubren con `ratio` o con comparación exacta.
- **FR-011**: Los tests de aprendizaje sensibles (GAN, VAE, ContrastiveLearning) MUST validarse con perturbaciones de `lr` y semilla: el peor caso medido debe quedar por debajo del `0.7 ×` del umbral del criterio (margen ≥ 30 %); las cifras se anotan en la cabecera del test.
- **FR-012**: Ninguna tarea modifica `package.json`, `package-lock.json`, `src/index.ts` ni el changelog; cada tarea toca solo sus `[files]`.
- **FR-013**: La suite MUST quedar en verde y estable: 20 ejecuciones seguidas, 5 con `--sequence.shuffle`, y la suite completa bajo Node 20.19; además el CI de GitHub (ubuntu, Node 20) en verde antes de cerrar la feature.

### Tabla de reparto (propuesta)

| Defecto | Modelo | 004 | Archivos de `src/` |
|---|---|---|---|
| I1 | `NetworkN` residual | sí | `src/NetworkN.ts` |
| método aditivo | `NetworkN.inputGradient` | sí | `src/NetworkN.ts` |
| C2 | `GAN` | sí | `src/GAN.ts` |
| C2 + N8 | `VAE` | sí | `src/VAE.ts` |
| N2 | `BatchNorm` | sí (backward exacto) | `src/BatchNorm.ts` |
| N9 (a), (b) | `ContrastiveLearning` | sí (defectos) | `src/ContrastiveLearning.ts` |
| N9 (c) | `ContrastiveLearning` | limitación documentada | comentarios |
| N10 | `Autoencoder` | sí, si Q8 | `src/Autoencoder.ts` |
| `dh` con `W2` movida | `Word2Vec` | sí (corrección pequeña); divergencia: medir y documentar | `src/Word2Vec.ts` |
| `NetworkN.train`/`Network` devuelven MSE y entrenan `½Σ` | | aplazado a 005 (Q8) | |
| Dropout sin oráculo de `dx`, `Conv` `dInput` | | 006 | |
| N7, N4/I5 | | 005 | |

### Cambios observables para el changelog (feature 007/008)

1. `NetworkN` con `residual`: el gradiente cambia (coseno 0.887 antes): las redes residuales entrenan distinto. Si la capa de salida tiene residual, `trainWithDeltas` lanza (Q2).
2. `NetworkN.inputGradient` (nuevo, aditivo).
3. `GAN.trainStep`: el generador pasa de subir la brecha a bajarla; el `lr` del generador y las apps que lo usan se reajustan. El método `generate`/`discriminate` no cambian.
4. `VAE`: (a) la salida del encoder pasa de sigmoide a lineal (`μ` y `logVar` ya no están en (0,1)); (b) el decoder da un paso `2·dfn(x̂)` veces el anterior (≈0.4× con sigmoide: un `lr` guardado aprende más despacio); (c) el encoder aprende de la reconstrucción; (d) opción de peso de KL (Q4, aditiva). Los pesos guardados no son comparables.
5. `BatchNorm.backward` con `momentum < 1`: el `dX` devuelto baja ≈ `m` veces (≈10× con 0.1); las capas anteriores reciben menos gradiente.
6. `ContrastiveLearning`: la cabeza `project()` puede devolver valores negativos; el paso es el gradiente simultáneo (antes Gauss-Seidel); las trayectorias cambian con la misma semilla.
7. `Autoencoder.train` (si Q8): el paso baja `d/2` veces; para conservar la dinámica anterior, multiplicar `lr` por `d/2`.
8. `Word2Vec`: el paso de los embeddings gana el término `lr·h·‖err‖²` que antes faltaba (efecto de segundo orden).

## Success Criteria *(mandatory)*

- **SC-001**: `rg "it\.fails" tests/oracle` solo devuelve los dos de `LogisticRegression.oracle.test.ts` (`// AUDIT N7`); ningún `AUDIT C2`, `AUDIT I1`, `AUDIT N2`, `AUDIT N8`, `AUDIT N9` ni `AUDIT N10` queda como defecto abierto.
- **SC-002**: Por modelo (`NetworkN`, `GAN`, `VAE`, `BatchNorm`, `ContrastiveLearning`, `Autoencoder`, `Word2Vec`) su oráculo pasa con coseno > 0.999 y `ratio` en [0.9, 1.1] donde la escala es convención de la tarea.
- **SC-003**: Todos los mutantes listados en `tasks.md` (que quitan un término) hacen fallar el archivo de oráculo correspondiente; revertidos, `git diff --quiet -- src`.
- **SC-004**: Los tests de aprendizaje de GAN, VAE y ContrastiveLearning mantienen margen ≥ 30 % bajo las perturbaciones de FR-011.
- **SC-005**: `npx tsc --noEmit -p .`, `npm run typecheck`, `npm run build` y `npx vitest run` en verde; 20 ejecuciones seguidas, 5 con `--sequence.shuffle` y la suite completa bajo Node 20.19 (`~/.nvm/versions/node/v20.19.0`) sin fallos; CI de GitHub en verde antes de dar la feature por cerrada.
- **SC-006**: La revisión `math-reviewer` no deja bloqueantes.

## Criterio de escalado a Opus (no se lanza nada)

Esta feature es de Sonnet (`math-dev`). GAN y VAE son las candidatas a Opus porque exigen un diseño de flujo de gradiente y de dinámica adversarial/variacional, no un arreglo local. Se pide permiso al usuario ANTES de usar Opus, explicando la tarea, qué falló en cada intento, por qué Opus lo resolvería y la alternativa. Se considera tras **dos intentos fallidos de Sonnet con medición por bloque**, donde fallido significa: el coseno del bloque (paso de G; decoder; encoder μ; encoder logVar) sigue < 0.999 tras corregir las causas identificadas, o los bloques pasan pero "learns" no cumple el margen de FR-011 y la causa no se identifica con la medición (p. ej. GAN: coseno > 0.999 pero la brecha no baja con ninguna combinación de `lr` ±30 %). Cada informe de intento trae: coseno y `ratio` por bloque, curva de pérdida a 4 puntos y el mutante que se probó. ContrastiveLearning (c) y VAE-colapso posterior (Q4) NO son motivo de Opus: son decisiones de diseño del usuario.

## Assumptions

- Los worktrees parten de `master` sin `node_modules`: cada tarea se ejecuta tras `git rebase fix/v0.4` y `npm ci`; `spec.md`, `plan.md` y `tasks.md` están commiteados antes de lanzar agentes (lo hace el orquestador).
- `NetworkN` solo lo importan modelos de `src/` que no usan `residual`: `GAN`, `VAE`, `Autoencoder`, `ContrastiveLearning`, `Network`, `TCN`, `Flatten`, `EarlyStopping`, `WeightInspector`, `NeuronN`, `optimizers`. Se comprueba con `rg "residual" src tests` antes de tocar (hoy solo `NetworkN.ts` y `tests/NetworkN.test.ts`).
- `tests/NetworkN.test.ts` (líneas 76-105, `residual trains and loss decreases`) y `tests/Integration.test.ts` pueden depender del comportamiento antiguo; la tarea los ajusta con justificación.
- `Math.random` está sembrado por `tests/setup.ts`; GAN y VAE reinician la semilla con `seedRandom` para fijar `z` y `ε` (el orden y número de llamadas a `Math.random` en `trainStep` y `forward` NO debe cambiar, porque el oráculo depende de él).

## Preguntas abiertas

- **Q1 (API nueva en `NetworkN`)**: GAN y VAE necesitan el gradiente respecto a la entrada de una red sin mover pesos; hoy no existe. **Propongo** añadir `NetworkN.inputGradient(inputs, outputDeltas)` (aditivo, mismo contrato de deltas y convención `−dL/dx`) en vez de repetir el recorrido manual de capas de `Autoencoder` en cada modelo. Coste: toca `src/NetworkN.ts` (por eso va tras I1 en serie) y se anota en el changelog. ¿Conforme con el nombre y la API pública?
- **Q2 (residual en la última capa con `trainWithDeltas`)**: con residual en la capa de salida el salto necesita `dL/dsalida`, que `trainWithDeltas` no recibe (solo el delta con `dfn`). `train` sí puede hacerlo bien. **Propongo** que `trainWithDeltas`/`inputGradient` lancen un error explícito en ese caso (Constitución V). Alternativa: aceptar el delta tal cual como gradiente del salto (silenciosamente incorrecto). Aparte: la reescritura guarda la salida pre-residual y pre-dropout, lo que corrige de paso el `dfn` sobre el valor escalado por dropout; sin oráculo de dropout (006). ¿Conforme?
- **Q3 (VAE: salidas de `μ` y `logVar`)**: el encoder de `VAE` usa sigmoide en todas las capas por defecto, así que `σ ≥ 1` y `μ ∈ (0,1)`: con ese rango no puede reconstruir mejor que la media aunque el gradiente sea exacto. **Propongo** que la última capa del encoder sea siempre lineal; si el usuario pasa `options.activations`, se usa para las capas ocultas y para la salida del decoder, y la salida del encoder se fuerza a lineal (se conserva la firma del constructor). Cambio incompatible aceptado (VII). ¿Conforme, o prefieres `encoderOptions`/`decoderOptions` separadas?
- **Q4 (VAE: balance reconstrucción/KL)**: con `L = MSE_medio + KL_suma` y datos en [0,1] el óptimo es el colapso posterior (argumento de cota de tasa-distorsión; pendiente de medir). Entonces "reconstruye mejor que la media" y "μ varía" no son alcanzables sin cambiar el objetivo. **Propongo** medir primero con el peso por defecto; si colapsa, añadir un peso de KL (`klWeight`, β-VAE, por defecto 1: la pérdida definida no cambia) y que los tests de aprendizaje usen un β pequeño (valor medido por la tarea, con margen 30 %). Alternativa: cambiar la reconstrucción a suma de cuadrados (cambia el valor de `reconLoss`). ¿Aceptas `klWeight` como opción aditiva?
- **Q5 (umbral de dispersión de `μ`, 0.05)**: lo puso un agente, no la spec. **Propongo** sustituirlo: la prueba "μ varía con la entrada" pasa a ser causal, sin número mágico: `MSE(x, decode(μ(x)))` es menor que `0.5 × MSE(x, decode(μ(x')))` con `x'` otra entrada del conjunto (el factor 0.5 es el criterio de `expectLearns`). Se conserva la dispersión como dato informativo en el informe. ¿Conforme?
- **Q6 (N2 BatchNorm)**: opciones: (a) `backward` exacto sobre el `forward` actual (**propuesto**: no cambia `forward`, elimina la excepción, el `dX` baja ≈ `m` veces); (b) documentar la excepción con umbral (coseno > 0.9, razón 8-13, lo que fija hoy el test); (c) cambiar `forward` para normalizar con las estadísticas previas a la actualización (cambio mayor de comportamiento, más parecido a BN estándar). (a) cumple la Constitución II sin excepción; (c) es un rediseño y lo aplazo salvo petición. ¿Conforme con (a)?
- **Q7 (N9: qué es defecto y qué limitación)**: **Propongo** (a) cabeza lineal y (b) paso simultáneo como defectos corregidos; (c) curvatura 1/‖z‖ como limitación documentada de la similitud coseno (no se arregla; se mide cuántas de las semillas 1..20 pasan el oráculo tras (a) y (b) y se informa; si alguna falla solo por (c) se decide con el usuario si se documenta un `lr` máximo). No se sustituyen las diferencias finitas por retropropagación analítica en 004 (trabajo grande, sería su propia feature). ¿Conforme?
- **Q8 (N10 `Autoencoder` y convención de escala)**: opción A (**propuesta**): `outputDeltas · 2/d`, de modo que el paso es el gradiente de la pérdida que devuelve `train` (misma regla que 003 en `NetworkTransformer`/`TCN`); coste: el paso baja `d/2` veces (con `d = 784`, ≈392×) y las apps deben subir `lr`. Opción B: dejar el paso y devolver `½Σ(x − x̂)²` (cambia el número que ve el usuario, no la dinámica). Opción C: solo documentar. Nota: `NetworkN.train`/`Network.train` tienen la misma inconsistencia (entrenan `½Σ`, devuelven MSE); **propongo aplazarla a 005** (documentar o alinear) para no mezclar. ¿A, B o C, y se aplaza `NetworkN.train`?
- **Q9 (Word2Vec en 004)**: **Propongo incluirlo**: la corrección de `dh` son dos líneas por método, el mismo patrón que N1 y N1-LayerNorm de 003. La divergencia de skip-gram con `lr = 0.1` se mide tras la corrección; si persiste, queda como limitación documentada (no cambia el algoritmo). ¿Conforme?
- **Q10 (GAN: contrato del discriminador)**: `trainStep` supone salida sigmoide en D (`[1 − dReal]`, `[−dFake]`). **Propongo** documentarlo y no validarlo (un D con otra salida hace que `log` pierda sentido y es uso fuera de contrato), y usar `dfn_D(D)/D` en el paso de G para que sea exacto con cualquier salida en (0,1). Orden de actualización: se conserva (D primero, luego G con el D actualizado y `z2` nuevo); el oráculo lo aproxima congelando D (`O(lr)`). ¿Conforme?
- **Q11 (candidatos a Opus, no lanzado)**: ver "Criterio de escalado". Se pregunta al usuario solo si se cumple el criterio; no antes.
