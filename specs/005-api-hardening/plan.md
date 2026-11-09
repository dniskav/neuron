# Implementation Plan: api-hardening

**Branch**: `fix/v0.4` | **Date**: 2026-11-08 | **Spec**: `specs/005-api-hardening/spec.md`

## Summary

Cerrar los hallazgos de API y validación de la auditoría (I2, I3, I4, I5, I6, I7) y los pendientes de 002–004 que son de contrato, no de gradiente: N7 y entrenamiento incremental de las regresiones, `trainWithDeltas` sin validar, máscaras de dropout distintas en `inputGradient`, `backprop` recurrente que devuelve ceros en silencio, y la escala ½Σ de `NetworkN.train`/`Network.train` (N10). Regla común: un test que FALLA antes del cambio y PASA después, con mensaje de error claro, y que no altera el comportamiento válido (pruebas golden).

## Technical Context

**Language/Version**: TypeScript ^5, Node 20 (CI) y Node 26 local; se verifica también con `~/.nvm/versions/node/v20.19.0`.
**Testing**: `vitest run` (`include: tests/**/*.test.ts`); helpers de 002 en `tests/helpers/` (`gradientOracle`, `expectGradientMatches`, `expectLearns`, `seedRandom`). Los tests nuevos de validación van en `tests/api/<tema>.test.ts`.
**Constraints**: se edita `src/` solo en los archivos declarados por tarea; `package.json`, `package-lock.json`, `src/index.ts`, el changelog y `tasks.md` son del orquestador; worktrees desde `master` sin `node_modules`; `tasks.md` commiteado antes de lanzar agentes. En macOS `wc -l` imprime espacios: `tr -d ' '`. Scripts temporales con nombres únicos (`t0NN_<tema>_<n>`) en el scratchpad compartido.

## Constitution Check

- **V (fallar con claridad):** núcleo de la feature: `setWeights`, `Trainer`, `trainWithDeltas`, `backprop`, `Network.train` con objetivo de tamaño erróneo, `fromJSON`. La opción `clipValue` que no hace nada se elimina (Q1).
- **II (gradiente verificable):** `Network` con 2 salidas y `Conv1D`/`Conv2D` `dInput` entran en el oráculo; el cambio de máscara de dropout se verifica con diferencias finitas a máscara fija. Sin excepciones previstas.
- **III (entrenar es aprender):** `Network` con varias salidas, regresiones incrementales y `Trainer.restoreBestWeights` llevan prueba de aprendizaje, validada con perturbaciones de `lr`/semilla (lección CI).
- **IV (tests deterministas y que detectan errores):** mutantes que QUITAN o ALTERAN una validación o un término (no solo invertir); 20 + 5 ejecuciones y Node 20.19.
- **VI:** se corrigen la cabecera de `Trainer` (`clipValue`), el comentario "Returns the squared error" de `Network.train`, los de `LinearRegression.oracle` ("no hay entrenamiento incremental") y `Network.oracle` (it.todo I5).
- **VII:** cambios incompatibles aceptados (Q1–Q10); lista en el spec para 007/008.
- **Operación:** nada se publica ni sube a `master`; integración en serie.

## Decisiones de diseño

1. **Validación de longitud en `setWeights`:** `validateArray(weights, esperada, 'Clase.setWeights')` como primera línea, ANTES de resetear máscaras o escribir. La longitud esperada se calcula de la estructura (no con `getWeights().length` salvo en clases compuestas, donde es aceptable porque `setWeights` es poco frecuente). Las clases compuestas validan la longitud total en el nivel exterior.
2. **`ModelSaver`:** `toJSON` → `{ format: 1, length, structure?, weights, timestamp }`; `fromJSON` valida contra el modelo (longitud y, si hay, `structure`) y contra el propio archivo (`length`, valores finitos) y solo después llama a `setWeights`. `Serializable.structure?` opcional.
3. **Copias en `LSTMLayer`/`GRULayer`:** `getWeights` clona matrices y vectores; `setWeights(data)` valida la forma y copia elemento a elemento en los arrays existentes (no reasigna referencias).
4. **`Network.train`:** `target: number | number[]`; normaliza a vector (error si es escalar con `nOutputs > 1`); calcula todos los deltas de salida y los ocultos con los pesos de salida originales (suma sobre salidas); actualiza todas. Devuelve el error cuadrático medio. Con `nOutputs = 1` el orden y número de operaciones es el mismo (trayectoria idéntica).
5. **`Trainer`:** una función privada `_validateConfig(dataset, netExt)` al inicio de `train` reúne todos los errores de Q2; `restoreBestWeights` guarda `getWeights()` (ya copia) cuando mejora la validación y lo restaura al terminar (por `earlyStopping` o por `maxEpochs`).
6. **Padding `same`:** cálculo único `total = k − 1; left = floor(total/2); right = total − left` por eje, usado por `forward`, `getOutputLength`/`outputShape` y el recorte de `backward` (una sola fuente de verdad). Validación de forma de `dOut` al principio de `backward`.
7. **`Value.backward`:** DFS iterativo con pila explícita de `[nodo, iterador de _prev]`; emite el nodo al agotar el iterador (post-orden idéntico al recursivo).
8. **Regresiones:** el número de características sale de `weights` (`Logistic`: `weights.length`; `Softmax`: `weights[0].length` y `biases.length === weights.length`; `Linear`: `weights.length − 1`); los campos privados `_nFeatures`/`_nClasses` se eliminan o quedan como caché derivada, y `train`/`fitGD` solo inicializan si no hay pesos de la forma esperada.
9. **Dropout:** `Dropout` gana la capacidad de reaplicar la máscara guardada (`reuseMask`, sin consumir RNG) y `NetworkN._forwardAll(inputs, training, reuseMasks)` la usa; `inputGradient`/`trainWithDeltas` reciben `reuseMasks = false`. `Autoencoder`, `VAE` y `GAN` hacen el forward de entrenamiento con `predict(x, true)` y pasan `reuseMasks = true`. Con `dropoutRate = 0` el código no se ejecuta distinto (golden).
10. **Escala (Q8-A):** `g = (2/n)·(t − p)` en `NetworkN.train`; `outputDelta` ×`2/nOutputs` en `Network.train`. Se mide primero qué tests existentes se mueven y solo se ajustan `lr`/épocas con justificación, nunca umbrales.
11. **Pruebas golden:** antes de tocar `src/`, el agente captura (script de nombre único en el scratchpad) los pesos/salidas de la base tras N pasos en un caso pequeño y los fija como constantes (tolerancia 1e-9) o escribe en el test una réplica del algoritmo antiguo; ese test debe seguir pasando tras el cambio.

## Orden de ejecución

Máximo 4 agentes a la vez, archivos disjuntos dentro de cada lote, integración en serie después de cada lote.

1. **Lote 1** (todo Haiku, `[P]`): T001 capas básicas (`MatMul`, `LayerNorm`, `BatchNorm`), T002 atención (`AttentionHead`, `MultiHeadAttention`, `TransformerBlock`), T003 transformers completos, T004 recurrentes (copias y `setWeights`).
2. **Lote 2** (`[P]`): T005 `Tape` (Sonnet), T006 `ModelSaver` (Haiku), T007 `Network` multi-salida (Haiku), T008 `Conv1D` (Sonnet).
3. **Lote 3** (`[P]`; T011 requiere T007 integrada): T009 `Conv2D` (Sonnet), T010 `NetworkN` validación (Haiku), T011 `Trainer` (Sonnet), T012 N7 (Haiku).
4. **Lote 4** (`[P]`; T013 requiere T004, T014 requiere T012, T015 requiere T010): T013 backprop recurrente (Haiku), T014 incremental (Sonnet), T015 dropout (Sonnet).
5. **Lote 5** (en serie; requiere T007, T010, T015): T016 escala ½Σ (Sonnet, según Q8).
6. **Lote 6** (cierre, en serie): T017 limpieza → T018 `math-reviewer` → T019 estabilidad (20 + 5 + Node 20.19). Huecos de T018 en el Lote 7 reservado; después espera del CI de GitHub antes de cerrar.
7. Preguntas a resolver antes de lanzar: Q4 → T008/T009; Q5 → T007; Q1–Q3 → T011; Q6 → T012; Q7 → T014; Q9 → T015; Q10 → T013; Q8 → T016; Q11 → T006; Q12 → T004. El Lote 1 solo depende de Q12 (T004); T001–T003 pueden lanzarse ya.
8. Escalado: Haiku falla una vez en el mismo criterio → Sonnet; Sonnet falla dos veces → permiso a Opus (solo T015 es candidata). Registrar escalados en `HANDOFF.md`.

## Project Structure

```text
specs/005-api-hardening/{spec.md,plan.md,tasks.md}
src/{MatMul,LayerNorm,BatchNorm,AttentionHead,MultiHeadAttention,TransformerBlock,NetworkTransformer,NetworkTransformerRL,
     LSTMLayer,GRU,NetworkLSTM,Tape,ModelSaver,Network,NetworkN,Dropout,Conv1D,Conv2D,Trainer,
     LogisticRegression,LinearRegression,Autoencoder,VAE,GAN}.ts
tests/api/*.test.ts                      (nuevos: un archivo por tarea)
tests/{Network,NetworkN,Trainer,ModelSaver,Conv1D,Dropout,Integration}.test.ts   (solo si el cambio los rompe)
tests/oracle/{Network,NetworkN,Conv1D,Conv2D,LogisticRegression,LinearRegression,Autoencoder,VAE,GAN}.oracle.test.ts
```

## Riesgos

- **Q8-A mueve la dinámica de `NetworkN`/`Network`:** los tests de `Integration`, `Trainer` y `NetworkN` entrenan con `lr` de la escala antigua; el `lr` doble puede volverlos inestables. Mitigación: T016 va al final de la cadena `NetworkN`, mide antes de tocar y se puede degradar a opción C (solo documentar) sin deshacer nada.
- **T015 (dropout) toca tres modelos de 004** con oráculos ajustados (VAE con margen 0.348 frente a 0.35 en 'reconstruye mejor que la media'). Con `dropoutRate = 0` no debe cambiar nada; golden obligatorio y suite completa en Node 20.19.
- **Validación de `setWeights` en composiciones:** una clase exterior que valide mal puede rechazar pesos válidos de `NetworkTransformerRL` (parámetros `inputProj`, `outputBias`); las pruebas de ida y vuelta de cada clase y `tests/Integration.test.ts` lo cubren.
- **Atomicidad de `NetworkN.setWeights` con dropout:** validar antes de resetear máscaras; el test existente `'getWeights and setWeights reset dropout masks'` debe seguir pasando.
- **`Trainer` con opciones que lanzan** puede romper código de usuario que dependía de que se ignorasen (cambio aceptado, VII); `tests/Trainer.test.ts` tiene 2–3 tests que cambian y lo declara cada informe.
- **Entrenamiento incremental (Q7):** los oráculos de 002 de `LinearRegression` y `Softmax` cambian de estructura (se simplifican); el riesgo es perder cobertura. Se exige que sigan fallando con los mismos mutantes de 002 (invertir signo, quitar término).
- **Lección CI (002/003/004):** pasar N veces en macOS/Node 26 no prueba estabilidad en ubuntu/Node 20: T019 corre Node 20.19 y la feature no se cierra sin el CI de GitHub en verde.
