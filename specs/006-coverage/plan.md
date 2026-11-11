# Implementation Plan: coverage

**Branch**: `fix/v0.4` | **Date**: 2026-11-09 | **Spec**: `specs/006-coverage/spec.md`

## Summary

Dar un archivo de test con valores calculados a mano y comprobación de mutante a cada módulo de `src/` que hoy no tiene (15 a 0 %) o solo tiene oráculo (17 con ramas < 90 %), añadir los oráculos de `dx` que faltan (`Dropout`, `Flatten`, `MaxPool2D`), decidir y fijar el caso XOR de `DecisionTree`, validar las entradas del resto de modelos con un helper compartido y cerrar los `getWeights` que desbordan la pila con `push(...fila)`. Casi todo es `tests/`; `src/` solo cambia en las tareas de validación (T033–T043) y de `getWeights` (T034, T035), con archivos declarados.

## Technical Context

**Language/Version**: TypeScript ^5, Node 20 (CI) y Node 26 local; se verifica también con `~/.nvm/versions/node/v20.19.0`.
**Testing**: `vitest run` (`include: tests/**/*.test.ts`, `tests/setup.ts`); helpers de 002 en `tests/helpers/` (`seed.ts`: `seedRandom`, `withSeed`; `gradientOracle.ts`; `learns.ts`: `expectLearns`). Cobertura: `@vitest/coverage-v8` ya instalado, script `npm run coverage`; falta `coverage.include` (T001).
**Constraints**: un archivo de test por módulo; tests deterministas; márgenes ≥ 30 % en tests de aprendizaje; `package.json`, `package-lock.json`, `src/index.ts`, el changelog y `tasks.md` son del orquestador; worktrees desde `master` sin `node_modules` (el prompt pide `git rebase fix/v0.4` y `npm ci`); `tasks.md` y la spec se commitean antes de lanzar agentes.

## Constitution Check

- **II (gradiente verificable):** `Dropout`, `Flatten`, `MaxPool2D` reciben oráculo de `dx` (T014–T016); los demás oráculos no cambian.
- **III (entrenar es aprender):** los modelos con archivo nuevo citan su "aprende" de 002–004; solo `TSNE`, `KMeans`, `Word2Vec`, `SOM`, `PCA` añaden una prueba de comportamiento aprendido, con margen ≥ 30 %.
- **IV (tests deterministas y que detectan errores):** FR-004 pide ≥ 3 mutantes por archivo (alterar, no solo invertir); 20 + 5 ejecuciones y Node 20.19; el CI de GitHub arbitra.
- **V (fallar con claridad):** T033–T043 (US5) convierten los casos sondeados en `Error`; lo que se mantiene (p. ej. `Tokenizer.decode` → `<UNK>`) queda decidido en la spec (Q3).
- **VI:** cabecera de `DecisionTree` (T003) y comentarios de `r2Score`/`Metrics` (T036) se corrigen si dejan de describir el comportamiento real.
- **VII:** los cambios incompatibles (nuevos `throw`) se listan en el informe de cada tarea para el changelog de 007.
- **Operación:** nada se publica ni sube a `master`; integración en serie por el orquestador.

## Decisiones de diseño

1. **Orden por riesgo.** Primero lo que cierra decisiones (T003, Q1), luego los 15 módulos de 0 % (T002–T016, T020), después los parciales (T017–T032), y al final las validaciones (T033–T043) que dependen de los tests ya escritos: un `throw` nuevo se prueba contra un archivo de test que ya fija el comportamiento válido (el golden está en el archivo de la tarea anterior).
2. **Un archivo, un módulo.** `tests/<Módulo>.test.ts` en la raíz de `tests/` (como `Conv1D.test.ts`); oráculos en `tests/oracle/<Módulo>.dx.oracle.test.ts`; validación en `tests/api/<tema>.validation.test.ts` (convención de 005). Las tareas de validación NO editan el archivo de la tarea de comportamiento, así `[files: ...]` queda disjunto.
3. **Valores a mano.** Cada `expect` numérico lleva su cálculo en el comentario. Ejemplos de partida (ya comprobables de cabeza): `accuracy([0,1,1,0],[0,1,0,0]) = 3/4`; `confusionMatrix([0,1,1],[0,1,0]) = [[1,0],[1,1]]`; `mae([1,2,3],[2,2,5]) = (1+0+2)/3 = 1`; `rmse` del mismo par `= √(5/3)`; `PositionalEncoding.encode(0, d)` = `[0,1,0,1,…]` (sin(0)=0, cos(0)=1); `DataAugmentation.normalize([[0,10],[5,20],[10,30]])` → `[[0,0],[0.5,0.5],[1,1]]`; `PCA` de puntos sobre la recta `y = x` → primer componente `±[1/√2, 1/√2]` y ratio de varianza 1. El agente verifica cada cifra a mano y la escribe; si una cifra de arriba no coincide con el código, es un hallazgo y se informa (no se ajusta el test).
4. **Mutantes.** Como en 005: el agente aplica el mutante en su worktree (nunca en el directorio principal), comprueba que el archivo falla, restaura con `git checkout -- src` y repite el gate. Lista mínima por módulo en cada tarea; el informe dice cuál usó.
5. **Estocásticos.** `seedRandom(seed)` en `beforeEach` y `restoreRandom()` en `afterEach` (como `tests/setup.ts` espera); los tests de "aprende" se validan con 5 semillas y `lr` ×0.7/×1.3 y el peor caso queda < 0.7 × el umbral (cifras en la cabecera).
6. **Salida por consola.** `vi.spyOn(console, 'log' | 'warn').mockImplementation(() => {})` y comprobación del contenido; se restaura en `afterEach`.
7. **`DecisionTree` y Q1.** T003 es `[S]` porque decide. Si Q1 = (a): test que fija 0.5 en XOR, control AND y XOR + característica `x1·x2` (acierto 1.0), cabecera en `src/DecisionTree.ts`. Si Q1 = (b): cambio de `<` a `<=` en `_bestSplit` con golden de árboles previos (AND, OR, regresión) antes de tocar `src/`; y el test de XOR afirma 1.0. T003 no se lanza hasta que el usuario responda Q1 (por defecto (a) si no hay respuesta, anotándolo en el informe).
8. **Helper de datos (T033).** `validateDataset` en `src/Validation.ts` reutiliza `validateArray`/`validateNumber` de la misma unidad; `LogisticRegression`, `LinearRegression` y `SoftmaxRegression` conservan sus mensajes (los tests de 005 `tests/api/Regression.*.test.ts` los fijan; se ejecutan en el gate). `KMeans`, `PCA`, `SOM`, `GaussianNaiveBayes`, `DecisionTree` lo adoptan en T038/T039 solo si el mensaje existente no cambia; si cambia, el test existente se actualiza en la misma tarea.
9. **`getWeights` grandes (T034, T035).** Sustituir `w.push(...row)` por `for (const v of row) w.push(v)` (o `Array.prototype.concat`), mismo orden; en los transformers el argumento es `block.getWeights()`. No se introducen utilidades nuevas (un bucle es legible; la librería es educativa). Los tests usan matrices de 200 000 elementos construidas con `new WeightMatrix(1, 200000)` (tamaño medido hoy: lanza `RangeError`).
10. **Umbral y meta-test al final.** T044 (meta-test de módulos) y T045 (umbral en `vitest.config.ts`) van después de todos los lotes; el umbral se calcula del informe real (valor medido − 2 puntos).

## Project Structure

```text
specs/006-coverage/{spec,plan,tasks}.md
vitest.config.ts                        (T001, T045)
tests/<Módulo>.test.ts                  (T002–T032)
tests/oracle/{Dropout,Flatten,MaxPool2D}.dx.oracle.test.ts   (T014–T016)
tests/api/*.validation.test.ts, tests/api/getWeights.large.*.test.ts   (T033–T043)
tests/coverage.modules.test.ts          (T044)
src/Validation.ts + src/<Módulo>.ts     (solo T003 si Q1=(b), T033–T043)
```

## Riesgos

- **Haiku inventa el valor esperado copiando la salida.** Mitigación: el gate pide el mutante y el revisor Sonnet repite una muestra (T046); cada `expect` lleva su derivación.
- **Tests de aprendizaje frágiles en CI** (pasó en 002/004). Mitigación: decisión 5 y Node 20.19; el CI arbitra.
- **Validación nueva rompe tests existentes** de 002–005 (p. ej. un oráculo que pasa filas irregulares). Mitigación: cada tarea de validación ejecuta `tests/oracle` y `tests/api` completos en su gate.
- **Alcance de `Metrics` (Q2).** Cambia valores de retorno degenerados a `throw`: incompatible; se anota para 007.
- **Cobertura por línea no es calidad.** Por eso el criterio es mutante, no porcentaje; el umbral es un seguro contra regresiones, no el objetivo.
