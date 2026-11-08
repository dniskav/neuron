# Implementation Plan: gradient-oracle

**Branch**: `fix/v0.4` | **Date**: 2026-11-08 | **Spec**: `specs/002-gradient-oracle/spec.md`

## Summary

Construir los instrumentos que dicen la verdad sobre el entrenamiento (semilla fija, oráculo de diferencias finitas sobre la actualización real, prueba de "aprende"), aplicarlos a todo modelo y capa entrenable, marcar con `it.fails` los defectos conocidos (C1, C2, I1) y dejar la suite determinista. Solo se escriben archivos en `tests/` (más `vitest.config.mts`); no se toca `src/`.

## Technical Context

**Language/Version**: TypeScript ^5, Node 20
**Testing**: `vitest run` (^4.1.12), `globals: true`, `include: ['tests/**/*.test.ts']` (`vitest.config.mts`)
**Constraints**: sin cambios en `src/`, `package.json`, `package-lock.json`, `src/index.ts` ni changelog; los worktrees parten de `master` sin `node_modules` (el agente hace `git rebase fix/v0.4` y `npm ci`); `tasks.md` y `spec.md` deben estar commiteados antes de lanzar agentes (lo hace el orquestador).

## Constitution Check

- II: el oráculo implementa la comparación contra diferencias finitas con coseno > 0.999 y fija la convención `Δw ∝ −∇L`. Las excepciones (`Perceptron`, `SOM`, `Seq2Seq` si se queda aproximado) quedan documentadas en la spec (Q3).
- III: `expectLearns` con pérdida medida fuera del `train`, razón `< 0.5` por defecto (ajustable por modelo y justificado en el test).
- IV: semilla global; mutante por tarea (FR-008) y verificación de 20 ejecuciones (T039).
- Operación: nadie publica ni sube a `master`; archivos exclusivos del orquestador intactos.

## Decisiones de diseño

1. **Ubicación.** Utilidades en `tests/helpers/` (con su propio `*.test.ts` de autoprueba). Pruebas por modelo en `tests/oracle/<Nombre>.oracle.test.ts`, un archivo por modelo, para que las tareas sean disjuntas y paralelizables. Sin colisión con los `tests/<Modulo>.test.ts` de la feature 006.
2. **`tests/helpers/seed.ts`**: `mulberry32(seed)`, `seedRandom(seed = 0x5eed)` (reemplaza `Math.random`), `restoreRandom()`, `withSeed(seed, fn)`.
3. **`tests/helpers/gradientOracle.ts`** (API propuesta; el autor puede ajustarla y documentarlo en la cabecera del archivo):
   ```ts
   interface OracleOptions {
     getParams(): number[]; setParams(p: number[]): void
     loss(): number                 // determinista; reinicia estado y ruido
     step(lr: number): void         // UNA actualización real del modelo
     lr?: number; eps?: number; maxParams?: number; resetRng?: () => void
   }
   gradientOracle(o): { cosine: number; ratio: number; n: number }
   expectGradientMatches(o, min = 0.999)
   bindParams(...tensors: (number[] | number[][] | number[][][])[]): { getParams, setParams }
   withPlainSGD(fn)                // sustituye Adam.prototype.step por w + lr·g (Q1)
   ```
   Algoritmo: `p0 = getParams()`; gradiente numérico central `g_i = (L(p0+εe_i) − L(p0−εe_i)) / 2ε` en los índices muestreados (todos si ≤ 400, si no paso fijo); restaura `p0`; `step(lr)`; `Δ = getParams() − p0` en los mismos índices; restaura `p0`; `cosine = Δ·(−g) / (‖Δ‖‖g‖)`; `ratio = ‖Δ‖ / (lr·‖g‖)`. Convención de la constitución: los optimizadores reciben `−dL/dw`, así que un modelo correcto da `Δ ∥ −∇L` y cosine ≈ +1. `bindParams` lee y escribe en sitio campos como `RNN.Wxh` o `CausalConv1D.kernels` para modelos sin `getWeights`.
4. **`tests/helpers/learns.ts`**: `expectLearns({ loss, step, steps, ratio = 0.5 })` mide `loss()` antes y después de `steps` llamadas a `step()`, exige finito y `final < ratio·initial`. Los modelos de actualización por muestra pasan un `step` que recorre el mini-conjunto.
5. **Semilla global**: `tests/setup.ts` con `beforeEach(() => seedRandom())` y `afterEach(restoreRandom)`, registrado en `vitest.config.mts` (`setupFiles`). Cada archivo de test corre en su propio worker, así que no hay estado compartido entre archivos.
6. **Estructura de cada archivo de oráculo** (plantilla creada en T005/T006 por Sonnet y copiada por Haiku): 1) test humo (construye, forward finito, entrena un paso con resultados finitos, nº de parámetros > 0); 2) `expectGradientMatches` (con `it.fails` y comentario `// AUDIT C1` si procede); 3) `expectLearns`. Datos mínimos, ε y entradas lejos de puntos no diferenciables.
7. **Mutante (Constitución IV)**: cada tarea de modelo debe, antes de terminar, introducir una mutación temporal (invertir el signo de un gradiente o quitar un término en el `train`/`backward` del modelo), comprobar que el archivo falla, revertir y dejar `git diff --quiet -- src`. La mutación se hace en el propio worktree, sin commit de `src/`.
8. **Meta-test de cobertura (T037)**: lista explícita de modelos entrenables ↔ archivos de `tests/oracle/`, y verificación de que toda clase exportada por `src/index.ts` con método `train`/`trainStep`/`backward`/`backprop`/`fit`/`fitGD`/`trainBatch`/`update`/`trainParams` está en la lista de cubiertos o en una lista de exclusiones con motivo.
9. **Limpieza**: eliminar `tests/GradientCheck.test.ts` y renombrar los dos tests engañosos de `tests/Integration.test.ts` (Q6, Q7).

## Orden de ejecución

1. Lote 1 (utilidades, disjuntas) → Lote 2 (semilla global, depende de T001) → Lote 3 (plantillas, Sonnet) → Lotes 4 a 6 (por modelo; dentro de cada lote, de 4 en 4 como máximo) → Lote 7 (limpieza y meta-test) → Lote 8 (revisión y estabilidad).
2. Escalado: si una tarea `[H]` falla una vez en el mismo criterio, pasa a Sonnet (ORCHESTRATION). Registrar los escalados en `HANDOFF.md`.

## Project Structure

```text
specs/002-gradient-oracle/{spec.md,plan.md,tasks.md}
tests/helpers/{seed,gradientOracle,learns}.ts  + .test.ts de cada una
tests/setup.ts
vitest.config.mts                  (solo setupFiles)
tests/oracle/<Nombre>.oracle.test.ts   (una por modelo; ver tasks.md)
tests/oracle/coverage.test.ts
tests/GradientCheck.test.ts        (se elimina)
tests/Integration.test.ts          (solo dos títulos)
```

## Riesgos

- La semilla global puede destapar tests existentes sensibles a la inicialización: T004 los arregla ajustando datos o semilla, nunca debilitando umbrales sin justificarlo.
- Con `Adam` fijo (Q1) el parche de `Adam.prototype.step` debe restaurarse siempre (`try/finally`) o contaminaría otros tests del mismo archivo.
- Modelos cuyo `train` hace varias actualizaciones por llamada (`GAN.trainStep`: D real, D falso, G por muestra; `LogisticRegression.train`) no cumplen "un paso = un gradiente": se usan lotes de 1 muestra y se marca la ambigüedad si aun así no encaja.
- `ContrastiveLearning` calcula su gradiente por diferencias finitas internas: el oráculo lo pasará de forma casi trivial. Es un resultado aceptable, pero el test debe señalarlo y exigir además que aprenda.
- Un `it.fails` mal diseñado pasa con cualquier excepción (Q8). Mitigado con el test humo.
- Riesgo de coste: ~35 tareas por modelo; las `[H]` son plantillas sobre T005/T006. Si más del 30 % de un tipo escala, pasa a Sonnet.
