# Implementation Plan: fast-core

**Rama**: `feat/0.5-core` (desde `origin/master`, 0.4.0) | **Fecha**: 2026-11-18 | **Spec**: `spec.md` | **Contratos**: `contracts.md`

## Resumen

Una clase rápida (`NetworkFast`) sobre un núcleo de `Float64Array` con backend inyectable, optimizadores vectoriales, lotes, semilla inyectable, formato de guardado 2 y una entrada solo de inferencia. Las clases educativas no se reescriben: ganan memoria por dentro (estado del optimizador en buffers) y sirven de oráculo exacto.

## Contexto técnico

- TypeScript ^5 estricto, Node ≥ 20 (CI 20, local 26). Sin dependencias de runtime.
- Tests: vitest. Utilidades de `tests/helpers/` (`gradientOracle`, `expectLearns`, `seedRandom`, `withSeed`, `mulberry32`).
- Build: tsup. La entrada `infer` exige cambiar el script `build` y `exports` de `package.json` (archivos del orquestador, T019).
- Benchmarks: `bench/run.mjs` en Node puro sobre `dist/`. Sin `tsx` ni otras herramientas nuevas.

## Comprobación de la Constitución

- **I:** sin dependencias. `bench/` no añade ninguna.
- **II:** `NetworkFast` tiene oráculo de diferencias finitas, además de la equivalencia con `NetworkN`. El gradiente del lote se compara con la media de los gradientes por ejemplo.
- **III:** `NetworkFast` aprende XOR y una regresión pequeña (`expectLearns`), con `train` y con `trainBatch`.
- **IV:** cada tarea declara sus mutantes. Las equivalencias exactas (`Object.is`) matan casi cualquier mutante de fórmula.
- **V:** `Trainer` con `batchSize > 1` sobre un modelo sin `trainBatch` lanza error. Guardar una activación sin nombre, también.
- **VI:** `NetworkFast` lleva comentarios que remiten a `NetworkN` para la explicación matemática y explican solo el diseño de buffers.
- **VII:** los cambios observables están en la sección 7 de la spec y van al changelog.
- **Operación:** pendiente de Q2. La Constitución nombra `fix/v0.4` como rama de integración. Hay que enmendarla con aprobación del usuario antes del Lote 1.

## Decisiones de diseño

1. **Oráculo por equivalencia.** Lo más barato y fuerte para probar `NetworkFast` es correr las dos clases con la misma semilla y comparar los pesos tras K pasos. Matriz mínima de casos: estructuras `[2,3,1]`, `[4,8,8,3]`, `[5,5,5,5]` (para residual); activaciones sigmoid, relu, tanh, linear, leakyRelu y elu; optimizadores SGD, Momentum, Adam y Clip(Adam); `residual` true, false y función; `dropoutRate` 0 y 0.3 (misma semilla, mismo orden de sorteo). Ayudante compartido en `tests/helpers/equivalence.ts` (T008).
2. **Orden de sorteo del dropout.** `NetworkFast` debe consumir `random()` en el mismo orden que `Dropout.forward`. Se reutiliza la clase `Dropout`, que ya existe, sobre `number[]` en el camino de un ejemplo. En el de lote, una máscara por ejemplo, en orden de ejemplo.
3. **Optimizador por capa.** Un `VecOptimizer` para `W` y otro para `b` en cada capa. Como el estado de cada elemento es independiente, el resultado es el mismo que con un objeto por escalar.
4. **Memoria en `NeuronN`.** `_opts: Optimizer[]` se sustituye por `_opt: VecOptimizer` de tamaño `nInputs + 1` (pesos más sesgo), creado con `vecFromFactory`. `_update` escribe en sitio en `this.weights` y después en el sesgo. Como `bias` es un número y no un array, el estado del sesgo va en un `VecOptimizer` aparte de tamaño 1 o se usa un buffer temporal de 1 elemento: la tarea elige y lo documenta.
5. **Backend.** `getBackend()` se lee **una vez por llamada pública** (al principio de `train`, `predict`…), no por operación, para no pagar la indirección en los bucles internos. La opción por instancia tiene prioridad (Q8).
6. **Formato 2.** El validador puro vive en `src/core/format2.ts`, sin importar ninguna clase. `ModelSaver` y `infer` lo comparten. El registro de clases está en `ModelSaver`; cada clase se registra en su propio archivo con un efecto al importarse. Ojo: el paquete declara `sideEffects: false`, así que el registro se hace en `src/index.ts` (orquestador), no en cada archivo.
7. **Benchmarks.** Cada medida se repite 5 veces y se informa la mediana. El `verifier` corre la línea base y la nueva en la misma máquina y en la misma sesión (compila `master` en un worktree aparte), para que la comparación sea justa.

## Orden de ejecución y paralelismo

```
Lote 0 (orquestador): Q1–Q8, rama, enmienda de la Constitución, commit de specs/010
Lote 1 (4 en paralelo):  T001 bench+línea base [H] | T002 random.ts [H] | T003 core/optim [S] | T004 core/tensor+backend [H]
Lote 2 (4 en paralelo):  T005 checkBackend [H] | T006 sustituir Math.random A [H] | T007 sustituir Math.random B [H] | T008 ayudante de equivalencia + nombres de activaciones [H]
Lote 3:                  T009 NeuronN memoria [S] | T010 MatMul memoria [H] | T011 NetworkFast un ejemplo [S]
Lote 4:                  T012 trainBatch/predictBatch [S] → T013 Trainer batchSize [H]
Lote 5 (formato 2):      T014 núcleo format2 + NetworkN/Network [S] → T015 NetworkFast [H] | T016 NetworkLSTM [H] | T017 Transformer y RL [H]
Lote 6:                  T018 infer.ts [S] → T019 build/exports (orquestador) | T027 embeddings propios [H] (tras T017: mismo archivo)
Lote 7 (opcional, [H] en paralelo): T020 LSTM+GRU | T021 Conv1D/Conv2D/TCN | T022 RNN+Seq2Seq (memoria de optimizadores)
Lote 8 (cierre):         T023 docs [H] → T025 revisión matemática [S] → T026 verificación y bench [H]
Opcional (Q7):           T024 informe del autograd vectorizado [S]
```

## Selección de modelo (prioridad del usuario: Haiku, luego Sonnet, Opus en último caso)

| Tipo de tarea | Modelo | Por qué |
|---|---|---|
| Firmas cerradas en `contracts.md`, criterio de igualdad exacta | Haiku | La equivalencia exacta hace que un error no pueda pasar el gate. |
| Sustituciones mecánicas (`Math.random`) | Haiku | El gate es `grep` más la suite completa sin cambios. |
| Docs, bench, verificación | Haiku | Sin diseño. |
| Diseño de buffers con varios contratos a la vez (`NetworkFast`, lotes, `NeuronN`, format 2 base, `infer`) | Sonnet | Varias invariantes a la vez; un error sutil cuesta más que el modelo. |
| Revisión matemática | Sonnet (`math-reviewer`) | |
| Opus | Solo con permiso | Si Sonnet falla dos veces en el mismo criterio. Candidatos: T011 y T012. |

Escalado: Haiku falla una vez → Sonnet. Sonnet falla dos veces → pedir permiso para Opus, explicando la tarea, qué falló y por qué. Si más del 30 % de un tipo de tarea escala en el primer lote, ese tipo pasa a Sonnet desde el siguiente.

## Riesgos

- **`NeuronN` en sitio (Q3):** `VAE`, `GAN` y `NetworkLSTM` leen `n.weights` durante el backward. Hoy `_update` crea un array nuevo, y escribir en sitio podría cambiar un resultado si alguien lee los pesos después de actualizarlos dentro del mismo paso. Mitigación: la suite completa sin cambios y los oráculos de `GAN`, `VAE` y `NetworkLSTM` en el gate de T009.
- **Sumas en otro orden:** la igualdad bit a bit no siempre es posible. Por eso el contrato pide ≤ 1e-12 relativo en `NetworkFast` y exactitud solo en los optimizadores.
- **Orden de `random()` con dropout en lote:** documentado en el contrato; el test de equivalencia con lote de 1 lo cubre.
- **`sideEffects: false` y el registro de clases:** si el registro está en el archivo de cada clase, el tree-shaking lo elimina. Se hace en `src/index.ts`.
- **Benchmarks ruidosos:** mediana de 5 y misma sesión. Los objetivos tienen margen (×5 frente a lo que se espera, ×10–20).

## Estructura

```text
specs/010-fast-core/{spec.md,plan.md,contracts.md,tasks.md,HANDOFF.md,bench-baseline.json}
bench/run.mjs
src/random.ts
src/core/{tensor.ts,backend.ts,optim.ts,format2.ts}
src/NetworkFast.ts
src/infer.ts
src/{NeuronN,MatMul,activations,ModelSaver,Trainer,NetworkN,Network,NetworkLSTM,NetworkTransformer,NetworkTransformerRL}.ts (cambios acotados)
tests/{random,core.tensor,core.backend,core.optim,NetworkFast,infer,ModelSaver.format2}.test.ts
tests/oracle/NetworkFast.oracle.test.ts
tests/helpers/equivalence.ts
```
