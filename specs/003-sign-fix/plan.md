# Implementation Plan: sign-fix

**Branch**: `fix/v0.4` | **Date**: 2026-11-08 | **Spec**: `specs/003-sign-fix/spec.md`

## Summary

Corregir en `src/` los defectos de dirección y escala de gradiente que el oráculo de 002 dejó marcados (`it.fails`): signo C1 en `NetworkTransformer`, `NetworkTransformerRL`, `RNN`, `TCN` y `Seq2Seq`; orden de actualización N1 en `AttentionHead`; N6 en `Autoencoder`; y los comentarios incorrectos (N3, "MSE loss"). Por cada modelo se retira el `it.fails` y se borra el control de signo. GAN, VAE, `NetworkN` residual, BatchNorm, ContrastiveLearning y Word2Vec pasan a 004.

## Technical Context

**Language/Version**: TypeScript ^5, Node 20
**Testing**: `vitest run` (^4.1.12); oráculo de 002 en `tests/helpers/` (`gradientOracle`, `expectGradientMatches`, `withPlainSGD`, `expectLearns`)
**Constraints**: se edita `src/` (a diferencia de 002), pero solo los archivos declarados en cada tarea; `package.json`, `package-lock.json`, `src/index.ts`, el changelog y `tasks.md` son del orquestador; los worktrees parten de `master` sin `node_modules` (`git rebase fix/v0.4` + `npm ci`); `tasks.md` y esta carpeta deben estar commiteados antes de lanzar agentes.

## Constitution Check

- II: cada modelo corregido pasa su oráculo con coseno > 0.999; solo `Seq2Seq` podría necesitar excepción, y el plan propone evitarla (BPTT completo, Q2).
- III: `expectLearns` pasa a test normal en los cinco modelos de C1.
- IV: mutante por tarea (revertir el arreglo en `src/` hace fallar el test) y 20 ejecuciones + 5 con orden aleatorio al final.
- VI: se corrigen los comentarios que no coinciden con el código (N3, "MSE loss").
- VII: los cambios observables incompatibles están listados en el spec para el changelog (007/008).
- Operación: ningún agente publica ni sube a `master`; archivos exclusivos del orquestador intactos; integración en serie.

## Decisiones de diseño

1. **Arreglo de signo en el origen del delta, no en el optimizador.** Se niega el delta de salida (`dLogits`, `dPred`, `dOutputs`, `dOut`, `dProjOut`) donde nace, de modo que todo `backward` posterior recibe `−dL/dout` como ya esperan `TransformerBlock`, `CausalConv1D` y `LSTMLayer`. Se actualiza el comentario del delta para decir `−dL/d…`.
2. **N1: "calcular todo, actualizar al final" solo donde haga falta** (`AttentionHead`): mover el cálculo de `dX` (paso 6) antes de las tres llamadas `update` (o diferir los `update` tras `dX`). No se reordena nada más; MHA, bloque y LayerNorm se verifican con tests de `dX` (lectura del código en spec).
3. **Escala** (Q1, Q3): el gradiente es siempre el de la pérdida que devuelve el método. `NetworkTransformer`: CE media (`dLogits / count`). `TCN`: `dOut = −2·diff/(T·outputSize)` y `lr` sin dividir en todas las capas. `RNN` y `Seq2Seq` ya usan `lr/T` con pérdida media y no cambian de escala.
4. **`Seq2Seq` con BPTT completo** (Q2): `LSTMLayer.backprop(dh_seq, lr, dcFinal?)` devuelve `{ dh0, dc0 }`. `trainStep`: (1) decodificador `backprop` con `dhSeq` negado → `dh0/dc0`, (2) encoder `backprop(encoderDhSeq, lr·T_in/T_out, dc0)` con `encoderDhSeq[T_in−1] = dh0` (compensa que `LSTMLayer` divide entre su propio `T`). Se conserva el valor devuelto.
5. **`Autoencoder`**: `dLdz[k] *= encoder.layers[last].neurons[k].activation.dfn(z[k])`; se verifica también con estructura sin capas ocultas.
6. **Oráculos**: se quitan los `it.fails` y los controles; se sube el `lr` de MHA/bloque/LayerNorm a 0.05 para que N1 deje de estar enmascarado; los demás conservan el `lr` que midió 002 salvo que cruce un quiebre de ReLU.
7. **Cabeceras de test**: los comentarios de cabecera de los archivos de oráculo describen los `it.fails` y controles; se reescriben al quitar el `it.fails` (tarea de limpieza final).

## Orden de ejecución

1. **Lote 1** (src disjuntos, en paralelo, máx. 4): T001 `AttentionHead` (N1), T002 `Autoencoder` (N6), T003 `RNN`, T004 `TCN`.
2. **Lote 2**: T005 `Seq2Seq` (+ `LSTMLayer`). T006 `NetworkTransformer` y T007 `NetworkTransformerRL` dependen de T001 (heredan `AttentionHead`) y van tras él; T008 (tests de `dX` de MHA/bloque/LayerNorm) también depende de T001.
3. **Lote 3**: T009 N3 (comentario de `LayerNorm`), puede ir en cualquier momento tras el lote 1 (sin dependencia de código).
4. **Lote 4**: T010 limpieza residual (cabeceras, `it.fails`/controles huérfanos) → T011 `math-reviewer` → T012 estabilidad (20 + 5 ejecuciones) → el orquestador actualiza HANDOFF.
5. Escalado según ORCHESTRATION: Haiku falla una vez → Sonnet; Sonnet falla dos veces → se pide permiso para Opus (candidatas: T004 `TCN`, T005 `Seq2Seq`). Registrar escalados en HANDOFF.

## Project Structure

```text
specs/003-sign-fix/{spec.md,plan.md,tasks.md}
src/{AttentionHead,Autoencoder,RNN,TCN,Seq2Seq,LSTMLayer,NetworkTransformer,NetworkTransformerRL,LayerNorm}.ts   (correcciones; LayerNorm solo comentarios)
tests/oracle/{AttentionHead,AttentionHead.dx,Autoencoder,RNN,TCN,Seq2Seq,NetworkTransformer,NetworkTransformerRL}.oracle.test.ts   (it.fails y controles fuera)
tests/oracle/{MultiHeadAttention,TransformerBlock,LayerNorm}.oracle.test.ts   (lr 0.05)
tests/oracle/{MultiHeadAttention,TransformerBlock,LayerNorm}.dx.oracle.test.ts   (nuevos, regresión N1)
tests/{NetworkTransformer,NetworkTransformerRL}.test.ts   (solo si el cambio de dirección los rompe)
```

## Riesgos

- **TCN**: el coseno +0.77 con solo signo no está explicado del todo por la escala; puede haber una tercera causa (Q6). Se mide por bloque.
- **Seq2Seq**: plumbing de `dc` final y escala `T_in/T_out`; error probable si se olvida que `LSTMLayer.backprop` divide entre su `T`. Verificar con encoder de longitud distinta a la del decodificador (el oráculo de 002 usa 3 y quizá 3: la tarea añade un caso con `T_in ≠ T_out`).
- **Tests existentes** (`tests/NetworkTransformer*.test.ts`, `Integration.test.ts`) fueron escritos con el entrenamiento al revés y pueden tener umbrales que ahora cambian; se ajustan con justificación, nunca relajando sin motivo.
- **Estabilidad**: al entrenar de verdad, los modelos pueden divergir con los `lr` actuales de los tests (lr 0.05 en MHA); T012 ejecuta 20 veces y 5 en orden aleatorio.
- **Adam real vs plain SGD** (Q5): el oráculo no detecta pasos múltiples de Adam por llamada (`NetworkTransformerRL.inputProj`).
