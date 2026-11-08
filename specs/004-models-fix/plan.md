# Implementation Plan: models-fix

**Branch**: `fix/v0.4` | **Date**: 2026-11-08 | **Spec**: `specs/004-models-fix/spec.md`

## Summary

Cerrar los últimos defectos de gradiente que el oráculo de 002 dejó marcados fuera de 003: I1 (`NetworkN` residual), C2 (GAN, VAE + N8), más N2 (BatchNorm), N9 (ContrastiveLearning), N10 (Autoencoder) y la corrección pequeña de Word2Vec. Cada corrección retira su `it.fails` y sus tests de excepción/control; tras 004 solo quedan los `it.fails` de N7 (005). GAN y VAE necesitan un método aditivo nuevo de `NetworkN` (gradiente respecto a la entrada, sin tocar pesos), por lo que dependen de I1 en serie.

## Technical Context

**Language/Version**: TypeScript ^5, Node 20 (CI) y Node 26 local; se verifica también con `~/.nvm/versions/node/v20.19.0`.
**Testing**: `vitest run`; oráculo de 002 en `tests/helpers/` (`gradientOracle`, `expectGradientMatches`, `withPlainSGD`, `expectLearns`, `seedRandom`, `mulberry32`).
**Constraints**: se edita `src/` solo en los archivos declarados por tarea; `package.json`, `package-lock.json`, `src/index.ts`, el changelog y `tasks.md` son del orquestador; worktrees desde `master` sin `node_modules` (`git rebase fix/v0.4` + `npm ci`); `tasks.md` y esta carpeta commiteados antes de lanzar agentes. En macOS `wc -l` imprime espacios: usar `tr -d ' '`.

## Constitution Check

- II: cada modelo pasa su oráculo con coseno > 0.999. No se prevé ninguna excepción: BatchNorm pasa a gradiente exacto (Q6); si Q6 se resuelve como (b) o Q7(c) deja semillas fuera, la excepción se documenta en el spec con aproximación y umbral.
- III: "learns" pasa a test normal en GAN y VAE; ContrastiveLearning conserva el suyo sin inicializaciones elegidas a mano.
- IV: mutantes que QUITAN un término (la revisión de 003 mostró que invertir el signo no basta); semillas y `lr` perturbados (FR-011); 20 + 5 ejecuciones y Node 20.19.
- V: `trainWithDeltas`/`inputGradient` lanzan error explícito ante residual en la última capa (Q2).
- VI: se corrigen comentarios falsos (GAN 117-129, `BatchNorm.backward`, cabeceras de test, `Word2Vec` "minor approx").
- VII: cambios incompatibles listados en el spec para 007/008.
- Operación: nada se publica ni sube a `master`; integración en serie.

## Decisiones de diseño

1. **`NetworkN`: guardar la salida de `f` antes de residual y dropout.** `_forwardAll` devuelve, además de `act`, el vector `pre[l] = f(z_l)`. El backward arrastra `g_l = dL/d(salida de la capa l)` (no el delta): `δ_l = g_l ⊙ dfn(pre[l])`, `g_{l−1} = Wᵀδ_l + (residual ? g_l : 0)`. Se aplica dropout a `g` antes de `dfn`. Para la última capa `train` parte de `g_L = (t − p)` y `trainWithDeltas` solo admite deltas (error si hay residual en la última, Q2). Se escribe un núcleo privado `_backward(act, pre, ..., lr | null)` que devuelve `−dL/dx` y actualiza pesos solo si `lr != null`; `trainWithDeltas` y `inputGradient` lo usan. Los pesos se leen siempre antes del `_update` de su capa (hoy ya es así).
2. **GAN**: dentro de `trainStep` y tras el paso de D, `xFake2 = G(z2)`, `D = D(xFake2)`, `dxHat = D.inputGradient(xFake2, [dfn_D(D)/D])` (para sigmoide, `[1 − D]`), `gDelta[i] = dxHat[i] · dfn_G(xFake2[i])`, `G.trainWithDeltas(z2, gDelta, lr)`. Se conserva el orden de las llamadas a `Math.random` (`z`, luego `z2`). Los pesos de D no se tocan en el paso de G.
3. **VAE**: `forward` guarda/devuelve `eps` (campo aditivo; mismo orden y número de muestras). `train`: `dRecon_i = 2(x_i − x̂_i)/n · dfn(x̂_i)`; `gz = decoder.inputGradient(z, dRecon)` ANTES de `decoder.trainWithDeltas(z, dRecon, lr)`; `−dMSE/dμ = gz`, `−dMSE/dlogVar = gz · 0.5σε`; KL: `−dKL/dμ = −μ`, `−dKL/dlogVar = −0.5(e^{lv} − 1)` (con peso `klWeight` si Q4); delta del encoder = suma × `dfn` de su salida (lineal: 1). Encoder: salida lineal (Q3).
4. **BatchNorm**: `forward` guarda `u = x − media_vieja` además de `xNorm`/`std`; `backward` devuelve `dOut·γ·(m/s − (1−m)·m³·u²/s³)`; se elimina el bucle muerto. Se verifica por componente contra diferencias finitas de `forward` con las estadísticas restauradas (como hace hoy el test de momentum).
5. **ContrastiveLearning**: cabeza `activations: [relu, linear]`; `trainStep` calcula primero `grad[k]` para todos los parámetros (restaurando cada uno a su valor exacto guardado, no con `+= eps / −= 2eps / += eps`) y después aplica `w += lr·(−grad)`; la limitación (c) queda en un comentario. Los tests dejan de depender de `semilla 9 × 3` y `semilla 3` si la corrección lo permite.
6. **Autoencoder** (Q8-A): `outputDeltas[i] = 2/d · (x_i − x̂_i) · dfn(x̂_i)`; oráculo con `ratio ∈ [0.9, 1.1]`, `d = 3` y `d = 5`; `learns` con `lr` reajustado y verificado con perturbaciones.
7. **Word2Vec**: primer bucle calcula `dh[d] = Σ_j W2[d][j]·err[j]` con `W2` original; segundo bucle actualiza `W2`. Test exacto de un paso con `lr = 0.5` vía casts a `unknown` (como `Internals` ya hace). La divergencia de skip-gram con `lr = 0.1` se mide antes y después.
8. **Tests de aprendizaje robustos** (FR-011): variantes de `lr` (×0.7, ×1.3) y de semilla; el umbral del criterio se fija para que el peor caso quede por debajo de `0.7 ×` el umbral; las cifras medidas van en la cabecera.
9. **Mutantes que quitan términos** (ver `tasks.md`), probados en `src/` y revertidos con `git checkout -- src`.

## Orden de ejecución

1. **Lote 1** (máx. 4, archivos disjuntos): T001 `NetworkN` I1, T002 `BatchNorm` (N2), T003 `Autoencoder` (N10, requiere Q8), T004 `Word2Vec`.
2. **Lote 2** (T005 requiere T001 integrada): T005 `NetworkN.inputGradient`, T006 `ContrastiveLearning` (independiente de `NetworkN`).
3. **Lote 3** (requiere T005 integrada): T007 GAN, T008 VAE.
4. **Lote 4** (cierre, en serie): T009 limpieza (cabeceras, controles huérfanos) → T010 `math-reviewer` → T011 estabilidad (20 + 5 + Node 20.19). Huecos de T010 en un Lote 5 reservado (como en 003) y espera del CI de GitHub antes de cerrar.
5. Escalado: Sonnet falla dos veces → permiso a Opus (candidatas T007 y T008, criterio en el spec). Registrar escalados en `HANDOFF.md`. Preguntas Q1-Q10 resueltas antes de lanzar las tareas que dependen de ellas (Q1/Q2 → T001/T005; Q3-Q5 → T008; Q6 → T002; Q7 → T006; Q8 → T003; Q9 → T004; Q10 → T007).

## Project Structure

```text
specs/004-models-fix/{spec.md,plan.md,tasks.md}
src/{NetworkN,GAN,VAE,BatchNorm,ContrastiveLearning,Autoencoder,Word2Vec}.ts
tests/oracle/{NetworkN,GAN,VAE,BatchNorm,BatchNorm.momentum,ContrastiveLearning,Autoencoder,Word2Vec}.oracle.test.ts
tests/{NetworkN,BatchNorm}.test.ts        (solo si el cambio los rompe)
```

## Riesgos

- **GAN "learns"**: la dinámica adversarial es sensible; con 1 muestra real por paso y `lr = 0.1` puede oscilar aunque el gradiente sea exacto. Mitigación: medición por bloque (coseno del paso de G, D estable, curva de la brecha) y variantes de `lr`/semilla; candidata a Opus.
- **VAE colapso posterior** (Q4): puede imponer `klWeight`; sin eso "learns" no es alcanzable.
- **VAE `logVar` lineal**: `exp` puede desbordar; medir con las semillas de prueba antes de acotar.
- **Orden D/G**: la diferencia `O(lr)` entre el D congelado del oráculo y el D actualizado puede restar margen a 0.999; medirlo con `lr` 1e-3 y 1e-4.
- **NetworkN**: reescribir `_backpropLayers` afecta a todo modelo que lo usa (`Autoencoder`, `GAN`, `VAE`, `Network`, `TCN`...); el gate ejecuta toda la carpeta `tests/oracle`.
- **ContrastiveLearning (c)**: puede quedar alguna semilla cruda con coseno < 0.999 solo por curvatura; se informa (Q7).
- **Word2Vec**: la divergencia con `lr = 0.1` puede persistir; se documenta.
- **Lección CI (002/003)**: pasar N veces localmente no prueba estabilidad entre plataformas: Node 20.19 en T011 y CI de GitHub antes del cierre.
