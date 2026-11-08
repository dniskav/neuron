# Estado actual

**Mantener este archivo al día después de cada feature.** Última actualización: 2026-11-08.

## Cómo retomar

1. Ejecuta `/model sonnet`.
2. Di: `Lee specs/HANDOFF.md y specs/ORCHESTRATION.md y retoma como orquestador.`

## Dónde estamos

- **Rama de trabajo:** `fix/v0.4`, ya en `origin` y fusionada una vez en `master` (PR #4). `master` está protegida: nada se publica salvo aprobación del usuario (versión nueva en `package.json`).
- **Prueba base:** typecheck, build y 327 tests en verde; `npm audit` sin vulnerabilidades. Los tests siguen siendo inestables hasta la feature 002.
- **Hecho en la preparación:**
  - Quitadas las dependencias de runtime sin uso; `vitest` subido y `npm audit fix` aplicado. Los cambios previos de los lockfiles se descartaron. Eran derivados de esas dependencias, y una copia queda solo en el scratchpad de la sesión del 2026-11-08.
  - Spec Kit 1.3.0 inicializado, sin tocar archivos existentes.
  - Constitución, roles de agentes, auditoría y este plan escritos.

## Siguiente acción

Empezar la **feature 002 `gradient-oracle`** con `spec-author`. Con `master` protegida, todo llega por PR desde `fix/v0.4` (o ramas por feature).

## Avance

| Feature | Estado |
|---|---|
| 001 `release-safety` | **Hecha.** PR #4 fusionado, `publish.yml` saltó la publicación (0.3.2 ya publicada) y `master` protegida (PR + check `ci`, sin exigir aprobaciones, admins exentos) |
| 002 `gradient-oracle` | Pendiente |
| 003 `sign-fix` | Pendiente |
| 004 `models-fix` | Pendiente |
| 005 `api-hardening` | Pendiente |
| 006 `coverage` | Pendiente |
| 007 `docs` | Pendiente |
| 008 `release-app` | Pendiente |

## Escalados de modelo

Ninguno en 001: 8 tareas Haiku (T001–T005, T007, T008), todas a la primera; T009 Sonnet (aprobada sin bloqueantes). Formato: tarea, modelo inicial, modelo final y motivo.

## Hallazgos nuevos (no estaban en la auditoría)

- **N1 `AttentionHead.backward`:** `dX` se calcula tras actualizar `Wq/Wk/Wv` (error hasta 85 % con lr 0.05). Probablemente hereda `MultiHeadAttention`/`TransformerBlock`. Cubierto por T040 (`it.fails`); la corrección va en la feature 003 o 004.
- **N2 `BatchNorm` con `momentum` 0.1:** aproximación de stop-gradient, no error de signo. `forward` normaliza con estadísticas ya actualizadas con `x`, así que la pendiente real es ≈ m·γ/std, y `backward` devuelve γ/std (las trata como constantes): dirección correcta, magnitud ≈ 1/m (~10x con m=0.1). Con m=1 es exacto. Test T041 fija coseno > 0.9 y razón de normas 8–13 (`// AUDIT N2`). (El 'coseno 0.933' inicial no se reproduce.) Decidir en 004 si se corrige.
- **N3 `LayerNorm.backwardOne`:** devuelve −dL/dx (convención de `TransformerBlock`) pero el comentario dice dL/dx. Corregir el comentario en 003.
- **N4 `Network` (I5):** `train` solo acepta un objetivo escalar y solo actualiza la primera salida; `it.todo` en `Network.oracle.test.ts`.
- **N5 `Conv1D`/`Conv2D`:** el oráculo cubre pesos, no `dInput`. Candidato para la feature 006.

- **N6 `Autoencoder.train`:** `dL/dz` se calcula sin la derivada de la activación de salida del encoder; el paso del encoder es ~4x el correcto (coseno global 0.998 con [4]/[4], 0.835 sin ocultas). Marcado `it.fails // AUDIT N6` en `Autoencoder.oracle.test.ts`; corregir en la feature 003.

- **N1 es más ancho de lo previsto:** no basta mover `update` en `AttentionHead`; `MultiHeadAttention` (Wo), el FFN y `LayerNorm.backwardOne` también actualizan antes de calcular `dX`. Arreglo en 003/004: calcular todos los gradientes y actualizar al final, en cada clase.
- **Controles de signo (T019, T020):** el test de control que niega Δw solo vale mientras el defecto C1 existe. Al corregir cada modelo en 003 hay que quitar los `it.fails` Y borrar el control.
- **`NetworkTransformer.train`:** ratio ≈ 3 entre el paso y la pérdida (gradiente de la suma sobre posiciones, pérdida media por posición): decidir la escala al corregir C1. El comentario 'Returns: MSE loss' es incorrecto (es cross-entropy).

- **Notas para 003 por modelo (medidas en 002):**
  - `RNN`: solo signo (`dOutputs` en `backward()`; coseno −1).
  - `TCN`: DOS defectos: signo (`dOut = 2*diff/outputSize`) y escala (la proyección de salida usa `lr/T`, las capas conv usan `lr` con `dOut` sumado sobre T). Con solo el signo el coseno queda en +0.77. `CausalConv1D` solo está correcto.
  - `Seq2Seq`: signo en `dProjOut` (decoder y W_out/b_out dan −1) MÁS gradiente del encoder aproximado (solo `dhSeq[0]`, ignora `dc`): con signo corregido el global queda en 0.996. Hace falta BPTT completo o excepción documentada con umbral.
  - `GAN` (C2): generador −0.948; discriminador correcto (0.99999996). Trampa: con salida de dimensión 1 el coseno sale +1 por casualidad; usar dimensión ≥ 3.
  - Al corregir cada modelo: quitar `it.fails` Y borrar los controles de signo (se ponen en rojo).

- **N7 `LogisticRegression.predict` / `SoftmaxRegression.predictProba`:** lanzan con `weights` asignados a mano sin `train()` previo (validan contra `_nFeatures` privado). `it.fails // AUDIT N7`; corregir en 005. Además `SoftmaxRegression.train` reinicia siempre pesos con `Math.random` y `LinearRegression.fitGD` reinicia a cero: no hay entrenamiento incremental (documentar o cambiar en 005).
- **N8 `VAE` (3 causas, 003/004):** (1) el encoder no recibe dMSE/dz por la reparametrización (C2; `mu` no varía con la entrada), (2) los deltas del decoder/encoder no llevan la derivada de la sigmoide de salida, (3) al delta del decoder le falta el factor 2 del MSE. Los bloques sueltos pasan; solo el coseno global detecta la escala relativa. El umbral de dispersión de `mu` (0.05) lo puso el agente, no la spec.
- **Limitación de 002 (Q9):** el coseno es invariante a la escala, así que un factor constante equivocado en el gradiente no lo detecta el oráculo. Se cubre con pruebas de solución exacta donde existe (p. ej. `LinearRegression`), no en general.

- **N9 `ContrastiveLearning`:** (a) la cabecera de proyección termina en ReLU con Xavier: con algunas semillas las proyecciones quedan todas a cero (pérdida = log(2N−1), gradiente 0, no se recupera); (b) `trainStep` actualiza en sitio mientras calcula las diferencias finitas (estilo Gauss-Seidel), así que el gradiente por parámetro se evalúa con pesos ya movidos; (c) con normas de z pequeñas la similitud coseno tiene curvatura ~1/|z| y el oráculo solo da >0.999 con ciertas semillas o lr muy pequeño. El test de 002 fija semilla 9 y 400 pasos y es frágil por diseño: revisar en 004/006.
- **`Word2Vec` (observación):** `_skipgramStep`/`_cbowStep` calculan `dh` con `W2` ya actualizado (mismo patrón que N1, error O(lr), comentario 'minor approx' en src). Skip-gram diverge con lr 0.1 en el corpus de prueba (0.6 → 18.4) y converge con 0.03; causa no medida. El mutante de T029 solo cubrió skip-gram, no CBOW.

## Avisos

- **`.claude/settings.local.json` permite `Bash(npm publish *)` sin preguntar.** La constitución lo prohíbe a los agentes, pero el permiso sigue activo. Recomendado: que el usuario lo quite antes de lanzar el enjambre. Un agente no debe editar sus propios permisos.
- **`isolation: "worktree"` probado:** los worktrees parten de `master` (723fa58) y sin `node_modules`. Cada prompt debe exigir `git rebase fix/v0.4` y `npm ci`; con eso funciona. Los archivos sin commit (p. ej. specs) no llegan al worktree: hacer commit antes de lanzar. Los agentes tropiezan con comandos git compuestos (`cd && git`, `for`): pedirles comandos simples.
- **Menores de T009 sin tratar (publish.yml):** `NODE_AUTH_TOKEN` de relleno puede dar 401 en `npm view` y bloquear el publish (falla cerrado; confirmar en la primera ejecución real); la comparación de versión es `=`, no `>`; sin `concurrency`.
- **Rendimiento de Haiku:** primer lote (tareas de configuración) 8/8 sin escalar. Aún no medido en tareas con tests o código.
