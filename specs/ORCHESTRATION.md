# Plan de orquestación: reparación hacia v0.4.0

Plan aprobado por el usuario el 2026-11-08. Los hallazgos están en `specs/000-audit/AUDIT.md`, las reglas en `.specify/memory/constitution.md` y el estado actual en `specs/HANDOFF.md`.

## Orquestador

La **sesión principal con Sonnet 5.5** hace de orquestador (`/model sonnet`). Precios por millón de tokens, entrada/salida: Haiku 5.5 $0.10/$0.50 (sube a $0.50/$2.50 con prompts de más de 100K tokens), Sonnet 5.5 $2/$10, Opus 5.5 $4/$20.

Por qué Sonnet y no Haiku: sus errores son los más caros, lee informes cortos y delega el trabajo pesado. Por qué no Opus: cuesta el doble y las decisiones difíciles se resuelven con pruebas automáticas.

El orquestador no implementa features. Reparte, integra, escala y mantiene al día `specs/HANDOFF.md`.

## Selección de modelo

1. **Haiku por defecto:** tareas con archivos concretos, criterio ejecutable y sin matemáticas nuevas. Prompt por debajo de 100K tokens.
2. **Sonnet:** gradientes, diseño de API, tests base, specs y revisión de código matemático.
3. **Escalado:** Haiku falla 1 vez en el mismo criterio → Sonnet. Sonnet falla 2 veces → se considera Opus.
4. **Opus requiere permiso:** el orquestador se detiene y le dice al usuario qué tarea es, qué falló en cada intento, por qué cree que Opus lo resolvería y qué alternativa hay. Sin aprobación no se usa.
5. **Ajuste:** si en el primer lote se escala más del 30 % de un tipo de tarea, ese tipo pasa directamente a Sonnet. Registrar los escalados en `HANDOFF.md`.

## Agentes (`.claude/agents/`)

| Agente | Modelo | Función | Edita |
|---|---|---|---|
| orquestador (sesión principal) | Sonnet | Reparte, integra, escala | Solo archivos compartidos |
| `spec-author` | Sonnet | Spec, plan y tareas con Spec Kit | Solo `specs/` |
| `math-dev` | Sonnet | Gradientes y modelos | Archivos de su tarea |
| `api-dev` | Haiku (Sonnet si hay diseño, con `model`) | Validaciones y correcciones de API | Archivos de su tarea |
| `test-writer` | Haiku | Tests de módulos | Solo `tests/` |
| `docs-chore` | Haiku | Documentación y configuración | Nunca `src/` |
| `verifier` | Haiku | Ejecuta criterios e informa | Nada |
| `math-reviewer` | Sonnet | Revisa cambios matemáticos | Nada |

## Spec Kit (v1.3.0, ya inicializado)

Por feature: `/speckit-specify` → `/speckit-clarify` → `/speckit-plan` → `/speckit-tasks` → `/speckit-analyze` → ejecución por el orquestador → `/speckit-converge`, repitiendo hasta que no añada tareas.

**No usar `/speckit-implement` como ejecutor del enjambre:** procesa las tareas en la propia sesión, sin elegir modelo ni usar subagentes. El orquestador reparte las tareas de `tasks.md`. Lo mismo vale para las skills: al ser de la sesión principal, las redacta `spec-author` siguiendo sus plantillas o el orquestador las invoca.

Formato obligatorio de cada tarea, además del estándar de Spec Kit:

```
- [ ] T012 [P] [H] [files: tests/PCA.test.ts] Tests de PCA
      Gate: npx vitest run tests/PCA.test.ts && mutante detectado
```

`[H]` Haiku, `[S]` Sonnet. `[P]` solo si los conjuntos `[files: ...]` del lote no se solapan.

## Cómo no pisarse

- Cada agente trabaja con `isolation: "worktree"` en la llamada a `Agent`, y su propia rama.
- **Comprobar la base:** el agente verifica que su worktree contiene la punta de `fix/v0.4` (`git merge-base --is-ancestor`). Si no, hace `git rebase fix/v0.4` antes de trabajar. Si `node_modules` falta, ejecuta `npm ci`.
- Archivos exclusivos del orquestador: `tasks.md`, `package.json`, `package-lock.json`, `src/index.ts` y el changelog. Un agente que necesite cambiarlos lo pide en su informe.
- Integración en serie: rebase sobre `fix/v0.4`, `verifier`, y merge de una en una.
- Máximo de 4 agentes a la vez.
- Nadie sube a `master`, porque ese push puede publicar en npm.

## Informe de cada agente (obligatorio y breve)

```
Tarea: T012 | Estado: done / failed / blocked | Rama: <rama> | Commit: <sha>
Archivos: <lista>
Criterio: <comando> → <resultado, últimas líneas>
Pendiente o riesgos: <o "ninguno">
```

No incluir código ni volcados largos.

## Features, en orden

Dependencias: 001 → 002 → 003 → 004. Después, 005 y 006 en paralelo, y al final 007 → 008.

### 001 `release-safety` (Haiku; Sonnet revisa CI)
- `ci.yml` con typecheck, build y tests en PR y push.
- `publish.yml` ejecuta los tests antes de publicar y publica solo si la versión local es nueva. Mantener 0.3.2 hasta la feature 008.
- `exports` con `types` primero (I8); añadir `LICENSE` MIT, `repository`, `homepage`, `bugs`, `sideEffects: false`, `engines`.
- Instalar `@vitest/coverage-v8`; script `typecheck` que incluya `tests/`; corregir el aviso de `vitest.config`.
- Quitar `bun.lock` y el `.tgz`; mover informes de IA y planes a `docs/archive/`.
- Al llegar CI a `master`: **activar la protección de rama** exigiendo PR y el check de CI (aprobado por el usuario). No antes: exigir un check inexistente bloquearía los merges.

### 002 `gradient-oracle` (Sonnet; condición previa de todo lo demás)
- Utilidades de test: semilla fija de `Math.random`, comparación con diferencias finitas de actualizaciones reales, prueba de "aprende".
- Reemplazar `signedPerturbationCheck` y `multiStepLossCheck` por pruebas reales.
- Cubrir todo modelo y capa entrenable. Las pruebas que hoy deben fallar (C1, C2, I1) se marcan con `it.fails`, para que CI siga en verde y se detecte el arreglo.
- Eliminar la inestabilidad: 20 ejecuciones seguidas sin fallos.

### 003 `sign-fix` (Sonnet)
- Corregir el origen en `NetworkTransformer`, `NetworkTransformerRL`, `RNN`, `TCN` y `Seq2Seq`, y quitar sus `it.fails`.
- Investigar el coseno de `TCN` y el gradiente del encoder de `Seq2Seq`. Si `Seq2Seq` se queda como aproximación, documentar la excepción y su umbral.
- Corregir comentarios incorrectos, como "Returns: MSE loss" en `NetworkTransformer`.

### 004 `models-fix` (Sonnet; posible candidato a Opus, con permiso)
- `GAN`: propagar a través del discriminador sin modificar sus pesos.
- `VAE`: llevar la reconstrucción al encoder a través de la reparametrización, con salidas de `mu` y `logVar` adecuadas.
- `residual` en `NetworkN` (I1).
- Criterio: `GAN` se acerca a la distribución objetivo; `VAE` reconstruye mejor que predecir la media y `mu` varía según la entrada.

### 005 `api-hardening` (Haiku y Sonnet)
- Haiku: longitud en `setWeights` y `ModelSaver` con metadatos de estructura (I2); copias en `getWeights` (I6); `Network` con varias salidas (I5: entrenarlas todas o lanzar error).
- Sonnet: `Trainer` (I3), `padding: 'same'` con kernel par (I4), `Value.backward()` iterativo (I7).

### 006 `coverage` (Haiku en paralelo; Sonnet revisa una muestra)
- Tests para los 29 módulos de la auditoría y `Metrics`, un archivo por módulo, con valores esperados calculados a mano y comprobación de mutante.
- Revisar `DecisionTree` con XOR y decidir si es una limitación documentada.

### 007 `docs` (Haiku)
- Actualizar `AGENTS.md`; README en un solo idioma con ejemplos ejecutables, comprobados con `specs/000-audit/evidence/extract.mjs`; changelog con los cambios incompatibles y su migración.

### 008 `release-app` (Sonnet)
- Subir a 0.4.0 y publicar **solo con aprobación del usuario**.
- Actualizar `neuron_app` (`/Users/daniel/Desktop/projects/experiments/neuron_app`). Ojo: su `main` está 1 commit adelante y 3 atrás, y tiene `bun.lock`, `package-lock.json` y `package.json` sin commitear. Revisar y preguntar antes de tocarlos, y trabajar en rama propia.
- Reajustar los `lr` en Sudoku, generación de texto y laberinto, porque se eligieron con el entrenamiento roto, y comprobar que aprenden de verdad.

## Decisiones del usuario (2026-11-08)

1. Quitar las dependencias de runtime añadidas sin commitear, usando `overrides` si hiciera falta. Hecho sin necesitar `overrides`.
2. El usuario aprueba la publicación de la 0.4.0. Protección de `master` aprobada.
3. Se aceptan cambios incompatibles: "tenemos que hacer esto bien, luego si algo arreglamos las apps que la usan".
4. `neuron_app` entra en el plan.
5. Opus solo con permiso, explicando por qué.
