# Estado actual

**Mantener este archivo al día después de cada feature.** Última actualización: 2026-11-08.

## Cómo retomar

1. Ejecuta `/model sonnet`.
2. Di: `Lee specs/HANDOFF.md y specs/ORCHESTRATION.md y retoma como orquestador.`

## Dónde estamos

- **Rama de trabajo:** `fix/v0.4`, solo local. Sin push, sin publicar y con `master` intacto.
- **Prueba base:** typecheck, build y 327 tests en verde; `npm audit` sin vulnerabilidades. Los tests siguen siendo inestables hasta la feature 002.
- **Hecho en la preparación:**
  - Quitadas las dependencias de runtime sin uso; `vitest` subido y `npm audit fix` aplicado. Los cambios previos de los lockfiles se descartaron. Eran derivados de esas dependencias, y una copia queda solo en el scratchpad de la sesión del 2026-11-08.
  - Spec Kit 1.3.0 inicializado, sin tocar archivos existentes.
  - Constitución, roles de agentes, auditoría y este plan escritos.

## Siguiente acción

Empezar la **feature 001 `release-safety`**: lanzar `spec-author` y seguir el flujo de `specs/ORCHESTRATION.md`.

## Avance

| Feature | Estado |
|---|---|
| 001 `release-safety` | Pendiente |
| 002 `gradient-oracle` | Pendiente |
| 003 `sign-fix` | Pendiente |
| 004 `models-fix` | Pendiente |
| 005 `api-hardening` | Pendiente |
| 006 `coverage` | Pendiente |
| 007 `docs` | Pendiente |
| 008 `release-app` | Pendiente |

## Escalados de modelo

Ninguno por ahora. Formato: tarea, modelo inicial, modelo final y motivo.

## Avisos

- **`.claude/settings.local.json` permite `Bash(npm publish *)` sin preguntar.** La constitución lo prohíbe a los agentes, pero el permiso sigue activo. Recomendado: que el usuario lo quite antes de lanzar el enjambre. Un agente no debe editar sus propios permisos.
- **`isolation: "worktree"` no está probado en este repo:** puede partir de otra base o sin `node_modules`. Comprobarlo en la primera tarea, como indica ORCHESTRATION.
- **Rendimiento de Haiku sin medir:** el reparto 60/40 es una estimación. Medirlo en el primer lote y ajustar.
