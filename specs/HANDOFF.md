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

Esperar el check `ci` en verde en el PR de `fix/v0.4`, que el usuario fusione, y entonces activar la protección de `master` (T011, requiere aprobación). Después, feature 002 `gradient-oracle`.

## Avance

| Feature | Estado |
|---|---|
| 001 `release-safety` | T001–T010 hechas y verificadas; PR abierto. Falta T011 (protección de `master`) tras el merge |
| 002 `gradient-oracle` | Pendiente |
| 003 `sign-fix` | Pendiente |
| 004 `models-fix` | Pendiente |
| 005 `api-hardening` | Pendiente |
| 006 `coverage` | Pendiente |
| 007 `docs` | Pendiente |
| 008 `release-app` | Pendiente |

## Escalados de modelo

Ninguno en 001: 8 tareas Haiku (T001–T005, T007, T008), todas a la primera; T009 Sonnet (aprobada sin bloqueantes). Formato: tarea, modelo inicial, modelo final y motivo.

## Avisos

- **`.claude/settings.local.json` permite `Bash(npm publish *)` sin preguntar.** La constitución lo prohíbe a los agentes, pero el permiso sigue activo. Recomendado: que el usuario lo quite antes de lanzar el enjambre. Un agente no debe editar sus propios permisos.
- **`isolation: "worktree"` probado:** los worktrees parten de `master` (723fa58) y sin `node_modules`. Cada prompt debe exigir `git rebase fix/v0.4` y `npm ci`; con eso funciona. Los archivos sin commit (p. ej. specs) no llegan al worktree: hacer commit antes de lanzar. Los agentes tropiezan con comandos git compuestos (`cd && git`, `for`): pedirles comandos simples.
- **Menores de T009 sin tratar (publish.yml):** `NODE_AUTH_TOKEN` de relleno puede dar 401 en `npm view` y bloquear el publish (falla cerrado; confirmar en la primera ejecución real); la comparación de versión es `=`, no `>`; sin `concurrency`.
- **Rendimiento de Haiku:** primer lote (tareas de configuración) 8/8 sin escalar. Aún no medido en tareas con tests o código.
