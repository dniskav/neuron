# HANDOFF 010 fast-core

**Estado (2026-11-18):** feature 010 TERMINADA en `feat/0.5-core` (sin push). Todas las tareas integradas y verificadas salvo T024 (aplazada a 011). NFR-1 a NFR-6 pasan (NFR-5 tras T031: 0,59). Suite: 153 archivos, 3269 tests; estable en Node 20.19 y 26. Pendiente SOLO con aprobación del usuario: push, PR a master, subir a 0.5.0 y publicar en npm.

## Cómo retomarlo (sesión nueva o tras un reset)

1. Leer, en orden: este archivo, `spec.md`, `plan.md`, `contracts.md`, `tasks.md`, `.specify/memory/constitution.md` y `specs/ORCHESTRATION.md` (reglas generales; en 010 la rama es `feat/0.5-core`, no `fix/v0.4`).
2. El orquestador es la sesión principal con **Sonnet** (`/model sonnet`). No implementa: reparte, integra, escala y mantiene este archivo al día. Opus solo con permiso del usuario.
3. Hacer T000. Después, lote a lote: crear un worktree por tarea (`git worktree add .claude/worktrees/010-T0NN -b 010/T0NN feat/0.5-core`), lanzar como máximo 4 agentes dentro de ellos → `verifier` sobre cada rama → merge en serie a `feat/0.5-core` → añadir las exportaciones pedidas a `src/index.ts` y sus filas a `AGENTS.md` → siguiente lote.

## Plantilla de lanzamiento de un agente

```
Agent({
  subagent_type: "<agente de la tarea>",          // api-dev, math-dev, docs-chore, test-writer, verifier, math-reviewer
  model: "<haiku|sonnet según [H]/[S]>",          // api-dev es Haiku por defecto: pasar model "sonnet" en las [S]
  // sin isolation: el agente trabaja en el worktree que le crea el orquestador
  description: "010 T0NN <título corto>",
  prompt: `
Feature 010 (fast-core) de @dniskav/neuron. Rama de integración: feat/0.5-core.
Trabaja SOLO en <ruta del worktree> (rama 010/T0NN); ejecuta ahí `npm ci` si falta node_modules.
Lee primero: specs/010-fast-core/contracts.md (firmas exactas), la tarea T0NN de specs/010-fast-core/tasks.md
(cabecera incluida: gate común, mutantes, base) y .specify/memory/constitution.md.
Haz SOLO la tarea T0NN y toca SOLO los archivos de su [files]. Si algo no encaja con contracts.md, informa blocked.
No edites src/index.ts, package.json, tasks.md, AGENTS.md ni CHANGELOG.md: pide los cambios en el informe.
Commit en tu rama, sin push. Termina con el informe de specs/ORCHESTRATION.md más las exportaciones que pides.`
})
```

## Plantilla del verifier antes de cada merge

```
Agent({ subagent_type: "verifier", model: "haiku", description: "010 verificar T0NN",
  prompt: "Rama <rama>. Tarea T0NN de specs/010-fast-core/tasks.md. Ejecuta su Gate, el gate común, `npx tsc --noEmit -p .` y `npm run build`. Comprueba con `git diff --name-only feat/0.5-core...<rama>` que solo cambiaron los archivos de [files]. Aplica uno de los mutantes listados y confirma que algún test falla (revierte después). PASS/FAIL por comprobación." })
```

## Paralelismo por lote

| Lote | Tareas | Agentes a la vez | Modelos |
|---|---|---|---|
| 1 | T001, T002, T003, T004 | 4 | H, H, S, H |
| 2 | T005, T006, T007, T008 | 4 | H, H, H, H |
| 3 | T009, T010, T011 | 3 | S, H, S |
| 4 | T012 → T013 | 1 + 1 | S, H |
| 5 | T014 → T015, T016, T017 | 1 + 3 | S, H, H, H |
| 6 | T018, T027; T019 (orquestador) | 2 | S, H |
| 7 (opcional) | T020, T021, T022 | 3 | H, H, H |
| 8 | T023 → T025 → T026 | en serie | H, S, H |

Reparto: 18 tareas Haiku, 7 Sonnet y 0 Opus previstas. Candidatas a Opus si Sonnet falla dos veces: T011 y T012.

## Decisiones del usuario

2026-11-18: el usuario acepta todas las recomendaciones ("Sí, todas").
- Q1: `NetworkFast` convive con `NetworkN`.
- Q2: rama `feat/0.5-core` desde `origin/master`; versión 0.5.0; Constitución enmendada (1.1.0) para integrar en la rama de la feature en curso.
- Q3: `NeuronN.weights` se escribe en sitio.
- Q4: el formato 2 no guarda el estado del optimizador.
- Q5: `infer` solo para modelos densos en 0.5.
- Q6: objetivos NFR-1 a NFR-5 aceptados.
- Q7: sin recomendación en la spec; el orquestador lo deja para 011 (T024 no se lanza) por coste. Se puede reabrir si el usuario lo pide.
- Q8: backend global y por instancia, con prioridad de la instancia.
- Q9: en 0.5 solo congelar y cargar la tabla de `NetworkTransformer`; la capa `Embedding` suelta, en 011.
- NFR-3 (2026-11-18): NetworkFast medía 32,4 bytes/param frente al objetivo ≤ 32; el usuario delegó ("toma como respuesta mía lo que tú consideres mejor") y el orquestador lo fija en ≤ 36.

## Escalados

(vacío)

## Notas

- Línea base medida en la conversación del 2026-11-18, antes de la spec: Adam ~90 bytes/parámetro, ~20 M de actualizaciones/s, `[512,512,512,10]` a 27,7 ms por ejemplo. T001 la vuelve a medir con el script oficial.
- El repo local tiene cambios sin commitear de 009 (juego) en `fix/v0.4`. No mezclarlos con 010: `specs/010-fast-core/` se copia a la rama nueva en T000.
- Las preguntas del usuario que dieron lugar a FR-9 (piezas intercambiables, backend inyectable) y FR-10 (embeddings propios) están en `spec.md`.
