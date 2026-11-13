# Implementation Plan: release-app

**Branch**: `fix/v0.4` (librería) y `feat/neuron-0.4.0` (app, por crear) | **Date**: 2026-11-12 | **Spec**: `specs/008-release-app/spec.md`

## Summary

Publicar `@dniskav/neuron` 0.4.0 y dejar `neuron_app` funcionando contra ella, de modo que la acción irreversible (publicar) sea lo último que ocurre y solo con aprobación explícita. Se prepara el release en una rama con un PR en borrador (FR-001/002), se audita el paquete y se prueba el tarball en un proyecto temporal y en la app ANTES de publicar (FR-004/006/010), se endurece `publish.yml` en un PR previo que no publica (FR-005), y la app se actualiza y se demuestra, demo por demo, que aprende (FR-011/012). Después de publicar se verifica en el registro, se etiqueta y la app pasa a depender del paquete del registro.

## Technical Context

**Librería**: TypeScript ^5, Node 20 (CI) y Node 26 local; `npm pack`/`npm install` del tarball en directorios temporales fuera de ambos repos.
**App**: React 19, Vite 8, Chakra; `tsc -b && vite build`; sin tests en marcha (T010 añade `vitest`, ver Q5).
**Constraints**: Constitución (publicación solo con aprobación; `package.json`, `package-lock.json`, `src/index.ts`, `CHANGELOG.md` y `tasks.md` exclusivos del orquestador, por eso el bump y la fecha son pasos O03/O06 y no tareas); comandos git sueltos, sin `cd &&` ni bucles en los prompts; la app se toca solo desde un worktree propio; las tareas de la app no editan la copia principal de la app.

## Qué dispara qué (leído de los workflows)

| Evento | Librería | App |
|---|---|---|
| push a cualquier rama | `ci.yml` (typecheck, build, coverage) | — |
| pull request | `ci.yml` | — |
| push a `master` (librería) | `ci.yml` + **`publish.yml` (publica si la versión es nueva)** | — |
| push a `main` (app) | — | **`deploy.yml`: `npm ci` + `npm run build` + despliegue a Pages** |

Consecuencias: (1) empujar `release/0.4.0` no publica; (2) un PR en borrador no se puede fusionar; (3) el merge del PR de release es el punto de no retorno; (4) empujar `main` de la app despliega y hoy el sitio publicado es de `7ed4b73` (sin Sudoku/texto/laberinto-Transformer).

## Orden de ramas y PR (librería)

```text
master (0.3.2) ──┬─ PR-A: specs/008 (+ handoff)            → fusionar con confirmación (no publica: 0.3.2)
                 ├─ PR-B: publish.yml endurecido + tools/   → fusionar con confirmación (no publica: 0.3.2 existe)
                 └─ release/0.4.0 (desde master DESPUÉS de PR-A y PR-B)
                       R1  docs: AGENTS.md versión actual
                       R2  chore: version 0.4.0 (package.json, package-lock.json)     ← base del tarball de pruebas
                       [aprobación del usuario]
                       R3  docs: CHANGELOG.md fecha real  → PR sale de borrador → merge → publish.yml publica
```

- El tarball de las pruebas (`npm pack` en R2) es byte a byte lo que `publish.yml` construiría, salvo la fecha del changelog (no entra en el paquete: `files` = `dist`, más `README.md`, `LICENSE`, `package.json`).
- Si las pruebas de la app destapan un defecto de la librería, el arreglo es un PR a `master` con decisión del usuario (FR-016) y `release/0.4.0` se rehace sobre el `master` nuevo: la versión 0.4.0 nunca está en `master` antes de la aprobación.

## Orden y dependencias globales

1. **Fase 0 (usuario):** T001, T002, T003.
2. **Fase A1 (librería, sin dependencias de respuestas):** Lote 1 = T004, T005, T006. O01 (PR-A).
3. **Fase A2:** T007 (necesita Q6c), O02 (PR-B), luego O03 (rama release, R1 con T008, R2), T009 (audita el tarball R2), O04 (tarball a un lugar fijo del scratchpad).
4. **Fase B (app):** necesita O04 y las respuestas Q1–Q3, Q5. O05 crea el worktree. Lote 3 = T010, T011 (en serie respecto a lo demás). Lotes 4–7 = las 16 demos en orden (T012–T027). Lote 8 = T028, T029, T030.
5. **Gate de publicación:** T031 [USUARIO], solo con T009, T028, T029, T030 en verde y T001/T002 hechos.
6. **Fase C:** O06 (R3), T031 (aprobación + merge), T032, T033, T034, T035, T036, T037.

## Decisiones de diseño

1. **El bump es lo último; el tarball se prueba antes.** R2 existe en una rama no fusionada; su tarball se instala en la app con `--no-save`. Ningún commit de la app apunta a 0.4.0 hasta la Parte C (FR-010).
2. **PR de release en borrador** como cerrojo: GitHub no deja fusionarlo. La decisión explícita del usuario (T031) lo saca de borrador.
3. **La decisión de publicar es «¿existe esta versión exacta?».** `publish.yml` pasa de `LOCAL = latest` a `npm view @dniskav/neuron@$LOCAL version`: 404 → publicar, versión devuelta → saltar, otro error → fallar. Evita publicar una versión menor y mover `latest`. `concurrency` evita dos jobs a la vez. Es un simulador con un `npm` falso en el `PATH` el que arbitra, no la lectura del YAML.
4. **El paquete se audita con una herramienta, no a ojo** (`pack-audit.mjs`): lista exacta de archivos, `version`, `exports`, tamaño, `dependencies` vacío. Con mutante.
5. **El consumo se prueba como lo haría un usuario** (`consume-smoke.mjs`): directorio vacío, `npm install`, ESM, CJS, `tsc` con 3 resoluciones, entrenamiento XOR con semilla. La misma herramienta sirve después contra el registro (T032).
6. **Una tarea por demo para el reajuste de `lr`**, cada una con su test, su tabla de barrido y su mutante (`DEMO_MUTANT`). Haiku para las mediciones repetitivas de redes `Adam` pequeñas (D01, D02, D04, D05, D06); Sonnet para las que requieren diagnóstico o diseño del criterio (Transformers, laberintos, RL, Sudoku MLP, Círculo/Arquitecto por el cambio de paso de `Momentum`/`SGD`). Regla de escalado de ORCHESTRATION: Haiku que falla un criterio una vez → Sonnet.
7. **Receta de `lr` solo donde es exacta.** `lr_nuevo = lr·n/2` aplica a `NetworkN` con `SGD`/`Momentum` sin `residual`/dropout (D03, D07 con esos optimizadores, D09 con ellos); con `Adam` se mantiene `lr` y se mide. Para `NetworkTransformer` no hay receta (A1): barrido empírico.
8. **Pesos viejos: se versionan las claves** de `localStorage` en lugar de intentar migrarlos (S06 del inventario): los pesos de 0.2.5 con el entrenamiento roto "cargan" pero no sirven.
9. **Pruebas de demos separadas del test rápido** (`test:demos`), con un test de versión que impide medir contra 0.2.5 por error.
10. **El cuello de la app es el `tsc`.** T011 deja `npm run build` verde antes de las demos para que la puerta de cada demo incluya el build.
11. **Verificación en navegador al final de la Parte B y otra después del deploy**; usa las herramientas de navegador (Chrome), no clics de pantalla.

## Puntos de no retorno

| # | Acción | Reversible | Cómo se evita un error |
|---|---|---|---|
| 1 | Merge del PR de release a `master` (publica) | **NO**: npm no permite republicar una versión ni reutilizarla tras `unpublish` | Borrador; T031 exige un mensaje de aprobación explícito; `NPM_TOKEN` y permiso verificados antes (T002); tarball auditado y probado (T009, T028) |
| 2 | Tag `v0.4.0` / release de GitHub | Parcial (se pueden borrar, pero ya notifica) | Solo después de T032 y con confirmación |
| 3 | Merge de la app a `main` (despliega a Pages) | Sí (revert + redeploy), pero el sitio público queda roto mientras tanto | Solo con 0.4.0 publicado y T034 verde; confirmación del usuario |
| 4 | PR-A/PR-B a `master` | Sí | No publican (versión 0.3.2 ya existe); confirmación por PR |

## Plan de reversión si la publicación sale mal

- **Antes de T031:** nada que revertir; se cierra el PR de release o se corrige en la rama.
- **El job de publicación falla antes de `npm publish`** (tests, typecheck, token): no hay nada publicado. Se corrige (p. ej. renovar `NPM_TOKEN`) y se vuelve a ejecutar el job desde la pestaña Actions; es idempotente (la decisión se toma de nuevo contra npm).
- **`npm publish` falla a medias:** se comprueba con `npm view @dniskav/neuron@0.4.0 version`. Si no existe, se reintenta; si existe, el paquete está publicado y manda el caso siguiente.
- **Publicado con un defecto:**
  1. `npm deprecate @dniskav/neuron@0.4.0 "<motivo y versión correcta>"` (reversible, no borra nada).
  2. Si hace falta que `npm install @dniskav/neuron` vuelva a 0.3.2 mientras se arregla: `npm dist-tag add @dniskav/neuron@0.3.2 latest` (reversible).
  3. Corregir en `master` por PR, subir a **0.4.1** (la numeración no se reutiliza) y publicar con el mismo flujo; actualizar la fecha y la sección del changelog.
  4. `npm unpublish` solo como último recurso (ventana de 72 horas y condiciones del registro; no libera el número de versión) y con permiso del usuario.
- **`master` revertido a 0.3.2 con 0.4.0 ya publicado:** con el `publish.yml` actual (`!=` frente a `latest`) el job intentaría publicar 0.3.2, fallaría con 403 y quedaría en rojo; con el endurecido salta. Otra razón para hacer PR-B antes.
- **App:** si falla tras publicar, no se fusiona a `main`; el sitio de Pages conserva el último despliegue (`7ed4b73`). La rama de la app se corrige con 0.4.1.
- **Tag/release de GitHub erróneos:** se borran y se recrean con confirmación (`git push --delete origin v0.4.0` es una operación de remoto que pide permiso).

## Riesgos

| Riesgo | Mitigación |
|---|---|
| `NPM_TOKEN` caducado o sin permiso (creado 2026-06-13; vigencia desconocida) | T002 lo comprueba en npmjs.com antes de T031; el fallo ocurre en `npm publish` y deja el job rojo sin publicar (reintentable) |
| Un defecto de la librería bloquea una demo (p. ej. un `lr` que ningún valor razonable estabiliza) | FR-016: se abre una tarea con evidencia; el usuario decide; `release/0.4.0` se rehace |
| Los umbrales de «aprende» son flojos y pasan un entrenamiento malo | Mutantes obligatorios (FR-012) y revisión por muestreo (T030, Sonnet distinto del autor) |
| Pruebas largas e inestables con RL | 3 semillas con 2 de 3, `Math.random` sustituido por un PRNG propio (mulberry32 de `tests/helpers/seed.ts` de la librería, copiado), y presupuestos medidos en la tarea; se separan del test rápido |
| Dependabot/CodeQL en la app añaden conflictos al traer los 3 commits remotos | La integración de esos commits es un paso explícito del orquestador (O05) con regeneración de locks; Q1 define quién decide |
| `npm ci` del CI de la app falla por el lock desincronizado (`file:../../neuron`) | El lock arreglado del usuario es parte de Q1; T034 regenera y comprueba `npm ci` limpio |
| Se fusiona la app a `main` antes de publicar y Pages despliega una dependencia inexistente | FR-010 + punto de no retorno 3 |
| Conflicto entre tareas por archivos compartidos (`LaberintoCard/rl.ts` lo importa D16; `SudokuSession.ts` y `DetectorArquitecto.tsx` los toca T011) | Orden de lotes: T011 antes que D11/D07; D16 después de D14 y D15; archivos disjuntos dentro de cada lote |
| Rutas con `ñ` (`CañonCard`) en NFD rompen globs o `[files:]` | Las tareas usan el glob `Ca*onCard` y comprueban con `git status --short` |
| Los agentes no pueden ejecutar `cd && git` ni bucles | Los prompts de lote piden comandos sueltos y `git -C <ruta>` para la app; el orquestador ya aplicó esta regla en 001–007 |
| Haiku inventa una cifra de resultado | Cada informe pega el valor medido y la salida del comando; T030 muestrea |
| Sesión sin navegador disponible para T029/T036 | Se marca como bloqueada y se pide al usuario que lo haga o conecte la extensión; no se sustituye por capturas de pantalla de la app de escritorio |

## Estructura de archivos de la feature

```text
specs/008-release-app/{spec,plan,tasks}.md
specs/008-release-app/app-inventory.md                     (reconocimiento, hecho)
specs/008-release-app/evidence/probe-text-targets.cjs      (Hallazgo 1, hecho)
specs/008-release-app/evidence/{pack-0.4.0,app-types,app-battery,browser-0.4.0,registry-0.4.0}.md   (informes de tareas)
specs/008-release-app/tools/{pack-audit,consume-smoke,publish-check-sim,app-typecheck}.mjs           (T004–T007)
.github/workflows/publish.yml                              (T007, PR-B)
AGENTS.md                                                  (T008, solo en release/0.4.0)
app: tests/demos/*, vitest.config.ts, package.json (solo vitest y scripts), src/** de cada demo (T010–T027)
```
