# Feature Specification: release-app

**Feature Branch**: `fix/v0.4` (no se ejecutó `create-new-feature.sh` porque crearía una rama; la numeración 008 viene de `specs/ORCHESTRATION.md`)
**Created**: 2026-11-12
**Status**: Draft (a la espera de las respuestas a «Preguntas abiertas al usuario»)
**Input**: `specs/ORCHESTRATION.md` sección «008 `release-app`», primer «PUNTO DE RETOMA VIGENTE» y «Avisos» de `specs/HANDOFF.md`, `.github/workflows/{publish,ci}.yml`, `package.json`, `CHANGELOG.md`, `specs/007-docs/breaking-changes.md`, Constitución, y el reconocimiento de `neuron_app` en `specs/008-release-app/app-inventory.md`.

## Estado verificado hoy

### Librería (`/Users/daniel/Desktop/projects/neuron`, `fix/v0.4` = `465a31b`, `master` = `0460a43`)

- **npm tiene 0.1.0 … 0.3.2; `latest` = 0.3.2** (`npm view @dniskav/neuron dist-tags versions`). `package.json` está en `0.3.2`; `CHANGELOG.md` encabeza `## 0.4.0 — unreleased` (línea 7).
- **`master` contiene 001–007** (PR #4–#11). Los 5 últimos runs de `publish.yml` (PR #7–#11) terminaron en verde con `Publish: skipped`: el paso que compara versiones ya funciona con el token de relleno (comprobado con `gh run list`).
- **`publish.yml` (leído entero):** se dispara con `push` a `master` (líneas 3-6). Ejecuta `npm ci`, `build`, `typecheck`, `test` (líneas 24-34) y luego «Check if version is already published» (36-57): compara con `npm view @dniskav/neuron version`, que devuelve la etiqueta **`latest`**, no «¿existe esta versión?»; si falla con 404 usa `0.0.0`; si falla con otro error aborta; **`skip=true` solo si `LOCAL = REMOTE`** (línea 53). Publica con `npm publish --access public` y `NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}` (líneas 59-63). Sin `concurrency`. Riesgos confirmados leyendo el archivo: ver FR-005.
- **`NPM_TOKEN`** existe como secreto del repositorio, creado/actualizado el **2026-06-13** (`gh secret list`). Si caducó o perdió permiso de escritura NO se sabe desde aquí: el único publish con esta versión del workflow aún no ha ocurrido (el último publish real fue con el workflow anterior, junio de 2026).
- **`npm pack --dry-run` hoy (0.3.2):** 7 archivos (`LICENSE`, `README.md` 81.7 kB, `package.json`, `dist/index.js` 327 kB, `dist/index.mjs` 323 kB, `dist/index.d.ts` y `dist/index.d.mts` 48 kB), 188.3 kB comprimido, 830.2 kB descomprimido. `package.json:8-14`: `exports["."]` con `types`, `import`, `require`; `files: ["dist"]`; `dependencies` sin definir (vacío).
- **Tags de git:** el último es `v0.2.5`; las versiones 0.2.6 a 0.3.2 no tienen tag. (Pregunta menor, no bloquea.)
- **`.claude/settings.local.json` permite `Bash(npm publish *)` sin preguntar** (HANDOFF, «Avisos»). Constitución: ningún agente publica.
- **`AGENTS.md:236`** dice «Current version in `package.json`: 0.3.2; the 0.4.0 changes are listed as unreleased in `CHANGELOG.md`»: hay que cambiarlo con la versión (README no menciona la versión actual; su texto «This changed in 0.4.0» sigue siendo cierto).

### App (`/Users/daniel/Desktop/projects/experiments/neuron_app`, solo lectura)

Detalle y citas en `app-inventory.md`. Resumen:

- `main` local: **1 commit por delante y 3 por detrás** de `origin/main`. El commit local `b4d139c` (sin subir) trae **todas las demos que nombra ORCHESTRATION que no son laberinto-LSTM**: Sudoku (MLP y Transformer), generación de texto, laberinto Transformer, CurveFitting, el sistema Ladle. `origin/main` (y el sitio publicado) no las tiene. Los 3 commits remotos son de Dependabot/CodeQL.
- **Cambios sin commitear del usuario** en `bun.lock`, `package-lock.json`, `package.json`, `vite.config.ts`: 13 paquetes añadidos a `dependencies` que el código no usa (parece subida de transitivas por alertas de seguridad), `vite` a `^8.3.5`, el lock arreglado (el committed apunta a `file:../../neuron`) y un `resolve.alias` de Vite hacia `node_modules/@dniskav/neuron/dist/index.mjs` sin explicación.
- Depende de `@dniskav/neuron` **0.2.5 exacto**. 27 archivos de `src/` importan la librería; **16 demos** (D01–D16 en el inventario) en 5 rutas. Ninguna usa `Trainer`, `ModelSaver`, `Metrics`, `Conv`, `DecisionTree`, `residual` ni dropout.
- **Los tipos de 0.4.0 no rompen la compilación** (comprobado: los mismos 8 errores de `tsc` con 0.2.5 y con 0.4.0; son defectos previos de la app). Los riesgos son de comportamiento.
- **Defecto previo de la app, confirmado con una sonda:** `TextGeneratorSession` pasa índices de clase como `targets`; la librería espera one-hot plano. Da `NaN` en el paso 2 con 0.2.5 y con 0.4.0, y la app reconstruye la red en cada `NaN`. Con one-hot y 0.4.0, `lr` 0.001 baja la pérdida de ≈3.5 a ≈0.04 en 2500 pasos; con 0.2.5 diverge (EMA 16.1).
- **Defecto latente que 0.4.0 convierte en excepción:** `useLaberintoRL.ts:237-248` (`handlePaso`) avanza el LSTM sin guardar el paso y `handleStart` no reinicia el episodio, así que `NetworkLSTM.train` recibirá menos filas que pasos y ahora lanza (V04).
- `deploy.yml` de la app despliega a Pages en **cada push a `main`**. Fusionar a `main` antes de publicar 0.4.0 dejaría el sitio dependiendo de un paquete que npm no tiene.
- `tests/ColorCard.test.ts` y `vitest.config.ts` existen pero `vitest` no está instalado ni hay script `test`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - El release 0.4.0 está preparado sin que nada con esa versión llegue a `master` (Priority: P1)

Existe una rama `release/0.4.0` con la versión 0.4.0 y un PR **en borrador** que nadie fusiona hasta la aprobación; `master` sigue en 0.3.2.

**Why this priority**: el merge a `master` de un `package.json` con versión nueva publica en npm de forma irreversible.

**Independent Test**: `git -C <lib> show master:package.json | grep '"version"'` da `0.3.2` y `gh pr view <n> --json isDraft` da `true` mientras no haya aprobación; `git diff master release/0.4.0 --stat` muestra solo `package.json`, `package-lock.json`, `AGENTS.md` y `CHANGELOG.md`.

**Acceptance Scenarios**:
1. **Given** la rama `release/0.4.0`, **When** se empuja, **Then** corre `ci.yml` y NO corre `publish.yml` (solo dispara `push` a `master`).
2. **Given** el PR en borrador, **When** alguien intenta fusionarlo, **Then** GitHub lo impide (es un borrador).
3. **Given** la aprobación explícita del usuario, **When** se marca como listo y se fusiona, **Then** `publish.yml` publica 0.4.0 una sola vez.

### User Story 2 - El paquete que se publicaría es el correcto y se puede consumir (Priority: P1)

`npm pack` de la rama release produce un tarball con exactamente los archivos esperados, y un proyecto temporal puede importarlo en ESM, CJS y con tipos.

**Independent Test**: `node specs/008-release-app/tools/pack-audit.mjs` y `node specs/008-release-app/tools/consume-smoke.mjs <tarball>` salen con 0; con un mutante (quitar `dist/index.mjs`, o `exports.types` mal puesto) salen con 1.

**Acceptance Scenarios**:
1. **Given** el tarball de 0.4.0, **When** se audita, **Then** contiene `LICENSE`, `README.md`, `package.json`, `dist/index.{js,mjs,d.ts,d.mts}` y nada más, con `version` 0.4.0, y pesa menos de 250 kB comprimido (referencia: 188.3 kB hoy).
2. **Given** un proyecto temporal fuera de ambos repos, **When** instala el tarball, **Then** `import { NetworkN } from '@dniskav/neuron'` (ESM), `require('@dniskav/neuron')` (CJS) y `tsc --noEmit` con `moduleResolution` `bundler`, `node16` y `nodenext` pasan, y una red `NetworkN` entrena XOR hasta pérdida < 0.05 con semilla.

### User Story 3 - `publish.yml` falla cerrado y no puede publicar dos veces ni una versión vieja (Priority: P1)

El workflow decide «publicar o no» por «¿existe exactamente esta versión en npm?» y no por igualdad con `latest`, y no se solapan dos ejecuciones.

**Independent Test**: un simulador (`publish-check-sim.mjs`) con un `npm` falso en el `PATH` reproduce 5 casos (versión nueva, misma versión, local menor que `latest`, 404 del paquete, error de red/401) y comprueba la decisión del script de decisión extraído del YAML.

**Acceptance Scenarios**:
1. **Given** local 0.4.0 y registro con `latest` 0.3.2 y sin 0.4.0, **Then** decide publicar.
2. **Given** local 0.3.2 y registro con 0.3.2, **Then** decide saltar (como hoy).
3. **Given** local 0.3.1 (p. ej. tras revertir `master`) y registro con 0.3.2 como `latest`, **Then** decide saltar (hoy con `!=` intentaría publicar 0.3.1 y movería `latest`).
4. **Given** un error que no es 404 al consultar, **Then** el job falla sin publicar.
5. **Given** dos pushes seguidos a `master`, **Then** el segundo espera al primero (`concurrency`).

### User Story 4 - La app compila y arranca contra 0.4.0 antes de que exista en npm (Priority: P1)

Una rama de la app, instalada con el tarball de 0.4.0, compila (`npm run build` verde) y sus 16 demos cargan sin excepciones.

**Independent Test**: en el worktree de la app, `npm install --no-save <tarball>` y `npm run build` salen con 0; el test de versión de la batería falla si `@dniskav/neuron` no es 0.4.0.

**Acceptance Scenarios**:
1. **Given** la rama de la app y el tarball, **When** se ejecuta `npm run build`, **Then** pasa sin errores (incluidos los 8 de tipos previos, que se corrigen).
2. **Given** las demos de laberinto, **When** se pasa a modo paso a paso y luego a entrenamiento, **Then** no lanza (el defecto `handlePaso`/`handleStart` está corregido).

### User Story 5 - Cada demo demuestra que aprende, de forma medible y reproducible (Priority: P1)

Cada una de las 16 demos tiene un test o script con semilla, un criterio numérico y una comprobación de que el criterio falla con un entrenamiento roto a propósito (`lr = 0` o gradiente invertido).

**Why this priority**: es la razón de la 008 (ORCHESTRATION: «comprobar que aprenden de verdad») y la Constitución III.

**Independent Test**: `npm run test:demos` en la app (a crear) ejecuta los 16 y sale con 0; con `DEMO_MUTANT=lr0` sale con distinto de 0 en cada uno.

**Acceptance Scenarios**:
1. **Given** una demo supervisada, **When** se entrena el presupuesto fijado, **Then** la pérdida final es menor que `ratio × pérdida inicial` y mejor que el predictor trivial (media), en las 3 semillas.
2. **Given** una demo de RL (laberinto ×2, Arkanoid, Snake, Robot), **Then** la tasa de éxito o la puntuación media supera la de la política sin entrenar con margen, en al menos 2 de 3 semillas.
3. **Given** `lr` reajustado, **Then** el valor y la medición que lo justifica (tabla de barrido) quedan en el informe de la tarea y el `lr` elegido es el que cumple el criterio con margen, no el límite.

### User Story 6 - Pesos de la 0.2.5/0.3.x no contaminan las demos (Priority: P2)

Las claves de `localStorage` de pesos entrenados con el entrenamiento roto cambian de versión (o se descartan) y ninguna carga parcial deja una red a medio cargar.

**Independent Test**: un test con `localStorage` simulado precargado con pesos viejos comprueba que la demo arranca con red nueva.

### User Story 7 - Las demos clave funcionan en un navegador real (Priority: P2)

Se recorren las 5 rutas con almacenamiento limpio y con almacenamiento «sucio» y no hay errores de consola, `NaN` en pantalla ni curvas de pérdida que suban.

**Independent Test**: lista de comprobación por ruta en `evidence/browser-0.4.0.md`, con capturas y la salida de consola.

### User Story 8 - Publicación aprobada, verificada y cerrada (Priority: P1, REQUIERE APROBACIÓN EXPLÍCITA DEL USUARIO)

Tras la aprobación: fecha en el changelog, merge, publicación, verificación en el registro, instalación limpia, tag y release de GitHub.

**Independent Test**: `npm view @dniskav/neuron@0.4.0 version` da `0.4.0`, `dist-tags.latest` es `0.4.0`, y `consume-smoke.mjs @dniskav/neuron@0.4.0` pasa desde un directorio vacío.

### User Story 9 - La app pasa a depender de 0.4.0 del registro (Priority: P1)

`neuron_app` declara `"@dniskav/neuron": "0.4.0"`, instala limpio con `npm ci` y su batería pasa contra el paquete publicado.

## Requirements *(mandatory)*

### Parte A: release de la librería

- **FR-001**: La versión 0.4.0 en `package.json` y `package-lock.json` (2 apariciones de `version`: líneas 3 y 9 del lock) vive SOLO en la rama `release/0.4.0`; `master` permanece en 0.3.2 hasta la aprobación.
- **FR-002**: El PR de release se abre y se mantiene **en borrador**. La fecha del `CHANGELOG.md` (`## 0.4.0 — unreleased` → `## 0.4.0 — AAAA-MM-DD`) se pone en el último commit, el día de la publicación.
- **FR-003**: `AGENTS.md:236` y cualquier frase de versión actual se actualizan en la rama release (no antes); `README.md` no se edita a mano (se regenera desde `specs/007-docs/readme-parts/`; hoy no menciona la versión actual).
- **FR-004**: `pack-audit.mjs` comprueba el contenido exacto del tarball (lista de 7 archivos, `version`, `exports`, `types` primero, `engines.node`, `dependencies` vacío, tamaño < 250 kB) y falla con un mutante.
- **FR-005**: `publish.yml` se endurece antes del release, en un PR propio que NO publica (la versión 0.3.2 ya existe): (a) decidir con `npm view @dniskav/neuron@$LOCAL version` (existe la versión exacta → saltar) en vez de comparar con `latest`; (b) `concurrency` con `cancel-in-progress: false`; (c) conservar el fallo cerrado ante errores distintos de 404 y dejar el motivo en el log; (d) documentar en un comentario que `NODE_AUTH_TOKEN` de `setup-node` es un relleno y el real solo se pasa al paso `Publish`. Riesgos que lo motivan, confirmados en el archivo: igualdad con `latest` (línea 53: una versión local menor que `latest` intentaría publicarse y movería la etiqueta), sin `concurrency`, el `NPM_TOKEN` real nunca ejercitado con este workflow, y la dependencia de `npm view` de `latest`. NO confirmado: que `npm view` falle con 401 por el relleno (5 ejecuciones lo contradicen).
- **FR-006**: `consume-smoke.mjs` instala un tarball o un `nombre@versión` en un directorio temporal fuera de ambos repos y verifica ESM, CJS y tipos (3 `moduleResolution`) más un entrenamiento XOR con semilla.
- **FR-007**: Ningún agente ejecuta `npm publish` ni `npm publish --dry-run`; la comprobación del paquete es `npm pack --dry-run`/`npm pack`. La publicación la dispara el merge del PR por el usuario.
- **FR-008**: Los archivos exclusivos del orquestador (`package.json`, `package-lock.json`, `src/index.ts`, `CHANGELOG.md`, `tasks.md`) no se asignan a ninguna tarea; el bump y la fecha son pasos del orquestador (O03, O06).

### Parte B: app

- **FR-009**: Todo el trabajo de la app ocurre en una rama propia (nombre propuesto `feat/neuron-0.4.0`) sobre una copia de trabajo separada (`git worktree add`), nunca en `main` y sin tocar los cuatro archivos sin commitear de la copia principal. La base se decide en Q2.
- **FR-010**: Hasta que 0.4.0 esté publicado, ningún commit de la app declara `"@dniskav/neuron": "0.4.0"`; las pruebas usan el tarball con `npm install --no-save`. El cambio de `package.json`/lock es un único commit de la Parte C.
- **FR-011**: Existe `npm run test:demos` (largo, fuera del `npm test` rápido) con un test o script por demo (D01–D16), semilla fija de `Math.random`, y un test que falla si la versión instalada no es 0.4.0.
- **FR-012**: Cada criterio de «aprende» incluye: línea base medida (política o predictor sin entrenar, o pérdida inicial), umbral con margen, semillas (3), y un **mutante** (`DEMO_MUTANT=lr0` y `DEMO_MUTANT=negate`) que debe hacer fallar el criterio. Los umbrales de la tabla de abajo son hipótesis iniciales; la tarea los fija con la medición y los justifica.
- **FR-013**: El reajuste de `lr` se documenta como tabla (valores probados × métrica × semillas) en el informe de cada tarea y el `lr` y el comentario del código (p. ej. `TransformerSudokuSession.ts:46`, que hoy habla de «CE ~10× menor que MSE») quedan coherentes con la medición. Textos de UI con el `lr` (`DetectorLaberinto.tsx:56`, `DetectorArkanoid.tsx:48`, `SudokuCard.tsx:167`, etc.) se actualizan.
- **FR-014**: Se corrigen en la app: los `targets` de `TextGeneratorSession` (one-hot plano de `seqLen × nClasses`), el desajuste trayectoria/buffer del LSTM en `useLaberintoRL.ts` (y `useMazeRL.ts` si aplica), los 8 errores de `tsc` previos, y los import sin uso.
- **FR-015**: Las claves de `localStorage` con pesos de redes cuyo entrenamiento cambió (`laberinto_lstm_v2`, `laberinto_transformer_v2`, `maze-*`) se versionan; `arkanoid_dqn_v1` y `snake-dqn-*` se versionan por limpieza (opcional). Los cargadores no dejan una red a medio cargar.
- **FR-016**: Si una demo no puede cumplir su criterio con ningún `lr` razonable porque la causa está en la librería, NO se cambia la librería en esta feature: se abre una tarea con la evidencia y se pregunta al usuario (ver «Fuera de alcance»).
- **FR-017**: Verificación en navegador de las 5 rutas con almacenamiento limpio y «sucio» (pesos 0.2.5 precargados en `localStorage`).
- **FR-018**: Las dos baterías de la app (tipos y demos) se repiten contra el paquete publicado en la Parte C; los resultados van a `evidence/`.

### Parte C: después de publicar

- **FR-019**: Tras el merge, `npm view @dniskav/neuron@0.4.0` coincide con el tarball auditado (lista de archivos y hash por archivo de `dist/`), `latest` = 0.4.0, y `consume-smoke.mjs @dniskav/neuron@0.4.0` pasa desde un directorio vacío.
- **FR-020**: Tag anotado `v0.4.0` sobre el commit de merge y release de GitHub con el texto de la sección 0.4.0 del changelog, solo con confirmación del usuario.
- **FR-021**: La app pasa a `"@dniskav/neuron": "0.4.0"` (pin exacto, como hoy), `npm ci` limpio, lock regenerado (y `bun.lock` si el usuario lo mantiene), batería completa y `npm run build` verdes, y PR a `main` de la app. El merge a `main` despliega a Pages: solo con confirmación.
- **FR-022**: Cierre: `specs/HANDOFF.md` actualizado, worktrees y ramas de agentes limpiados, `Bash(npm publish *)` sigue fuera de `settings.local.json`.

### Fuera de alcance

- Cambios de comportamiento nuevos en la librería (solo defectos que bloqueen la app, como tarea con decisión del usuario, FR-016), `rng` inyectable, T066 (ejes nulos de `PCA`) y validación de `lr` en las piezas de bajo nivel (documentado en `AGENTS.md`).
- Retro-etiquetar 0.2.6–0.3.2, migrar la app a otro empaquetador o framework, arreglar las alertas de seguridad de Dependabot de la app (salvo lo que decida Q1), rediseñar las demos.
- Escribir `neuron_app/README.md` nuevo.

## Criterios de «aprende» propuestos por demo (hipótesis a calibrar con la línea base medida)

Semillas 1, 2, 3. «Supervisada» = 3 de 3 semillas; «RL» = 2 de 3. Todos con mutante `lr0` y `negate` que debe fallar. `L0` = pérdida inicial medida.

| ID | Demo | Presupuesto (a fijar) | Criterio inicial |
|---|---|---|---|
| D01 | Neuron | 500 épocas sobre `data/datos.json` | exactitud ≥ 0.95 y pérdida final ≤ 0.5 L0 |
| D02 | Color (`NeuronN` ×2) | 100 épocas × 500 muestras (el test existente) | el test existente pasa con las 3 semillas; sin `NaN` |
| D03 | Círculo (`Momentum`, n = 1) | N épocas × 500 puntos | exactitud ≥ 0.9; barrido de `TASA` ∈ {0.025, 0.05} |
| D04 | CurveFitting (`Adam`) | 2000 pasos de 32 muestras | MSE ≤ 0.25 × MSE del predictor constante, para `sin`, `x²`, `tanh` |
| D05 | AngryBird (`Adam`, 1 salida) | N épocas × 400 ejemplos | MSE del ángulo ≤ 0.25 × MSE del predictor constante |
| D06 | Cañón (`Adam`, 2 salidas) | N épocas × 800 ejemplos | ídem para ángulo y `v0` |
| D07 | Arquitecto clasificación (`sgd`/`momentum`/`adam`) | por problema | cada problema alcanza el umbral propio de `problems.ts` con el `lr` por defecto y `adam`; para `sgd`/`momentum` se documenta el rango estable |
| D08 | Arkanoid DQN | E episodios | supervivencia media de los últimos 50 episodios ≥ 2 × la de la red sin entrenar |
| D09 | Snake DQN | E episodios | puntuación media de los últimos 100 ≥ 2 × la de la red sin entrenar |
| D10 | Robot (niveles 1–2) | E episodios | tasa de éxito ≥ `GOAL_SUCCESS_RATE` (0.80, `RobotSession.ts:34`) |
| D11 | Sudoku MLP (`Adam`, 729 salidas) | N pasos | MSE en celdas vacías ≤ 0.9 × L0 en puzzles nuevos **y** memoriza 20 puzzles fijos (exactitud por celda ≥ 0.9) |
| D12 | Sudoku Transformer | N pasos | CE enmascarada ≤ 0.8 × L0 (ln 9 ≈ 2.20) en puzzles nuevos, memoriza 20 puzzles fijos (≥ 0.9), `resets` = 0 |
| D13 | Texto (Transformer) | N pasos | EMA de la pérdida ≤ 0.1 (inicio ≈ 3.3 = ln 27), `resets` = 0, la generación voraz desde `"the"` reproduce ≥ 20 caracteres del corpus |
| D14 | Laberinto LSTM | E episodios | tasa de éxito en 20 episodios de demo (`ε` 0.04) ≥ 0.5 y > la de la red sin entrenar + 0.3 |
| D15 | Laberinto Transformer | E episodios | ídem |
| D16 | Arquitecto laberinto (LSTM y Transformer) | E episodios | ídem para la arquitectura por defecto de cada tipo |

La generalización de Sudoku sobre puzzles aleatorios puede ser débil por diseño del problema; por eso D11/D12 exigen también la memorización de un conjunto fijo, que prueba que el optimizador y el gradiente funcionan. Si la generalización no supera el azar con ningún `lr`, el informe lo dice y la demo se documenta como «memoriza, no generaliza» (decisión del usuario sobre el texto de la UI).

## Success Criteria *(mandatory)*

- **SC-001**: `master` de la librería sigue en `0.3.2` hasta el momento de la aprobación (`git show master:package.json`), y el PR de release es borrador hasta entonces.
- **SC-002**: `pack-audit.mjs` y `consume-smoke.mjs` salen con 0 sobre el tarball de la rama release y con 1 sobre un mutante.
- **SC-003**: `publish-check-sim.mjs` pasa sus 5 casos y falla sobre el `publish.yml` actual en el caso «local menor que `latest`».
- **SC-004**: En el worktree de la app con el tarball: `npm run build` sale 0; `npm run test:demos` sale 0 (16 demos, 3 semillas) y sale ≠ 0 con `DEMO_MUTANT=lr0` en cada una.
- **SC-005**: 0 errores de consola y 0 `NaN` visibles en las 5 rutas, con almacenamiento limpio y sucio.
- **SC-006**: Tras publicar: `npm view @dniskav/neuron dist-tags.latest` = `0.4.0`; instalación limpia en un directorio vacío pasa; la app con `"0.4.0"` instalada desde el registro pasa `build` y `test:demos`.
- **SC-007**: La publicación no ocurre sin un mensaje del usuario que la apruebe de forma explícita en la sesión (T031).

## Preguntas abiertas al usuario

1. **¿Qué hago con tus cambios sin commitear de la app (`package.json`, `package-lock.json`, `bun.lock`, `vite.config.ts`)?** Son: 13 paquetes añadidos a `dependencies` que el código no usa, `vite ^8.3.5`, el lock arreglado (el committed apunta a `file:../../neuron`) y un alias de Vite a `node_modules/@dniskav/neuron/dist/index.mjs`.
   - (a) Los dejo donde están y trabajo en un worktree separado, sin tocarlos; tú decides después. Opciones para ti: commitearlos tú, descartarlos o pedirme que los lleve a la rama.
   - (b) Los commiteo tal cual en una rama aparte antes de empezar.
   - (c) Los descarto (los tendrías que confirmar tú; no ejecuto nada destructivo).
   - **Recomendación: (a).** Y, al llevarlos a la rama, lo mismo que decidiste en la librería: las 13 dependencias no van a `dependencies` (usar `overrides` o subir las herramientas que las arrastran), el lock sincronizado sí es necesario, y el alias de Vite probablemente sobra con 0.4.0 del registro (se verifica en T034).
2. **¿Desde qué base creo la rama de la app?** `main` local tiene `b4d139c` (68 archivos, sin subir) y va 3 por detrás de `origin/main`.
   - (a) Desde `main` local, y después traer los 3 commits remotos (conflicto probable en `package.json`/locks).
   - (b) Desde `origin/main`: pierde Sudoku, texto, laberinto Transformer y CurveFitting (no hay nada que reajustar).
   - **Recomendación: (a)**, en un worktree nuevo, y NO empujar `main` (el push despliega a Pages). `b4d139c` viaja en la rama de la app y llega a `main` por PR cuando 0.4.0 esté publicado.
3. **¿Cómo pruebo la app antes de publicar?** (a) Tarball de `npm pack` instalado con `npm install --no-save` en el worktree de la app; (b) `npm link`; (c) `file:` en `package.json`.
   - **Recomendación: (a).** Ejercita exactamente lo que se publicaría (`files`, `exports`, tipos CJS/ESM). `npm link` resuelve a la carpeta de la librería saltándose `files`/`exports` y su `node_modules` de desarrollo; `file:` ensucia el `package.json` y el lock (justo lo que dejó el lock committed desincronizado).
4. **¿La app pasa de 0.2.5 a 0.4.0 directamente?** (a) Directo, pin exacto como hoy; (b) con `^`.
   - **Recomendación: (a).** 0.3.0–0.3.2 también tienen el entrenamiento roto y no aportan un paso intermedio útil; un pin exacto evita que una 0.4.x cambie el `lr` ajustado.
5. **¿Cómo y dónde viven las pruebas de las demos?** (a) `vitest` como devDependency de la app (hay un `vitest.config.ts` y un `tests/ColorCard.test.ts` huérfanos) con `npm run test:demos` aparte del `test` rápido; (b) un script de Vite SSR sin dependencias nuevas.
   - **Recomendación: (a).** Es lo que la app ya intentó; implica tocar `package.json` de la app (en la rama, no en tu copia). Las pruebas largas no entran en el CI de la app salvo que lo pidas.
6. **¿Cómo publico y cómo cierro?** (a) PR de release en borrador que fusionas tú cuando apruebes (el CI de `master` publica) y tag/release después; (b) publicas a mano desde tu máquina; (c) además, ¿hago antes el PR que endurece `publish.yml` (FR-005)?
   - **Recomendación: (a) y (c).** El endurecimiento no publica (0.3.2 ya existe) y elimina el riesgo de publicar una versión menor o dos veces. Para publicar a mano habría que mantener el permiso `Bash(npm publish *)`, que recomiendas quitar.

Preguntas menores (no bloquean, valores por defecto propuestos): ¿retro-etiquetar `v0.2.6`–`v0.3.2` (no, fuera de alcance)? ¿Texto de UI de Sudoku si no generaliza (lo decides al ver el informe)?

## Ambigüedades señaladas (no resueltas en esta spec)

- **A1 de `app-inventory.md`**: un solo `lr` gobierna Adam y el SGD de los embeddings en `NetworkTransformer`; no hay receta numérica, solo medición.
- El `npm run build` real de la app no se ejecutó (se aproximó con `tsc`): T011 confirma si ya falla.
- Si `NPM_TOKEN` sigue vigente solo se sabe publicando; el usuario debe comprobarlo en npmjs.com (T002) antes de T031.
- No se sabe por qué existe el alias de `vite.config.ts` ni por qué el lock committed apuntaba a `../../neuron`.
- El criterio de D07 depende de los umbrales de `problems.ts`, que no se leyeron en este reconocimiento.
