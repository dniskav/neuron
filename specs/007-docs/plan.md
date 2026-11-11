# Implementation Plan: docs

**Branch**: `fix/v0.4` | **Date**: 2026-11-11 | **Spec**: `specs/007-docs/spec.md`

## Summary

Documentar lo que 002–006 cambió y hacer que la documentación no pueda desfasarse otra vez: un inventario único de cambios incompatibles (`breaking-changes.md`) alimenta `CHANGELOG.md` y el README; el README se compone en partes disjuntas y se ensambla en una sola tarea; los ejemplos se ejecutan contra `dist/` con `extract.mjs`; un meta-check en `npm test` exige cada export en el README (y en `AGENTS.md`). `src/` solo cambia en comentarios. `AGENTS.md` queda BLOQUEADO hasta que el usuario decida sobre su cambio sin commitear.

## Technical Context

**Language/Version**: TypeScript ^5, Node 20 (CI) y Node 26 local; se verifica también con `~/.nvm/versions/node/v20.19.0`.
**Testing**: `vitest run` (`tests/**/*.test.ts`, `tests/setup.ts`). Cobertura: `npm run coverage` (umbral 97/94/97/97, ya en `ci.yml`). Ejemplos del README: `specs/000-audit/evidence/extract.mjs` (ver «Cómo funciona el extractor»).
**Constraints**: sin dependencias de runtime (Constitución I); `package.json`, `package-lock.json`, `src/index.ts` y `tasks.md` exclusivos del orquestador; el changelog también, **salvo en 007 donde es el entregable de UNA tarea (T005): excepción expresa**; worktrees desde `master` sin `node_modules` (el prompt pide `git rebase fix/v0.4` y `npm ci`, y `npm run build` antes de cualquier gate de ejemplos); comandos git sueltos, sin `cd &&` ni `for`.

### Cómo funciona el extractor (leído en `extract.mjs`)

`node extract.mjs <raíz> <scratch>`: lee `<raíz>/README.md`, extrae bloques ```` ```ts|typescript|js|javascript ````, sustituye `'@dniskav/neuron'` por `<scratch>/dist/index.mjs`, transpila con `esbuild` (resuelto con `createRequire(<raíz>/)`), escribe `<scratch>/readme/bN.mjs` y lo ejecuta con `node` (30 s). Imprime `OK`/`FAIL` por bloque y `bloques=N ok=M fail=K`; su código de salida es siempre 0. Limitaciones que el envoltorio (T002) resuelve: (1) la ruta del README está fija: el envoltorio crea una raíz temporal con el `.md` pedido como `README.md` y `node_modules` enlazado, para poder probar PARTES; (2) exige `<scratch>/dist` y `<scratch>/readme`: el envoltorio los crea (enlace simbólico a `dist/`); (3) no devuelve error: el envoltorio analiza la línea final; (4) no detecta azar: lo ejecuta 3 veces.

## Constitution Check

- **VI (comentarios y ejemplos veraces):** es el objetivo; cada ejemplo se ejecuta y cada comentario de `src/` incorrecto se corrige.
- **VII (incompatibles con migración):** el inventario y la guía de migración son el entregable principal; las recetas numéricas se verifican contra 0.3.2.
- **I (sin dependencias):** nada nuevo; `esbuild` ya llega transitivamente por `tsup`.
- **IV (tests que detectan errores):** los meta-checks (T015, T016) exigen mutante (quitar un nombre del README debe fallarlos).
- **Operación:** no se sube versión ni se publica; `package.json` queda en 0.3.2; nada va a `master` salvo por el PR de cierre y con confirmación del usuario.

## Decisiones de diseño

1. **Inventario primero (T001).** Tabla única `breaking-changes.md` (área | clase/método | antes | ahora | migración | feature | origen). Parte del apéndice A de la spec, verifica cada fila contra el código (`grep`, `git diff 723fa58..HEAD`) y marca lo no verificable. `CHANGELOG.md`, README y `AGENTS.md` solo citan filas del inventario: así dos agentes no inventan dos versiones de lo mismo.
2. **README por partes, ensamblado por un paso.** `README.md` es un solo archivo y varios agentes no pueden editarlo a la vez sin conflicto. Cada agente escribe su parte en `specs/007-docs/readme-parts/NN-nombre.md` (archivos disjuntos, `[P]` legítimo) y solo T013 escribe `README.md` concatenando las partes en orden con `tools/assemble-readme.mjs`. Las partes: `01-intro-tablas` (cabecera, mermaid, «What's inside», tipos, «Install»), `02-uso-basico` (Neuron … Hopfield), `03-uso-modelos` (Autoencoder, GAN, VAE, Word2Vec, t-SNE, PositionalEncoding, ContrastiveLearning), `04-uso-secuencias` (Value, Conv, RNN, TCN, LSTM), `05-uso-herramientas` (Metrics, EarlyStopping, LossPlotter, DataAugmentation, WeightInspector y nuevos: Trainer, ModelSaver, DataLoader, Tokenizer, DatasetLoader), `06-cierre` («How it works», «Upgrading to 0.4», Build, «For AI agents», Roadmap, enlace al changelog, License). Cada parte pasa el gate de ejemplos por separado (`readme-gate.mjs <parte>`).
3. **Idioma:** el de la pregunta 1 (recomendado inglés). Las tareas llevan el idioma como parámetro del prompt; mientras no haya respuesta no se lanza ninguna tarea de redacción del README ni del changelog (T005–T011, T013).
4. **Ejemplos autosuficientes, rápidos y deterministas.** Cada bloque define sus datos, usa tamaños pequeños (segundos), y si usa `Math.random` o afirma una cifra, lo hace con una semilla propia en el bloque (un PRNG de 3 líneas, sin importar `tests/helpers`) o con un rango. El agente compara la salida real con cada comentario de resultado y lo anota en su informe. Un ejemplo ilustrativo no ejecutable lleva ```` ```text ```` y se justifica.
5. **Firmas y `lr` desde el código, no de la memoria.** Antes de escribir un ejemplo, el agente lee la firma real en `src/<Clase>.ts` y el inventario; en particular `Autoencoder`, `VAE` (`klWeight`), `GAN`, `NetworkN.train` (escala `2/n`), `Trainer`, `ModelSaver`, `ContrastiveLearning` (constructor `(inputSize, encoderHidden, projectionDim, options)`).
6. **Verificación de migraciones (T014).** Un agente distinto del autor del changelog crea un worktree en `723fa58` (0.3.2), construye una copia del `dist` viejo y del nuevo y compara las recetas con fórmula: `NetworkN.train` con `lr_viejo` frente a `lr·n/2` nuevo, `Network.train`, `Autoencoder` con `lr·d/2`. Con semilla fija y tolerancia 1e-9 (o documenta por qué no son idénticos). Si falla, se corrige el changelog.
7. **Comentarios de `src/` (T003, T004).** Solo líneas de comentario. Criterio ejecutable: el diff de `src/` contra la base, sin líneas de comentario, está vacío; la suite pasa. T004 (`TSNE`, `TCN`) reproduce primero el caso medido del `lr` (script temporal con prefijo `t004_` en el scratchpad) antes de escribir la cifra.
8. **Meta-checks como tests (T015, T016).** `tests/docs.readme.exports.test.ts` lee `src/index.ts`, extrae los nombres de cada `export { … }`/`export type { … }` (incluye `as`) y exige coincidencia de palabra completa en `README.md`. `tests/docs.agents.exports.test.ts`, igual con `AGENTS.md`. Mutante: borrar un nombre de una copia en memoria del README. Sensibles a mayúsculas.
9. **`AGENTS.md` bloqueado (T012, T016).** Ni siquiera se hace una versión provisional: el usuario tiene cambios sin commitear en la copia principal y cualquier edición en un worktree genera un conflicto de merge. T012 parte de la decisión (pregunta 2).
10. **Cierre al estilo T047 (T017).** Serie: typecheck, build, `npm run coverage`, suite ×20, ×5 con `--sequence.shuffle`, ×3 en Node 20.19, gate de ejemplos ×3 sobre el `dist` recién construido, `git diff` de `src/` solo comentarios, versión 0.3.2, `grep -r "it.fails\|it.todo" tests` = 0, actualizar `specs/HANDOFF.md`, PR contra `master` (con confirmación del usuario) y CI en verde.

## Project Structure

```text
specs/007-docs/{spec,plan,tasks}.md
specs/007-docs/breaking-changes.md              (T001)
specs/007-docs/migration-evidence.md            (T014)
specs/007-docs/tools/{readme-gate,assemble-readme}.mjs   (T002)
specs/007-docs/readme-parts/0N-*.md             (T006–T011)
CHANGELOG.md                                    (T005; excepción: único archivo exclusivo del orquestador que una tarea escribe)
README.md                                       (T013, solo por ensamblado)
AGENTS.md                                       (T012, BLOQUEADA)
src/{Trainer,NetworkN,Autoencoder,ContrastiveLearning,TSNE,TCN}.ts   (T003, T004: solo comentarios)
tests/docs.{readme,agents}.exports.test.ts      (T015, T016)
```

## Orden y dependencias

- Lote 1: T001, T002, T003, T004 (archivos disjuntos).
- Lote 2: T005, T006, T007, T008 (necesitan T001, T002; T008 además T004). T005 es la única tarea que escribe `CHANGELOG.md`.
- Lote 3: T009, T010, T011, T012 (T011 necesita T005; T012 BLOQUEADA).
- Lote 4: T013 (ensambla las partes; necesita T006–T011) y T014 (necesita T005).
- Lote 5: T015 (necesita T013) y T016 (necesita T012 y T013; BLOQUEADA).
- Lote 6: T017 (cierre, en serie).
- Cada rama la verifica el orquestador antes del merge `--no-ff`, uno a uno.

## Riesgos

- **Ejemplos con azar en cifras:** un ejemplo puede pasar en Node 26 y fallar en 20. Mitigación: semilla propia o rango; el gate corre 3 veces; T017 lo corre también bajo Node 20.19; el CI arbitra.
- **README largo y tres tablas a la vez:** conflictos de merge si dos agentes editan `README.md`. Mitigación: partes disjuntas + ensamblado en serie (decisión 2).
- **Haiku inventa una cifra de resultado** (comentario `// ~10` sin ejecutar). Mitigación: el informe incluye valor impreso frente a comentario; T017 vuelve a ejecutar cada bloque y T014/revisor Sonnet muestrea.
- **Receta de migración falsa** (p. ej. `d/2` con `d` mal definido). Mitigación: T014 la verifica numéricamente contra 0.3.2; si no coincide se corrige el changelog.
- **Inventario incompleto:** los cambios están dispersos en `HANDOFF.md` y 63 tareas. Mitigación: apéndice A de la spec como punto de partida, comprobación con `git diff --stat 723fa58..HEAD -- src` por clase modificada y revisión final de T017 (cada archivo de `src/` cambiado en ese diff está en el inventario o marcado «solo interno»).
- **Meta-check frágil por la forma de `src/index.ts`:** `export { A as B }` o exports multilínea. Mitigación: el test cubre `as` y bloques multilínea; hoy hay un `export {…}` multilínea (Metrics).
- **`esbuild` no resoluble desde la raíz temporal:** el envoltorio enlaza `node_modules` y avisa; sin él, informar (no añadir dependencias).
- **Desfase futuro:** cada export nuevo exige mención en el README (coste aceptado, pregunta 6).
- **`AGENTS.md` bloqueado puede retrasar el cierre:** si el usuario no responde, 007 se cierra sin T012/T016 y se registra en `HANDOFF.md`.
- **Mensajes de error en español en `src/ModelSaver.ts`** (`:70` «el texto no es JSON válido», `:78` «formato … no soportado», `:109` «structure debe ser un array de números»; el resto de `src/` está en inglés): el changelog los cita tal cual; unificarlos es un cambio de `src/` y queda fuera de alcance (anotado para la 008).
