# Feature Specification: release-safety

**Feature Branch**: `fix/v0.4` (se trabaja directamente en ella; no se crea rama por feature)
**Created**: 2026-11-08
**Status**: Draft
**Input**: `specs/ORCHESTRATION.md` sección "001 release-safety" y hallazgo C4 e I8 de `specs/000-audit/AUDIT.md`

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ningún cambio roto llega a `master` ni a npm (Priority: P1)

Hoy el único workflow (`.github/workflows/publish.yml`) se dispara con push a `master`, ejecuta `npm ci`, `npm run build`, comprueba la versión y publica. No ejecuta ni typecheck ni tests, y no hay CI en PR. `master` no tiene protección de rama (AUDIT C4).

**Why this priority**: es la condición previa de todas las demás features; sin ella, un arreglo mal integrado se publica.

**Independent Test**: abrir un PR con un test roto y comprobar que el job de CI falla; hacer push a `master` con un test roto y comprobar que `publish.yml` se detiene antes de publicar.

**Acceptance Scenarios**:

1. **Given** un PR o push a cualquier rama, **When** se dispara `ci.yml`, **Then** corre typecheck (incluye `tests/`), build y tests, y falla si cualquiera falla.
2. **Given** un push a `master` con tests rotos, **When** corre `publish.yml`, **Then** el paso de publicar no se ejecuta.
3. **Given** un push a `master` cuya versión local ya existe en npm, **When** corre `publish.yml`, **Then** no publica.
4. **Given** la versión `0.3.2` de `package.json`, **When** termina la feature, **Then** sigue siendo `0.3.2`.

---

### User Story 2 - Paquete correcto y metadatos completos (Priority: P2)

`package.json` pone `types` después de `import`/`require` en `exports` (I8, aviso de esbuild), no tiene `repository`, `homepage`, `bugs`, `sideEffects` ni `engines`, y el repositorio no tiene `LICENSE` aunque declara `"license": "MIT"`.

**Independent Test**: `npm pack --dry-run` lista `LICENSE` y `dist/*`, y `npm run build` no emite el aviso de `types`.

**Acceptance Scenarios**:

1. **Given** `exports["."]`, **When** se leen sus claves en orden, **Then** `types` es la primera.
2. **Given** el paquete, **When** se empaqueta, **Then** incluye `LICENSE` (MIT) y no incluye `.tgz`.

---

### User Story 3 - Herramientas de desarrollo sin avisos (Priority: P3)

`npm run coverage` no tiene proveedor instalado; `tsconfig.json` solo incluye `src` (los tests no se comprueban); `vitest.config.ts` da el aviso de carga ESM en CJS (confirmado hoy: "ESM syntax in a file loaded as CommonJS (vitest.config.ts:1:1)").

**Acceptance Scenarios**:

1. **Given** `npm run typecheck`, **When** hay un error de tipos en `tests/`, **Then** el comando falla.
2. **Given** `npm run coverage`, **When** se ejecuta, **Then** produce un informe sin error de proveedor.
3. **Given** `npx vitest run`, **When** carga la configuración, **Then** no imprime el aviso de `configLoader`.

---

### User Story 4 - Raíz del repositorio limpia (Priority: P3)

Rastreados hoy en la raíz: `bun.lock`, `dniskav-neuron-0.2.1.tgz` y seis documentos de IA/planes: `NEURON_UPGRADE_PLAN.md`, `NEURON_v0.2.5_PLAN.md`, `REFACTOR_SUGGESTIONS_CLAUDE_SONNET_4_5.md`, `REFACTOR_SUGGESTIONS_minimax-m2.7_cloud.md`, `kimi-k2.5.md`, `refactor-suggestions-qwen3.5.md`. Ninguno es referenciado desde `README.md`, `AGENTS.md` ni `SECURITY.md` (comprobado con grep).

**Acceptance Scenarios**:

1. **Given** el repo, **When** se listan los archivos rastreados, **Then** no hay `bun.lock` ni `*.tgz`, y los seis documentos están en `docs/archive/` con su historial (`git mv`).

---

### User Story 5 - Protección de `master` (Priority: P2, tarea del usuario)

Solo después de que `ci.yml` exista en `master` y haya corrido al menos una vez, el usuario (o el orquestador con su aprobación) activa la protección exigiendo PR y el check de CI. Antes, exigir un check inexistente bloquearía los merges.

### Edge Cases

- `npm view @dniskav/neuron version` falla por red: el script actual cae a `0.0.0` y publicaría una versión ya existente (npm la rechazaría con error, pero el workflow quedaría en rojo). Ver pregunta Q3.
- `.gitignore` ignora `*.js` y `*.d.ts`: no afecta a los `.yml`, `.mts` ni `.json` nuevos.
- Renombrar `vitest.config.ts` a `.mts` no cambia `include: ['tests/**/*.test.ts']`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `.github/workflows/ci.yml` MUST ejecutar `npm ci`, typecheck, build y tests en `pull_request` y `push`, con Node 20 y `permissions: contents: read`.
- **FR-002**: `.github/workflows/publish.yml` MUST ejecutar typecheck y tests antes de `npm publish`, y MUST publicar solo si la versión local difiere de la publicada.
- **FR-003**: `package.json` MUST mantener `version: 0.3.2` y `dependencies` vacío (Constitución I).
- **FR-004**: `exports["."]` MUST declarar `types` antes de `import` y `require`.
- **FR-005**: `package.json` MUST añadir `repository` (`https://github.com/dniskav/neuron.git`, tomado de `git remote`), `homepage`, `bugs`, `sideEffects: false` y `engines`.
- **FR-006**: MUST existir `LICENSE` con el texto MIT.
- **FR-007**: `@vitest/coverage-v8` MUST instalarse en `devDependencies` con la misma línea mayor que `vitest` (`^4.1.12`).
- **FR-008**: MUST existir el script `typecheck` que compruebe `src/` y `tests/` sin emitir; `tsconfig.json` no se modifica (el build usa `rootDir: src`).
- **FR-009**: `vitest.config` MUST cargarse sin aviso (renombrar a `vitest.config.mts`).
- **FR-010**: `bun.lock` y `dniskav-neuron-0.2.1.tgz` MUST salir del control de versiones y no volver a entrar (`.gitignore`).
- **FR-011**: Los seis documentos listados MUST moverse a `docs/archive/`.
- **FR-012**: La protección de `master` MUST activarse solo después de FR-001 en `master`.

### Key Entities

- **Workflow CI** (`ci.yml`): check obligatorio para PR; su nombre de job es el que se exigirá en la protección.
- **Workflow publish** (`publish.yml`): único camino a npm; sigue disparándose solo con push a `master`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `npm run typecheck && npm run build && npx vitest run` termina con código 0 y sin avisos de esbuild ni vitest.
- **SC-002**: `npm pack --dry-run` incluye `LICENSE`, `dist/` y `package.json`, y ningún `.tgz`.
- **SC-003**: Un test roto introducido a propósito hace fallar `ci.yml` y `publish.yml` antes de `npm publish` (verificable por inspección del orden de pasos; la ejecución real solo es posible tras el push, ver Q1).
- **SC-004**: `git ls-files` no contiene `bun.lock` ni `*.tgz` en la raíz.

## Assumptions

- Node 20 es la versión de CI, igual que `publish.yml` actual.
- Todo se hace en `fix/v0.4` sin push; los workflows no se pueden ejecutar hasta que la rama llegue a GitHub.
- `npm run coverage` ya existe como script (`vitest run --coverage`).

## Preguntas abiertas

- **Q1**: ¿Cómo probar `ci.yml` antes de `master`? Opción: push de `fix/v0.4` a origin (no publica, porque `publish.yml` solo escucha `master`) y abrir PR. Requiere permiso del usuario para el push.
- **Q2**: `engines.node`: se propone `>=20` (vitest 4 y CI usan 20). ¿Se quiere declarar soporte de 18 para consumidores? El código compilado (target ES2020) probablemente funciona en 18, pero no está probado.
- **Q3**: ¿Endurecer la comprobación de versión para que solo caiga a `0.0.0` ante un 404 y falle ante otros errores de red?
- **Q4**: Año y titular del `LICENSE` (se propone `Copyright (c) <año del primer commit> dniskav`).
- **Q5**: ¿`docs/archive/` lleva un `README.md` que explique que son informes de IA históricos y no vigentes? (propuesto: sí, una línea).
- **Q6**: Nombre exacto del check a exigir en la protección: se propone el job `ci` del workflow `CI`.
