# Implementation Plan: release-safety

**Branch**: `fix/v0.4` | **Date**: 2026-11-08 | **Spec**: `specs/001-release-safety/spec.md`

## Summary

Poner red de seguridad antes de tocar código matemático: CI en PR/push, publicación condicionada a tests, metadatos y exports correctos, herramientas sin avisos y raíz limpia. No se toca `src/`.

## Technical Context

**Language/Version**: TypeScript ^5, Node 20 en CI
**Primary Dependencies**: ninguna de runtime; dev: tsup ^8, typescript ^5, vitest ^4.1.12 (+ `@vitest/coverage-v8`)
**Testing**: `vitest run` (27 archivos, 327 tests hoy; inestables hasta la feature 002)
**Target Platform**: npm (CJS + ESM + d.ts)
**Constraints**: versión 0.3.2 intacta; sin `npm publish`; sin push a `master`; `package.json`/`package-lock.json` solo el orquestador.

## Estado verificado hoy

- `.github/workflows/` contiene solo `publish.yml` (push a `master`: checkout, setup-node 20, `npm ci`, build, comprobación de versión con `npm view`, `npm publish --access public`).
- Scripts actuales: `build`, `dev`, `test`, `test:watch`, `coverage`. No hay `typecheck`.
- `tsconfig.json`: `include: ["src"]`, `rootDir: "src"`. Una comprobación de `src` + `tests` con la misma configuración, `noEmit` y `rootDir` en la raíz pasa sin errores (probado con un tsconfig temporal fuera del repo).
- `vitest.config.ts`: ESM en paquete CJS; produce el aviso. Fix: `.mts`.
- Sin `LICENSE`, sin `docs/`. `.gitignore`: `node_modules/`, `dist/`, `app/`, `*.js`, `*.d.ts`, `*.js.map`, `.rimuru/`.
- `dist/` no está rastreado.

## Constitution Check

- I (sin runtime): `@vitest/coverage-v8` va en `devDependencies`; verificar `dependencies` ausente. OK.
- IV: no cambia tests; el CI refuerza su ejecución.
- Operación: nadie publica ni sube a `master`; la protección de rama es acción del usuario.
- Propiedad de archivos: los cambios de `package.json` y `package-lock.json` se agrupan en una sola tarea del orquestador (T006).

## Decisiones de diseño

1. **Typecheck con `tsconfig.test.json`** (nuevo, extiende `tsconfig.json`, `noEmit`, `rootDir: "."`, `include: ["src","tests"]`). Evita tocar `tsconfig.json`, que gobierna el build. Script: `tsc --noEmit -p tsconfig.test.json`. La definición de terminado de la constitución (`npx tsc --noEmit -p .`) sigue válida.
2. **`vitest.config.mts`** por `git mv`, contenido idéntico.
3. **`ci.yml`**: triggers `pull_request` y `push` (todas las ramas), un job `ci`, pasos `npm ci` → `npm run typecheck` → `npm run build` → `npm test`. Concurrency por ref con cancelación opcional.
4. **`publish.yml`**: mismo trigger; añade `npm run typecheck` y `npm test` antes de la comprobación de versión y de publicar; el paso publicar mantiene `if: skip == 'false'`.
5. **`package.json`** (T006): `exports["."]` = `types`, `import`, `require`; `sideEffects: false`; `engines.node` (Q2); `repository`, `homepage` (`https://github.com/dniskav/neuron#readme`), `bugs` (`https://github.com/dniskav/neuron/issues`); script `typecheck`; devDependency vía `npm i -D @vitest/coverage-v8@^4.1.12`. `files` ya es `["dist"]`; npm incluye `LICENSE` automáticamente.
6. **Limpieza**: `git rm` de `bun.lock` y el `.tgz`, y entradas en `.gitignore`; `git mv` de los seis documentos a `docs/archive/`.

## Project Structure

```text
specs/001-release-safety/{spec.md,plan.md,tasks.md}
.github/workflows/{ci.yml,publish.yml}
tsconfig.test.json
vitest.config.mts          (antes .ts)
LICENSE
docs/archive/*.md          (6 archivos movidos + README.md opcional)
package.json, package-lock.json   (orquestador)
```

## Riesgos

- Los workflows no se pueden ejecutar localmente; se valida sintaxis YAML y orden de pasos, y la ejecución real requiere push (Q1).
- Hasta la feature 002 la suite es inestable (5 de 25 ejecuciones fallan): el CI puede dar rojo intermitente. No se arregla aquí.
- Si la protección exige el check antes de que corra en `master`, bloquea los merges (FR-012).
- `engines` demasiado estricto puede romper a consumidores (Q2).
