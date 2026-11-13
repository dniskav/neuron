# Comprobación de tipos de `neuron_app` con 0.2.5 frente a la librería actual (T006)

Herramienta: `specs/008-release-app/tools/app-typecheck.mjs`. Lee la app y no escribe nada en ella.

```
node specs/008-release-app/tools/app-typecheck.mjs --app /Users/daniel/Desktop/projects/experiments/neuron_app
```

## Entradas

- **App:** `/Users/daniel/Desktop/projects/experiments/neuron_app`, commit `b4d139c` (`git -C <app> rev-parse --short HEAD`), rama local `main`, sin subir.
- **Copia de trabajo de la app: NO está limpia.** Cuatro archivos modificados sin commitear:
  - `package.json`: añade a `dependencies` paquetes transitivos (`@babel/core`, `postcss`, `zod`, etc.) y sube `vite` de `^8.0.1` a `^8.3.5`.
  - `vite.config.ts`: añade un alias `resolve.alias['@dniskav/neuron']` hacia `node_modules/@dniskav/neuron/dist/index.mjs`.
  - `package-lock.json` y `bun.lock`: subidas transitivas y el lock regenerado.

  Esto afecta a la reproducibilidad. `tsc` no lee `vite.config.ts`, pero sí resuelve tipos de `react`, `vite` y demás según el `node_modules` instalado, que depende de `package.json` y del lock sin commitear. El resultado vale para el árbol instalado en esta máquina, no solo para el commit `b4d139c`.
- **0.2.5:** `node_modules/@dniskav/neuron/dist/index.d.ts` de la app (`package.json` dice 0.2.5).
- **Librería actual:** `dist/index.d.ts` del worktree de la librería, construido con `npm run build` desde la punta de `fix/v0.4` (`fa3f6d3`). Su `package.json` dice 0.3.2 porque el bump a 0.4.0 es el paso O03.

## Configuración de tsc

- Hereda el `tsconfig.app.json` REAL de la app mediante `extends` (solo lectura). Es más estricto que la aproximación del inventario: `verbatimModuleSyntax`, `erasableSyntaxOnly`, `noUnusedParameters`, `noUncheckedSideEffectImports`, etc.
- `types: vite/client` se resuelve con un enlace simbólico a `node_modules/vite` de la app dentro del directorio temporal (`os.tmpdir()`), que se borra al terminar.
- `@dniskav/neuron` se fija con `paths` a cada `dist/index.d.ts`.
- `include`: el `src/` de la app.

## Errores de tipos

Con 0.2.5 (8 errores):

```
src/cards/ArquitectoCard/CurriculumArquitecto.tsx:373:TS2322
src/cards/ArquitectoCard/DetectorArquitecto.tsx:255:TS2322
src/cards/ArquitectoCard/SnakeArquitecto.tsx:229:TS2322
src/cards/RobotCard/RobotAgent/RobotAgent.tsx:169:TS2322
src/cards/SudokuCard/SudokuCard.tsx:156:TS2322
src/cards/SudokuCard/SudokuCard.tsx:9:TS6133
src/core/sudoku/SudokuSession.ts:17:TS6133
src/lab/stories/hooks.stories.tsx:8:TS6133
```

Con la librería actual (8 errores): la misma lista, línea a línea.

- Solo con la librería actual: ninguno.
- Solo con 0.2.5: ninguno.
- Estos 8 coinciden con los de `app-inventory.md` (Hallazgo 2): 5 `TS2322` de `Partial<LayerConfig>` frente a `Partial<HiddenLayerConfig>` y 3 `TS6133` de imports sin uso. No dependen de la librería: ya fallan con 0.2.5.

## Mutante

Copias en el scratchpad (`t006_app_mutant*`, con `src/` y `tsconfig.app.json` copiados y `node_modules` enlazado a la app). El archivo añadido es `src/t006_mutant.ts`.

1. **Opción retirada entre 0.2.5 y la librería actual** (`new Trainer(net, { epochs: 1, clipValue: 1 })`). `clipValue` existía en `TrainerOptions` de 0.2.5 y ya no existe. Resultado: 8 errores con 0.2.5, 9 con la librería actual, `src/t006_mutant.ts:4:TS2353` solo con 0.4.0. Sale **1**.
2. **Método que no existe en ninguna de las dos versiones** (`new Trainer(...).metodoInexistente()`). Resultado: 9 errores con 0.2.5 y 9 con la librería actual, diferencia 0. Sale **0**.

## Limitación del criterio

El criterio del Gate («errores NUEVOS con la librería actual») solo detecta lo que cambió entre 0.2.5 y la librería actual. Un método inexistente en ambas versiones ya falla con 0.2.5 y no es un problema de la publicación, así que sale 0. Para detectar un método que nunca existió, habría que comparar también con el total de errores de la app sin mutar. Se deja documentado y no se cambia el criterio.

## Gate

`0.2.5: 8 errores | 0.4.0: 8 errores | solo en 0.4.0: 0`, sale 0. Mutante con API retirada: sale 1. Gate común: `npm run typecheck` sin errores y `git diff --quiet master -- src package.json package-lock.json` con código 0.
