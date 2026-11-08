# Constitución de @dniskav/neuron

## Principios

### I. Sin dependencias de runtime
`dependencies` en `package.json` permanece vacío. Las herramientas van en `devDependencies`; un parche de seguridad transitivo se resuelve subiendo la herramienta o con `overrides`.

### II. Un gradiente correcto es verificable
Todo modelo o capa entrenable tiene un test que compara su actualización real con diferencias finitas, con coseno mayor que 0.999. Una excepción debe estar justificada y documentada en su spec, indicando la aproximación y el umbral usados. Convención: los optimizadores hacen `w + lr·g`, así que reciben el gradiente negado (`-dL/dw`).

### III. Entrenar significa aprender
Todo modelo entrenable tiene un test con semilla fija que demuestra que la pérdida baja claramente en una tarea pequeña. Tolerar que la pérdida suba no cuenta como prueba.

### IV. Tests deterministas y que detectan errores
Los tests fijan la semilla de `Math.random`. La suite debe pasar 20 ejecuciones seguidas. Un test nuevo debe fallar cuando se introduce un error a propósito en el código que cubre; si no falla, se rechaza.

### V. Fallar con claridad
Las entradas inválidas lanzan un error explícito en lugar de producir `NaN` o ignorarse en silencio. Una opción que no hace nada se implementa o se elimina.

### VI. Librería educativa y legible
El código sigue siendo autocontenido y escrito a mano, con las matemáticas explicadas en comentarios. Un comentario o ejemplo que no coincide con el comportamiento real es un defecto.

### VII. Cambios incompatibles permitidos en 0.x
Se aceptan cambios incompatibles si corrigen un comportamiento incorrecto. Deben documentarse en el changelog con su migración. Las apps que usen la librería se actualizan después.

## Restricciones de operación

- **Publicación:** solo con aprobación explícita del usuario. Ningún agente ejecuta `npm publish`, sube a `master` ni cambia la versión por su cuenta.
- **Git:** el trabajo se integra en `fix/v0.4`; solo el orquestador hace merge. Ningún agente fuerza un push ni reescribe historia compartida.
- **Propiedad de archivos:** cada tarea declara sus archivos y un agente solo toca esos. `tasks.md`, `package.json`, `package-lock.json`, `src/index.ts` y el changelog son exclusivos del orquestador.
- **Modelos:** Haiku por defecto, Sonnet cuando hace falta y Opus solo con permiso del usuario, según `specs/ORCHESTRATION.md`.

## Definición de terminado

Una tarea está terminada cuando pasan `npx tsc --noEmit -p .`, `npm run build` y `npx vitest run`, además de su criterio propio, comprobado por un agente distinto del que la implementó. Ningún agente da por buena su propia tarea.

## Gobernanza

Esta constitución prevalece sobre las specs y los planes. Cambiarla requiere aprobación del usuario.

**Versión**: 1.0.0 | **Ratificada**: 2026-11-08 | **Última enmienda**: 2026-11-08
