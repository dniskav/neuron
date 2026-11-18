# Benchmark de `@dniskav/neuron`

Mide memoria y tiempo de `NetworkN` (y de `NetworkFast` cuando exista) sobre el build de `dist/index.mjs`. Es Node puro: sin dependencias, sin `tsx`.

## Cómo se ejecuta

Desde la raíz del repo, después de compilar:

```sh
npm run build
node --expose-gc bench/run.mjs            # estructuras [256,256,256,10] y [512,512,512,10]
node --expose-gc bench/run.mjs --quick    # solo [256,256,256,10]
```

`--expose-gc` es obligatorio: la medición de memoria fuerza el GC. Sin él, el script termina con código 1 y un aviso.

Solo el JSON va a stdout, así que se puede encadenar. El progreso va a stderr.

```sh
node --expose-gc bench/run.mjs > salida.json
```

## Qué mide

Para cada estructura y cada optimizador (`sgd` con lr 0.01, `adam` con lr 0.001), activaciones `relu, relu, sigmoid`:

- `params`: número de pesos, `nOut·(nIn+1)` por capa, con sesgo.
- `bytesPerParam`: diferencia de `heapUsed` antes y después de construir la red, con GC previo, dividida entre `params`.
- `trainMs`: milisegundos por ejemplo en `train`.
- `predictMs`: milisegundos por ejemplo en `predict`.
- `batchMsPerSample` (solo `NetworkFast`): tiempo de `trainBatch` con 32 ejemplos, dividido entre 32.

Los tiempos son la mediana de 5 tandas de 20 llamadas, tras 2 tandas de calentamiento que se descartan. Los datos son fijos (mulberry32, semilla 42) y se generan antes de medir.

Si el módulo exporta `NetworkFast`, el script añade sus filas (`train`, `predict` y `trainBatch`). Si no, las omite. Hoy `NetworkFast` no existe, así que solo hay filas de `NetworkN`.

## Formato de salida

```json
{ "node": "v20.x", "cpu": "...", "date": "ISO-8601",
  "rows": [{ "model": "NetworkN", "structure": [256,256,256,10], "optimizer": "sgd",
             "params": 134154, "bytesPerParam": 40.7, "trainMs": 2.8, "predictMs": 0.17 }] }
```

## Cómo comparar

- La línea base está en `specs/010-fast-core/bench-baseline.json` y corresponde a la 0.4.0 (`npm run build` antes de cualquier cambio en `src/`).
- Compara solo mediciones de la **misma máquina y la misma versión de Node**. Los tiempos varían entre ejecuciones: en dos pasadas completas de la línea base, `trainMs` difirió entre el 3 % y el 30 %, y una primera ejecución en frío llegó a ser unas cuatro veces más lenta. Repite la medición antes de sacar conclusiones de diferencias pequeñas.
- Para una comparación con `master`, compila la otra versión en un worktree aparte y ejecuta el script en las dos, en la misma sesión.
