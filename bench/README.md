# Benchmark de `@dniskav/neuron`

Mide memoria y tiempo de `NetworkN` y de `NetworkFast` sobre el build de `dist/index.mjs`. Es Node puro: sin dependencias, sin `tsx`.

## Cómo se ejecuta

Desde la raíz del repo, después de compilar:

```sh
npm run build
node --expose-gc bench/run.mjs            # estructuras [256,256,256,10] y [512,512,512,10]
node --expose-gc bench/run.mjs --quick    # solo [256,256,256,10]
node --expose-gc bench/run.mjs --dist <ruta>   # otro build; <ruta> es el directorio que contiene index.mjs
```

`--expose-gc` es obligatorio: la medición de memoria fuerza el GC. Sin él, el script termina con código 1 y un aviso.

Sin `--dist` usa `../dist` respecto a este script. Con `--dist` se puede medir el build de otra versión con el mismo script (así se generó la línea base, ver más abajo).

Solo el JSON va a stdout, así que se puede encadenar. El progreso va a stderr.

```sh
node --expose-gc bench/run.mjs > salida.json
```

## Qué mide

Para cada estructura y cada optimizador (`sgd` con lr 0.01, `adam` con lr 0.001), activaciones `relu, relu, sigmoid`:

- `params`: número de pesos, `nOut·(nIn+1)` por capa, con sesgo.
- `bytesPerParam`: memoria por parámetro, medida como **`heapUsed` más `arrayBuffers`**. Con GC previo (dos veces), se toma `process.memoryUsage()` antes y después de construir la red, se suma la diferencia de `heapUsed` y la de `arrayBuffers`, y se divide entre `params`. `arrayBuffers` cuenta la memoria de los `Float64Array`, que vive fuera del heap de V8; por eso `heapUsed` solo subestima las redes que usan buffers.
- `heapOnlyBytesPerParam`: la misma diferencia, pero solo de `heapUsed`, dividida entre `params`. Es el dato que daba el script antes de T001b. Como `arrayBuffers` no baja, siempre es menor o igual que `bytesPerParam`; la diferencia es la memoria que el heap no ve.
- `trainMs`: milisegundos por ejemplo en `train`.
- `predictMs`: milisegundos por ejemplo en `predict`.
- `batchMsPerSample` (solo `NetworkFast`): tiempo de `trainBatch` con 32 ejemplos, dividido entre 32.

Por qué hay dos columnas de memoria: en 0.4 cada peso de Adam era un objeto JS, así que `heapUsed` lo veía entero. Desde T009 el estado de `NetworkN` con Adam vive en `Float64Array` (fuera del heap). Medir solo `heapUsed` daba unos 14,6 bytes/param para `NetworkN` con Adam, cuando la memoria real es unos 30,6.

Los tiempos son la mediana de 5 tandas de 20 llamadas, tras 2 tandas de calentamiento que se descartan. Los datos son fijos (mulberry32, semilla 42) y se generan antes de medir.

Si el módulo exporta `NetworkFast` (desde 0.5), el script añade sus filas (`train`, `predict` y `trainBatch`). Si no, las omite.

## Formato de salida

```json
{ "node": "v20.x", "cpu": "...", "date": "ISO-8601",
  "rows": [{ "model": "NetworkN", "structure": [256,256,256,10], "optimizer": "sgd",
             "params": 134154, "bytesPerParam": 40.8, "heapOnlyBytesPerParam": 40.8,
             "trainMs": 3.5, "predictMs": 0.17 }] }
```

## Línea base

`specs/010-fast-core/bench-baseline.json` es la salida completa del script sobre el build de `bd479a1`, la última integración de `feat/0.5-core` antes de T009 (la 0.4 más el lote 2 de la feature 010). Se midió con `node v26.12.1` en Apple M3. Para regenerarla:

```sh
git worktree add --detach <tmp> bd479a1
(cd <tmp> && npm ci && npm run build)
node --expose-gc bench/run.mjs --dist <tmp>/dist > specs/010-fast-core/bench-baseline.json
git worktree remove --force <tmp>
```

`bench/run.mjs` se toma del checkout actual, no del commit medido.

## Cómo comparar

La tabla oficial de cada versión la mide el verifier (T026) **sin ningún otro agente ni proceso pesado en marcha**: con agentes corriendo tests en paralelo, los tiempos varían hasta ×2 (T001b). Hasta que T026 la publique, las cifras de `CHANGELOG.md` son mediciones de desarrollo, no la tabla oficial.

- Compara solo mediciones de la **misma máquina y la misma versión de Node**.
- La memoria se repite entre pasadas (dos pasadas completas de la línea base dieron los mismos bytes/parámetro con dos decimales). Los tiempos no: en la línea base, `trainMs` de `[256,256,256,10]` con SGD fue 3,5 ms en una pasada y 8,2 ms en otra, y el de `[512,512,512,10]` con Adam fue 44 ms y 22 ms. Repite la medición antes de sacar conclusiones de diferencias pequeñas, sobre todo en tiempos.
- Para una comparación con otra versión, compila esa versión en un worktree aparte y ejecuta el script con `--dist` en las dos, en la misma sesión.
