# Evidencia de las recetas de migración (T014)

Verificación independiente de las recetas numéricas de `CHANGELOG.md` (sección 0.4.0). Todo se midió ejecutando la misma llamada contra la 0.3.2 y contra el `dist` actual (`a95eba8`).

## Método

- **0.3.2**: el commit `723fa58` tiene `package.json` con `"version": "0.3.2"` (comprobado). Se construyó en un worktree aparte (`npm ci`, `npm run build`), se copió `dist/index.mjs` a `t014_old.mjs` y el worktree se borró (`git worktree list` ya no lo muestra).
- **Actual**: `dist/index.mjs` de `a95eba8` (`npm ci`, `npm run build`).
- **Semilla**: PRNG propio (mulberry32) asignado a `Math.random` justo antes de construir cada modelo, con la misma semilla en ambas versiones. Los pesos iniciales (`getWeights()`) coinciden con diferencia máxima 0 en todos los casos (columna «init» de los scripts). Los datos salen de otro PRNG independiente.
- **Diferencia máxima**: `max |w_viejo − w_nuevo|` sobre todos los pesos y sesgos. Una receta es «exacta» si ≤ 1e-9.
- Los scripts `t014_*.mjs` están en el scratchpad y no se commitean.
- El orden de actualización es siempre: una muestra por paso, el mismo orden de muestras en las dos versiones, N pasos desde los pesos iniciales (se reconstruye el modelo para cada N).
- Nota de convención: en la 0.3.2 los optimizadores reciben `-dL/dw` (`w + lr·g`) igual que ahora; la diferencia entre versiones está en el delta de salida y, para los modelos de secuencia, en el signo (ver «Sube → baja»).

## (a) `NetworkN.train`: `lr_nuevo = lr_viejo · n/2`

Datos: 8 muestras aleatorias en [−1,1] (entradas) y [0,1] (objetivos), recorridas en ciclo. Semillas 1, 2, 3; `lr_viejo` 0.05 y 0.3; N = 1, 10, 200. Activaciones por defecto (sigmoide) y `[relu, sigmoid]`. La 0.3.2 acepta las mismas opciones (`activations`, `optimizer`, `residual`, `dropoutRate`), no hubo que adaptar nada.

| Configuración | Semillas | `lr_viejo` → `lr_nuevo` | Pasos | Diferencia máxima con la receta | Con el MISMO `lr` (control) | Veredicto |
|---|---|---|---|---|---|---|
| `[2,3,1]`, sigmoide | 1,2,3 | 0.05→0.025; 0.3→0.15 | 1, 10, 200 | 0 | 2.6e-3 a 2.9e-1 | **exacta** |
| `[2,3,2]`, sigmoide | 1,2,3 | 0.05→0.05; 0.3→0.3 | 1, 10, 200 | 0 | 0 (n = 2: el factor es 1) | **exacta** |
| `[3,4,3]`, sigmoide | 1,2,3 | 0.05→0.075; 0.3→0.45 | 1, 10, 200 | 2.5e-16 | 1.8e-3 a 1.2e-1 | **exacta** |
| `[2,3,1]`, relu + sigmoid | 1,2,3 | igual | 1, 10, 200 | 0 | 1.9e-3 a 7.9e-1 | **exacta** |
| `[2,3,2]`, relu + sigmoid | 1,2,3 | igual | 1, 10, 200 | 0 | 0 | **exacta** |
| `[3,4,3]`, relu + sigmoid | 1,2,3 | igual | 1, 10, 200 | 3.3e-16 | 1.5e-3 a 2.5e-1 | **exacta** |
| `[2,3,n]`, n = 1,2,3,4, XOR, 20 épocas, `lr_viejo` 0.2 (el caso del changelog) | 1,2,3 | 0.2→0.2·n/2 | 80 | n=1: 0; n=2: 0; n=3: 1.4e-16; n=4: 0 | 1.1e-2 a 5.8e-2 (n=2: 0) | **exacta** (confirma las cifras del changelog) |

**Condiciones en las que la receta NO es exacta** (el changelog la presenta sin condiciones). Estructura `[3,3,3,3]`, 8 muestras aleatorias, `lr_viejo` 0.1, `lr_nuevo` 0.15, semillas 1 y 2, N = 1, 10, 200:

| Configuración | Diferencia máxima con `lr·n/2` | Veredicto |
|---|---|---|
| optimizador `Momentum` | 1.7e-18 a 2.2e-16 | exacta (el optimizador es lineal en el gradiente) |
| optimizador `Adam` | 5.0e-2 (N=1) a 6.6 (N=200) | **falsa**. Con `Adam` el mismo `lr` queda a 2.6e-6 a 4.2e-5 (no es exacto por el `eps` de Adam, pero es prácticamente la dinámica anterior): con `Adam` NO se debe reescalar `lr` |
| `dropoutRate` 0.2 | 2.7e-5 (N=1) a 4.4e-3 (N=200) | **aproximada**. Con `dropoutRate` 1e-9 la diferencia baja a 6e-13 a 1.8e-11, así que el residuo viene del dropout. Mecanismo (leído en el diff de `src/NetworkN.ts`, no aislado por separado): la 0.3.2 evaluaba la derivada de la activación sobre la salida ya pasada por dropout |
| `residual: true` | 0.31 a 7.5e3 (diverge) | **falsa** (G08 cambió el gradiente; la receta de escala no aplica) |

## (b) `Network.train`

Datos y semillas como en (a); `Network(2,3,1)`; `lr_viejo` 0.05 y 0.3; `lr_nuevo = lr_viejo/2`; N = 1, 10, 200; también el caso XOR de 20 épocas con `lr` 0.2 (semillas 1, 2, 3).

| Configuración | Semillas | `lr` | Pasos | Diferencia máxima | Control (mismo `lr`) | Veredicto |
|---|---|---|---|---|---|---|
| `Network(2,3,1)`, `lr/2` | 1,2,3 | 0.05, 0.3 | 1, 10, 200 | 0 | 2.6e-3 a 2.9e-1 | **exacta** |
| `Network(2,3,1)`, XOR, 20 épocas | 1,2,3 | 0.2 | 80 | 0 | no medido | **exacta** |
| `Network(2,3,2)`, varias salidas | 5 | 0.3 | 1 | n/a | n/a | **sin receta** (confirmado, ver abajo) |

Varias salidas, `Network(2,3,2)` (17 pesos: 9 de la capa oculta, 4 de la salida 0, 4 de la salida 1), una llamada con `lr` 0.3:

- 0.3.2: `train(x, 0.7, lr)` (solo admite `number`). Pesos que cambian: índices 0 a 12 (los 9 de la capa oculta y los 4 de la salida 0). **Los 4 de la salida 1 (índices 13 a 16) no cambian.** Devuelve 0.01352 (error cuadrático de la salida 0).
- Actual: `train(x, 0.7, lr)` lanza `Network.train: expected 2 targets (number[]), got a single number`; con `[0.7, 0.2]` cambian los 17 pesos y devuelve 0.06066 (MSE medio).
- Como la salida 1 no aprendía antes, ningún `lr` reproduce la dinámica anterior. El texto de G21 es correcto.

## (c) `Autoencoder.train`: `lr_nuevo = lr_viejo · inputSize/2` (APROXIMADA)

`d` = `inputSize`. Datos: 6 muestras aleatorias en [−1,1], recorridas en ciclo. Semillas 1, 2, 3; `lr_viejo` 0.1 y 0.5; `lr_nuevo = lr_viejo·d/2`. Los pesos iniciales de encoder y decoder coinciden (diferencia 0). Rangos sobre las 6 combinaciones semilla × `lr` de cada fila:

| Configuración (`inputSize`, ocultas enc, latente, ocultas dec) | Pasos | Dif. máx. decoder | Dif. máx. encoder |
|---|---|---|---|
| 3, [], 2, [] | 1 | 0 a 1.4e-17 (exacta) | 5.0e-3 a 3.9e-2 |
| 3, [], 2, [] | 50 | 1.3e-2 a 2.9e-1 | 2.9e-1 a 1.15 |
| 3, [4], 2, [4] | 1 | 2e-19 a 1.4e-17 (exacta) | 1.6e-4 a 1.8e-2 |
| 3, [4], 2, [4] | 50 | 1.4e-4 a 1.6e-2 | 2.1e-2 a 1.7e-1 |
| 5, [4], 2, [4] | 1 | 8.7e-19 a 5.6e-17 (exacta) | 1.2e-3 a 1.5e-2 |
| 5, [4], 2, [4] | 50 | 8.3e-4 a 2.7e-2 | 1.5e-2 a 3.3e-1 |
| 4, [6], 2, [6] | 1 | 0 (exacta) | 1.1e-3 a 1.0e-2 |
| 4, [6], 2, [6] | 50 | 5.9e-4 a 5.3e-3 | 1.0e-2 a 1.2e-1 |

Conclusiones:

- **Tras 1 paso el decoder coincide exactamente** (máximo 5.6e-17 en las 24 combinaciones). El encoder NO coincide (1.6e-4 a 3.9e-2): su gradiente cambió (G07).
- **Tras 50 pasos tampoco coincide el decoder** (5.9e-4 a 2.9e-1), porque recibe como entrada la salida del encoder, que ya se ha separado. La frase del changelog «keeps the step of the decoder» es cierta para UN paso con los mismos pesos, no para una trayectoria.
- Veredicto: **aproximada** (la afirmación del changelog se confirma). Las cifras del changelog (tras 30 épocas de 3 muestras, hasta 1.3e-1 en el encoder y 4.8e-3 en el decoder en el caso más pequeño) son del mismo orden que las medidas aquí con otro protocolo (50 pasos): el caso más pequeño da encoder 0.29 a 1.15 y decoder 1.3e-2 a 2.9e-1 (sin capas ocultas) y 2.1e-2 a 1.7e-1 / 1.4e-4 a 1.6e-2 (con [4]).

## (d) `trainWithDeltas` e `inputGradient`

- `trainWithDeltas` existía en la 0.3.2 (`trainWithDeltas(inputs, outputDeltas, lr)`); `inputGradient` **no existía** (`typeof NetworkN.prototype.inputGradient` = `undefined`; es nuevo en 0.4.0, A01).
- `trainWithDeltas`, mismos deltas y mismo `lr` en ambas versiones, estructuras `[2,3,1]`, `[2,3,2]`, `[3,4,3]`, semillas 1, 2, 3, `lr` 0.05 y 0.3, N = 1, 10, 200 (54 combinaciones): diferencia máxima de pesos **0** en todas. **No cambia de escala.**
- Ancla de escala en la versión actual: `train` equivale a `trainWithDeltas` con delta `(2/n)(t − p)·dfn` (diferencia 3.5e-18), es decir, la escala de `train` cambió y la de `trainWithDeltas` no.
- `inputGradient` (solo actual) frente a una derivada numérica centrada (`eps` 1e-6) de `L = −Σ g_k·out_k` con `delta_k = g_k·out_k(1−out_k)`: devuelve `−dL/dx` (convención de signo de la biblioteca) con diferencia máxima 1.8e-11, 7.8e-11 y 2.6e-10 en `[2,3,1]`, `[2,3,2]`, `[3,4,3]`; no modifica los pesos y es determinista. Como no existía antes no hay comparación entre versiones, y «NO cambia de escala» solo se puede afirmar de `trainWithDeltas`.
- Veredicto: **exacta** para `trainWithDeltas`; **no aplicable** para `inputGradient` (nueva).

## (e) Otras afirmaciones cuantitativas del changelog

| Afirmación (id) | Cómo se midió | Resultado | Veredicto |
|---|---|---|---|
| `Seq2Seq`: el encoder recibe el gradiente exacto con escala `lr·T_in/T_out` (G04) | `lr` 1e-6, SGD, `Seq2Seq(1,3,1)`; para cada peso del encoder y del decoder se compara el paso `Δw` con `−lr·dL/dw`, con `dL/dw` por diferencias centradas (`eps` 1e-6) de la pérdida que devuelve `trainStep` con `lr = 0`. Regresión de `Δw` sobre `−lr·g` | Actual: pendiente 1.000000 para encoder y decoder con (T_in, T_out) = (5,3) y (2,6). 0.3.2: decoder −1.000000 (el paso SUBE la pérdida) y encoder −0.8116 / −3.1024 (ni siquiera es un múltiplo constante del gradiente) | **exacta** (la escala `lr·T_in/T_out` hace que todo el modelo dé exactamente `−lr·dL/dw`) |
| `NetworkTransformer.train`: el paso de SGD directo baja `count` veces (G02). Medido en el changelog: `tokenEmb` 4.162e-2 → 1.038e-2 con `count` = 4 | Cambio de `tokenEmb` tras 1 paso, `NetworkTransformer(4)` con `d_model` 8, 1 bloque, `lr` 0.01, máscara con k = 1, 2, 4 posiciones, semillas 1 y 2 | Cociente norma(viejo)/norma(actual): k=1: 1.000 y 1.002; k=2: 1.994 y 1.997; k=4: 4.002 y 4.008. Con `lr·k` el cociente es 0.997 a 1.002. Norma con k=4: 3.9e-2 → 9.8e-3 (misma proporción que 4.162e-2 → 1.038e-2). **Pero el coseno entre el cambio viejo y el nuevo es −1.000**: en la 0.3.2 `tokenEmb` se movía en la dirección contraria | **exacta en magnitud**; el changelog no dice que la dirección era la opuesta (ver correcciones) |
| «Adam on the weight matrices is invariant to this scale» (G02) | Mecanismo de Adam, y consistente con el hallazgo de (a) con `Adam` | No medido de forma directa en `NetworkTransformer` | **no verificada** |
| `AttentionHead`: `dX` dependía de `lr` (G05). Cifras del changelog: 0.012523, 0.018809, 0.075385 | `AttentionHead(4,2,2)` con semilla 3, una entrada de 3×4 y `dOut` fijos, `lr` 0, 0.05, 0.5 | 0.3.2: −0.083783, −0.034972, 0.404321. Actual: −0.083783 en los tres. Los valores del changelog NO se reproducen (pesos y entradas sin especificar), la conclusión sí | **cualitativa confirmada**; cifras exactas **no verificadas** |
| `BatchNorm.backward` (G09): `[3.03, 2.71]` → `[0.278, 0.199]` con `m` 0.1; con `momentum` 0 el gradiente es 0 | `BatchNorm(2, m)`, `forward([1,2])`, `backward([1,1])` | 0.3.2: [3.029, 2.712]. Actual: [0.278, 0.199]. Con `momentum` 0: 0.3.2 [316.228, 316.228], actual [0, 0] | **exacta** |
| `MaxPool2D.backward` (G16): `MaxPool2D(2,1)` sobre 3×3 daba una suma de gradiente de 5 | `MaxPool2D(2,1)`, `backward` con unos, 300 entradas 3×3 aleatorias (con y sin empates) y una entrada creciente | Actual: siempre 4 (nº de ventanas). 0.3.2: entre 4 y 9 según la entrada (9 con 1..9 creciente; 5 en 94 de 300 casos con valores continuos) | **imprecisa**: «5» depende de la entrada (ver correcciones) |
| `tanh(355)` (G19): NaN antes, 1 ahora | `tanh.fn(355)` | 0.3.2: NaN. Actual: 1 | **exacta** |
| `Metrics.f1Score` macro (G20): 0.6933 → 0.6556 | `f1Score([0,0,1,1,2,2],[0,1,1,1,2,0])` | 0.3.2: 0.6933. Actual: 0.6556 | **exacta** |
| `TCN` `lr` estable (E04): baja con `lr` 0.21 a 0.65, sube con 1.0, diverge desde 2.0 | `TCN(1,4,2,2,1)`, `x = sin(1.3t+0.4)`, T = 16, objetivo `x[t−1]` (con `x[−1] = sin(−0.9)`), 300 pasos, semillas 1 a 5, solo la versión actual | `final/inicial`: 0.21: 1.1e-3 a 2.6e-2; 0.3: 2.4e-4 a 3.1e-2; 0.4: 1.6e-2 a 3.4e-2; 0.5: 2.2e-2 a 5.3e-2; 0.65: 6.3e-2 a 0.56; **1.0: 3.8 a 15.0 (sube)**; 2.0: 5e284 a 2e288; 4.0: Infinity | **cualitativa confirmada** (misma frontera). Cifras exactas **no verificadas**: aquí 0.4 y 0.5 llegan a 3.4e-2 y 5.3e-2 y el changelog dice ≤ 0.02; los rangos de 0.65 y 1.0 también difieren (otra elección de `x[−1]` y de PRNG) |
| G01: «`RNN` 0.5701 a 6.6e14; `TCN` NaN; `Seq2Seq` 0.6106 a 9.0e27» y «0.0290; 0.0369; 0.1253» | Datos del changelog sin especificar; se usaron datos propios (sección siguiente) | Misma conducta: la pérdida sube en 0.3.2 (hasta 1e17 o NaN) y baja ahora | **cualitativa confirmada**; cifras exactas **no verificadas** (datos no especificados) |
| `lr = 0` lanza en `Trainer` (la 0.3.2 lo aceptaba) y no lanza en las redes (punto 4 de «Read this first», V27) | No medido en esta tarea | La verificación es de T001 (inventario) | **no verificada** aquí |
| E05 `VAE` `klWeight`, E06 `TSNE` `lr`, G10 `Word2Vec` (1.821 a 25.631 y 1.814 a 1.681), G17 `PCA` (`[0.6798]`), `PCA` ~1e-111, G25 `Trainer` f1 (7/9 a 2/3), V26 (125 000 elementos) | No se repitieron | Requieren corpus o datos que el changelog no especifica del todo, 3000 épocas, o no son recetas de migración | **no verificadas** (datos no especificados o fuera de alcance) |
| G03 (la proyección de `TCN` iba con `lr/T` y las capas conv con `lr`) y G06 (un solo paso de Adam en `inputProj`) | No se repitieron | Son descripciones de código, no recetas numéricas; G03 se apoya en T028 (coseno 1.000000) | **no verificadas** |

## Sube → baja: la pérdida en 0.3.2 frente a la actual

Misma llamada en ambas versiones, pesos iniciales idénticos (misma semilla; la pérdida del primer paso es igual en las dos columnas), una secuencia fija repetida en cada paso. Modelos: `RNN`, `TCN` y `Seq2Seq` con 1 entrada, 4 ocultas y 1 salida sobre 6 pasos de tiempo con entrada y objetivo aleatorios; `NetworkTransformer` con `vocabSize` 5, `d_model` 8, 2 cabezas, `d_ff` 16, 1 bloque y tarea «token siguiente»; `NetworkTransformerRL` con `seqLen` 4, `inputDim` 3, 3 acciones y objetivo fijo. «Inicial» es la pérdida devuelta por la primera llamada y «final» la de la última. Se muestra la semilla 1; con las semillas 2 y 3 ocurre lo mismo (sube en 0.3.2 y baja ahora en las 18 parejas: 6 configuraciones × 3 semillas).

| Modelo | `lr`, pasos | 0.3.2 (inicial → final) | Actual (inicial → final) |
|---|---|---|---|
| `RNN(1,4,1)` | 0.05, 60 | 0.2632 → 4.07e18 | 0.2632 → 0.0791 |
| `RNN(1,4,1)` | 0.01, 20 | 0.2632 → 1.143 | 0.2632 → 0.1387 |
| `TCN(1,4,2,2,1)` | 0.1, 60 | 0.8017 → NaN (máx. 4.7e244) | 0.8017 → 0.0495 |
| `TCN(1,4,2,2,1)` | 0.01, 20 | 0.8017 → NaN (máx. 1.6e155) | 0.8017 → 0.2137 |
| `Seq2Seq(1,4,1)` | 0.1, 60 | 0.4973 → 8.8e30 | 0.4973 → 0.0997 |
| `Seq2Seq(1,4,1)` | 0.01, 20 | 0.4973 → 1.164 | 0.4973 → 0.2436 |
| `NetworkTransformer` | 0.01, 100 | 1.942 → 16.12 | 1.942 → 0.0169 |
| `NetworkTransformer` | 0.003, 40 | 1.942 → 5.891 | 1.942 → 0.6646 |
| `NetworkTransformerRL` | 0.01, 100 | 0.8886 → 2.58e19 | 0.8886 → 3.3e-6 |
| `NetworkTransformerRL` | 0.003, 40 | 0.8886 → 11.43 | 0.8886 → 2.8e-4 |

Notas: el 16.12 de `NetworkTransformer` en 0.3.2 es `−ln(1e-7)`, el tope del `Math.max(prob, 1e-7)` de la entropía cruzada (satura, no diverge más). La conclusión central del changelog («the loss went up instead of down» en `RNN`, `TCN`, `Seq2Seq`, `NetworkTransformer` y `NetworkTransformerRL`, y «in 0.4.0 the same calls lower the loss») **queda confirmada**, incluso con `lr` 0.01 y 20 pasos. La medida de `Seq2Seq` en (e) muestra el mecanismo: en 0.3.2 el paso del decoder tiene pendiente −1 sobre `−lr·dL/dw`, es decir, sube la pérdida. `Autoencoder`, `VAE`, `GAN`, `ContrastiveLearning` y `Word2Vec` no se midieron aquí.

## Correcciones que necesita el changelog

Ninguna afirmación es falsa en el caso por defecto; hay cinco imprecisiones en recetas o cifras. Se propone el texto exacto a cambiar.

1. **Receta de `NetworkN.train` presentada como «exact» sin condiciones** («Read this first» punto 3, comentario `// NetworkN.train and Network.train with n outputs (exact):`; E01 «Migrate: `lr_new = lr_old * n / 2` reproduces the old dynamics»; «Migration guide» punto 2 «(exact; E01, E02)»).
   - Falsa con `Adam` (diferencia 5e-2 a 6.6 con la receta; con el mismo `lr` queda en 2.6e-6 a 4.2e-5), falsa con `residual: true` (G08 cambió el gradiente) y aproximada con `dropoutRate > 0` (2.7e-5 a 4.4e-3 con `dropoutRate` 0.2).
   - Exacta (≤ 3.3e-16) con el optimizador por defecto y con `Momentum`, sin `residual` y sin dropout.
   - Formulación propuesta: «exact with the default optimizer (SGD) or `Momentum`, no `residual` and `dropoutRate` 0; with `Adam` keep `lr` (Adam is scale-free, old and new runs differ by about 1e-5); with `residual` there is no recipe (G08); with `dropoutRate` > 0 it is approximate».
2. **`Network.train` en el mismo bloque**: el comentario dice «NetworkN.train and Network.train with n outputs», pero `Network.train` solo tiene receta con n = 1 (`lr/2`). Formulación propuesta: «NetworkN.train with n outputs, and Network.train with n = 1 (exact)». El párrafo siguiente ya lo dice bien; basta ajustar el comentario.
3. **Receta de `Autoencoder` («keeps the step of the decoder»)**: es cierta solo para el primer paso. Tras 1 paso el decoder coincide exactamente (≤ 5.6e-17) y el encoder no (1.6e-4 a 3.9e-2); tras 50 pasos el decoder también difiere (5.9e-4 a 2.9e-1) porque su entrada, la salida del encoder, ya se separó. Formulación propuesta: «`lr * d / 2` gives the same decoder update for the FIRST step from identical weights (difference below 1e-16), but not the encoder update, and after a few dozen steps the decoder weights differ too (6e-4 to 3e-1 after 50 steps)».
4. **G02, «raise `lr` by about the number of masked positions to keep the direct SGD step»**: la magnitud se confirma (cociente 0.997 a 1.002 con `lr·count`), pero el cambio de `tokenEmb` en la 0.3.2 iba en la dirección CONTRARIA (coseno −1.000; la pérdida subía). Formulación propuesta: «...to keep the SIZE of the direct SGD step (in 0.3.2 that step pointed the other way, see G01)».
5. **G16, «`MaxPool2D(2, 1)` on a 3x3 input gave a gradient sum of 5»**: la suma en la 0.3.2 depende de la entrada (entre 4 y 9 con 300 entradas aleatorias; 9 con la entrada 1..9). Formulación propuesta: indicar la entrada del caso medido, o escribir «gave a gradient sum above the number of windows (for example 5 or 9 depending on the input; now always 4)».

Cifras del changelog que no se pudieron reproducir exactamente por falta de datos o semilla (no son errores): G01 (0.5701, 6.6e14, 0.0290, 0.6199, 0.0369, 0.6106, 9.0e27, 0.1253), G05 (0.012523, 0.018809, 0.075385), E04 (rangos de `final/inicial`), E05, E06, G10, G17.
