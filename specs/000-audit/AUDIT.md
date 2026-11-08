# Auditoría de `@dniskav/neuron` (2026-11-08)

Revisión de la v0.3.2 (commit `723fa58`). Medido con semillas fijas contra diferencias finitas. Los scripts están en `evidence/`.

**Convención usada en las medidas:** `cos` es el coseno entre la actualización real de los pesos y `-∇L` numérico. `1` es descenso correcto y `-1` es ascenso.

## Causa raíz principal

`SGD` y `Adam` hacen `w + lr·g` (`src/optimizers.ts:30` y `:99`), y `EmbeddingMatrix.update` hace `W += lr·grad`. Por tanto esperan **el gradiente ya negado** (`-dL/dw`). `NetworkN`, `Network` y `NetworkLSTM` lo respetan; los modelos de abajo no.

## Críticos

### C1. Cinco modelos entrenan en la dirección equivocada

| Modelo | Línea (`v0.3.2`) | Pérdida al entrenar | Control con signo invertido | cos |
|---|---|---|---|---|
| `NetworkTransformer` | `src/NetworkTransformer.ts:137` | 1.704 → 203.8 (`lr` .001) | 1.704 → 0.015 | -1.0000 |
| `NetworkTransformerRL` | `src/NetworkTransformerRL.ts:127` | 0.440 → 74.9 (`lr` .001) | 0.440 → 0.000 | -1.0000 |
| `RNN` | `src/RNN.ts:187` | 0.325 → 1.9e103 (`lr` .02) | 0.325 → 8.7e-10 | -1.0000 |
| `TCN` | `src/TCN.ts:318` | 0.183 → NaN (`lr` .02) | 0.183 → 1.5e-9 | -0.8945 |
| `Seq2Seq` | `src/Seq2Seq.ts:176` | 0.172 → 0.299 (`lr` .0005); diverge con `lr` mayor | no probado de forma válida | -0.98 |

- La corrección va en el origen de cada cadena: `target - pred` en lugar de `pred - target`. Las capas internas (`AttentionHead`, `Conv1D`, `LSTMLayer`, etc.) ya son correctas con la convención `dOut = -dL`.
- `Seq2Seq` además aproxima el gradiente del encoder: solo envía `dhSeq[0]` al último paso e ignora la celda `c`.
- El coseno de TCN no es -1 exacto: revisar el camino residual en `src/TCN.ts`.
- **`neuron_app` usa la 0.2.5, que da los mismos números.** Sudoku, generación de texto y laberinto con Transformer llaman a `train` con `lr` positivo. No se ejecutó la app. Las líneas son de `v0.3.2`: tras `fix/v0.4` pueden cambiar.

### C2. `GAN` y `VAE` no implementan su algoritmo

- `GAN` (`src/GAN.ts:137`): `gDelta` es uniforme y positivo, no propaga a través del discriminador. Con objetivo 0.20 la media generada acabó en 1.000 con 3 semillas.
- `VAE` (`src/VAE.ts:140`): el encoder solo recibe el gradiente de KL. `mu` = 0 para toda entrada y reconstrucción 0.1875, igual a predecir la media, también con `tanh`/`linear`.

### C3. Los tests no detectan estos errores

- `signedPerturbationCheck` (`tests/GradientCheck.test.ts`) nunca llama a `train` ni a `backward`. `multiStepLossCheck` tolera subir hasta un 50 %.
- 29 de 56 módulos no aparecen en ningún test: Autoencoder, ContrastiveLearning, Conv2D, DataAugmentation, DatasetLoader, DecisionTree, EarlyStopping, Flatten, GAN, HopfieldNetwork, KMeans, LinearRegression, LogisticRegression, LossPlotter, MaxPool2D, NaiveBayes, PCA, Perceptron, PositionalEncoding, RNN, SOM, Seq2Seq, TCN, TSNE, Tape, Tokenizer, VAE, WeightInspector, Word2Vec. `Metrics` solo aparece en un comentario.
- La suite falló en 5 de 25 ejecuciones (`NetworkTransformer` multi-step y dropout de `NetworkN`).

### C4. Publicación sin red de seguridad

- `.github/workflows/publish.yml` publica al subir a `master` si cambia la versión, ejecutando solo el build. No hay workflow de tests.
- `master` no tiene protección de rama.

## Importantes

| ID | Problema | Evidencia |
|---|---|---|
| I1 | `residual` en `NetworkN` (`src/NetworkN.ts:195`): `dfn` se evalúa sobre el valor ya sumado y el salto mezcla deltas previos a la activación | cos 0.86; -0.89 si la salida es residual |
| I2 | `setWeights` y `ModelSaver.fromJSON` no validan longitud | `[NaN]` con una capa; acepta pesos de otro modelo |
| I3 | `Trainer`: `clipValue` no hace nada; `earlyStopping` sin validación se ignora sin aviso; no restaura los mejores pesos | 0 avisos, `stop=maxEpochs` |
| I4 | `padding: 'same'` con kernel par | `Conv2D` 4×4 → 5×5; `Conv1D` anuncia 4 y devuelve 3 |
| I5 | `Network` con varias salidas solo entrena la primera | 2ª salida sin cambios tras 100 pasos |
| I6 | `LSTMLayer.getWeights()` devuelve referencias vivas | la copia cambia al entrenar; revisar también `GRULayer` y `NetworkLSTM` |
| I7 | `Value.backward()` es recursivo | desborda la pila con 20 000 nodos |
| I8 | `exports` en `package.json`: `types` va después de `import`/`require` | aviso de esbuild al compilar |

## Mantenimiento

- Falta `LICENSE`; faltan `repository`, `homepage`, `bugs`, `sideEffects: false` y `engines`.
- `AGENTS.md` dice que no hay convoluciones, autograd ni funciones de pérdida, y lista 15 clases.
- `npm run coverage` no tiene proveedor instalado; `tsconfig` no comprueba `tests/`; `vitest.config.ts` da un aviso de carga ESM.
- Raíz con `dniskav-neuron-0.2.1.tgz`, cuatro informes de IA, dos planes y dos lockfiles.
- README mezcla inglés y español; 15 de 28 ejemplos no corren tal cual porque usan datos sin definir (no son fallos de API).
- `DecisionTree` da 0.5 de acierto en XOR. Probablemente sea la limitación de una división voraz con ganancia cero; sin confirmar.

## Lo que está bien (no regresar)

- `cos` = 1.0000 en `Network`, `NetworkN` sin `residual`, `NetworkLSTM`, `LSTMLayer`, `GRULayer`, `Conv1D`, `Conv2D` y `AttentionHead`. `Autoencoder` 0.998 a 1.000 y aprende.
- Resultados correctos o razonables en `LogisticRegression`, `SoftmaxRegression`, `GaussianNaiveBayes`, `KMeans`, `PCA`, `HopfieldNetwork`, `Word2Vec` y `TSNE`.
- Las sondas de `Perceptron`, `LinearRegression` y `auc` lanzaron error por mal uso en el propio script; sus API no se validaron así y no hay que tratarlo como un fallo.

## Cómo repetir las medidas

```bash
npx tsup src/index.ts --format cjs,esm --dts --out-dir /tmp/neuron-dist
mkdir -p /tmp/neuron-probe && ln -sfn /tmp/neuron-dist /tmp/neuron-probe/dist
node specs/000-audit/evidence/probe.cjs /tmp/neuron-probe
node specs/000-audit/evidence/probe2.cjs /tmp/neuron-dist/index.js
node specs/000-audit/evidence/probe3.cjs /tmp/neuron-dist/index.js
```

Son scripts de diagnóstico puntuales. Los tests definitivos los crea la feature `gradient-oracle`.
