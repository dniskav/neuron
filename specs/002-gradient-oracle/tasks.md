# Tasks: gradient-oracle

**Input**: `specs/002-gradient-oracle/spec.md`, `plan.md`
**Formato**: `- [ ] T0NN [P?] [H|S] [files: ...]` + línea `Gate:`. `[P]` solo si los archivos no se solapan con otras tareas `[P]` del mismo lote. `package.json`, `package-lock.json`, `src/index.ts`, el changelog y `tasks.md` son exclusivos del orquestador. Ninguna tarea modifica `src/` (el gate de cada tarea de modelo incluye `git diff --quiet -- src`). Ningún agente ejecuta `npm publish` ni sube a `master`.
**Requisitos de cada agente**: comprobar la base (`git merge-base --is-ancestor fix/v0.4 HEAD`, si no, `git rebase fix/v0.4`), `npm ci` si falta `node_modules`, y usar comandos git simples (sin `cd && git` ni `for`).
**Nota sobre mutantes** (Constitución IV): en las tareas de modelo, antes de dar la tarea por terminada se introduce una mutación temporal en el `src/` del modelo (invertir el signo de un gradiente o quitar un término), se comprueba que el archivo de la tarea falla, y se revierte con `git checkout -- src`. Tras eso `git diff --quiet -- src` debe pasar. En las tareas `it.fails`, el "mutante" es el inverso: el test debe seguir pasando como `it.fails` y el humo debe pasar.
**Plantilla de archivo de oráculo**: la crean T005 y T006; las tareas `[H]` la copian cambiando el modelo, la pérdida y los datos.
**Orden dentro de un lote**: ejecutar como máximo 4 agentes a la vez.

## Lote 1 (utilidades base; en paralelo: archivos disjuntos)

- [x] T001 [P] [S] [files: tests/helpers/seed.ts, tests/helpers/seed.test.ts] Crear `seed.ts` con `mulberry32(seed)`, `seedRandom(seed = 0x5eed)` (sustituye `Math.random`), `restoreRandom()` y `withSeed(seed, fn)`. La autoprueba comprueba: misma semilla ⇒ misma secuencia; semillas distintas ⇒ secuencias distintas; valores en `[0, 1)`; `restoreRandom` devuelve el `Math.random` original; dos `new NeuronN(3)` construidos tras la misma semilla tienen pesos idénticos.
      Gate: `npx vitest run tests/helpers/seed.test.ts && npm run typecheck`

- [x] T002 [P] [S] [files: tests/helpers/gradientOracle.ts, tests/helpers/gradientOracle.test.ts] Crear el oráculo con la API del plan (`gradientOracle`, `expectGradientMatches`, `bindParams`, `withPlainSGD`, error tipado `OracleError`). Reglas: diferencias finitas centrales, muestreo determinista si hay más de 400 parámetros, `resetRng` antes de cada `loss()`/`step()`, fallo si `‖Δw‖ = 0`, `NaN`, o menos de 3 gradientes numéricos no nulos; `withPlainSGD` restaura `Adam.prototype.step` en `finally` (Q1). Autoprueba con modelos de juguete: (a) regresión lineal con paso `w += lr·(−dL/dw)` ⇒ `cosine > 0.9999`; (b) el mismo con signo invertido ⇒ `cosine < −0.9999` y `expectGradientMatches` lanza; (c) un modelo con un término del gradiente omitido ⇒ coseno < 0.999 y lanza; (d) paso nulo ⇒ lanza `OracleError`; (e) `bindParams` sobre `number[][]` anidados lee y escribe en sitio; (f) `withPlainSGD` hace que un `WeightMatrix.update` mueva `W` exactamente `lr·g` y restaura `Adam` después (comparar `Adam.prototype.step` antes y después).
      Gate: `npx vitest run tests/helpers/gradientOracle.test.ts && npm run typecheck`

- [x] T003 [P] [S] [files: tests/helpers/learns.ts, tests/helpers/learns.test.ts] Crear `expectLearns({ loss, step, steps, ratio = 0.5 })`: mide `loss()` antes y después, exige finito y `final < ratio·initial`. Autoprueba: acepta una regresión que converge; rechaza un modelo que no cambia, uno que diverge y uno que da `NaN`; rechaza una bajada del 10 % con `ratio` 0.5.
      Gate: `npx vitest run tests/helpers/learns.test.ts && npm run typecheck`

## Lote 2 (depende de T001)

- [x] T004 [S] [files: tests/setup.ts, vitest.config.mts] Crear `tests/setup.ts` con `beforeEach(() => seedRandom())` y `afterEach(restoreRandom)`, y registrarlo en `vitest.config.mts` (`setupFiles`). Ejecutar la suite existente: si algún test previo (excepto `tests/GradientCheck.test.ts`, que se elimina en T035) falla con la semilla fija, ajustar la entrada o la semilla de ESE test y anotarlo en el informe; no relajar umbrales sin justificarlo. Solo se editan `tests/setup.ts` y `vitest.config.mts` salvo los tests sensibles, que el agente lista en su informe y el orquestador reasigna.
      Gate: `npx vitest run --exclude tests/GradientCheck.test.ts && npx vitest run --exclude tests/GradientCheck.test.ts && npm run typecheck`

## Lote 3 (plantillas de referencia, Sonnet; depende de T001 a T004)

- [x] T005 [P] [S] [files: tests/oracle/NetworkN.oracle.test.ts] Plantilla principal. `NetworkN` sin `residual` y con `dropoutRate: 0` (estructura pequeña, p. ej. `[3,4,3,1]`, entrada y objetivo fijos): humo, `expectGradientMatches` (esperado: pasa), `expectLearns` (MSE sobre un par fijo). Con `residual: true` (estructura donde se aplique, como `[3,3,3,1]`): `it.fails` del oráculo con comentario `// AUDIT I1` (coseno 0.86); medir además si `expectLearns` con residual pasa y, si no, marcarlo `it.fails` e informarlo. Dropout: el caso "entrenamiento estable" de `GradientCheck.test.ts` (pérdida finita y sin crecer más de 20x en 20 pasos) con semilla fija. Documentar la plantilla en un comentario de cabecera.
      Gate: `npx vitest run tests/oracle/NetworkN.oracle.test.ts && git diff --quiet -- src`

- [x] T006 [P] [S] [files: tests/oracle/Network.oracle.test.ts] `Network` (2 capas): humo, oráculo (esperado: pasa, `train(inputs, target, lr)` con un solo objetivo) y `expectLearns`. Añadir un test `it.fails` o `it.todo` documentado (`// AUDIT I5`) para varias salidas solo si la API actual lo permite; si `Network` solo admite una salida, anotarlo y no inventar el test.
      Gate: `npx vitest run tests/oracle/Network.oracle.test.ts && git diff --quiet -- src`

## Lote 4 (modelos que deben pasar hoy; `[H]`, plantilla de T005/T006; archivos disjuntos)

- [x] T007 [P] [H] [files: tests/oracle/Neuron.oracle.test.ts, tests/oracle/NeuronN.oracle.test.ts, tests/oracle/Layer.oracle.test.ts] `Neuron` (`weight`, `bias`, `train(input, target, lr)`), `NeuronN` (`weights`, `bias`) y `Layer` (neuronas de `NeuronN`; pesos vía `bindParams` o `neurons[i].weights/bias`). Humo, oráculo (pérdida cuadrática de la salida frente al objetivo; el cociente de escala no importa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/Neuron.oracle.test.ts tests/oracle/NeuronN.oracle.test.ts tests/oracle/Layer.oracle.test.ts && git diff --quiet -- src`

- [x] T008 [P] [H] [files: tests/oracle/LSTMLayer.oracle.test.ts] `LSTMLayer` con `getWeightsFlat/setWeightsFlat`, `reset()` antes de cada `loss`/`step`, secuencia de 3 pasos, `backprop(dh_seq, lr)` con `dh = −dL/dh` (la capa interna espera la convención negada; ver tests existentes). Humo, oráculo (esperado: pasa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/LSTMLayer.oracle.test.ts && git diff --quiet -- src`

- [x] T009 [P] [H] [files: tests/oracle/GRU.oracle.test.ts] `GRULayer` igual que T008, con `getWeightsFlat/setWeightsFlat`. Humo, oráculo (esperado: pasa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/GRU.oracle.test.ts && git diff --quiet -- src`

- [x] T010 [P] [H] [files: tests/oracle/Conv1D.oracle.test.ts] `Conv1D` con `getWeights/setWeights`, `backward(dOut, lr)` con `dOut = −dL/dout`. Entrada sin cercanía a cero si hay activación ReLU. Humo, oráculo (esperado: pasa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/Conv1D.oracle.test.ts && git diff --quiet -- src`

- [x] T011 [P] [H] [files: tests/oracle/Conv2D.oracle.test.ts] `Conv2D` (`getWeights/setWeights`, `backward(dOutput, lr)`), entrada 4×4 pequeña, `padding: 'valid'` (evita I4). Humo, oráculo (esperado: pasa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/Conv2D.oracle.test.ts && git diff --quiet -- src`

- [x] T012 [P] [H] [files: tests/oracle/BatchNorm.oracle.test.ts] `BatchNorm` (`gamma`, `beta`; `backward` y `trainParams(dOut, lr)`). Cubrir la actualización de `gamma/beta` con el oráculo y, si la API lo permite, `backward` hacia la entrada frente a diferencias finitas de la entrada. Humo, oráculo y `expectLearns` (ajusta la salida a un objetivo fijo). Si `BatchNorm` no tiene modo de entrenamiento utilizable con una muestra, informar de la ambigüedad.
      Gate: `npx vitest run tests/oracle/BatchNorm.oracle.test.ts && git diff --quiet -- src`

- [x] T013 [P] [H] [files: tests/oracle/LayerNorm.oracle.test.ts] `LayerNorm` (`resetCache(seqLen)`, `predictOne(x, pos)`, `backwardOne(dOut, pos, lr)`; `gamma/beta` por SGD). Oráculo sobre `gamma/beta` (`getWeights/setWeights`) con 2 posiciones. Humo, oráculo y `expectLearns`.
      Gate: `npx vitest run tests/oracle/LayerNorm.oracle.test.ts && git diff --quiet -- src`

- [x] T014 [P] [H] [files: tests/oracle/AttentionHead.oracle.test.ts] `AttentionHead` (`Wq, Wk, Wv` son `WeightMatrix` ⇒ usar `withPlainSGD`, Q1). Secuencia de 2 a 3 tokens, `causal` en `false`, pérdida de suma de cuadrados de la salida con `dOut = −dL`. Humo, oráculo (esperado: pasa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/AttentionHead.oracle.test.ts && git diff --quiet -- src`

- [x] T015 [P] [H] [files: tests/oracle/MultiHeadAttention.oracle.test.ts] `MultiHeadAttention` (2 cabezas, `d_model` 4) con `withPlainSGD`. Humo, oráculo (esperado: pasa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/MultiHeadAttention.oracle.test.ts && git diff --quiet -- src`

- [x] T016 [P] [H] [files: tests/oracle/TransformerBlock.oracle.test.ts] `TransformerBlock` (`d_model` 4, 1 a 2 cabezas, `d_ff` 8, `backward(dOut, lr)`) con `withPlainSGD`. Entradas lejos de cero por ReLU. Humo, oráculo (esperado: pasa; si el coseno no llega a 0.999 informar, no marcar `it.fails`) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/TransformerBlock.oracle.test.ts && git diff --quiet -- src`

- [x] T017 [P] [H] [files: tests/oracle/NetworkLSTM.oracle.test.ts] `NetworkLSTM` con `getWeightsFlat/setWeightsFlat`, `resetState()` antes de cada `loss`/`step`, episodio de 3 pasos, `train(targets, lr)` (gradiente promediado por `T`: el coseno es invariante a la escala). Humo, oráculo (esperado: pasa) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/NetworkLSTM.oracle.test.ts && git diff --quiet -- src`

- [x] T018 [P] [H] [files: tests/oracle/Autoencoder.oracle.test.ts] `Autoencoder` (encoder y decoder `NetworkN` públicos; pesos con `encoder.getWeights()` + `decoder.getWeights()` concatenados; `train(x, lr)`). Pérdida de reconstrucción del propio `train` como cruce. Humo, oráculo (esperado: pasa, 0.998 a 1.000 en la auditoría; si queda por debajo de 0.999 informar el valor) y `expectLearns`.
      Gate: `npx vitest run tests/oracle/Autoencoder.oracle.test.ts && git diff --quiet -- src`

## Lote 4b (hallazgos nuevos del lote 4; Sonnet; no estaban en la auditoría)

- [x] T040 [P] [S] [files: tests/oracle/AttentionHead.dx.oracle.test.ts] Hallazgo N1: en `AttentionHead.backward` se llama a `Wq/Wk/Wv.update(...)` ANTES de calcular `dX`, así que `dX` usa pesos ya actualizados (error relativo hasta 85 % con lr 0.05; exacto con lr = 0). Test de control sin `it.fails`: con `lr = 0`, `dX` coincide con −dL/dX por diferencias finitas (coseno > 0.999). Test `it.fails` (`// AUDIT N1`): con `lr > 0`, dos cabezas idénticas (misma semilla) deben devolver el mismo `dX` que con `lr = 0`. Comprobar también `MultiHeadAttention` y `TransformerBlock` y reportar si heredan el defecto (con `it.fails` propio si sí). Mutante inverso: mover `update` después de `dX` en `src` y comprobar que el `it.fails` se pone en rojo; revertir.
      Gate: `npx vitest run tests/oracle/AttentionHead.dx.oracle.test.ts && git diff --quiet -- src`

- [x] T041 [P] [S] [files: tests/oracle/BatchNorm.momentum.oracle.test.ts] Hallazgo N2: con `momentum` 0.1 (por defecto) `forward` actualiza las estadísticas con `x` y `backward` las trata como constantes: coseno 0.933 frente a −dL/dx. Con `momentum = 1` es exacto (1.0, ya cubierto en T012). Documentar como excepción justificada (Constitución II) con umbral: test que fija 0.9 < coseno < 0.999 con comentario `// AUDIT N2`, y que explique que la corrección (si se quiere) va a la feature 004. No tocar `src/`.
      Gate: `npx vitest run tests/oracle/BatchNorm.momentum.oracle.test.ts && git diff --quiet -- src`

## Lote 5 (defectos conocidos con `it.fails`: C1 y C2; Sonnet por la pérdida y los supuestos de cada modelo)

- [x] T019 [P] [S] [files: tests/oracle/NetworkTransformer.oracle.test.ts] `NetworkTransformer` (`getWeights/setWeights`, `train(tokens, targets, lr)`: cross-entropy con softmax por token). Pérdida del test = la que implementa `train` (cruce con el valor devuelto por `train`, ver el comentario incorrecto "Returns: MSE loss" de 003). Humo (sin `it.fails`), oráculo con `withPlainSGD` en `it.fails` (`// AUDIT C1`, cos −1.0000) y `expectLearns` en `it.fails`. Además, verificar con una prueba de control (no `it.fails`) que el oráculo, aplicado a un `NetworkTransformer` cuyo `Δw` se niega manualmente, daría coseno ≈ +1: sirve de prueba de que el defecto es solo de signo.
      Gate: `npx vitest run tests/oracle/NetworkTransformer.oracle.test.ts && grep -c "it.fails" tests/oracle/NetworkTransformer.oracle.test.ts && git diff --quiet -- src`

- [x] T020 [P] [S] [files: tests/oracle/NetworkTransformerRL.oracle.test.ts] `NetworkTransformerRL` (`train(sequence, target, lr)`, `getWeights/setWeights`). Misma estructura que T019: humo, oráculo `it.fails` (C1) y `expectLearns` `it.fails`, con el control de signo.
      Gate: `npx vitest run tests/oracle/NetworkTransformerRL.oracle.test.ts && grep -c "it.fails" tests/oracle/NetworkTransformerRL.oracle.test.ts && git diff --quiet -- src`

- [x] T021 [P] [S] [files: tests/oracle/RNN.oracle.test.ts] `RNN` (`Wxh, Whh, Why, bh, by` públicos ⇒ `bindParams`; `backward(sequence, targets, lr)` devuelve la pérdida). Humo, oráculo `it.fails` (C1; sin `withPlainSGD`, usa SGD por defecto) y `expectLearns` `it.fails`, con control de signo. `lr` pequeño para no producir `Infinity` en el humo.
      Gate: `npx vitest run tests/oracle/RNN.oracle.test.ts && grep -c "it.fails" tests/oracle/RNN.oracle.test.ts && git diff --quiet -- src`

- [x] T022 [P] [S] [files: tests/oracle/TCN.oracle.test.ts] `CausalConv1D` (`kernels`, `biases`, `backward`) y `TCN` (capas en `layers`; proyección en `_outputW/_outputB` privados, acceso con `as any`, Q5; `train(sequence, targets, lr)`). `CausalConv1D` aislada: oráculo esperado que pase (si no, informar). `TCN`: oráculo `it.fails` (C1; auditoría cos −0.8945, no −1 exacto) y `expectLearns` `it.fails`. Informar de la diferencia con −1 como pista para 003.
      Gate: `npx vitest run tests/oracle/TCN.oracle.test.ts && grep -c "it.fails" tests/oracle/TCN.oracle.test.ts && git diff --quiet -- src`

- [x] T023 [P] [S] [files: tests/oracle/Seq2Seq.oracle.test.ts] `Seq2Seq` (`encoder`, `decoder` son `LSTMLayer` con `getWeightsFlat`; `W_out`, `b_out` públicos; `trainStep(inputSeq, targetSeq, lr)`). Oráculo `it.fails` (C1, cos −0.98; aproxima además el gradiente del encoder: ver Q3) y `expectLearns` `it.fails`. Separar, si es posible, un oráculo solo de `W_out/b_out` para que 003 vea qué parte falla.
      Gate: `npx vitest run tests/oracle/Seq2Seq.oracle.test.ts && grep -c "it.fails" tests/oracle/Seq2Seq.oracle.test.ts && git diff --quiet -- src`

- [x] T024 [P] [S] [files: tests/oracle/GAN.oracle.test.ts] `GAN` (`generator`, `discriminator`: `NetworkN`; `trainStep(realBatch, lr)`: D real, D falso y G por muestra). Oráculo de la actualización del generador con pérdida `−log D(G(z))`, `D` sin cambios y `z` fijo (Q4; adaptar con `resetRng`; lote de 1 muestra): `it.fails` (C2). `expectLearns` en `it.fails`: la media generada se acerca a un objetivo (p. ej. 0.20, como en la auditoría) en lugar de saturar a 1.0 con 3 semillas. Humo sin `it.fails`. Registrar como ambigüedad cualquier supuesto sobre la pérdida.
      Gate: `npx vitest run tests/oracle/GAN.oracle.test.ts && grep -c "it.fails" tests/oracle/GAN.oracle.test.ts && git diff --quiet -- src`

- [x] T025 [P] [S] [files: tests/oracle/VAE.oracle.test.ts] `VAE` (`encoder`, `decoder`: `NetworkN`; `forward`, `train(x, lr)` devuelve las pérdidas). Oráculo con ELBO = reconstrucción (MSE) + KL y `ε` de la reparametrización fijado por `resetRng` (Q4): `it.fails` (C2: el encoder solo recibe el gradiente de KL). `expectLearns` `it.fails`: reconstrucción mejor que predecir la media y `mu` que varía según la entrada. Humo sin `it.fails`.
      Gate: `npx vitest run tests/oracle/VAE.oracle.test.ts && grep -c "it.fails" tests/oracle/VAE.oracle.test.ts && git diff --quiet -- src`

## Lote 6 (resto de entrenables)

- [x] T026 [P] [H] [files: tests/oracle/Perceptron.oracle.test.ts] `Perceptron` (`weights`, `bias`, `train(inputs, target, lr)` devuelve el error). Sin pérdida diferenciable: sin oráculo (excepción de Constitución II, anotada en el archivo con el motivo "regla del perceptrón"). Humo y `expectLearns` con AND o un conjunto linealmente separable: el número de errores por época llega a 0.
      Gate: `npx vitest run tests/oracle/Perceptron.oracle.test.ts && git diff --quiet -- src`

- [x] T027 [P] [H] [files: tests/oracle/LogisticRegression.oracle.test.ts] `LogisticRegression` y `SoftmaxRegression` (`train(X, y, lr, epochs)`; `weights`/`biases` públicos ⇒ `bindParams`). Con 1 muestra y 1 época, para que un `step` sea una sola actualización. Humo, oráculo (entropía cruzada) y `expectLearns` con un conjunto separable. Si `weights` de `LogisticRegression` incluye o no el sesgo, leerlo del código y documentarlo.
      Gate: `npx vitest run tests/oracle/LogisticRegression.oracle.test.ts && git diff --quiet -- src`

- [x] T028 [P] [H] [files: tests/oracle/LinearRegression.oracle.test.ts] `LinearRegression` (`fitGD` frente a `fitNormal`). Oráculo del paso de `fitGD` con 1 muestra y 1 época si la firma lo permite (`weights` incluye el sesgo en la última posición), y `expectLearns`: `fitGD` converge a la solución de `fitNormal` con tolerancia explícita. Si la API no permite un único paso, informar y limitar a `expectLearns`.
      Gate: `npx vitest run tests/oracle/LinearRegression.oracle.test.ts && git diff --quiet -- src`

- [x] T029 [P] [S] [files: tests/oracle/Word2Vec.oracle.test.ts] `Word2Vec` (`train(sentences, lr, epochs)` devuelve pérdidas por época; `embeddings` públicos). Derivar del código la pérdida (skip-gram con negativos o softmax) y ver si hay muestreo negativo aleatorio (`resetRng`). Humo y `expectLearns` (la pérdida por época baja con semilla fija); oráculo solo si la pérdida es determinista y reproducible, si no anotar la excepción.
      Gate: `npx vitest run tests/oracle/Word2Vec.oracle.test.ts && git diff --quiet -- src`

- [x] T030 [P] [S] [files: tests/oracle/SOM.oracle.test.ts] `SOM` (`train(X, epochs)`). Sin gradiente: excepción de Constitución II. Humo y `expectLearns` con el error de cuantización medio sobre datos fijos.
      Gate: `npx vitest run tests/oracle/SOM.oracle.test.ts && git diff --quiet -- src`

- [x] T031 [P] [S] [files: tests/oracle/ContrastiveLearning.oracle.test.ts] `ContrastiveLearning` (`encoder`, `projectionHead`: `NetworkN`; `trainStep(pairs, lr)`; `computeLoss(pairs)`). Su paso usa diferencias finitas internas: oráculo con `computeLoss` (esperado: pasa, trivial; anotarlo en el archivo) y `expectLearns` con la pérdida NT-Xent. Si el paso no es una actualización limpia (p. ej. estado aleatorio interno), informarlo como ambigüedad.
      Gate: `npx vitest run tests/oracle/ContrastiveLearning.oracle.test.ts && git diff --quiet -- src`

- [x] T032 [P] [S] [files: tests/oracle/Tape.oracle.test.ts] `Value` de `src/Tape.ts` (autograd; `backward()`). Compara `grad` de cada hoja con diferencias finitas en 3 expresiones (suma/producto, `tanh`/`relu`/`exp` o las operaciones que existan, y una expresión compuesta con reutilización de nodos). Aquí el criterio es `grad ≈ ∇L` (relativo < 1e-5), no hay paso de optimizador. No incluir el desbordamiento de pila (I7, de 005).
      Gate: `npx vitest run tests/oracle/Tape.oracle.test.ts && git diff --quiet -- src`

- [x] T033 [P] [S] [files: tests/oracle/MatMul.oracle.test.ts] `WeightMatrix.update` (con `withPlainSGD`), `BiasVector.update` y `EmbeddingMatrix.update` (SGD directo, `W += lr·grad`): la actualización real contra la convención `−dL/dw`. Humo y `expectLearns` ajustando `W·x` a un objetivo. Documentar la convención de signo y la de `clipValue`.
      Gate: `npx vitest run tests/oracle/MatMul.oracle.test.ts && git diff --quiet -- src`

- [x] T034 [P] [H] [files: tests/oracle/PositionalEncoding.oracle.test.ts] `LearnedPositionalEncoding` (`src/PositionalEncoding.ts`): si expone `update`/`backward`, humo, oráculo y `expectLearns`; si no, un único test que documente que no es entrenable y avisar en el informe. `PositionalEncoding` fijo no entra.
      Gate: `npx vitest run tests/oracle/PositionalEncoding.oracle.test.ts && git diff --quiet -- src`

## Lote 7 (limpieza y meta-test; depende de los lotes 4 a 6)

- [x] T035 [P] [H] [files: tests/GradientCheck.test.ts] `git rm tests/GradientCheck.test.ts` (elimina `signedPerturbationCheck` y `multiStepLossCheck`; el caso de dropout ya vive en T005, Q6).
      Gate: `test ! -f tests/GradientCheck.test.ts && ! grep -rn "signedPerturbationCheck\|multiStepLossCheck" tests src && npx vitest run && npm run typecheck`

- [x] T036 [P] [H] [files: tests/Integration.test.ts] Renombrar los dos tests "gradients match finite differences for Network/NetworkN" a "moves prediction toward target for Network/NetworkN" sin cambiar su lógica (Q7).
      Gate: `! grep -n "finite differences" tests/Integration.test.ts && npx vitest run tests/Integration.test.ts`

- [x] T037 [P] [S] [files: tests/oracle/coverage.test.ts] Meta-test de cobertura (plan, decisión 8): lista explícita de modelos entrenables con su archivo en `tests/oracle/` (que debe existir), y comprobación de que toda clase exportada por `src/index.ts` con método `train|trainStep|backward|backprop|fit|fitGD|trainBatch|trainParams|update` está en esa lista o en una lista de exclusiones con motivo (las de la sección "sin gradiente" de la spec). Lee `src/index.ts` y `src/*.ts`, no los modifica. Debe fallar si se borra uno de los archivos de oráculo (comprobarlo renombrando uno temporalmente y revirtiendo).
      Gate: `npx vitest run tests/oracle/coverage.test.ts && git diff --quiet -- src`

## Lote 8 (revisión y estabilidad; el verificador es distinto de quien implementó)

- [x] T038 [S] [files: ] `math-reviewer` (sin editar): revisar `tests/helpers/*.ts` (convención de signo, `ratio`, muestreo, que `withPlainSGD` no deje fugas), la corrección de la pérdida de cada archivo de `tests/oracle/` (la que se optimiza coincide con la que usa el oráculo), que cada `it.fails` corresponda a C1, C2 o I1 (SC-004) y que ninguno pase por una excepción ajena. Muestrear al menos 6 mutantes en modelos distintos (`NetworkN`, `LSTMLayer`, `Conv1D`, `AttentionHead`, `NetworkTransformer` con el signo arreglado temporalmente para comprobar que el `it.fails` se pone en rojo, `Autoencoder`) y confirmar que los tests fallan. Informe sin hallazgos bloqueantes.
      Gate: informe del revisor sin bloqueantes; `git diff --quiet -- src` tras sus mutaciones

- [x] T039 [H] [files: ] `verifier` (sin editar): definición de terminado y estabilidad. 20 ejecuciones seguidas de la suite, más 5 con orden aleatorio.
      Gate: `npx tsc --noEmit -p . && npm run typecheck && npm run build && for i in $(seq 1 20); do npx vitest run > /dev/null 2>&1 || { echo FAIL $i; exit 1; }; done && for i in $(seq 1 5); do npx vitest run --sequence.shuffle > /dev/null 2>&1 || { echo FAIL shuffle $i; exit 1; }; done && git diff --quiet -- src package.json && echo OK`

## Dependencias

- T004 ← T001. T005/T006 ← T001 a T004. Lotes 4 a 6 ← T005/T006 (plantilla) y T001 a T004. T034 depende además de que `LearnedPositionalEncoding` exponga una API de entrenamiento. T035, T036 ← lotes 4 a 6 (T035 ← T005 para el caso de dropout). T037 ← lotes 4 a 6. T038 ← T035 a T037. T039 ← T038.
- Fuera de las tareas (orquestador): commit de `specs/002-gradient-oracle/` antes de lanzar agentes, merges en serie con rebase y verificador, y actualización de `specs/HANDOFF.md`.
- Resultado esperado de la feature para 003/004: lista de `it.fails` a retirar = NetworkTransformer, NetworkTransformerRL, RNN, TCN, Seq2Seq (003); GAN, VAE y `NetworkN` con `residual` (004).
