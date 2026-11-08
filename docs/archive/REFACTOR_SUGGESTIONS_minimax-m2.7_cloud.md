# Refactoring Suggestions — minimax-m2.7:cloud

## 1. Extraer `FFN` (Feed-Forward Network) desde `TransformerBlock`

**Problema:** `TransformerBlock` (~212 líneas) contiene lógica de FFN duplicada inline (ff1, ff2, b1, b2, forward y backward). Esto hace que el bloque sea ~60% más largo de lo necesario.

**Solución:** Crear una clase `FFN` autocontenida:

```ts
// d_model → d_ff (ReLU) → d_model
class FFN {
  W1, W2: WeightMatrix
  b1: number[], b2: number[]
  // predict(X): number[][]
  // backward(dOut, lr): dX
}
```

`TransformerBlock` tendría:
```ts
ffn: FFN
// en predict: h1 = this.norm1.predictOne(...)
//             out = this.norm2.predictOne(...)
```

**Impacto:** Reduce `TransformerBlock` de ~212 a ~80 líneas.

---

## 2. Unificar `EmbeddingMatrix` y `WeightMatrix` en `Linear`

**Problema:** `EmbeddingMatrix` y `WeightMatrix` tienen la misma estructura (2D weights + bias + per-scalar Adam optimizers). Solo cambian en el método de update (`sgdUpdate` vs `update` con clip opcional).

**Solución:**
```ts
class Linear {
  constructor(readonly rows: number, readonly cols: number)
  W: number[][]
  bias: number[]
  biasOpts: Adam[]
  // forward(x: number[]): number[]
  // update(dW, lr, clipValue?)
  // sgdUpdate(dW, lr)  ← para embeddings
}
```

`EmbeddingMatrix` y `WeightMatrix` se convierten en aliases o subclases.

**Impacto:** Elimina duplicación de ~100 líneas entre ambos archivos.

---

## 3. Clase base abstracta `Module`

**Problema:** El patrón `predict()` guarda caches → `backward()` consume caches se repite en `LSTMLayer`, `TransformerBlock`, `NetworkTransformer`. No hay enforced contract.

**Solución:**
```ts
abstract class Module {
  abstract predict(...): number[]
  abstract backward(dout, lr): number[]
  protected caches: Map<string, any>
}
```

**Impacto:** Fuerza interfaz consistente, facilita testing y composición.

---

## 4. Helper `updateBias()`

**Problema:** Código repetitivo en todos lados:
```ts
for (let i = 0; i < n; i++) bias[i] = opts[i].step(bias[i], grad[i], lr)
```

**Solución:**
```ts
function updateBias(bias: number[], grad: number[], opts: Optimizer[], lr: number): void {
  for (let i = 0; i < bias.length; i++)
    bias[i] = opts[i].step(bias[i], grad[i], lr)
}
```

**Impacto:** Reduce ~30 líneas duplicadas dispersas en 5 archivos.

---

## 5. Extraer `CrossEntropyLoss` desde `NetworkTransformer`

**Problema:** La lógica de softmax + CE + gradiente combinado (prob - target) está inline en `NetworkTransformer.train()`. No es testeable independientemente.

**Solución:**
```ts
class CrossEntropyLoss {
  // forward(logits, targets, mask?): { loss, dLogits }
}
```

**Impacto:** Separa concerns, mejora testabilidad.

---

## 6. Consistencia: `LSTMLayer` no usa `LayerNorm`

**Problema:** `TransformerBlock` usa LayerNorm después de cada sub-capa. `LSTMLayer` no tiene normalización. Esto dificulta abstracciones comunes y puede causar inconsistency en training stability.

**Nota:** Puede ser intencional (LSTM no requiere LayerNorm en la misma medida). Sugerencia solo si se quiere uniformidad.

---

## Prioridades recomendadas

| # | Refactor | Prioridad | Riesgo | Impacto |
|---|----------|-----------|--------|---------|
| 1 | Extraer FFN | **Alta** | Bajo | Alto |
| 2 | Unificar Linear | **Alta** | Medio | Alto |
| 3 | Module base | Media | Bajo | Medio |
| 4 | Helper updateBias | **Alta** | Muy bajo | Medio |
| 5 | CrossEntropyLoss | Media | Bajo | Medio |
| 6 | LayerNorm en LSTM | Baja | Medio | Bajo |

Empezar por #1 y #4 (alto impacto, bajo riesgo).
