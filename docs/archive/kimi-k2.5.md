# Sugerencias de Refactor - kimi-k2.5

Análisis realizado el 2026-04-01 sobre el proyecto `@dniskav/neuron` v0.2.1.

---

## 1. Eliminar duplicación de `sigmoid` (Alta prioridad)

**Problema:** `Neuron.ts` define su propia función `sigmoid` cuando ya existe en `activations.ts`.

**Ubicación:** `src/Neuron.ts`, líneas 1-5

**Código actual:**
```typescript
// ─── ACTIVATION FUNCTION ─────────────────────────────────────────────────────
// Squashes any number into a value between 0 and 1
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
```

**Sugerencia:** Importar desde `activations.ts`:
```typescript
import { sigmoid } from "./activations";
```

**Beneficio:** Elimina duplicación de código, centraliza la definición.

---

## 2. `Neuron` debería reutilizar `NeuronN` (Alta prioridad)

**Problema:** `Neuron` es esencialmente `NeuronN` con `nInputs=1`. Hay lógica duplicada.

**Ubicación:** `src/Neuron.ts`

**Sugerencia:** Implementar como wrapper:
```typescript
import { NeuronN } from "./NeuronN";
import { sigmoid } from "./activations";
import { SGD } from "./optimizers";

export class Neuron {
  private neuron: NeuronN;
  
  constructor() {
    this.neuron = new NeuronN(1, sigmoid, () => new SGD());
  }
  
  predict(input: number): number {
    return this.neuron.predict([input]);
  }
  
  train(input: number, target: number, lr: number): void {
    this.neuron.train([input], target, lr);
  }
}
```

**Beneficio:** Elimina duplicación, hereda optimizaciones y mejoras de `NeuronN`.

---

## 3. Separar `MatMul.ts` (Media-Alta prioridad)

**Problema:** El archivo viola el principio de responsabilidad única. Contiene:
- 2 clases (`WeightMatrix`, `EmbeddingMatrix`)
- 4 funciones (`matMul`, `transpose`, `softmax`, `softmaxBackward`)

**Ubicación:** `src/MatMul.ts`

**Sugerencia:** Dividir en:
- `src/utils/matrix.ts` → `matMul`, `transpose`
- `src/utils/softmax.ts` → `softmax`, `softmaxBackward`
- `src/layers/WeightMatrix.ts` → clase `WeightMatrix`
- `src/layers/EmbeddingMatrix.ts` → clase `EmbeddingMatrix`

**Beneficio:** Mejor organización, separación de concerns, más fácil de mantener.

---

## 4. Método `_update` en `NeuronN` (Media prioridad)

**Problema:** Usa prefijo underscore (convención de "privado") pero es público en TypeScript.

**Ubicación:** `src/NeuronN.ts`, línea 41

**Sugerencia:** Usar modificador `private` real de TypeScript:
```typescript
// Opción A
private update(weightGrads: number[], biasGrad: number, lr: number): void

// Opción B (si es API pública intencional)
update(...)
```

**Beneficio:** Claridad semántica, el compilador garantiza el encapsulamiento.

---

## 5. Centralizar tipos de opciones (Media prioridad)

**Problema:** Los tipos `NetworkNOptions`, `NetworkLSTMOptions`, etc. están dispersos en diferentes archivos.

**Ubicación:** 
- `src/NetworkN.ts`
- `src/NetworkLSTM.ts`
- `src/NetworkTransformer.ts`
- `src/TransformerBlock.ts`

**Sugerencia:** Crear `src/types/` o `src/options.ts`:
```typescript
// src/types/network-options.ts
export type { NetworkNOptions } from "./NetworkN";
export type { NetworkLSTMOptions } from "./NetworkLSTM";
export type { NetworkTransformerOptions } from "./NetworkTransformer";
export type { TransformerBlockOptions } from "./TransformerBlock";
```

**Beneficio:** Centralización de tipos públicos, mejor discoverability.

---

## 6. Validación consistente de inputs (Baja prioridad)

**Problema:** Algunas clases validan longitud de arrays de entrada, otras no.

**Ubicación:** Múltiples clases (`NeuronN`, `Layer`, `NetworkN`, etc.)

**Sugerencia:** Agregar validación consistente:
```typescript
if (inputs.length !== this.weights.length) {
  throw new Error(
    `Expected ${this.weights.length} inputs, got ${inputs.length}`
  );
}
```

**Beneficio:** Mejor debugging, falla rápido con mensajes claros.

---

## 7. Considerar eliminar `Neuron` (Opcional)

**Problema:** `Neuron` es redundante dado que `NeuronN(1)` cubre el mismo caso.

**Sugerencia:** 
- Opción A: Deprecar y remover en versión mayor
- Opción B: Convertir en alias: `export const Neuron = () => new NeuronN(1, sigmoid)`

**Beneficio:** API más simple y minimalista (alineado con la filosofía del proyecto).

---

## Prioridad General

| # | Sugerencia | Prioridad | Esfuerzo | Impacto |
|---|------------|-----------|----------|---------|
| 1 | Eliminar duplicación `sigmoid` | Alta | Bajo | Medio |
| 2 | `Neuron` reusa `NeuronN` | Alta | Medio | Alto |
| 3 | Separar `MatMul.ts` | Media-Alta | Medio | Alto |
| 4 | Método `_update` privado | Media | Bajo | Bajo |
| 5 | Centralizar tipos | Media | Bajo | Medio |
| 6 | Validación consistente | Baja | Medio | Medio |
| 7 | Eliminar `Neuron` | Opcional | Bajo | Medio |

---

## Notas del Modelo

- Todas las sugerencias respetan la filosofía del proyecto: minimalismo, educativo, sin dependencias.
- Las sugerencias 1 y 2 son las más importantes y seguras de implementar.
- La sugerencia 3 requiere actualizar imports en varios archivos.
- Las sugerencias 4-7 son mejoras de calidad de código más que correcciones.
