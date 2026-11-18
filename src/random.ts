/**
 * Generador de números aleatorios inyectable.
 *
 * Por defecto `random()` llama a `Math.random()` en cada llamada, así que el
 * comportamiento de la librería no cambia. `setRandom` instala otro generador
 * (por ejemplo `seededRandom(42)`) para que los modelos sean reproducibles sin
 * tocar el global `Math.random`.
 */

/** Función que devuelve un número en [0, 1). */
export type RandomFn = () => number

let active: RandomFn | null = null

/**
 * Devuelve un número en [0, 1) del generador activo.
 * Sin generador instalado, llama a `Math.random()` en ese momento.
 */
export function random(): number {
  return active !== null ? active() : Math.random()
}

/** Instala un generador, o `null` para volver a `Math.random`. */
export function setRandom(fn: RandomFn | null): void {
  active = fn
}

/** Devuelve el generador instalado, o `null` si se usa `Math.random`. */
export function getRandom(): RandomFn | null {
  return active
}

/**
 * Generador determinista mulberry32 (el mismo que `tests/helpers/seed.ts`).
 * Misma semilla, misma secuencia de valores en [0, 1).
 */
export function seededRandom(seed: number): RandomFn {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Ejecuta `body` con el generador `fn` instalado y restaura el anterior al
 * terminar, también si `body` lanza una excepción.
 */
export function withRandom<T>(fn: RandomFn, body: () => T): T {
  const previous = active
  active = fn
  try {
    return body()
  } finally {
    active = previous
  }
}
