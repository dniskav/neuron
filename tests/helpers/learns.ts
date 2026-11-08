/**
 * expectLearns: "entrenar significa aprender" (Constitucion III).
 *
 * Mide `loss()` ANTES y DESPUES de `steps` llamadas a `step()` (la perdida se
 * calcula fuera del `train` del modelo, sobre datos fijos) y exige que:
 *   - ambas perdidas sean finitas (sin NaN ni Infinity),
 *   - final < ratio * initial   (por defecto ratio = 0.5).
 * Los modelos de actualizacion por muestra pasan un `step` que recorre el mini-conjunto.
 * Lanza `LearnsError` con las cifras medidas si falla.
 */
export class LearnsError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'LearnsError'
  }
}

export interface LearnsOptions {
  loss: () => number
  step: () => void
  steps: number
  ratio?: number
}

export function expectLearns({ loss, step, steps, ratio = 0.5 }: LearnsOptions): {
  initial: number
  final: number
  ratio: number
} {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new LearnsError(`expectLearns: steps debe ser un entero >= 1 (recibido ${steps})`)
  }
  if (!(ratio > 0)) {
    throw new LearnsError(`expectLearns: ratio debe ser > 0 (recibido ${ratio})`)
  }
  const initial = loss()
  if (!Number.isFinite(initial)) {
    throw new LearnsError(`expectLearns: perdida inicial no finita (${initial})`)
  }
  for (let i = 0; i < steps; i++) step()
  const final = loss()
  if (!Number.isFinite(final)) {
    throw new LearnsError(`expectLearns: perdida final no finita (${final}); inicial ${initial}`)
  }
  if (!(final < ratio * initial)) {
    throw new LearnsError(
      `expectLearns: no aprende; inicial ${initial}, final ${final} ` +
        `(final/inicial = ${initial !== 0 ? final / initial : NaN}, exigido < ${ratio})`
    )
  }
  return { initial, final, ratio: final / initial }
}
