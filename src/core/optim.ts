// ─── VECTOR OPTIMIZERS ────────────────────────────────────────────────────────
//
// Array versions of the scalar optimizers in `optimizers.ts`. One instance holds
// the state of `size` weights in typed arrays instead of `size` objects.
// Formulas and operation order are copied literally from the scalar classes so
// the results are bit-identical (Object.is) to N scalar instances.
// Convention: `grads` is -dL/dw and `step` writes `w + lr·g` into `params`.
import {
  Adam,
  ClipOptimizer,
  Momentum,
  SGD,
  type Optimizer,
  type OptimizerFactory,
} from '../optimizers'

type NumArray = { [i: number]: number; length: number }

export interface VecOptimizer {
  readonly size: number
  step(params: NumArray, grads: NumArray, lr: number): void
}

function checkLengths(who: string, size: number, params: NumArray, grads: NumArray): void {
  if (params.length !== size || grads.length !== size) {
    throw new Error(
      `${who}.step: params and grads must have length ${size}, got ${params.length} and ${grads.length}`,
    )
  }
}

function checkSize(who: string, size: number): void {
  if (!Number.isInteger(size) || size < 0) {
    throw new Error(`${who}: size must be an integer >= 0, got ${size}`)
  }
}

// w ← w + lr·g
export class VecSGD implements VecOptimizer {
  constructor(readonly size: number) {
    checkSize('VecSGD', size)
  }

  step(params: NumArray, grads: NumArray, lr: number): void {
    checkLengths('VecSGD', this.size, params, grads)
    for (let i = 0; i < this.size; i++) {
      params[i] = params[i] + lr * grads[i]
    }
  }
}

// v ← β·v + lr·g ; w ← w + v
export class VecMomentum implements VecOptimizer {
  private v: Float64Array

  constructor(
    readonly size: number,
    readonly beta = 0.9,
  ) {
    checkSize('VecMomentum', size)
    this.v = new Float64Array(size)
  }

  step(params: NumArray, grads: NumArray, lr: number): void {
    checkLengths('VecMomentum', this.size, params, grads)
    const v = this.v
    for (let i = 0; i < this.size; i++) {
      v[i] = this.beta * v[i] + lr * grads[i]
      params[i] = params[i] + v[i]
    }
  }
}

// Adam. `t` is shared by all elements: valid because every step updates all of them.
export class VecAdam implements VecOptimizer {
  private m: Float64Array
  private v: Float64Array
  private t = 0

  constructor(
    readonly size: number,
    readonly beta1 = 0.9,
    readonly beta2 = 0.999,
    readonly epsilon = 1e-8,
  ) {
    checkSize('VecAdam', size)
    this.m = new Float64Array(size)
    this.v = new Float64Array(size)
  }

  step(params: NumArray, grads: NumArray, lr: number): void {
    checkLengths('VecAdam', this.size, params, grads)
    this.t++
    const m = this.m
    const v = this.v
    const c1 = 1 - Math.pow(this.beta1, this.t)
    const c2 = 1 - Math.pow(this.beta2, this.t)
    for (let i = 0; i < this.size; i++) {
      const gradient = grads[i]
      m[i] = this.beta1 * m[i] + (1 - this.beta1) * gradient
      v[i] = this.beta2 * v[i] + (1 - this.beta2) * gradient * gradient
      const mHat = m[i] / c1
      const vHat = v[i] / c2
      params[i] = params[i] + lr * mHat / (Math.sqrt(vHat) + this.epsilon)
    }
  }
}

// Clips each gradient to [-clipValue, clipValue] BEFORE delegating. Does not modify `grads`.
export class VecClip implements VecOptimizer {
  private clipped: Float64Array

  constructor(
    readonly inner: VecOptimizer,
    readonly clipValue: number,
  ) {
    this.clipped = new Float64Array(inner.size)
  }

  get size(): number {
    return this.inner.size
  }

  step(params: NumArray, grads: NumArray, lr: number): void {
    checkLengths('VecClip', this.size, params, grads)
    const clipped = this.clipped
    for (let i = 0; i < this.size; i++) {
      clipped[i] = Math.max(-this.clipValue, Math.min(this.clipValue, grads[i]))
    }
    this.inner.step(params, clipped, lr)
  }
}

// Fallback for any scalar optimizer: calls `factory()` `size` times, in order.
export class ScalarArrayOptimizer implements VecOptimizer {
  private opts: Optimizer[]

  constructor(factory: OptimizerFactory, readonly size: number) {
    checkSize('ScalarArrayOptimizer', size)
    this.opts = []
    for (let i = 0; i < size; i++) this.opts.push(factory())
  }

  step(params: NumArray, grads: NumArray, lr: number): void {
    checkLengths('ScalarArrayOptimizer', this.size, params, grads)
    for (let i = 0; i < this.size; i++) {
      params[i] = this.opts[i].step(params[i], grads[i], lr)
    }
  }
}

function fromInstance(inst: Optimizer, size: number): VecOptimizer | null {
  if (inst instanceof SGD) return new VecSGD(size)
  if (inst instanceof Momentum) return new VecMomentum(size, inst.beta)
  if (inst instanceof Adam) return new VecAdam(size, inst.beta1, inst.beta2, inst.epsilon)
  if (inst instanceof ClipOptimizer) {
    const inner = fromInstance(inst.inner, size)
    return inner ? new VecClip(inner, inst.clipValue) : null
  }
  return null
}

// Picks the vector class that matches the built-in optimizer the factory makes
// (copying its hyperparameters), or ScalarArrayOptimizer for anything else.
// Side effect: it calls `factory()` ONE extra time to inspect the instance; a
// factory that counts its instances will see one more call.
export function vecFromFactory(factory: OptimizerFactory, size: number): VecOptimizer {
  return fromInstance(factory(), size) ?? new ScalarArrayOptimizer(factory, size)
}
