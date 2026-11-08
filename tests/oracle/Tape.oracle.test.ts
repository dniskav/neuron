// Gradient oracle for the autograd tape (feature 002, T032).
//
// Criterion here is grad ~= dL/dx (relative error < 1e-5) from central finite
// differences, not a one-step optimizer oracle: Value has no optimizer step.
// Each expression is a builder over a numeric input vector, so the same graph is
// rebuilt at x +/- h for the finite differences. Stack overflow on deep graphs
// (I7) belongs to feature 005 and is not covered.
import { describe, it, expect } from 'vitest';
import { Value } from '../../src/Tape';

type Builder = (v: Value[]) => Value;

const H = 1e-6;
const TOL = 1e-5;

function numeric(build: Builder, x: number[], i: number): number {
  const at = (d: number) => build(x.map((xi, j) => new Value(j === i ? xi + d : xi))).data;
  return (at(H) - at(-H)) / (2 * H);
}

function relErr(a: number, b: number): number {
  return Math.abs(a - b) / Math.max(1, Math.abs(a), Math.abs(b));
}

// Returns the worst relative error over all leaves; asserts each is below TOL.
function check(build: Builder, x: number[]): number {
  const leaves = x.map(xi => new Value(xi));
  build(leaves).backward();
  let worst = 0;
  x.forEach((_, i) => {
    const err = relErr(leaves[i].grad, numeric(build, x, i));
    worst = Math.max(worst, err);
    expect(err).toBeLessThan(TOL);
  });
  return worst;
}

describe('Tape (Value) gradients vs central finite differences', () => {
  it('sum and product: (a*b + c) * a - b', () => {
    check(([a, b, c]) => a.mul(b).add(c).mul(a).sub(b), [1.5, -0.7, 0.4]);
  });

  it('non-linear ops: tanh, relu, sigmoid, exp, pow, div', () => {
    check(([a, b]) => a.tanh().add(b.relu()).add(a.mul(b).sigmoid()), [0.6, 0.9]);
    check(([a, b]) => a.exp().mul(b.pow(3)).div(b.add(2)), [0.3, 1.2]);
    // relu on the inactive side must give zero gradient
    check(([a, b]) => a.relu().mul(b), [-0.8, 1.1]);
  });

  it('compound expression reusing nodes accumulates gradients', () => {
    check(([x, w, b]) => {
      const z = x.mul(w).add(b); // z used several times below
      const t = z.tanh();
      return z.mul(z).add(t.mul(z)).add(t).add(x.mul(x)).add(w.exp().mul(z));
    }, [0.5, -0.6, 0.2]);
    // same leaf as both operands
    check(([a]) => a.mul(a).add(a), [1.3]);
  });
});
