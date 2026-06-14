// ─── AUTOMATIC DIFFERENTIATION (AUTOGRAD / TAPE) ─────────────────────────────
//
// Implements reverse-mode automatic differentiation via a dynamic computational
// graph — the same principle behind PyTorch's autograd and Karpathy's micrograd.
//
// How it works:
//   Every scalar is a `Value` node. Arithmetic operations create new `Value`
//   nodes that remember their inputs (children) and the operation used to
//   produce them. This forms a directed acyclic graph (DAG).
//
//   Calling `.backward()` on a leaf node:
//     1. Topologically sorts all ancestor nodes.
//     2. Seeds the output gradient: output.grad = 1.
//     3. Visits each node in reverse topological order and calls its
//        `_backward` closure, which accumulates gradients into the children.
//
// Chain rule (the engine of backprop):
//   If z = f(a, b), then:
//     ∂L/∂a += ∂L/∂z · ∂z/∂a
//     ∂L/∂b += ∂L/∂z · ∂z/∂b
//   Each `_backward` closure implements these local partial derivatives
//   and accumulates into `child.grad` ("+=" because a node may be used
//   multiple times — the gradients from each usage must be summed).
//
// Example — simple expression:
//
//   const a = new Value(2);
//   const b = new Value(3);
//   const c = a.mul(b);    // c = 6,  ∂c/∂a = b = 3, ∂c/∂b = a = 2
//   const d = c.add(1);    // d = 7,  ∂d/∂c = 1
//   d.backward();
//   // a.grad = 3, b.grad = 2, c.grad = 1
//
// Example — mini 2-input neuron:
//
//   const x1 = new Value(1.0);
//   const x2 = new Value(0.5);
//   const w1 = new Value(0.8);
//   const w2 = new Value(-0.3);
//   const b  = new Value(0.1);
//   const z  = w1.mul(x1).add(w2.mul(x2)).add(b);
//   const out = z.tanh();
//   out.backward();
//   // w1.grad, w2.grad, b.grad now hold ∂out/∂wᵢ and ∂out/∂b
//   // Update: w1.data += -lr * w1.grad  (gradient descent step)
//

// ─── Value ────────────────────────────────────────────────────────────────────

export class Value {
  data: number;
  grad: number;
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  private _backward: () => void = () => {};
  private _prev: Set<Value>;
  private _op: string;

  constructor(data: number, children: Value[] = [], op = '') {
    this.data  = data;
    this.grad  = 0;
    this._prev = new Set(children);
    this._op   = op;
  }

  // ── Arithmetic Operations ────────────────────────────────────────────────

  // z = a + b   →   ∂z/∂a = 1,  ∂z/∂b = 1
  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], '+');
    out._backward = () => {
      this.grad += out.grad;   // ∂z/∂a = 1
      o.grad    += out.grad;   // ∂z/∂b = 1
    };
    return out;
  }

  // z = a * b   →   ∂z/∂a = b,  ∂z/∂b = a
  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o], '*');
    out._backward = () => {
      this.grad += o.data    * out.grad;  // ∂z/∂a = b
      o.grad    += this.data * out.grad;  // ∂z/∂b = a
    };
    return out;
  }

  // z = aⁿ   →   ∂z/∂a = n·aⁿ⁻¹
  pow(exp: number): Value {
    const out = new Value(Math.pow(this.data, exp), [this], `**${exp}`);
    out._backward = () => {
      this.grad += exp * Math.pow(this.data, exp - 1) * out.grad;
    };
    return out;
  }

  // z = max(0, a)   →   ∂z/∂a = a > 0 ? 1 : 0
  relu(): Value {
    const out = new Value(Math.max(0, this.data), [this], 'ReLU');
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  // z = tanh(a)   →   ∂z/∂a = 1 - tanh(a)² = 1 - z²
  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'tanh');
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  // z = σ(a) = 1/(1+e⁻ᵃ)   →   ∂z/∂a = z·(1-z)
  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this], 'sigmoid');
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  // z = eᵃ   →   ∂z/∂a = eᵃ = z
  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], 'exp');
    out._backward = () => {
      this.grad += e * out.grad;
    };
    return out;
  }

  // ── Derived Operations (built from primitives) ───────────────────────────

  // a / b = a * b⁻¹
  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  // a - b = a + (b * -1)
  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.mul(-1));
  }

  // -a = a * -1
  neg(): Value {
    return this.mul(-1);
  }

  // ── Backward Pass ────────────────────────────────────────────────────────
  //
  // Propagates gradients from this node (treated as the loss L) back through
  // the entire computational graph.
  //
  // Steps:
  //   1. Build a topological ordering of all ancestor nodes.
  //   2. Set this.grad = 1  (∂L/∂L = 1).
  //   3. Visit nodes in reverse topological order, calling each _backward.
  //
  backward(): void {
    const topo: Value[]   = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value): void => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) buildTopo(child);
        topo.push(v);
      }
    };

    buildTopo(this);
    this.grad = 1;  // seed: ∂L/∂L = 1

    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  toString(): string {
    return `Value(data=${this.data.toFixed(4)}, grad=${this.grad.toFixed(4)}, op='${this._op}')`;
  }
}
