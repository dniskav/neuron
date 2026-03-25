// ─── ACTIVATION FUNCTIONS ─────────────────────────────────────────────────────
//
// Each activation is an object with two functions:
//   fn(x)   — the activation applied to the pre-activation sum z
//   dfn(out) — the derivative expressed in terms of the POST-activation output
//              (avoids recomputing z during backprop)
//
// All built-ins use the output form so backprop stays O(1) per neuron:
//   sigmoid: σ'(z) = σ(z)·(1-σ(z))  →  dfn(out) = out*(1-out)
//   tanh:    tanh'(z) = 1-tanh(z)²   →  dfn(out) = 1-out²
//   relu:    relu'(z) = z>0 ? 1 : 0  →  dfn(out) = out>0 ? 1 : 0
//   linear:  1                        →  dfn()    = 1
//
export interface Activation {
  fn(x: number): number;
  dfn(out: number): number;
}

export const sigmoid: Activation = {
  fn:  (x) => 1 / (1 + Math.exp(-x)),
  dfn: (out) => out * (1 - out),
};

export const tanh: Activation = {
  fn: (x) => {
    const e = Math.exp(2 * x);
    return (e - 1) / (e + 1);
  },
  dfn: (out) => 1 - out * out,
};

export const relu: Activation = {
  fn:  (x) => Math.max(0, x),
  dfn: (out) => out > 0 ? 1 : 0,
};

export const linear: Activation = {
  fn:  (x) => x,
  dfn: () => 1,
};
