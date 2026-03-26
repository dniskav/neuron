// ─── ACTIVATION FUNCTIONS ─────────────────────────────────────────────────────
//
// Each activation is an object with two functions:
//   fn(x)    — the activation applied to the pre-activation sum z
//   dfn(out) — the derivative expressed in terms of the POST-activation output
//              (avoids recomputing z during backprop)
//
// All built-ins use the output form so backprop stays O(1) per neuron:
//   sigmoid:    σ'(z) = σ(z)·(1-σ(z))         →  dfn(out) = out*(1-out)
//   tanh:       tanh'(z) = 1-tanh(z)²          →  dfn(out) = 1-out²
//   relu:       relu'(z) = z>0 ? 1 : 0         →  dfn(out) = out>0 ? 1 : 0
//   leakyRelu:  f'(z) = z>0 ? 1 : α            →  dfn(out) = out>0 ? 1 : α
//   elu:        f'(z) = z>0 ? 1 : α·eˣ         →  dfn(out) = out>0 ? 1 : out+α
//   linear:     1                               →  dfn()    = 1
//
// Parametric variants: makeLeakyRelu(α) and makeElu(α) accept a custom α.
// The default exports leakyRelu (α=0.01) and elu (α=1.0) cover the typical cases.
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

// ─── Leaky ReLU ───────────────────────────────────────────────────────────────
//
// Solves the "dying ReLU" problem by allowing a small gradient (α) when x ≤ 0,
// so neurons that receive negative inputs can still receive weight updates.
//
//   fn(x)    =  x        if x > 0
//               α·x      if x ≤ 0
//
//   dfn(out) =  1        if out > 0   (input was positive → slope = 1)
//               α        if out ≤ 0   (input was negative/zero → slope = α)
//
// dfn uses the output directly: since α < 1, out ≤ 0 iff the input was ≤ 0.
// No need to store the pre-activation value.
//
// Typical α: 0.01 (default). Larger values (0.1–0.3) = "very leaky relu".
//
export function makeLeakyRelu(alpha = 0.01): Activation {
  return {
    fn:  (x)   => x > 0 ? x : alpha * x,
    dfn: (out) => out > 0 ? 1 : alpha,
  };
}

/** Leaky ReLU with the standard α = 0.01. */
export const leakyRelu: Activation = makeLeakyRelu(0.01);

// ─── ELU (Exponential Linear Unit) ───────────────────────────────────────────
//
// Smooth alternative to ReLU. Saturates to −α for large negative inputs,
// which keeps the mean activation close to zero and speeds up training.
//
//   fn(x)    =  x              if x > 0
//               α·(eˣ − 1)    if x ≤ 0
//
//   dfn(out) =  1              if out > 0
//               out + α        if out ≤ 0
//
// Derivation of the output form for x ≤ 0:
//   out = α·(eˣ−1)  →  eˣ = out/α + 1  →  α·eˣ = out + α
// So the derivative α·eˣ can be recovered from `out` alone — O(1), no exp() call.
//
// Typical α: 1.0 (default). Controls the saturation floor (−α).
//
export function makeElu(alpha = 1.0): Activation {
  return {
    fn:  (x)   => x > 0 ? x : alpha * (Math.exp(x) - 1),
    dfn: (out) => out > 0 ? 1 : out + alpha,
  };
}

/** ELU with the standard α = 1.0. */
export const elu: Activation = makeElu(1.0);
