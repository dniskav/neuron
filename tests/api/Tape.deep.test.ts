// T005 (I7): Value.backward() iterativo. Dos propiedades:
//  1) cadenas muy largas no desbordan la pila (20 000 y 100 000 nodos; la
//     recursión de Node 20 y de Node 26 se agota mucho antes de 20 000).
//  2) resultado numérico IDÉNTICO (toBe) al backward recursivo antiguo, cuya
//     copia vive en este archivo como referencia (oldBackward).
import { describe, it, expect } from 'vitest';
import { Value } from '../../src/Tape';

type Internals = { _prev: Set<Value>; _backward: () => void };

// Copia literal de la implementación recursiva anterior a T005.
function oldBackward(root: Value): void {
  const topo: Value[] = [];
  const visited = new Set<Value>();
  const buildTopo = (v: Value): void => {
    if (!visited.has(v)) {
      visited.add(v);
      for (const child of (v as unknown as Internals)._prev) buildTopo(child);
      topo.push(v);
    }
  };
  buildTopo(root);
  root.grad = 1;
  for (let i = topo.length - 1; i >= 0; i--) {
    (topo[i] as unknown as Internals)._backward();
  }
}

// Construye el mismo grafo dos veces (nuevo y antiguo) y compara grad por nodo.
function compare(build: () => { nodes: Value[]; out: Value }): void {
  const a = build();
  const b = build();
  a.out.backward();
  oldBackward(b.out);
  expect(a.nodes.length).toBe(b.nodes.length);
  expect(a.nodes.length).toBeGreaterThan(0);
  let nonZero = 0;
  for (let i = 0; i < a.nodes.length; i++) {
    expect(a.nodes[i].data).toBe(b.nodes[i].data);
    expect(a.nodes[i].grad).toBe(b.nodes[i].grad);
    if (a.nodes[i].grad !== 0) nonZero++;
  }
  expect(nonZero).toBeGreaterThan(0);
}

describe('Value.backward: cadenas profundas sin desbordar la pila', () => {
  for (const n of [20000, 100000]) {
    it(`suma de ${n} nodos: dL/dx = 1`, () => {
      const x = new Value(0.5);
      let v = x;
      for (let i = 0; i < n; i++) v = v.add(1);
      expect(() => v.backward()).not.toThrow();
      expect(x.grad).toBe(1);
      expect(v.data).toBe(0.5 + n);
    });

    it(`producto de ${n} nodos: dL/dx = 1.0001^n`, () => {
      const x = new Value(1);
      let v = x;
      for (let i = 0; i < n; i++) v = v.mul(1.0001);
      expect(() => v.backward()).not.toThrow();
      const expected = Math.pow(1.0001, n);
      expect(Math.abs(x.grad - expected) / expected).toBeLessThan(1e-9);
    });

    it(`tanh de ${n} nodos: dL/dx = producto de (1 - t²)`, () => {
      const x = new Value(0.3);
      let v = x;
      let expected = 1;
      let t = 0.3;
      for (let i = 0; i < n; i++) {
        t = Math.tanh(t);
        expected *= 1 - t * t;
        v = v.tanh();
      }
      expect(() => v.backward()).not.toThrow();
      expect(expected).toBeGreaterThan(0);
      expect(Math.abs(x.grad - expected) / expected).toBeLessThan(1e-9);
    });
  }
});

describe('Value.backward: idéntico al recursivo antiguo (toBe por nodo)', () => {
  it('diamante repetido (~200 nodos) con ramas compartidas', () => {
    compare(() => {
      const x = new Value(0.7);
      const nodes: Value[] = [x];
      let v = x;
      for (let i = 0; i < 40; i++) {
        const l = v.mul(1.1 + i * 0.01);
        const r = v.add(0.3 * i);
        const j = l.mul(r).tanh();
        v = j.add(v); // v reutilizado: aparece en l, r y aquí
        nodes.push(l, r, j, v);
      }
      return { nodes, out: v };
    });
  });

  it('a.add(a) y a.mul(a)', () => {
    compare(() => {
      const a = new Value(1.7);
      const s = a.add(a);
      const p = a.mul(a);
      const out = s.mul(p).add(s);
      return { nodes: [a, s, p, out], out };
    });
  });

  it('a.add(a): el mismo nodo dos veces acumula 2', () => {
    const a = new Value(3);
    const out = a.add(a);
    out.backward();
    expect(a.grad).toBe(2);
  });

  it('red de 2 neuronas con tanh/sigmoid/relu/exp/pow', () => {
    compare(() => {
      const xs = [new Value(0.9), new Value(-0.4), new Value(0.25)];
      const w1 = [0.3, -0.8, 0.5].map((d) => new Value(d));
      const w2 = [-0.6, 0.2, 0.7].map((d) => new Value(d));
      const b1 = new Value(0.1);
      const b2 = new Value(-0.2);
      const z1 = xs[0].mul(w1[0]).add(xs[1].mul(w1[1])).add(xs[2].mul(w1[2])).add(b1);
      const z2 = xs[0].mul(w2[0]).add(xs[1].mul(w2[1])).add(xs[2].mul(w2[2])).add(b2);
      const h1 = z1.tanh();
      const h2 = z2.sigmoid();
      const h3 = z1.relu().add(z2.relu());
      const mix = h1.mul(h2).add(h3.exp()).add(h2.pow(3)).add(h1.pow(2));
      const out = mix.div(h2.add(2)).sub(h1);
      return { nodes: [...xs, ...w1, ...w2, b1, b2, z1, z2, h1, h2, h3, mix, out], out };
    });
  });

  it('cadena larga que cabe en la recursión (500 nodos) con reutilización', () => {
    compare(() => {
      const x = new Value(0.4);
      const nodes: Value[] = [x];
      let v = x;
      for (let i = 0; i < 500; i++) {
        v = i % 3 === 0 ? v.tanh() : i % 3 === 1 ? v.mul(x).add(0.01) : v.sigmoid();
        nodes.push(v);
      }
      return { nodes, out: v };
    });
  });

  it('llamadas repetidas acumulan grad igual que antes', () => {
    const build = () => {
      const a = new Value(2);
      const out = a.mul(a).add(a);
      return { a, out };
    };
    const n = build();
    const o = build();
    n.out.backward();
    n.out.backward();
    oldBackward(o.out);
    oldBackward(o.out);
    expect(n.a.grad).toBe(o.a.grad);
  });
});
