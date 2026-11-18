// Benchmark de la librería compilada (dist/index.mjs). Node puro, sin dependencias.
//
// Uso (desde la raíz del repo, tras `npm run build`):
//   node --expose-gc bench/run.mjs            # línea base completa
//   node --expose-gc bench/run.mjs --quick    # solo [256,256,256,10]
//
// Solo el JSON va a stdout (así se puede encadenar con otro proceso);
// el progreso va a stderr.

import { cpus } from 'node:os';
import { performance } from 'node:perf_hooks';
import * as lib from '../dist/index.mjs';

if (typeof globalThis.gc !== 'function') {
  process.stderr.write('Falta --expose-gc: node --expose-gc bench/run.mjs\n');
  process.exit(1);
}
const gc = globalThis.gc;

const quick = process.argv.includes('--quick');
const STRUCTURES = quick
  ? [[256, 256, 256, 10]]
  : [[256, 256, 256, 10], [512, 512, 512, 10]];

// Tres capas de pesos: relu, relu, sigmoid (una activación por capa no de entrada).
const ACTIVATIONS = [lib.relu, lib.relu, lib.sigmoid];

const OPTIMIZERS = [
  { name: 'sgd', factory: () => new lib.SGD(), lr: 0.01 },
  { name: 'adam', factory: () => new lib.Adam(), lr: 0.001 },
];

// Parámetros de la medición de tiempo (ver bench/README.md).
const WARMUP_BATCHES = 2;
const BATCHES = 5;
const CALLS_PER_BATCH = 20;
const TRAIN_BATCH_SIZE = 32; // tamaño de lote para NetworkFast.trainBatch
const N_SAMPLES = 64;        // ejemplos pregenerados, se reutilizan en cada tanda

// PRNG determinista (mulberry32) para que los datos sean iguales en cada ejecución.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeData(nIn, nOut, n, rnd) {
  const X = [];
  const T = [];
  for (let i = 0; i < n; i++) {
    X.push(Array.from({ length: nIn }, () => rnd()));
    T.push(Array.from({ length: nOut }, () => rnd()));
  }
  return { X, T };
}

// Número de pesos de una red densa: nOut * (nIn + 1) por capa (incluye sesgo).
function countParams(structure) {
  let total = 0;
  for (let i = 1; i < structure.length; i++) {
    total += structure[i] * (structure[i - 1] + 1);
  }
  return total;
}

// Heap usado tras forzar GC dos veces (la segunda limpia lo que la primera deja pendiente).
function heapAfterGc() {
  gc();
  gc();
  return process.memoryUsage().heapUsed;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// Mide la mediana (sobre BATCHES tandas) del tiempo medio por llamada de `callFn`.
// Cada tanda hace CALLS_PER_BATCH llamadas; las WARMUP_BATCHES primeras se descartan.
function medianMsPerCall(callFn) {
  const runBatch = () => {
    const t0 = performance.now();
    for (let c = 0; c < CALLS_PER_BATCH; c++) callFn();
    return (performance.now() - t0) / CALLS_PER_BATCH;
  };
  for (let w = 0; w < WARMUP_BATCHES; w++) runBatch();
  const times = [];
  for (let b = 0; b < BATCHES; b++) times.push(runBatch());
  return median(times);
}

// Heap por parámetro: diferencia de heap antes y después de construir la red.
function measureBytesPerParam(build, params) {
  const before = heapAfterGc();
  const net = build();
  const after = heapAfterGc();
  return { bytesPerParam: (after - before) / params, net };
}

function measureModel(model, structure, opt, data, rnd) {
  const params = countParams(structure);
  const build = () => {
    if (model === 'NetworkN') {
      return new lib.NetworkN(structure, { activations: ACTIVATIONS, optimizer: opt.factory });
    }
    return new lib.NetworkFast(structure, { activations: ACTIVATIONS, optimizer: opt.factory });
  };
  const { bytesPerParam, net } = measureBytesPerParam(build, params);

  const { X, T } = data;
  let idx = 0;
  const next = () => {
    const i = idx;
    idx = (idx + 1) % N_SAMPLES;
    return i;
  };

  const trainMs = medianMsPerCall(() => {
    const i = next();
    net.train(X[i], T[i], opt.lr);
  });
  const predictMs = medianMsPerCall(() => {
    net.predict(X[next()]);
  });

  const row = { model, structure, optimizer: opt.name, params, bytesPerParam, trainMs, predictMs };

  // Solo NetworkFast tiene trainBatch. Se mide por ejemplo: tiempo de la llamada / tamaño del lote.
  if (model === 'NetworkFast' && typeof net.trainBatch === 'function') {
    const XB = X.slice(0, TRAIN_BATCH_SIZE);
    const TB = T.slice(0, TRAIN_BATCH_SIZE);
    const batchCallMs = medianMsPerCall(() => {
      net.trainBatch(XB, TB, opt.lr);
    });
    row.batchMsPerSample = batchCallMs / TRAIN_BATCH_SIZE;
  }
  return row;
}

const rnd = mulberry32(42);
const rows = [];
const hasNetworkFast = typeof lib.NetworkFast === 'function';

for (const structure of STRUCTURES) {
  const data = makeData(structure[0], structure[structure.length - 1], N_SAMPLES, rnd);
  for (const opt of OPTIMIZERS) {
    process.stderr.write(`NetworkN ${structure.join('-')} ${opt.name}\n`);
    rows.push(measureModel('NetworkN', structure, opt, data, rnd));
    if (hasNetworkFast) {
      process.stderr.write(`NetworkFast ${structure.join('-')} ${opt.name}\n`);
      rows.push(measureModel('NetworkFast', structure, opt, data, rnd));
    }
  }
}

const out = {
  node: process.version,
  cpu: cpus()[0]?.model ?? 'unknown',
  date: new Date().toISOString(),
  rows,
};
process.stdout.write(JSON.stringify(out, null, 2) + '\n');
