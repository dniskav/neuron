#!/usr/bin/env node
// consume-smoke.mjs — feature 008, tarea T005 (FR-006).
//
// Prueba el paquete como lo haría un usuario: un directorio temporal vacío fuera
// de los repositorios, `npm install --ignore-scripts`, importación ESM y CJS,
// comprobación de tipos con tres resoluciones y un entrenamiento XOR determinista.
// Solo usa módulos `node:`. Nunca ejecuta `npm publish` ni escribe en el registro.
//
// Uso:
//   node consume-smoke.mjs <tarball.tgz>              instala el tarball local
//   node consume-smoke.mjs @dniskav/neuron@<versión>  instala desde el registro (solo lectura)
//   opción --keep: no borra el directorio temporal (se imprime su ruta)
//
// Códigos de salida: 0 = todas las comprobaciones OK; 1 = alguna FALLO; 2 = uso incorrecto.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PKG = '@dniskav/neuron';
const HERE = path.dirname(fileURLToPath(import.meta.url));
// tools/ -> 008-release-app/ -> specs/ -> raíz de la librería
const LIB_ROOT = path.resolve(HERE, '..', '..', '..');
const TSC = path.join(LIB_ROOT, 'node_modules', 'typescript', 'bin', 'tsc');

// PRNG reproducible (mulberry32) que sustituye a Math.random en los scripts hijos.
const PRNG_SRC = `
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const XOR_DATA = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]];
function trainXor() {
  Math.random = mulberry32(1);
  const net = new NetworkN([2, 3, 1]);
  for (let e = 0; e < 1500; e++) for (const [x, y] of XOR_DATA) net.train(x, y, 0.5);
  let loss = 0;
  for (const [x, y] of XOR_DATA) { const p = net.predict(x); loss += (y[0] - p[0]) ** 2; }
  return loss / XOR_DATA.length;
}
`;

const ESM_SRC = `import { NetworkN, NetworkTransformer, NetworkLSTM, Neuron } from '@dniskav/neuron';
${PRNG_SRC}
const names = [typeof NetworkN, typeof NetworkTransformer, typeof NetworkLSTM, typeof Neuron];
console.log(JSON.stringify({ loss: trainXor(), names }));
`;

const CJS_SRC = `const { NetworkN, NetworkTransformer, NetworkLSTM, Neuron } = require('@dniskav/neuron');
${PRNG_SRC}
const names = [typeof NetworkN, typeof NetworkTransformer, typeof NetworkLSTM, typeof Neuron];
console.log(JSON.stringify({ loss: trainXor(), names, resolved: require.resolve('@dniskav/neuron') }));
`;

// Consumidor TypeScript: la misma superficie que usa una app (tipos de NetworkN,
// NetworkTransformer.train con máscara y Activation).
const CONSUMER_TS = `import { NetworkN, NetworkTransformer, NetworkLSTM, Neuron, relu, sigmoid } from '@dniskav/neuron';
import type { Activation, NetworkNOptions } from '@dniskav/neuron';

const act: Activation = relu;
const opts: NetworkNOptions = { activations: [act, sigmoid] };
const net: NetworkN = new NetworkN([2, 3, 1], opts);
const mlpLoss: number = net.train([0, 1], [1], 0.5);
const tr: NetworkTransformer = new NetworkTransformer(4, { vocabSize: 10, nClasses: 9 });
const trLoss: number = tr.train([0, 1, 2, 3], [1, 2, 3, 4], 0.001, [true, true, false, false]);
export const refs = [mlpLoss, trLoss, NetworkLSTM, Neuron];
`;

const TSCONFIGS = {
  bundler: { module: 'ESNext', moduleResolution: 'bundler' },
  node16: { module: 'node16', moduleResolution: 'node16' },
  nodenext: { module: 'NodeNext' },
};

function usage(msg) {
  if (msg) console.error(`consume-smoke: ${msg}`);
  console.error('uso: node consume-smoke.mjs <tarball.tgz | @dniskav/neuron@<versión>> [--keep]');
  return 2;
}

// Interpreta el argumento: tarball local o especificador de registro con versión.
function parseSpec(input) {
  if (/\.tgz$/i.test(input) || fs.existsSync(input)) {
    const tgz = path.resolve(input);
    if (!fs.existsSync(tgz) || !fs.statSync(tgz).isFile()) {
      return { error: `no existe el tarball: ${input}` };
    }
    return { kind: 'tarball', spec: tgz, label: input };
  }
  const at = input.lastIndexOf('@');
  if (at <= 0 || at === input.length - 1) {
    return { error: 'falta la versión: use <tarball.tgz> o @dniskav/neuron@<versión>' };
  }
  if (input.slice(0, at) !== PKG) {
    return { error: `solo se admite ${PKG}` };
  }
  return { kind: 'registry', spec: input, label: input, expected: input.slice(at + 1) };
}

// Lee package.json del tarball sin instalarlo (tar es del sistema; solo lectura).
function readTarballManifest(tgz) {
  const r = spawnSync('tar', ['-xzOf', tgz, 'package/package.json'], { encoding: 'utf8' });
  if (r.status !== 0) return { error: (r.stderr || 'tar falló').trim().split('\n').pop() };
  try {
    return { manifest: JSON.parse(r.stdout) };
  } catch (e) {
    return { error: `package.json ilegible: ${e.message}` };
  }
}

function lastLine(text) {
  const lines = String(text || '').trim().split('\n').filter(Boolean);
  return lines.length ? lines[lines.length - 1].slice(0, 300) : '';
}

// Primera línea que describe el error (no el pie de Node ni el resumen de tsc).
function errorLine(text) {
  const lines = String(text || '').trim().split('\n').filter(Boolean);
  const hit = lines.find((l) => /error|cannot find|does not provide|not exported/i.test(l));
  return (hit || lines[lines.length - 1] || '').slice(0, 300);
}

// Rutas que el manifiesto declara para el paquete (exports, main, module, types).
function manifestPaths(m) {
  const out = [];
  const walk = (v) => {
    if (typeof v === 'string') out.push(v);
    else if (v && typeof v === 'object') Object.values(v).forEach(walk);
  };
  walk(typeof m.exports === 'string' ? m.exports : m.exports && m.exports['.']);
  for (const k of ['main', 'module', 'types']) if (typeof m[k] === 'string') out.push(m[k]);
  return out;
}

function runNode(file, cwd) {
  return spawnSync(process.execPath, [file], { cwd, encoding: 'utf8' });
}

function parseChild(r) {
  if (r.status !== 0) return { error: errorLine(r.stderr) || `salida ${r.status}` };
  try {
    return { data: JSON.parse(lastLine(r.stdout)) };
  } catch {
    return { error: `salida no JSON: ${lastLine(r.stdout)}` };
  }
}

function main() {
  const args = process.argv.slice(2);
  const keep = args.includes('--keep');
  const positional = args.filter((a) => a !== '--keep');
  if (positional.length !== 1 || positional[0].startsWith('-')) return usage();

  const spec = parseSpec(positional[0]);
  if (spec.error) return usage(spec.error);

  const results = [];
  const check = (ok, label, detail = '') => {
    results.push(ok);
    console.log(`${ok ? 'OK   ' : 'FALLO'} ${label}${detail ? ` — ${detail}` : ''}`);
    return ok;
  };

  if (!fs.existsSync(TSC)) {
    console.log(`FALLO tsc de la librería no encontrado: ${TSC} (ejecute npm ci)`);
    return 1;
  }

  // Directorio temporal fuera de los repositorios (os.tmpdir()).
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'neuron-consume-'));
  console.log(`consume-smoke: ${spec.label} | Node ${process.version} | tmp: ${work}`);

  let fatal = false;
  try {
    // 1. Versión esperada: en tarball, la del propio paquete; en registro, la pedida.
    let expectedVersion = spec.expected;
    if (spec.kind === 'tarball') {
      const m = readTarballManifest(spec.spec);
      if (m.error) {
        check(false, 'tarball legible (package/package.json)', m.error);
        return 1;
      }
      const okName = m.manifest.name === PKG;
      check(okName, 'tarball: name', okName ? PKG : `name = ${m.manifest.name}`);
      expectedVersion = m.manifest.version;
    }

    // 2. Proyecto temporal y instalación sin scripts.
    const init = spawnSync('npm', ['init', '-y'], { cwd: work, encoding: 'utf8' });
    if (init.status !== 0) {
      check(false, 'npm init -y', lastLine(init.stderr));
      return 1;
    }
    const inst = spawnSync(
      'npm',
      ['install', '--ignore-scripts', '--no-audit', '--no-fund', spec.spec],
      { cwd: work, encoding: 'utf8' },
    );
    if (!check(inst.status === 0, `instalación: npm install --ignore-scripts ${spec.label}`, inst.status === 0 ? '' : lastLine(inst.stderr))) {
      fatal = true;
      return 1;
    }

    // 3. Versión instalada y resolución.
    const installedPath = path.join(work, 'node_modules', ...PKG.split('/'), 'package.json');
    let installed = null;
    try {
      installed = JSON.parse(fs.readFileSync(installedPath, 'utf8'));
    } catch (e) {
      check(false, 'package.json instalado legible', e.message);
      fatal = true;
      return 1;
    }
    check(
      installed.name === PKG && installed.version === expectedVersion,
      `versión instalada ${installed.version} (esperada ${expectedVersion})`,
      installed.name === PKG ? '' : `name = ${installed.name}`,
    );

    // Las rutas declaradas deben existir. TypeScript con bundler cae al condition
    // `import` si `exports.types` apunta a un archivo inexistente, y el tsc no lo ve.
    const pkgDir = path.dirname(installedPath);
    const missing = manifestPaths(installed).filter((p) => !fs.existsSync(path.join(pkgDir, p)));
    check(
      missing.length === 0,
      'rutas del manifiesto existen (exports, main, module, types)',
      missing.length ? `no existen: ${missing.join(', ')}` : '',
    );

    // 4. ESM y CJS: importación de nombres y entrenamiento XOR.
    fs.writeFileSync(path.join(work, 'esm-check.mjs'), ESM_SRC);
    fs.writeFileSync(path.join(work, 'cjs-check.cjs'), CJS_SRC);
    const esm = parseChild(runNode(path.join(work, 'esm-check.mjs'), work));
    const cjs = parseChild(runNode(path.join(work, 'cjs-check.cjs'), work));

    const namesOk = (d) => d && d.names.every((t) => t === 'function');
    check(!esm.error && namesOk(esm.data), 'ESM: import { NetworkN, NetworkTransformer, NetworkLSTM, Neuron }', esm.error || '');
    check(!cjs.error && namesOk(cjs.data), 'CJS: require de NetworkN, NetworkTransformer, NetworkLSTM, Neuron', cjs.error || '');

    const esmLoss = esm.data ? esm.data.loss : NaN;
    const cjsLoss = cjs.data ? cjs.data.loss : NaN;
    check(
      Number.isFinite(esmLoss) && esmLoss < 0.05,
      `ESM: XOR NetworkN([2,3,1]) 1500 épocas lr 0.5, pérdida ${esmLoss.toFixed(6)} < 0.05`,
      esm.error || '',
    );
    check(
      Number.isFinite(cjsLoss) && cjsLoss < 0.05,
      `CJS: XOR NetworkN([2,3,1]) 1500 épocas lr 0.5, pérdida ${cjsLoss.toFixed(6)} < 0.05`,
      cjs.error || '',
    );
    check(
      esmLoss === cjsLoss,
      'determinista: ESM y CJS dan la misma pérdida',
      `${esmLoss} / ${cjsLoss}`,
    );

    const resolvedOk = cjs.data && cjs.data.resolved &&
      fs.realpathSync(cjs.data.resolved).startsWith(fs.realpathSync(path.join(work, 'node_modules')) + path.sep);
    check(!!resolvedOk, 'require.resolve dentro del node_modules temporal', cjs.data ? cjs.data.resolved : '');

    // 5. Tipos con tres resoluciones (tsc de la librería, ruta absoluta).
    fs.writeFileSync(path.join(work, 'consumer.ts'), CONSUMER_TS);
    for (const [mode, opts] of Object.entries(TSCONFIGS)) {
      const cfg = {
        compilerOptions: {
          target: 'ES2020',
          strict: true,
          noEmit: true,
          skipLibCheck: false,
          types: [],
          ...opts,
        },
        files: ['consumer.ts'],
      };
      const cfgName = `tsconfig.${mode}.json`;
      fs.writeFileSync(path.join(work, cfgName), JSON.stringify(cfg, null, 2));
      const r = spawnSync(process.execPath, [TSC, '-p', cfgName], { cwd: work, encoding: 'utf8' });
      const out = `${r.stdout || ''}${r.stderr || ''}`;
      check(r.status === 0, `tipos ${mode}: tsc --noEmit sobre consumer.ts`, r.status === 0 ? '' : errorLine(out));
    }
  } finally {
    if (keep) {
      console.log(`--keep: directorio conservado en ${work}`);
    } else {
      fs.rmSync(work, { recursive: true, force: true });
    }
  }

  const okCount = results.filter(Boolean).length;
  const all = results.length > 0 && okCount === results.length && !fatal;
  console.log(`Resultado: ${okCount}/${results.length} comprobaciones OK${all ? '' : ' — FALLO'}`);
  return all ? 0 : 1;
}

let code;
try {
  code = main();
} catch (e) {
  console.error(`FALLO inesperado: ${e && e.stack ? e.stack : e}`);
  code = 1;
}
process.exitCode = code;
