#!/usr/bin/env node
// Gate de ejemplos del README (T002, feature 007).
//
// Uso: node specs/007-docs/tools/readme-gate.mjs <archivo.md> [--runs N] [--allow-empty]
//
// Ejecuta specs/000-audit/evidence/extract.mjs sobre el .md indicado, N veces (3 por defecto).
// Sale con 0 solo si TODAS las ejecuciones dan fail=0 y hay bloques (o se pasó --allow-empty).
// Sale con 1 si: falta el archivo, falta dist/index.mjs (npm run build), no se puede resolver
// esbuild, el extractor falla, no hay bloques (sin --allow-empty) o algún ejemplo falla.
//
// No modifica el repositorio: trabaja en un directorio temporal que borra al terminar,
// también cuando falla. Solo usa builtins de Node (sin dependencias).

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const EXTRACT = path.join(REPO, 'specs/000-audit/evidence/extract.mjs');
const DIST_ENTRY = path.join(REPO, 'dist/index.mjs');
const REPO_NODE_MODULES = path.join(REPO, 'node_modules');
const USAGE = 'Uso: node specs/007-docs/tools/readme-gate.mjs <archivo.md> [--runs N] [--allow-empty]';

function parseArgs(argv) {
  const opts = {file: null, runs: 3, allowEmpty: false};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--runs') {
      const v = argv[++i];
      if (!/^[1-9]\d*$/.test(v ?? '')) throw new Error(`--runs requiere un entero >= 1 (recibido: ${v ?? '(nada)'})`);
      opts.runs = Number(v);
    } else if (a === '--allow-empty') {
      opts.allowEmpty = true;
    } else if (a.startsWith('--')) {
      throw new Error(`opción desconocida: ${a}`);
    } else if (opts.file === null) {
      opts.file = a;
    } else {
      throw new Error(`argumento sobrante: ${a}`);
    }
  }
  if (opts.file === null) throw new Error('falta el archivo .md');
  return opts;
}

// Crea <tmp>/README.md, <tmp>/node_modules (enlace al del repo) y <tmp>/s/{dist,readme}.
function prepareLayout(src, tmp) {
  const scratch = path.join(tmp, 's');
  fs.mkdirSync(path.join(scratch, 'readme'), {recursive: true});
  fs.copyFileSync(src, path.join(tmp, 'README.md'));
  fs.symlinkSync(REPO_NODE_MODULES, path.join(tmp, 'node_modules'), 'dir');
  fs.symlinkSync(path.join(REPO, 'dist'), path.join(scratch, 'dist'), 'dir');
  return scratch;
}

// Líneas de inicio de los bloques ts/js, con la misma lógica que extract.mjs (índice i = bN.mjs).
function blockStarts(md) {
  const starts = [];
  let inb = false;
  let start = 0;
  md.split('\n').forEach((l, i) => {
    if (/^```(ts|typescript|js|javascript)\s*$/.test(l)) {
      inb = true;
      start = i + 1;
      return;
    }
    if (inb && /^```/.test(l)) {
      inb = false;
      starts.push(start);
    }
  });
  return starts;
}

// Reejecuta un bloque ya extraído y devuelve la línea de error de su stderr.
function blockError(readmeDir, index) {
  const file = path.join(readmeDir, `b${index}.mjs`);
  if (!fs.existsSync(file)) return 'sin archivo extraído (error de sintaxis del extractor)';
  const r = spawnSync(process.execPath, [file], {encoding: 'utf8', cwd: readmeDir, timeout: 30000});
  const lines = (r.stderr || '').split('\n').map((x) => x.trim()).filter(Boolean);
  const msg = lines.find((x) => /^(\w*Error|Uncaught|AssertionError)\b.*/.test(x)) || lines[lines.length - 1];
  return msg || `código de salida ${r.status}`;
}

function gate(argv) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (e) {
    console.error(`readme-gate: ${e.message}\n${USAGE}`);
    return 1;
  }

  const src = path.resolve(process.cwd(), opts.file);
  if (!fs.existsSync(src) || !fs.statSync(src).isFile()) {
    console.error(`readme-gate: no existe el archivo ${opts.file}`);
    return 1;
  }
  if (!fs.existsSync(DIST_ENTRY)) {
    console.error(`readme-gate: falta dist/index.mjs (${DIST_ENTRY}). Ejecuta npm run build antes del gate.`);
    return 1;
  }
  if (!fs.existsSync(REPO_NODE_MODULES)) {
    console.error('readme-gate: falta node_modules en el repositorio. Ejecuta npm ci.');
    return 1;
  }
  if (!fs.existsSync(EXTRACT)) {
    console.error(`readme-gate: no existe el extractor ${EXTRACT}`);
    return 1;
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'neuron-readme-gate-'));
  try {
    return runGate(src, opts, tmp);
  } finally {
    // rmSync no sigue enlaces simbólicos: borra solo los enlaces, no node_modules ni dist.
    fs.rmSync(tmp, {recursive: true, force: true});
  }
}

function runGate(src, opts, tmp) {
  const scratch = prepareLayout(src, tmp);
  const readmeDir = path.join(scratch, 'readme');

  try {
    createRequire(tmp + path.sep).resolve('esbuild');
  } catch {
    console.error('readme-gate: no se puede resolver esbuild desde node_modules del repositorio. Ejecuta npm ci.');
    return 1;
  }

  console.log(`readme-gate: ${opts.file} (${opts.runs} ejecución(es))`);

  let bloques = null;
  let totalFail = 0;
  // Fallos por línea de inicio del bloque: { heading, runs: [n, ...] }
  const failures = new Map();

  for (let run = 1; run <= opts.runs; run++) {
    fs.rmSync(readmeDir, {recursive: true, force: true});
    fs.mkdirSync(readmeDir, {recursive: true});

    const r = spawnSync(process.execPath, [EXTRACT, tmp, scratch], {encoding: 'utf8', maxBuffer: 64 * 1024 * 1024});
    const out = `${r.stdout || ''}${r.stderr || ''}`;
    if (r.error || r.status !== 0) {
      const tail = (r.stderr || '').trim().split('\n').slice(-3).join('\n    ');
      console.error(`readme-gate: el extractor falló en la ejecución ${run} (código ${r.status}${r.error ? `, ${r.error.message}` : ''})`);
      if (tail) console.error(`    ${tail}`);
      return 1;
    }

    const m = out.match(/bloques=(\d+) ok=(\d+) fail=(\d+)/);
    if (!m) {
      console.error(`readme-gate: el extractor no imprimió el resumen en la ejecución ${run}`);
      return 1;
    }
    const [nBloques, nOk, nFail] = m.slice(1).map(Number);
    console.log(`  ejecución ${run}/${opts.runs}: bloques=${nBloques} ok=${nOk} fail=${nFail}`);

    for (const line of out.split('\n')) {
      const m2 = line.match(/^(?:FAIL )?L(\d+) \[([^\]]*)\]/);
      if (m2 && /^FAIL |SYNTAX:/.test(line)) {
        const key = Number(m2[1]);
        if (!failures.has(key)) failures.set(key, {heading: m2[2], runs: []});
        failures.get(key).runs.push(run);
      }
    }

    if (bloques !== null && nBloques !== bloques) {
      console.error(`readme-gate: el número de bloques cambió entre ejecuciones (${bloques} y ${nBloques}).`);
      return 1;
    }
    bloques = nBloques;
    totalFail += nFail;

    if (bloques === 0) {
      if (!opts.allowEmpty) {
        console.error('readme-gate: el archivo no tiene bloques ts/js ejecutables (usa --allow-empty solo para partes sin ejemplos).');
        return 1;
      }
      console.log('readme-gate: sin bloques ejecutables (--allow-empty).');
      return 0;
    }
  }

  if (totalFail > 0) {
    // Los bloques de la última ejecución siguen en readmeDir: se reejecutan para leer el error real.
    const starts = blockStarts(fs.readFileSync(path.join(tmp, 'README.md'), 'utf8'));
    console.error(`readme-gate: FALLO — ${totalFail} fallo(s) de ejemplo en ${opts.runs} ejecución(es):`);
    for (const [line, info] of [...failures].sort((a, b) => a[0] - b[0])) {
      const index = starts.indexOf(line);
      const err = index < 0 ? 'bloque no localizado' : blockError(readmeDir, index);
      console.error(`  L${line} [${info.heading}] ejecuciones ${info.runs.join(',')}: ${err}`);
    }
    return 1;
  }

  console.log(`readme-gate: OK — ${opts.runs} ejecución(es), ${bloques} bloque(s), fail=0.`);
  return 0;
}

process.exitCode = gate(process.argv.slice(2));
