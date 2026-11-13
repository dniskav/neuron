#!/usr/bin/env node
// app-typecheck.mjs (feature 008, T006)
//
// Compara los errores de tipos (archivo:línea:código TS) de una app que usa
// @dniskav/neuron contra dos resoluciones del paquete:
//   - 0.2.5: dist/index.d.ts del node_modules de la app (lo que compila hoy);
//   - actual: dist/index.d.ts de la librería (por defecto ../../../dist/index.d.ts
//     respecto a este script; se cambia con --lib-types).
//
// Uso: node app-typecheck.mjs --app <ruta de la app> [--lib-types <ruta a index.d.ts>]
//      (si falta --app se usa la variable de entorno APPWT)
//
// Garantías:
//   - no escribe NADA dentro de la app. Solo lee su tsconfig.app.json, su src/,
//     su node_modules y ejecuta `git -C <app> rev-parse/status` (lectura).
//   - usa el tsc de node_modules de la librería por ruta absoluta.
//   - los tsconfig temporales y la salida de tsc van a un directorio de os.tmpdir()
//     que se borra al terminar.
//
// Salida: 0 si no hay errores NUEVOS con la librería actual; 1 si los hay;
//         2 si no se puede preparar la comprobación (falta tsc, dist, app, etc.).
// La última línea es siempre el resumen:
//   0.2.5: N errores | 0.4.0: M errores | solo en 0.4.0: K

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const libRoot = path.resolve(here, '..', '..', '..');
const tscBin = path.join(libRoot, 'node_modules', 'typescript', 'bin', 'tsc');

function fail(msg) {
  console.error(`app-typecheck: ${msg}`);
  process.exit(2);
}

function parseArgs(argv) {
  const opts = {
    app: process.env.APPWT || '',
    libTypes: path.join(libRoot, 'dist', 'index.d.ts'),
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--app') opts.app = argv[++i] ?? '';
    else if (a === '--lib-types') opts.libTypes = argv[++i] ?? '';
    else if (a === '--help' || a === '-h') {
      console.log('uso: node app-typecheck.mjs --app <ruta de la app> [--lib-types <index.d.ts>]');
      process.exit(0);
    } else fail(`argumento desconocido: ${a}`);
  }
  if (!opts.app) fail('falta --app <ruta de la app> (o la variable APPWT)');
  if (!opts.libTypes) fail('--lib-types sin valor');
  return opts;
}

function git(cwd, args) {
  const r = spawnSync('git', ['-C', cwd, ...args], { encoding: 'utf8' });
  return r.status === 0 ? r.stdout.trim() : null;
}

// Ruta relativa a la app (si cae dentro); si no, la absoluta.
// tsc imprime las rutas relativas a su cwd (el directorio temporal): se resuelven contra él.
function relToApp(file, cwd, appReal) {
  let abs = path.resolve(cwd, file);
  try {
    abs = fs.realpathSync(abs);
  } catch {
    // el archivo puede no existir en disco; se usa tal cual
  }
  const rel = path.relative(appReal, abs);
  return rel === '' || rel.startsWith('..') || path.isAbsolute(rel) ? abs : rel;
}

// Ejecuta tsc con el tsconfig dado y devuelve la lista "archivo:línea:código".
function tscErrors(configPath, cwd, appReal) {
  const r = spawnSync(process.execPath, [tscBin, '-p', configPath, '--pretty', 'false'], {
    cwd,
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  });
  if (r.error) fail(`no se pudo ejecutar tsc: ${r.error.message}`);
  const lines = `${r.stdout}\n${r.stderr}`.split('\n');
  const fileErr = /^(.+?)\((\d+),\d+\): error (TS\d+):/;
  const errors = [];
  const configErrors = [];
  for (const line of lines) {
    const m = fileErr.exec(line);
    if (m) errors.push(`${relToApp(m[1], cwd, appReal)}:${m[2]}:${m[3]}`);
    else if (/error TS\d+:/.test(line)) configErrors.push(line);
  }
  if (configErrors.length > 0) {
    fail(`errores de configuración de tsc (no son de la app):\n${configErrors.join('\n')}`);
  }
  return errors;
}

// tsconfig temporal: hereda el tsconfig.app.json REAL de la app (solo lectura)
// y fija la resolución de @dniskav/neuron a un dist/index.d.ts concreto.
function writeConfig(dir, name, appAbs, neuronTypes) {
  const cfg = {
    extends: path.join(appAbs, 'tsconfig.app.json'),
    compilerOptions: {
      noEmit: true,
      incremental: false,
      composite: false,
      tsBuildInfoFile: path.join(dir, `${name}.tsbuildinfo`),
      // sin la extensión: TS prueba index.ts, index.tsx y después index.d.ts
      paths: { '@dniskav/neuron': [neuronTypes.replace(/\.d\.ts$/, '')] },
    },
    include: [path.join(appAbs, 'src')],
  };
  const file = path.join(dir, `${name}.tsconfig.json`);
  fs.writeFileSync(file, JSON.stringify(cfg, null, 2));
  return file;
}

function readVersion(pkgPath) {
  try {
    return JSON.parse(fs.readFileSync(pkgPath, 'utf8')).version;
  } catch {
    return 'desconocida';
  }
}

const opts = parseArgs(process.argv.slice(2));
const appAbs = path.resolve(opts.app);
if (!fs.existsSync(path.join(appAbs, 'tsconfig.app.json'))) fail(`no hay tsconfig.app.json en ${appAbs}`);
if (!fs.existsSync(path.join(appAbs, 'src'))) fail(`no hay src/ en ${appAbs}`);
const appReal = fs.realpathSync(appAbs);

const libTypesAbs = path.resolve(opts.libTypes);
if (!fs.existsSync(libTypesAbs)) fail(`no existe ${libTypesAbs} (ejecuta npm run build en la librería)`);
if (!fs.existsSync(tscBin)) fail(`no existe el tsc de la librería en ${tscBin} (ejecuta npm ci)`);

const appNeuronDir = path.join(appAbs, 'node_modules', '@dniskav', 'neuron');
const appNeuronTypes = path.join(appNeuronDir, 'dist', 'index.d.ts');
if (!fs.existsSync(appNeuronTypes)) fail(`la app no tiene @dniskav/neuron instalado (${appNeuronTypes})`);

const appVersion = readVersion(path.join(appNeuronDir, 'package.json'));
const libVersion = readVersion(path.join(libRoot, 'package.json'));
const appHead = git(appAbs, ['rev-parse', '--short', 'HEAD']) ?? '(sin git)';
const appBranch = git(appAbs, ['branch', '--show-current']) ?? '(sin rama)';
const appStatus = git(appAbs, ['status', '--short']) ?? '(sin git)';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 't006-'));
try {
  // tsconfig.app.json declara "types": ["vite/client"]; TS resuelve ese tipo desde
  // el directorio del tsconfig temporal, que no tiene node_modules. Se enlaza
  // SOLO el paquete vite de la app dentro del directorio temporal (la app no se toca).
  const appVite = path.join(appAbs, 'node_modules', 'vite');
  if (fs.existsSync(appVite)) {
    fs.mkdirSync(path.join(dir, 'node_modules'));
    fs.symlinkSync(appVite, path.join(dir, 'node_modules', 'vite'), 'dir');
  }
  const cfgOld = writeConfig(dir, 'app-0.2.5', appAbs, appNeuronTypes);
  const cfgNew = writeConfig(dir, 'app-0.4.0', appAbs, libTypesAbs);
  const errOld = tscErrors(cfgOld, dir, appReal);
  const errNew = tscErrors(cfgNew, dir, appReal);

  const setOld = new Set(errOld);
  const setNew = new Set(errNew);
  const onlyNew = [...setNew].filter((e) => !setOld.has(e)).sort();
  const onlyOld = [...setOld].filter((e) => !setNew.has(e)).sort();

  console.log(`app: ${appAbs}`);
  console.log(`  HEAD ${appHead} (rama ${appBranch})`);
  console.log(`  cambios sin commitear: ${appStatus === '' ? 'ninguno' : appStatus.split('\n').map((l) => l.trim()).join(', ')}`);
  console.log(`0.2.5 = ${appNeuronTypes} (versión ${appVersion})`);
  console.log(`0.4.0 = ${libTypesAbs} (package.json de la librería ${libVersion})`);
  console.log(`errores con 0.2.5 (${errOld.length}):`);
  for (const e of [...errOld].sort()) console.log(`  ${e}`);
  console.log(`errores con 0.4.0 (${errNew.length}):`);
  for (const e of [...errNew].sort()) console.log(`  ${e}`);
  console.log(`solo con 0.4.0 (${onlyNew.length}):`);
  for (const e of onlyNew) console.log(`  ${e}`);
  console.log(`solo con 0.2.5 (${onlyOld.length}):`);
  for (const e of onlyOld) console.log(`  ${e}`);
  console.log(`0.2.5: ${errOld.length} errores | 0.4.0: ${errNew.length} errores | solo en 0.4.0: ${onlyNew.length}`);
  process.exitCode = onlyNew.length > 0 ? 1 : 0;
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}
