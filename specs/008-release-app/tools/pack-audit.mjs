#!/usr/bin/env node
// pack-audit.mjs: auditoría del paquete que publicaría `npm publish` (feature 008, tarea T004).
//
// Uso:
//   node pack-audit.mjs                       audita la raíz de este repo
//   node pack-audit.mjs --cwd <dir>           audita el repo en <dir> (npm pack --dry-run --json)
//   node pack-audit.mjs --tarball <a.tgz>     audita un tarball ya construido (se extrae en un temporal)
//
// Salida: una línea por comprobación (OK / FALLO) y un resumen final.
// Códigos de salida: 0 = todas las comprobaciones pasan; 1 = alguna falla; 2 = error de uso.
// Solo usa módulos node: y nunca ejecuta `npm publish` ni escribe en el registro.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const EXPECTED_FILES = [
  'LICENSE',
  'README.md',
  'package.json',
  'dist/index.js',
  'dist/index.mjs',
  'dist/index.d.ts',
  'dist/index.d.mts',
];
const EXPECTED_NAME = '@dniskav/neuron';
const EXPORT_KEYS = ['types', 'import', 'require'];
const ENGINES_NODE = '>=20';
const MAX_COMPRESSED = 250000; // bytes
const MAX_UNPACKED = 1000000; // bytes (1 MB, SI)
const DEFAULT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

const USAGE = `Uso:
  node pack-audit.mjs [--cwd <dir>] [--tarball <archivo.tgz>]
  (sin argumentos audita la raiz de este repo; --cwd y --tarball no se combinan)`;

class UsageError extends Error {}

function parseArgs(argv) {
  const opts = { cwd: null, tarball: null, help: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') {
      opts.help = true;
    } else if (arg === '--cwd' || arg === '--tarball') {
      const value = argv[i + 1];
      if (value === undefined || value.startsWith('--')) {
        throw new UsageError(`${arg} necesita un valor`);
      }
      opts[arg.slice(2)] = value;
      i++;
    } else {
      throw new UsageError(`argumento desconocido: ${arg}`);
    }
  }
  if (opts.cwd !== null && opts.tarball !== null) {
    throw new UsageError('--cwd y --tarball no se pueden usar juntos');
  }
  return opts;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function isFileAt(base, rel) {
  try {
    return fs.statSync(path.join(base, rel)).isFile();
  } catch {
    return false;
  }
}

// Recorre un directorio y devuelve los archivos regulares y los enlaces o entradas raras.
function walk(dir, base = dir, out = { files: [], odd: [] }) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, ent.name);
    const rel = path.relative(base, abs).split(path.sep).join('/');
    if (ent.isDirectory()) {
      walk(abs, base, out);
    } else if (ent.isFile()) {
      out.files.push({ rel, size: fs.lstatSync(abs).size });
    } else {
      out.odd.push(rel);
    }
  }
  return out;
}

// Fuente de datos para las comprobaciones: el repo (npm pack --dry-run) o un tarball extraído.
function fromRepo(root) {
  const pkg = readJson(path.join(root, 'package.json'));
  const out = execFileSync('npm', ['pack', '--dry-run', '--json'], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    maxBuffer: 64 * 1024 * 1024,
  });
  const packed = JSON.parse(out);
  if (!Array.isArray(packed) || packed.length !== 1) {
    throw new Error('npm pack --dry-run --json no devolvió exactamente un paquete');
  }
  const entry = packed[0];
  return {
    origen: `repo ${root} (npm pack --dry-run)`,
    pkgRoot: root,
    pkg,
    shipped: entry.files.map((f) => f.path),
    compressed: entry.size,
    unpacked: entry.unpackedSize,
    packMeta: { name: entry.name, version: entry.version },
    tarballName: null,
    structureProblems: [],
  };
}

function fromTarball(tarball, tmp) {
  const listing = execFileSync('tar', ['-tzf', tarball], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    maxBuffer: 64 * 1024 * 1024,
  })
    .split('\n')
    .filter((l) => l.length > 0);
  const structureProblems = [];
  for (const entry of listing) {
    if (entry.endsWith('/')) continue;
    if (!entry.startsWith('package/')) structureProblems.push(`fuera de package/: ${entry}`);
    if (path.posix.isAbsolute(entry) || entry.split('/').includes('..')) {
      structureProblems.push(`ruta insegura: ${entry}`);
    }
  }
  execFileSync('tar', ['-xzf', tarball, '-C', tmp], { stdio: ['ignore', 'pipe', 'pipe'] });
  const pkgRoot = path.join(tmp, 'package');
  if (!fs.existsSync(path.join(pkgRoot, 'package.json'))) {
    throw new Error('el tarball no contiene package/package.json');
  }
  const tree = walk(pkgRoot);
  for (const rel of tree.odd) structureProblems.push(`no es archivo regular: ${rel}`);
  const pkg = readJson(path.join(pkgRoot, 'package.json'));
  return {
    origen: `tarball ${tarball}`,
    pkgRoot,
    pkg,
    shipped: tree.files.map((f) => f.rel),
    compressed: fs.statSync(tarball).size,
    unpacked: tree.files.reduce((sum, f) => sum + f.size, 0),
    packMeta: null,
    tarballName: path.basename(tarball),
    structureProblems,
  };
}

// `exports` exige "./"; `main`, `module` y `types` admiten la ruta sin "./" (npm la acepta).
function targetProblem(ctx, target, requireDot) {
  if (typeof target !== 'string') return 'no es texto';
  if (requireDot && !target.startsWith('./')) return 'debe empezar por ./';
  const rel = path.posix.normalize(target.replace(/^\.\//, ''));
  if (rel.startsWith('..') || path.posix.isAbsolute(rel)) return 'sale del paquete';
  if (!ctx.shipped.includes(rel)) return 'no está en la lista de archivos del paquete';
  if (!isFileAt(ctx.pkgRoot, rel)) return 'no existe en disco';
  return null;
}

// Quita comentarios para que un texto dentro de un comentario no cuente como código.
function codeOnly(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('\n')
    .filter((line) => !/^\s*\/\//.test(line))
    .join('\n');
}

function runChecks(ctx, record) {
  const { pkg, shipped } = ctx;

  // (0) Solo con tarball: estructura del archivo.
  if (ctx.tarballName !== null) {
    record(
      ctx.structureProblems.length === 0,
      'estructura del tarball',
      ctx.structureProblems.length === 0
        ? 'todas las entradas están bajo package/ y son archivos regulares'
        : ctx.structureProblems.slice(0, 5).join('; ')
    );
  }

  // (1) Lista exacta de archivos.
  const expected = [...EXPECTED_FILES].sort();
  const actual = [...shipped].sort();
  const missing = expected.filter((f) => !actual.includes(f));
  const extra = actual.filter((f) => !expected.includes(f));
  record(
    missing.length === 0 && extra.length === 0,
    '(1) archivos del paquete',
    missing.length === 0 && extra.length === 0
      ? `${actual.length} archivos, exactamente los esperados`
      : `faltan: [${missing.join(', ')}] sobran: [${extra.join(', ')}]`
  );

  // (2) Nombre y versión.
  let identityOk = pkg.name === EXPECTED_NAME;
  let identityDetail = `name=${pkg.name} version=${pkg.version}`;
  if (ctx.packMeta !== null) {
    identityOk = identityOk && ctx.packMeta.name === pkg.name && ctx.packMeta.version === pkg.version;
    identityDetail += ` (npm pack: ${ctx.packMeta.name}@${ctx.packMeta.version})`;
  } else {
    const fromFile = ctx.tarballName.match(/^dniskav-neuron-(.+)\.tgz$/);
    if (fromFile) {
      identityOk = identityOk && fromFile[1] === pkg.version;
      identityDetail += ` (nombre de archivo: ${fromFile[1]})`;
    } else {
      identityDetail += ' (nombre de archivo sin versión reconocible; no se compara)';
    }
  }
  record(identityOk, '(2) name y version', identityDetail);

  // (3) package.json publicado.
  const exp = pkg.exports && typeof pkg.exports === 'object' ? pkg.exports['.'] : undefined;
  const hasDot = exp !== null && typeof exp === 'object';
  const keys = hasDot ? Object.keys(exp) : [];
  const keysOk = hasDot && keys.length === EXPORT_KEYS.length && keys.every((k, i) => k === EXPORT_KEYS[i]);
  record(
    keysOk,
    '(3) exports["."] claves en orden',
    keysOk ? `${keys.join(', ')}` : `encontrado: [${keys.join(', ') || 'ninguna'}], esperado: [${EXPORT_KEYS.join(', ')}]`
  );

  const targets = [
    ['exports.types', hasDot ? exp.types : undefined, true],
    ['exports.import', hasDot ? exp.import : undefined, true],
    ['exports.require', hasDot ? exp.require : undefined, true],
    ['main', pkg.main, false],
    ['module', pkg.module, false],
    ['types', pkg.types, false],
  ];
  const problems = [];
  for (const [label, value, requireDot] of targets) {
    const p = targetProblem(ctx, value, requireDot);
    if (p !== null) problems.push(`${label} (${JSON.stringify(value)}): ${p}`);
  }
  record(
    problems.length === 0,
    '(3) rutas de exports, main, module y types',
    problems.length === 0 ? 'las 6 rutas están en el paquete y existen' : problems.join('; ')
  );

  record(
    pkg.engines !== undefined && pkg.engines.node === ENGINES_NODE,
    '(3) engines.node',
    `engines.node=${JSON.stringify(pkg.engines && pkg.engines.node)} (esperado ${ENGINES_NODE})`
  );

  record(pkg.sideEffects === false, '(3) sideEffects', `sideEffects=${JSON.stringify(pkg.sideEffects)} (esperado false)`);

  const deps = pkg.dependencies;
  const depsOk = deps === undefined || (deps !== null && typeof deps === 'object' && Object.keys(deps).length === 0);
  record(
    depsOk,
    '(3) dependencies vacío',
    deps === undefined ? 'ausente' : depsOk ? 'vacío' : `tiene ${Object.keys(deps).join(', ')}`
  );

  // (4) Tamaños.
  record(
    ctx.compressed < MAX_COMPRESSED,
    '(4) tamaño comprimido',
    `${ctx.compressed} bytes (límite < ${MAX_COMPRESSED})`
  );
  record(
    ctx.unpacked < MAX_UNPACKED,
    '(4) tamaño descomprimido',
    `${ctx.unpacked} bytes (límite < ${MAX_UNPACKED})`
  );

  // (5) Formatos: CJS sin import/export al inicio; ESM con export.
  let cjsOk = false;
  let cjsDetail;
  try {
    const first = codeOnly(fs.readFileSync(path.join(ctx.pkgRoot, 'dist/index.js'), 'utf8')).trimStart();
    cjsOk = !/^(import|export)\b/.test(first);
    cjsDetail = cjsOk ? 'no empieza por import ni export' : `empieza por: ${first.slice(0, 30)}`;
  } catch (err) {
    cjsDetail = `no se puede leer: ${err.message}`;
  }
  record(cjsOk, '(5) dist/index.js es CJS', cjsDetail);

  let esmOk = false;
  let esmDetail;
  try {
    esmOk = /^export\s/m.test(codeOnly(fs.readFileSync(path.join(ctx.pkgRoot, 'dist/index.mjs'), 'utf8')));
    esmDetail = esmOk ? 'tiene export de nivel superior' : 'no tiene export de nivel superior';
  } catch (err) {
    esmDetail = `no se puede leer: ${err.message}`;
  }
  record(esmOk, '(5) dist/index.mjs es ESM', esmDetail);
}

function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    if (err instanceof UsageError) {
      console.error(`Error de uso: ${err.message}\n${USAGE}`);
      return 2;
    }
    throw err;
  }
  if (opts.help) {
    console.log(USAGE);
    return 0;
  }

  let failed = 0;
  let total = 0;
  const record = (ok, label, detail) => {
    total++;
    if (!ok) failed++;
    console.log(`${ok ? 'OK   ' : 'FALLO'} ${label}: ${detail}`);
  };

  let tmp = null;
  try {
    if (opts.tarball !== null) {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'pack-audit-'));
    }
    const root = opts.cwd !== null ? path.resolve(opts.cwd) : DEFAULT_ROOT;
    const ctx = opts.tarball !== null ? fromTarball(path.resolve(opts.tarball), tmp) : fromRepo(root);
    console.log(`Auditando ${ctx.origen}`);
    runChecks(ctx, record);
  } catch (err) {
    record(false, 'ejecución', err.message.split('\n')[0]);
  } finally {
    if (tmp !== null) fs.rmSync(tmp, { recursive: true, force: true });
  }

  const passed = total - failed;
  console.log(`Resumen: ${passed} de ${total} comprobaciones OK`);
  if (failed === 0 && total > 0) {
    console.log('RESULTADO: PASA');
    return 0;
  }
  console.log('RESULTADO: FALLA');
  return 1;
}

process.exitCode = main();
