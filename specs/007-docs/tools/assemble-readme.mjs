#!/usr/bin/env node
// Ensambla README.md a partir de las partes de specs/007-docs/readme-parts (T002, feature 007).
//
// Uso: node specs/007-docs/tools/assemble-readme.mjs [--check] [--parts <dir>] [--out <archivo>]
//
// Concatena en orden numérico las 6 partes esperadas (NN-*.md) en el archivo de salida.
// Determinista: misma entrada, misma salida byte a byte.
// Sale con 1 si falta una parte, hay huecos o repeticiones en la numeración, hay partes
// no esperadas, o el directorio no existe.
// Con --check no escribe: sale 1 si el archivo de salida difiere de lo que generaría.
// Solo usa builtins de Node (sin dependencias).

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const DEFAULT_PARTS = path.join(REPO, 'specs/007-docs/readme-parts');
const DEFAULT_OUT = path.join(REPO, 'README.md');
const EXPECTED = [
  '01-intro-tablas',
  '02-uso-basico',
  '03-uso-modelos',
  '04-uso-secuencias',
  '05-uso-herramientas',
  '06-cierre',
];
const USAGE = 'Uso: node specs/007-docs/tools/assemble-readme.mjs [--check] [--parts <dir>] [--out <archivo>]';

class AssembleError extends Error {}

function parseArgs(argv) {
  const opts = {check: false, parts: DEFAULT_PARTS, out: DEFAULT_OUT};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--check') {
      opts.check = true;
    } else if (a === '--parts' || a === '--out') {
      const v = argv[++i];
      if (v === undefined || v.startsWith('--')) throw new Error(`${a} requiere un valor`);
      opts[a.slice(2)] = path.resolve(process.cwd(), v);
    } else {
      throw new Error(`opción desconocida: ${a}`);
    }
  }
  return opts;
}

// Devuelve los nombres (sin .md) de las partes en orden, validando numeración y nombres.
function listParts(dir) {
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    throw new AssembleError(`no existe el directorio de partes: ${dir}`);
  }
  const files = fs.readdirSync(dir).filter((f) => /^\d{2}-.+\.md$/.test(f)).sort();
  const byNum = new Map();
  for (const f of files) {
    const n = Number(f.slice(0, 2));
    if (byNum.has(n)) throw new AssembleError(`número de parte repetido: ${String(n).padStart(2, '0')} (${byNum.get(n)} y ${f})`);
    byNum.set(n, f);
  }
  for (let n = 1; n <= EXPECTED.length; n++) {
    if (!byNum.has(n)) throw new AssembleError(`hueco en la numeración: falta la parte ${String(n).padStart(2, '0')} (${EXPECTED[n - 1]}.md)`);
  }
  for (const [n, f] of byNum) {
    if (n > EXPECTED.length) throw new AssembleError(`parte no esperada: ${f} (solo ${EXPECTED.length} partes)`);
  }
  return EXPECTED.map((name, i) => {
    const f = byNum.get(i + 1);
    if (f !== `${name}.md`) {
      throw new AssembleError(`falta la parte ${name}.md (en su lugar hay ${f})`);
    }
    return f;
  });
}

function assemble(dir) {
  const files = listParts(dir);
  const chunks = files.map((f) => {
    const text = fs.readFileSync(path.join(dir, f), 'utf8');
    return text.endsWith('\n') ? text : `${text}\n`;
  });
  return Buffer.from(chunks.join(''), 'utf8');
}

function run(argv) {
  let opts;
  try {
    opts = parseArgs(argv);
  } catch (e) {
    console.error(`assemble-readme: ${e.message}\n${USAGE}`);
    return 1;
  }

  let content;
  try {
    content = assemble(opts.parts);
  } catch (e) {
    if (!(e instanceof AssembleError)) throw e;
    console.error(`assemble-readme: ${e.message}`);
    return 1;
  }

  if (opts.check) {
    const current = fs.existsSync(opts.out) ? fs.readFileSync(opts.out) : null;
    if (current === null || !current.equals(content)) {
      console.error(`assemble-readme: ${opts.out} difiere de las partes. Regenera con: node specs/007-docs/tools/assemble-readme.mjs`);
      return 1;
    }
    console.log(`assemble-readme: ${opts.out} coincide con las partes (${content.length} bytes).`);
    return 0;
  }

  fs.writeFileSync(opts.out, content);
  console.log(`assemble-readme: escrito ${opts.out} desde ${EXPECTED.length} partes (${content.length} bytes).`);
  return 0;
}

process.exitCode = run(process.argv.slice(2));
