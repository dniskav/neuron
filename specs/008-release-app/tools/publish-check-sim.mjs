#!/usr/bin/env node
// Simulator for the "Check if version is already published" step of publish.yml (T007).
//
// Usage: node publish-check-sim.mjs <path to publish.yml>
//
// It does NOT reimplement the decision. It extracts the REAL `run:` block of the
// step from the YAML text, runs it with bash (`-eo pipefail`, as GitHub Actions
// does for `run:` on bash) in a temp directory that has a package.json with the
// case's local version and a FAKE `npm` first in PATH. The decision is read from
// the file pointed to by $GITHUB_OUTPUT. The fake npm never touches the network
// and the simulator asserts that `npm publish` was never invoked.
//
// Extraction (plain text, no YAML library): find the line `- name: <STEP_NAME>`
// (any indent), take the first following line that is `run: |` or `run: >` at
// the step's key indent (the indent of `name:`), and collect every following
// line that is blank or indented deeper than `run:`. Stop at the first
// non-blank line with indent <= the `run:` indent. The block is dedented by the
// indent of its first non-blank line. Only the literal block form `run: |` is
// supported; anything else (or a missing step) is an error, never a silent pass.
import { spawnSync } from 'node:child_process'
import { mkdtempSync, writeFileSync, readFileSync, chmodSync, mkdirSync, rmSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const STEP_NAME = 'Check if version is already published'

function die(msg) {
  console.error(`publish-check-sim: ${msg}`)
  process.exit(2)
}

export function extractRunBlock(yamlText, stepName = STEP_NAME) {
  const lines = yamlText.split('\n')
  const indentOf = (l) => l.length - l.trimStart().length
  const start = lines.findIndex((l) => /^\s*-\s+name:\s*/.test(l) && l.replace(/^\s*-\s+name:\s*/, '').trim() === stepName)
  if (start < 0) throw new Error(`step "${stepName}" not found`)
  // keys of the step are indented like `name:` (after the "- ")
  const keyIndent = indentOf(lines[start]) + 2
  let runIdx = -1
  for (let i = start + 1; i < lines.length; i++) {
    const l = lines[i]
    if (l.trim() === '') continue
    if (indentOf(l) < keyIndent) break // next step or end of the job
    if (indentOf(l) === keyIndent && /^run:\s*\|\s*$/.test(l.trim())) {
      runIdx = i
      break
    }
  }
  if (runIdx < 0) throw new Error(`step "${stepName}" has no "run: |" block`)
  const runIndent = indentOf(lines[runIdx])
  const body = []
  for (let i = runIdx + 1; i < lines.length; i++) {
    const l = lines[i]
    if (l.trim() !== '' && indentOf(l) <= runIndent) break
    body.push(l)
  }
  while (body.length && body[body.length - 1].trim() === '') body.pop()
  const first = body.find((l) => l.trim() !== '')
  if (!first) throw new Error('empty run block')
  const base = indentOf(first)
  return body.map((l) => (l.trim() === '' ? '' : l.slice(base))).join('\n') + '\n'
}

// Fake npm. Behaviour comes from env: FAKE_LATEST and FAKE_EXACT, each one of
//   ok:<version>  prints <version>, exit 0     404  E404 on stderr, exit 1
//   net           ECONNRESET, exit 1           401  E401 on stderr, exit 1
//   empty         prints nothing, exit 0       junk prints "<html>oops</html>", exit 0
// FAKE_EXACT may be `exact` (prints the requested version, exit 0).
// Every call is appended to $FAKE_LOG.
const FAKE_NPM = `#!/bin/sh
echo "$*" >> "$FAKE_LOG"
if [ "$1" != "view" ]; then
  echo "fake npm: unexpected command: $*" >&2
  exit 99
fi
case "$2" in
  @dniskav/neuron@*) MODE="$FAKE_EXACT"; REQ="\${2#@dniskav/neuron@}" ;;
  @dniskav/neuron)   MODE="$FAKE_LATEST"; REQ="" ;;
  *) echo "fake npm: unexpected package: $2" >&2; exit 99 ;;
esac
case "$MODE" in
  exact) echo "$REQ"; exit 0 ;;
  ok:*)  echo "\${MODE#ok:}"; exit 0 ;;
  404)   printf 'npm error code E404\\nnpm error 404 No match found for version\\n' >&2; exit 1 ;;
  net)   printf 'npm error code ECONNRESET\\nnpm error network aborted\\n' >&2; exit 1 ;;
  401)   printf 'npm error code E401\\nnpm error 401 Unauthorized\\n' >&2; exit 1 ;;
  empty) exit 0 ;;
  junk)  echo '<html>oops</html>'; exit 0 ;;
  *) echo "fake npm: bad mode $MODE" >&2; exit 99 ;;
esac
`

// expect.skip: 'true' | 'false' (decision written, exit 0) | null (must exit != 0 and write no skip=)
const CASES = [
  { id: 1, name: 'local 0.4.0 new (404), latest 0.3.2 -> publish', local: '0.4.0', exact: '404', latest: 'ok:0.3.2', skip: 'false' },
  { id: 2, name: 'local 0.3.2 already published -> skip', local: '0.3.2', exact: 'exact', latest: 'ok:0.3.2', skip: 'true' },
  { id: 3, name: 'local 0.3.1 published, latest 0.3.2 (lower than latest) -> skip', local: '0.3.1', exact: 'exact', latest: 'ok:0.3.2', skip: 'true' },
  { id: 4, name: 'package never published (404 for everything) -> publish', local: '0.1.0', exact: '404', latest: '404', skip: 'false' },
  { id: 5, name: 'ECONNRESET on exact view -> fails, no decision', local: '0.4.0', exact: 'net', latest: 'ok:0.3.2', skip: null },
  { id: 6, name: '401 on exact view -> fails, no decision', local: '0.4.0', exact: '401', latest: '401', skip: null },
  { id: 7, name: 'empty output with exit 0 -> fails closed', local: '0.4.0', exact: 'empty', latest: 'ok:0.3.2', skip: null },
  { id: 8, name: 'unexpected output with exit 0 -> fails closed', local: '0.4.0', exact: 'junk', latest: 'ok:0.3.2', skip: null },
  { id: 9, name: 'exact view 404 but reading latest fails (network) -> still publish', local: '0.4.0', exact: '404', latest: 'net', skip: 'false' },
]

function runCase(script, c, root) {
  const dir = mkdtempSync(join(root, `case${c.id}-`))
  const bin = join(dir, 'bin')
  mkdirSync(bin)
  writeFileSync(join(bin, 'npm'), FAKE_NPM)
  chmodSync(join(bin, 'npm'), 0o755)
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: '@dniskav/neuron', version: c.local }))
  const out = join(dir, 'github_output')
  const log = join(dir, 'npm_calls.log')
  writeFileSync(out, '')
  writeFileSync(log, '')
  const file = join(dir, 'step.sh')
  writeFileSync(file, script)
  const r = spawnSync('bash', ['--noprofile', '--norc', '-eo', 'pipefail', file], {
    cwd: dir,
    encoding: 'utf8',
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      GITHUB_OUTPUT: out,
      FAKE_LOG: log,
      FAKE_EXACT: c.exact,
      FAKE_LATEST: c.latest,
      TMPDIR: dir,
    },
  })
  const output = readFileSync(out, 'utf8')
  const calls = readFileSync(log, 'utf8').split('\n').filter(Boolean)
  const m = output.match(/^skip=(.*)$/m)
  const skip = m ? m[1] : null
  const problems = []
  if (calls.some((x) => /^publish\b/.test(x))) problems.push('npm publish was invoked')
  if (calls.some((x) => !/^view @dniskav\/neuron(@\S+)? version$/.test(x))) problems.push(`unexpected npm calls: ${calls.join(' | ')}`)
  if (c.skip === null) {
    if (r.status === 0) problems.push('expected non-zero exit, got 0')
    if (skip !== null) problems.push(`a decision was written (skip=${skip})`)
  } else {
    if (r.status !== 0) problems.push(`expected exit 0, got ${r.status}`)
    if (skip !== c.skip) problems.push(`expected skip=${c.skip}, got ${skip}`)
    if ((output.match(/^skip=/gm) || []).length !== 1) problems.push('skip= must be written exactly once')
  }
  return { problems, status: r.status, skip, stdout: r.stdout, stderr: r.stderr }
}

function main() {
  const path = process.argv[2]
  if (!path) die('usage: publish-check-sim.mjs <path to publish.yml>')
  if (!existsSync(path)) die(`file not found: ${path}`)
  let script
  try {
    script = extractRunBlock(readFileSync(path, 'utf8'))
  } catch (e) {
    die(e.message)
  }
  const root = mkdtempSync(join(tmpdir(), 'publish-sim-'))
  let failed = 0
  try {
    for (const c of CASES) {
      const res = runCase(script, c, root)
      const ok = res.problems.length === 0
      if (!ok) failed++
      const got = c.skip === null ? `exit=${res.status}` : `exit=${res.status} skip=${res.skip}`
      console.log(`${ok ? 'PASS' : 'FAIL'} case ${c.id}: ${c.name} [${got}]`)
      if (!ok) for (const p of res.problems) console.log(`       - ${p}`)
    }
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
  console.log(failed === 0 ? `all ${CASES.length} cases passed` : `${failed} of ${CASES.length} cases FAILED`)
  process.exit(failed === 0 ? 0 : 1)
}

main()
