# Tasks: release-safety

**Input**: `specs/001-release-safety/spec.md`, `plan.md`
**Formato**: `- [ ] T0NN [P?] [H|S] [files: ...]` + línea `Gate:`. `[P]` solo si los archivos no se solapan con otras tareas `[P]` del mismo lote. `package.json` y `package-lock.json` son exclusivos del orquestador (T006). Ningún agente ejecuta `npm publish` ni sube a `master`.

## Lote 1 (en paralelo: archivos disjuntos)

- [x] T001 [P] [H] [files: tsconfig.test.json] Crear `tsconfig.test.json` que extienda `./tsconfig.json` con `noEmit: true`, `declaration: false`, `declarationMap: false`, `sourceMap: false`, `rootDir: "."` e `include: ["src", "tests"]`. No modificar `tsconfig.json`.
      Gate: `npx tsc --noEmit -p tsconfig.test.json` (código 0)

- [x] T002 [P] [H] [files: vitest.config.ts, vitest.config.mts] `git mv vitest.config.ts vitest.config.mts` sin cambiar el contenido.
      Gate: `test -f vitest.config.mts && ! test -f vitest.config.ts && ! (npx vitest run 2>&1 | grep -q "configLoader") && npx vitest run 2>&1 | grep -q "Tests  .*passed"`

- [x] T003 [P] [H] [files: LICENSE] Crear `LICENSE` con el texto estándar MIT. Titular y año según Q4 (por defecto `dniskav`, año del primer commit: `git log --reverse --format=%ad --date=format:%Y | head -1`).
      Gate: `head -1 LICENSE | grep -q "MIT License" && grep -q "Permission is hereby granted, free of charge" LICENSE`

- [x] T004 [P] [H] [files: bun.lock, dniskav-neuron-0.2.1.tgz, .gitignore] `git rm bun.lock dniskav-neuron-0.2.1.tgz` y añadir `bun.lock` y `*.tgz` a `.gitignore`.
      Gate: `test -z "$(git ls-files bun.lock '*.tgz')" && grep -qx 'bun.lock' .gitignore && grep -qx '\*.tgz' .gitignore`

- [x] T005 [P] [H] [files: NEURON_UPGRADE_PLAN.md, NEURON_v0.2.5_PLAN.md, REFACTOR_SUGGESTIONS_CLAUDE_SONNET_4_5.md, REFACTOR_SUGGESTIONS_minimax-m2.7_cloud.md, kimi-k2.5.md, refactor-suggestions-qwen3.5.md, docs/archive/README.md] `mkdir -p docs/archive`, `git mv` de los seis documentos y, si Q5 se confirma, `docs/archive/README.md` de una línea (informes históricos de IA, no vigentes).
      Gate: `test "$(ls docs/archive/*.md | grep -v README | wc -l | tr -d ' ')" = 6 && test -z "$(git ls-files 'NEURON_*.md' 'REFACTOR_*.md' kimi-k2.5.md refactor-suggestions-qwen3.5.md)"`

## Lote 2 (orquestador, depende de T001)

- [x] T006 [H] [files: package.json, package-lock.json] TAREA DEL ORQUESTADOR. En `package.json`: reordenar `exports["."]` a `types`, `import`, `require`; añadir `repository`, `homepage`, `bugs`, `sideEffects: false`, `engines.node` (Q2); script `"typecheck": "tsc --noEmit -p tsconfig.test.json"`. Ejecutar `npm i -D @vitest/coverage-v8@^4.1.12`. No cambiar `version` (0.3.2) ni añadir `dependencies`.
      Gate: `node -e "const p=require('./package.json');if(Object.keys(p.exports['.'])[0]!=='types'||p.version!=='0.3.2'||p.sideEffects!==false||!p.engines||!p.repository||!p.homepage||!p.bugs||!p.scripts.typecheck||p.dependencies||!p.devDependencies['@vitest/coverage-v8'])process.exit(1)" && npm run typecheck && ! (npm run build 2>&1 | grep -qi "never be used") && npm run coverage >/dev/null && npm pack --dry-run 2>&1 | grep -q LICENSE && npm audit --omit=dev`

## Lote 3 (depende de T006; archivos disjuntos)

- [x] T007 [P] [H] [files: .github/workflows/ci.yml] Crear `ci.yml` (name `CI`, job `ci`, `ubuntu-latest`, `permissions: contents: read`, triggers `pull_request` y `push`, Node 20 con `cache: npm`): `npm ci`, `npm run typecheck`, `npm run build`, `npm test`.
      Gate: `ruby -ryaml -e 'y=YAML.load_file(".github/workflows/ci.yml"); t=y[true]||y["on"]; abort unless t.key?("pull_request") && t.key?("push"); s=y["jobs"]["ci"]["steps"].map{|x|x["run"]}.compact; abort unless s==["npm ci","npm run typecheck","npm run build","npm test"]'`

- [x] T008 [P] [H] [files: .github/workflows/publish.yml] Añadir `npm run typecheck` y `npm test` antes de la comprobación de versión y de `npm publish`; mantener el trigger (push a `master`), `npm ci`, build y `if: steps.version_check.outputs.skip == 'false'` en publicar. Tratar Q3 solo si el usuario lo confirma. No cambiar versión.
      Gate: `ruby -ryaml -e 'y=YAML.load_file(".github/workflows/publish.yml"); s=y["jobs"]["publish"]["steps"]; i=->(r){s.index{|x|x["run"].to_s.include?(r)}}; abort unless i["npm test"] && i["npm run typecheck"] && i["npm test"]<i["npm publish"] && i["npm run typecheck"]<i["npm publish"]; abort unless s.last["if"].to_s.include?("skip == '"'"'false'"'"'")'`

## Lote 4 (verificación y revisión)

- [x] T009 [S] [files: .github/workflows/ci.yml, .github/workflows/publish.yml] Revisión de Sonnet (sin editar): orden de pasos, permisos mínimos, que publicar es inalcanzable si falla typecheck/test/build, nombre del job para la protección (Q6), y que ningún paso ejecuta `npm publish` fuera de `publish.yml`.
      Gate: informe del revisor sin hallazgos bloqueantes; `grep -rn "npm publish" .github/workflows | grep -v publish.yml` sin resultados

- [x] T010 [H] [files: ] Verificador (sin editar): definición de terminado completa tras T001 a T008.
      Gate: `npx tsc --noEmit -p . && npm run typecheck && npm run build && npx vitest run && test "$(node -p "require('./package.json').version")" = 0.3.2`

## Lote 5 (acción del usuario, no del enjambre)

- [ ] T011 [S] [files: ] TAREA DEL USUARIO/ORQUESTADOR con aprobación: solo después de que `ci.yml` esté en `master` y haya corrido en verde, activar la protección de `master` exigiendo PR y el check `ci`. Requiere push de `fix/v0.4` y merge a `master` (ver Q1), que ningún agente hace.
      Gate: `gh api repos/dniskav/neuron/branches/master/protection --jq '.required_status_checks.contexts, .required_pull_request_reviews != null'` muestra `ci` y `true`

## Dependencias

T001 → T006 → T007, T008 → T009 → T010 → T011. T002 a T005 no dependen de nadie, pero T010 las exige. T006 también requiere T003 para que `npm pack` liste `LICENSE`.
