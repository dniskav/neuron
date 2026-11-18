import { describe, it, expect } from "vitest";

// Contenido del paquete publicado (T030): package.json, build minificado de la entrada de inferencia y mapas.
// Los casos sobre dist/ necesitan un `npm run build` previo; si dist/infer.min.mjs no existe, se omiten
// con skipIf (un build sin este cambio no lo genera). `npm run build` es el primer paso de la puerta de T030.
// No importa nada de src/ ni ejecuta npm pack: solo lee archivos. El proyecto no declara @types/node,
// así que node:fs se carga con un tipo mínimo propio (como tests/coverage.modules.test.ts).

interface FsLike {
  existsSync(p: URL): boolean;
  readFileSync(p: URL, enc: "utf8"): string;
  readdirSync(p: URL): string[];
  statSync(p: URL): { size: number };
}
// Carga de node:fs con import dinámico (como en coverage.modules.test.ts). Va al nivel superior porque
// skipIf necesita saber ya si existe el build.
const modFs = "node:fs";
const fs = (await import(/* @vite-ignore */ modFs)) as FsLike;

const ROOT = new URL("../", import.meta.url);
const DIST = (f: string) => new URL("dist/" + f, ROOT);
const LIMITE_MIN_BYTES = 12288;

interface PackageJson {
  files: string[];
  scripts: Record<string, string>;
  exports: Record<string, { types?: string; import?: string; require?: string }>;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
}

interface SourceMap {
  version: number;
  sources: string[];
  sourcesContent?: (string | null)[];
}

function leerJson<T>(p: URL): T {
  return JSON.parse(fs.readFileSync(p, "utf8")) as T;
}

// Un import o export-from estático en el bundle significaría que no es autocontenido.
function tieneImportEstatico(codigo: string): boolean {
  return /(^|[;}\s])import\s*[("'{*\w]|\bfrom\s*["']/.test(codigo);
}

const minEsCompilado = fs.existsSync(DIST("infer.min.mjs"));
const pkg = leerJson<PackageJson>(new URL("package.json", ROOT));

describe("contenido del paquete (T030)", () => {
  it("package.json publica src/ junto con dist/", () => {
    expect(pkg.files).toContain("dist");
    expect(pkg.files).toContain("src");
  });

  // ./infer/min es solo ESM (sin entrada require): el bundle minificado no tiene build CJS. Aceptado por el
  // orquestador; `require('@dniskav/neuron/infer/min')` falla con ERR_PACKAGE_PATH_NOT_EXPORTED.
  it("package.json exporta ./infer/min (solo ESM) hacia dist/infer.min.mjs", () => {
    const entrada = pkg.exports["./infer/min"];
    expect(entrada, "falta exports['./infer/min']").toBeDefined();
    expect(entrada.import).toBe("./dist/infer.min.mjs");
    expect(entrada.types).toBe("./dist/infer.d.ts");
    expect(entrada.require, "./infer/min no tiene build CJS").toBeUndefined();
  });

  it("esbuild está declarado en devDependencies (versión exacta) y no en dependencies", () => {
    expect(pkg.devDependencies?.esbuild).toBe("0.27.2");
    expect(pkg.dependencies ?? {}).not.toHaveProperty("esbuild");
  });

  it("el script build usa tsup.config.ts (con sourcemap y sin sourcesContent) y esbuild para el minificado", () => {
    const [tsup, esbuild, ...resto] = pkg.scripts.build.split("&&").map((s) => s.trim());
    expect(resto, "el build tiene pasos de más").toEqual([]);
    expect(tsup).toBe("tsup");
    const config = fs.readFileSync(new URL("tsup.config.ts", ROOT), "utf8");
    expect(config).toContain("sourcemap: true");
    expect(config).toContain("options.sourcesContent = false");
    expect(esbuild).toMatch(/^esbuild src\/infer\.ts /);
    expect(esbuild).toContain("--minify");
    expect(esbuild).toContain("--sourcemap");
    expect(esbuild).toContain("--sources-content=false");
    expect(esbuild).toContain("--outfile=dist/infer.min.mjs");
  });

  it.skipIf(!minEsCompilado)("dist/infer.min.mjs existe y mide como mucho 12 288 bytes", () => {
    const size = fs.statSync(DIST("infer.min.mjs")).size;
    expect(size, `dist/infer.min.mjs mide ${size} bytes`).toBeLessThanOrEqual(LIMITE_MIN_BYTES);
    expect(size).toBeGreaterThan(0);
  });

  it.skipIf(!minEsCompilado)("dist/infer.min.mjs es autocontenido (sin imports estáticos) y exporta loadForInference", () => {
    const codigo = fs.readFileSync(DIST("infer.min.mjs"), "utf8");
    expect(tieneImportEstatico(codigo)).toBe(false);
    expect(codigo).toMatch(/export\{[^}]*\bas loadForInference\b[^}]*\}/);
  });

  it.skipIf(!minEsCompilado)("hay sourcemaps para cada salida, sin sourcesContent, y apuntan a ../src/", () => {
    const salidas = ["index.js", "index.mjs", "infer.js", "infer.mjs", "infer.min.mjs"];
    const mapas = fs.readdirSync(new URL("dist/", ROOT)).filter((f) => f.endsWith(".map"));
    for (const s of salidas) {
      expect(mapas, `falta dist/${s}.map`).toContain(s + ".map");
    }
    for (const f of mapas) {
      const m = leerJson<SourceMap>(DIST(f));
      expect(m.sources.length, `${f} sin fuentes`).toBeGreaterThan(0);
      const conContenido = (m.sourcesContent ?? []).filter((c) => typeof c === "string" && c.length > 0);
      expect(conContenido.length, `${f} lleva sourcesContent (duplica src/)`).toBe(0);
      for (const src of m.sources) {
        expect(src.startsWith("../src/"), `${f} referencia ${src}`).toBe(true);
      }
    }
  });

  it.skipIf(!minEsCompilado)("dist/infer.min.mjs enlaza su sourcemap", () => {
    const codigo = fs.readFileSync(DIST("infer.min.mjs"), "utf8");
    expect(codigo).toContain("//# sourceMappingURL=infer.min.mjs.map");
  });
});
