import { describe, it, expect, beforeAll } from "vitest";
import { seedRandom } from "./helpers/seed";

// Test de documentación (T015, feature 007).
// 1. Todo export de src/index.ts aparece como palabra completa en README.md.
// 2. README.md es exactamente lo que genera specs/007-docs/tools/assemble-readme.mjs a partir de las 6 partes.
// Lee texto con node:fs; el proyecto no declara @types/node, así que node:fs se carga con un tipo mínimo propio
// (mismo patrón que tests/coverage.modules.test.ts). No importa src/ ni modifica nada.

interface FsLike {
  existsSync(p: URL): boolean;
  readFileSync(p: URL, enc: "utf8"): string;
}
let fs: FsLike;

const INDEX = new URL("../src/index.ts", import.meta.url);
const README = new URL("../README.md", import.meta.url);
const PARTS_DIR = new URL("../specs/007-docs/readme-parts/", import.meta.url);
// Mismas partes y mismo orden que EXPECTED en specs/007-docs/tools/assemble-readme.mjs.
const PARTES = [
  "01-intro-tablas",
  "02-uso-basico",
  "03-uso-modelos",
  "04-uso-secuencias",
  "05-uso-herramientas",
  "06-cierre",
];
const MINIMO_EXPORTS = 100;
const MSG_ENSAMBLADO =
  "README.md was edited by hand or the parts changed: run node specs/007-docs/tools/assemble-readme.mjs";

/** Quita comentarios de bloque y de línea. Las rutas de módulo ("./Foo") no contienen "//", así que es seguro. */
function sinComentarios(texto: string): string {
  return texto.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, "");
}

/**
 * Nombres exportados por `export { … } from "…"` y `export type { … } from "…"`, en orden de aparición.
 * Multilínea, con comentarios dentro de las llaves y con `X as Y` (cuenta `Y`).
 */
function extraerExports(texto: string): string[] {
  const re = /export\s+(?:type\s+)?\{([^}]*)\}\s*from\s*["'][^"']*["']/g;
  const limpio = sinComentarios(texto);
  const nombres: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(limpio)) !== null) {
    for (const parte of m[1].split(",")) {
      const item = parte.trim().replace(/^type\s+/, "");
      if (item === "") continue;
      const partes = item.split(/\s+as\s+/);
      nombres.push(partes[partes.length - 1].trim());
    }
  }
  return nombres;
}

/** Nombres que aparecen más de una vez en la lista. */
function duplicados(nombres: string[]): string[] {
  const vistos = new Set<string>();
  const repetidos = new Set<string>();
  for (const n of nombres) {
    if (vistos.has(n)) repetidos.add(n);
    vistos.add(n);
  }
  return [...repetidos];
}

/** `nombre` como palabra completa: `Network` no cuenta dentro de `NetworkN`; sí entre backticks, seguido de `(`, `.` o `<`. */
function aparecePalabraCompleta(nombre: string, texto: string): boolean {
  const escapado = nombre.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`\\b${escapado}\\b`).test(texto);
}

/** Nombres de `nombres` que no aparecen como palabra completa en `texto`. */
function faltantes(nombres: string[], texto: string): string[] {
  return nombres.filter((n) => !aparecePalabraCompleta(n, texto));
}

/** Misma concatenación que `assemble` en assemble-readme.mjs: cada parte termina en salto de línea y se unen sin separador. */
function ensamblar(partes: string[]): string {
  return partes.map((t) => (t.endsWith("\n") ? t : `${t}\n`)).join("");
}

describe("exports de src/index.ts frente a README.md (T015)", () => {
  let indexTexto = "";
  let readme = "";
  let partesTexto: string[] = [];
  let nombres: string[] = [];

  beforeAll(async () => {
    seedRandom(0x5eed);
    const mod = "node:fs";
    fs = (await import(/* @vite-ignore */ mod)) as FsLike;
    indexTexto = fs.readFileSync(INDEX, "utf8");
    readme = fs.readFileSync(README, "utf8");
    partesTexto = PARTES.map((nombre) => {
      const ruta = new URL(`${nombre}.md`, PARTS_DIR);
      expect(fs.existsSync(ruta), `falta la parte ${nombre}.md en specs/007-docs/readme-parts/`).toBe(true);
      return fs.readFileSync(ruta, "utf8");
    });
    nombres = extraerExports(indexTexto);
  });

  it("el extractor encuentra al menos 100 exports en src/index.ts y ninguno está duplicado", () => {
    expect(nombres.length, `el extractor solo encontró ${nombres.length} exports; se esperan al menos ${MINIMO_EXPORTS}`).toBeGreaterThanOrEqual(MINIMO_EXPORTS);
    expect(duplicados(nombres), "exports duplicados en src/index.ts").toEqual([]);
  });

  it("cada export de src/index.ts aparece como palabra completa en README.md", () => {
    const faltan = faltantes(nombres, readme);
    expect(
      faltan,
      `README.md no menciona como palabra completa estos exports: ${faltan.join(", ")}. ` +
        "Arreglo: añádelos a la parte correspondiente de specs/007-docs/readme-parts/ y regenera con: " +
        "node specs/007-docs/tools/assemble-readme.mjs",
    ).toEqual([]);
  });

  it("README.md es exactamente la concatenación de las 6 partes", () => {
    expect(readme === ensamblar(partesTexto), MSG_ENSAMBLADO).toBe(true);
  });

  it("las 6 partes están contenidas literalmente en README.md, en orden", () => {
    let desde = 0;
    for (let i = 0; i < partesTexto.length; i++) {
      const pos = readme.indexOf(partesTexto[i], desde);
      expect(pos, `${MSG_ENSAMBLADO} (parte ${PARTES[i]}.md no aparece después de la anterior)`).toBeGreaterThanOrEqual(0);
      desde = pos + partesTexto[i].length;
    }
  });

  describe("lógica pura con casos sintéticos", () => {
    it("extrae exports multilínea", () => {
      const t = 'export {\n  Neuron,\n  NeuronN\n} from "./x";\n';
      expect(extraerExports(t)).toEqual(["Neuron", "NeuronN"]);
    });

    it("extrae exports de tipos (export type), también multilínea", () => {
      const t = 'export type {\n  TipoA,\n  TipoB\n} from "./t";\n';
      expect(extraerExports(t)).toEqual(["TipoA", "TipoB"]);
    });

    it("cuenta el alias Y de `X as Y`, en exports de valor y de tipo", () => {
      const t = 'export { Foo as Bar, Baz } from "./f";\nexport type { Qux as Quux } from "./q";\n';
      expect(extraerExports(t)).toEqual(["Bar", "Baz", "Quux"]);
    });

    it("ignora comentarios dentro de las llaves", () => {
      const t = 'export {\n  // primero\n  Uno, /* dos */ Dos,\n  /* tres\n  */ Tres // cuatro\n} from "./c";\n';
      expect(extraerExports(t)).toEqual(["Uno", "Dos", "Tres"]);
    });

    it("no toma como export líneas que no son `export { … } from`", () => {
      const t = 'const x = { A };\nexport const z = 1;\nexport function f() {}\n';
      expect(extraerExports(t)).toEqual([]);
    });

    it("detecta duplicados", () => {
      expect(duplicados(["A", "B", "A"])).toEqual(["A"]);
      expect(duplicados(["A", "B"])).toEqual([]);
    });

    it("no confunde un nombre con otro que lo contiene como prefijo", () => {
      expect(aparecePalabraCompleta("Network", "Usa NetworkN aquí.")).toBe(false);
      expect(aparecePalabraCompleta("NetworkN", "Usa NetworkN aquí.")).toBe(true);
      expect(aparecePalabraCompleta("Network", "xNetwork")).toBe(false);
    });

    it("reconoce el nombre entre backticks o seguido de ( . <", () => {
      expect(aparecePalabraCompleta("Network", "usa `Network` aquí")).toBe(true);
      expect(aparecePalabraCompleta("Network", "new Network(2)")).toBe(true);
      expect(aparecePalabraCompleta("Network", "Network.train")).toBe(true);
      expect(aparecePalabraCompleta("Network", "Network<T>")).toBe(true);
    });

    it("lista como faltante un nombre ausente y no uno presente", () => {
      expect(faltantes(["Foo", "Bar"], "Bar")).toEqual(["Foo"]);
    });

    it("la concatenación de partes añade el salto de línea que falte y no duplica el existente", () => {
      expect(ensamblar(["a", "b\n", "c"])).toBe("a\nb\nc\n");
    });
  });
});
