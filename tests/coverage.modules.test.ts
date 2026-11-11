import { describe, it, expect, beforeAll } from "vitest";

// Meta-test de cobertura por módulo (FR-011, T044).
// Lee los nombres de src/*.ts y de tests/ (nivel superior, oracle/ y api/); no importa ni modifica nada.
// El proyecto no declara @types/node: node:fs se carga con un tipo mínimo propio (como tests/oracle/coverage.test.ts).

interface FsLike {
  existsSync(p: URL): boolean;
  readdirSync(p: URL): string[];
}
let fs: FsLike;

const SRC = new URL("../src/", import.meta.url);
const TESTS = new URL("./", import.meta.url);
// Directorios donde se busca un archivo de test con el nombre del módulo (no recursivo).
const TEST_DIRS = ["", "oracle/", "api/"];

/*
 * Reglas de coincidencia por nombre (decididas aquí):
 * 1. DISTINGUE MAYÚSCULAS. `activations` no se cubre con `Activations.test.ts`. Así el resultado es el mismo
 *    en macOS (sistema de archivos sin distinción) y en Linux (CI, con distinción); una comparación laxa daría
 *    verde en local y rojo en CI, o al revés. Por eso `activations` y `optimizers` van en EXCEPCIONES.
 * 2. LÍMITE ESTRICTO. Tras el nombre del módulo debe venir "." o el fin del nombre base (sin ".test.ts").
 *    `Network.test.ts` y `Network.oracle.test.ts` cubren `Network`; `NetworkN.test.ts` no.
 * 3. Solo cuenta el nombre del archivo. Un test que importe un módulo no lo cubre por nombre.
 */

/** Excepción: módulo sin archivo de test con su nombre, cubierto por `testFile` (relativo a tests/). */
interface Excepcion {
  testFile: string;
  motivo: string;
}

/** Excepciones verificadas en el propio test: cada una debe seguir siendo necesaria, existir y nombrar un módulo de src/. */
const EXCEPCIONES: Record<string, Excepcion> = {
  activations: {
    testFile: "Activations.test.ts",
    motivo: "El módulo es activations.ts y su test se llama Activations.test.ts; la comparación distingue mayúsculas, así que no coinciden por nombre.",
  },
  optimizers: {
    testFile: "Optimizers.test.ts",
    motivo: "El módulo es optimizers.ts y su test se llama Optimizers.test.ts; la comparación distingue mayúsculas, así que no coinciden por nombre.",
  },
};

/** Nombre base de un archivo de test, sin directorio ni ".test.ts": "oracle/Network.oracle.test.ts" -> "Network.oracle". */
function baseSinTest(rel: string): string {
  const nombre = rel.slice(rel.lastIndexOf("/") + 1);
  return nombre.replace(/\.test\.ts$/, "");
}

/** Coincidencia estricta en el límite (regla 2). Sensible a mayúsculas salvo que `ignorarMayusculas` sea true. */
function cubre(stem: string, mod: string, ignorarMayusculas = false): boolean {
  const s = ignorarMayusculas ? stem.toLowerCase() : stem;
  const m = ignorarMayusculas ? mod.toLowerCase() : mod;
  return s === m || s.startsWith(m + ".");
}

/** Lógica pura: módulos sin test (ni por nombre ni por excepción) y problemas de las excepciones. */
function analizar(
  modulos: string[],
  rutasTest: string[],
  excepciones: Record<string, Excepcion>,
  existe: (rel: string) => boolean,
): { sinTest: string[]; problemas: string[] } {
  const stems = rutasTest.map(baseSinTest);
  const tieneTest = (m: string) => stems.some((s) => cubre(s, m));
  const sinTest = modulos.filter((m) => !tieneTest(m) && !(m in excepciones)).sort();

  const problemas: string[] = [];
  for (const [mod, ex] of Object.entries(excepciones)) {
    if (!modulos.includes(mod)) problemas.push(`${mod}: no existe en src/`);
    else if (tieneTest(mod)) problemas.push(`${mod}: excepción innecesaria, ya tiene test con su nombre`);
    if (!existe(ex.testFile)) problemas.push(`${mod}: no existe tests/${ex.testFile}`);
    else if (!cubre(baseSinTest(ex.testFile), mod, true)) problemas.push(`${mod}: tests/${ex.testFile} no nombra a ese módulo`);
    if (ex.motivo.trim().length < 30) problemas.push(`${mod}: motivo demasiado corto`);
  }
  return { sinTest, problemas };
}

function listarModulos(): string[] {
  return fs
    .readdirSync(SRC)
    .filter((f: string) => f.endsWith(".ts") && f !== "index.ts")
    .map((f: string) => f.slice(0, -3))
    .sort();
}

function listarTests(): string[] {
  const out: string[] = [];
  for (const dir of TEST_DIRS) {
    for (const f of fs.readdirSync(new URL(dir, TESTS))) {
      if (f.endsWith(".test.ts")) out.push(dir + f);
    }
  }
  return out.sort();
}

describe("meta-test de cobertura por módulo (FR-011)", () => {
  let modulos: string[] = [];
  let rutasTest: string[] = [];
  const existe = (rel: string) => fs.existsSync(new URL(rel, TESTS));

  beforeAll(async () => {
    const mod = "node:fs";
    fs = (await import(/* @vite-ignore */ mod)) as FsLike;
    modulos = listarModulos();
    rutasTest = listarTests();
  });

  it("lee un conjunto no trivial de módulos y de archivos de test", () => {
    expect(modulos.length).toBeGreaterThan(40);
    expect(modulos).toContain("NetworkN");
    expect(rutasTest.length).toBeGreaterThan(40);
  });

  it("aplica la regla de nombre estricta (límite y mayúsculas)", () => {
    expect(cubre("Network", "Network")).toBe(true);
    expect(cubre("Network.oracle", "Network")).toBe(true);
    expect(cubre("Network.validation", "Network")).toBe(true);
    expect(cubre("NetworkN", "Network")).toBe(false);
    expect(cubre("LayerNorm", "Layer")).toBe(false);
    expect(cubre("Activations", "activations")).toBe(false);
    expect(cubre("Activations", "activations", true)).toBe(true);
  });

  it("todo módulo de src/ tiene un archivo de test con su nombre o una excepción", () => {
    const { sinTest } = analizar(modulos, rutasTest, EXCEPCIONES, existe);
    expect(sinTest, `módulos de src/ sin ningún archivo de test: ${sinTest.join(", ")}`).toEqual([]);
  });

  it("cada excepción sigue siendo necesaria, existe y nombra un módulo real", () => {
    const { problemas } = analizar(modulos, rutasTest, EXCEPCIONES, existe);
    expect(problemas, `excepciones inválidas: ${problemas.join(" | ")}`).toEqual([]);
  });

  it("la lógica falla cuando debe: sin test, excepción obsoleta, módulo inexistente y archivo ausente", () => {
    const motivo = "motivo de prueba suficientemente largo para la comprobación";

    // Límite estricto: NetworkN no cubre Network.
    const frontera = analizar(["Network", "NetworkN"], ["NetworkN.test.ts"], {}, () => true);
    expect(frontera.sinTest).toEqual(["Network"]);

    // Foo ya tiene test (excepción obsoleta), Baz no existe en src/, Bar apunta a un archivo ausente.
    const r = analizar(
      ["Foo", "Bar"],
      ["Foo.test.ts"],
      {
        Foo: { testFile: "Foo.test.ts", motivo },
        Baz: { testFile: "Baz.test.ts", motivo },
        Bar: { testFile: "Nope.test.ts", motivo },
      },
      (rel) => rel === "Foo.test.ts" || rel === "Baz.test.ts",
    );
    expect(r.sinTest).toEqual([]);
    expect(r.problemas.map((p) => p.split(":")[0])).toEqual(["Foo", "Baz", "Bar"]);
  });
});
