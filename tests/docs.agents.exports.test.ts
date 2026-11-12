import { describe, it, expect, beforeAll } from "vitest";

// Meta-test de AGENTS.md frente a src/index.ts (FR-013, T016). Lee ambos archivos como texto; no importa ni modifica nada.
// Los nombres de un export se leen de `export { … } from` y `export type { … } from` (multilínea, con comentarios y `as`).
// El proyecto no declara @types/node: node:fs se carga con un tipo mínimo propio (como tests/coverage.modules.test.ts).

interface FsLike {
  existsSync(p: URL): boolean;
  readFileSync(p: URL, encoding: "utf8"): string;
}
let fs: FsLike;

const INDEX = new URL("../src/index.ts", import.meta.url);
const AGENTS = new URL("../AGENTS.md", import.meta.url);
const ROOT = new URL("../", import.meta.url);

/** Mínimo de exports que el extractor debe encontrar (el número exacto no se fija). */
const MIN_EXPORTS = 100;

/** Afirmaciones que ya fueron falsas y no deben volver (cadenas exactas, sin distinguir mayúsculas). */
const FRASES_FALSAS = [
  "Active repair",
  "All activations are sigmoid",
  "No convolutional layers",
  "No data loaders",
  "No loss functions",
];

/** Bloque de reexport: `export { … } from "…"` o `export type { … } from "…"`. El grupo 1 son las entradas. */
const BLOQUE_EXPORT = /export\s+(?:type\s+)?\{([^}]*)\}\s*from\s*["'][^"']+["']/g;

/** Cita de archivo `.ts` entre backticks: `Neuron.ts` o `tests/helpers/seed.ts`. El grupo 1 es la ruta. */
const CITA_ARCHIVO = /`([A-Za-z0-9_./-]+\.ts)`/g;

/** Quita comentarios `// …` y `/* … *\/` antes de extraer. index.ts no tiene cadenas que contengan `//`. */
function quitarComentarios(texto: string): string {
  return texto.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, "");
}

/** Nombres exportados (el nombre público: `Y` en `X as Y`), en el orden del archivo, con repeticiones. */
function extraerExports(texto: string): string[] {
  const nombres: string[] = [];
  for (const m of quitarComentarios(texto).matchAll(BLOQUE_EXPORT)) {
    for (const entrada of (m[1] ?? "").split(",")) {
      const sinTipo = entrada.trim().replace(/^type\s+/, "");
      if (sinTipo === "") continue;
      const partes = sinTipo.split(/\s+as\s+/);
      nombres.push((partes[partes.length - 1] ?? "").trim());
    }
  }
  return nombres;
}

/** Nombres que aparecen más de una vez. */
function duplicados(nombres: string[]): string[] {
  const vistos = new Set<string>();
  const repetidos = new Set<string>();
  for (const n of nombres) {
    if (vistos.has(n)) repetidos.add(n);
    vistos.add(n);
  }
  return [...repetidos].sort();
}

/** `nombre` aparece como palabra completa: no rodeado de letras, dígitos, `_` o `$`. Sensible a mayúsculas. */
function aparece(texto: string, nombre: string): boolean {
  // Escapa TODOS los metacaracteres de regex (incluida la barra invertida), no solo `$`.
  const escapado = nombre.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?<![\\w$])${escapado}(?![\\w$])`).test(texto);
}

/** Exports que no aparecen en el texto como palabra completa. */
function faltan(nombres: string[], texto: string): string[] {
  return nombres.filter((n) => !aparece(texto, n));
}

/** Frases de la lista que aparecen en el texto (sin distinguir mayúsculas). */
function frasesEncontradas(texto: string, frases: string[]): string[] {
  const minusculas = texto.toLowerCase();
  return frases.filter((f) => minusculas.includes(f.toLowerCase()));
}

/** Rutas únicas citadas con el patrón `Nombre.ts` entre backticks. */
function citasDeArchivos(texto: string): string[] {
  const citas = new Set<string>();
  for (const m of texto.matchAll(CITA_ARCHIVO)) citas.add(m[1] ?? "");
  return [...citas].sort();
}

/** Ruta desde la raíz del repo: sin `/` se busca en src/; con `/` se toma tal cual. */
function rutaDeCita(cita: string): string {
  return cita.includes("/") ? cita : `src/${cita}`;
}

/** Rutas resueltas de las citas que no existen. */
function archivosNoExistentes(citas: string[], existe: (rel: string) => boolean): string[] {
  return citas.map(rutaDeCita).filter((rel) => !existe(rel));
}

describe("AGENTS.md frente a src/index.ts (FR-013)", () => {
  let indexTxt = "";
  let agentsTxt = "";
  const existe = (rel: string) => fs.existsSync(new URL(rel, ROOT));

  beforeAll(async () => {
    const mod = "node:fs";
    fs = (await import(/* @vite-ignore */ mod)) as FsLike;
    indexTxt = fs.readFileSync(INDEX, "utf8");
    agentsTxt = fs.readFileSync(AGENTS, "utf8");
  });

  it("lee src/index.ts y encuentra al menos 100 exports, sin duplicados", () => {
    const nombres = extraerExports(indexTxt);
    expect(nombres.length, `solo se extrajeron ${nombres.length} exports de src/index.ts`).toBeGreaterThanOrEqual(MIN_EXPORTS);
    const rep = duplicados(nombres);
    expect(rep, `exports duplicados en src/index.ts: ${rep.join(", ")}`).toEqual([]);
  });

  it("todo export de src/index.ts aparece en AGENTS.md como palabra completa", () => {
    const nombres = extraerExports(indexTxt);
    const sinCubrir = faltan(nombres, agentsTxt);
    expect(
      sinCubrir,
      `exports que no aparecen en AGENTS.md: ${sinCubrir.join(", ")} — add them to the module map in AGENTS.md`,
    ).toEqual([]);
  });

  it("AGENTS.md no vuelve a contener las afirmaciones ya falsas", () => {
    const encontradas = frasesEncontradas(agentsTxt, FRASES_FALSAS);
    expect(encontradas, `AGENTS.md contiene afirmaciones ya falsas: ${encontradas.join(" | ")}`).toEqual([]);
  });

  it("cada archivo de src/ citado en AGENTS.md existe", () => {
    const citas = citasDeArchivos(agentsTxt);
    // Guarda contra un patrón de cita roto: AGENTS.md cita más de 40 archivos distintos.
    expect(citas.length).toBeGreaterThan(40);
    const noExisten = archivosNoExistentes(citas, existe);
    expect(noExisten, `archivos citados en AGENTS.md que no existen: ${noExisten.join(", ")}`).toEqual([]);
  });

  it("el extractor de exports reconoce multilínea, export type, alias y comentarios", () => {
    const sintetico = `
export { Alpha, Beta as Gamma } from "./a";
/* bloque comentado
   export { NoDebe } from "./x";
*/
export type {
  Tipo1, // comentario dentro de las llaves
  Tipo2 as Tipo3,
} from "./b";
export {
  Multi1,
  // Multi2 comentado
  Multi3,
} from "./c";
export { type Inline } from "./e";
export { Network, NetworkN } from "./d";
`;
    expect(extraerExports(sintetico)).toEqual([
      "Alpha",
      "Gamma",
      "Tipo1",
      "Tipo3",
      "Multi1",
      "Multi3",
      "Inline",
      "Network",
      "NetworkN",
    ]);
  });

  it("la búsqueda exige palabra completa, sensible a mayúsculas", () => {
    expect(aparece("Usa `Network` aquí", "Network")).toBe(true);
    expect(aparece("Usa `NetworkN` aquí", "Network")).toBe(false);
    expect(aparece("`Layer`.", "Layer")).toBe(true);
    expect(aparece("`Layer(`", "Layer")).toBe(true);
    expect(aparece("Tensor<Layer>", "Layer")).toBe(true);
    expect(aparece("`LSTMLayer`", "Layer")).toBe(false);
    expect(aparece("network", "Network")).toBe(false);
    expect(aparece("ausente por completo", "Network")).toBe(false);
  });

  it("detecta exports que faltan y repeticiones", () => {
    expect(faltan(["A", "B", "C"], "texto con `A` y `C`")).toEqual(["B"]);
    expect(duplicados(["A", "B", "A", "C", "A"])).toEqual(["A"]);
    expect(duplicados(["A", "B"])).toEqual([]);
  });

  it("detecta las frases falsas sin distinguir mayúsculas y nombra la que encuentra", () => {
    expect(frasesEncontradas("... active REPAIR ... nada más", FRASES_FALSAS)).toEqual(["Active repair"]);
    expect(frasesEncontradas("all activations are SIGMOID", FRASES_FALSAS)).toEqual(["All activations are sigmoid"]);
    expect(frasesEncontradas("Convolutions and data loaders exist.", FRASES_FALSAS)).toEqual([]);
  });

  it("resuelve citas: bare en src/, con ruta desde la raíz, y reporta las inexistentes", () => {
    const citas = citasDeArchivos("`Neuron.ts`, `src/Tape.ts`, `Neuron.ts`, `tests/helpers/seed.ts`, `no.tsx`, `x.mts`");
    expect(citas).toEqual(["Neuron.ts", "src/Tape.ts", "tests/helpers/seed.ts"]);
    expect(rutaDeCita("Neuron.ts")).toBe("src/Neuron.ts");
    expect(rutaDeCita("src/Tape.ts")).toBe("src/Tape.ts");
    const noExisten = archivosNoExistentes(["Neuron.ts", "NoExiste.ts", "tests/x.ts"], (rel) => rel === "src/Neuron.ts");
    expect(noExisten).toEqual(["src/NoExiste.ts", "tests/x.ts"]);
  });
});
