import { describe, it, expect, beforeAll } from "vitest";
import { seedRandom } from "./helpers/seed";

// Enlaces de README.md a docs/ (0.5.1). La carpeta docs/ no va en el paquete npm, así que los enlaces
// deben ser absolutos a GitHub, y el archivo enlazado debe existir en el repositorio. Lee texto con
// node:fs; no importa src/ ni modifica nada.

interface FsLike {
  existsSync(p: URL): boolean;
  readFileSync(p: URL, enc: "utf8"): string;
  readdirSync(p: URL): string[];
}
let fs: FsLike;

const ROOT = new URL("../", import.meta.url);
const README = new URL("README.md", ROOT);
const PREFIJO = "https://github.com/dniskav/neuron/blob/master/docs/";

/** Destinos de los enlaces markdown `[texto](destino)` que mencionan docs/, con la ancla quitada. */
function destinosDocs(texto: string): string[] {
  const out: string[] = [];
  for (const m of texto.matchAll(/\]\(([^)\s]+)\)/g)) {
    const destino = m[1] ?? "";
    if (destino.includes("docs/")) out.push(destino);
  }
  return out;
}

/** Slug de GitHub para un encabezado: minúsculas, sin signos, espacios a guiones. */
function slug(encabezado: string): string {
  return encabezado.toLowerCase().replace(/[^\p{L}\p{N} _-]/gu, "").trim().replace(/ /g, "-");
}

/** Ruta del repositorio (`docs/<archivo>`) de una URL absoluta a docs/ en GitHub, sin ancla. */
function rutaDeUrl(url: string): string {
  return `docs/${url.slice(PREFIJO.length).split("#")[0] ?? ""}`;
}

describe("enlaces de README.md a docs/ (0.5.1)", () => {
  let readme = "";
  let destinos: string[] = [];

  beforeAll(async () => {
    seedRandom(0x5eed);
    const mod = "node:fs";
    fs = (await import(/* @vite-ignore */ mod)) as FsLike;
    readme = fs.readFileSync(README, "utf8");
    destinos = destinosDocs(readme);
  });

  it("README.md enlaza al menos una guía de docs/", () => {
    expect(destinos.length, "README.md no enlaza ninguna guía de docs/").toBeGreaterThan(0);
  });

  it("todo enlace a docs/ es absoluto a github.com/dniskav/neuron/blob/master/docs/", () => {
    const relativos = destinos.filter((d) => !d.startsWith(PREFIJO));
    expect(
      relativos,
      `enlaces a docs/ que no son absolutos (la carpeta no va en el paquete npm): ${relativos.join(", ")}. ` +
        `Usa ${PREFIJO}<archivo>.md`,
    ).toEqual([]);
  });

  it("cada archivo de docs/ enlazado desde README.md existe en el repositorio", () => {
    const rutas = [...new Set(destinos.filter((d) => d.startsWith(PREFIJO)).map(rutaDeUrl))];
    const noExisten = rutas.filter((r) => !fs.existsSync(new URL(r, ROOT)));
    expect(noExisten, `README.md enlaza archivos que no existen en el repositorio: ${noExisten.join(", ")}`).toEqual([]);
  });

  it("cada ancla de un enlace a docs/ corresponde a un encabezado del archivo", () => {
    const rotas: string[] = [];
    for (const d of destinos.filter((x) => x.startsWith(PREFIJO) && x.includes("#"))) {
      const [, ancla = ""] = d.split("#");
      const archivo = new URL(rutaDeUrl(d), ROOT);
      const encabezados = fs
        .readFileSync(archivo, "utf8")
        .split("\n")
        .filter((l) => /^#{1,6}\s/.test(l))
        .map((l) => slug(l.replace(/^#{1,6}\s+/, "")));
      if (!encabezados.includes(ancla)) rotas.push(d);
    }
    expect(rotas, `anclas que no existen en el archivo enlazado: ${rotas.join(", ")}`).toEqual([]);
  });

  it("cada guía de docs/*.md está enlazada desde README.md", () => {
    const guias = docsMd();
    const enlazadas = new Set(destinos.map(rutaDeUrl));
    const sinEnlace = guias.filter((g) => !enlazadas.has(`docs/${g}`));
    expect(sinEnlace, `guías de docs/ sin enlace desde README.md: ${sinEnlace.join(", ")}`).toEqual([]);
  });

  /** Archivos .md de primer nivel en docs/ (sin archive/). */
  function docsMd(): string[] {
    return fs.readdirSync(new URL("docs/", ROOT)).filter((f) => f.endsWith(".md"));
  }
});
