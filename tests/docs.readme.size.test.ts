import { describe, it, expect, beforeAll } from "vitest";
import { seedRandom } from "./helpers/seed";

// Límite de tamaño de README.md (0.5.1). npm guarda solo los primeros 64 KB de la descripción del paquete:
// lo que queda después se corta y no aparece en la página de npmjs.com. Por eso el README es la
// página corta y las guías largas están en docs/. Lee el archivo con node:fs; no importa src/.

interface FsLike {
  statSync(p: URL): { size: number };
}
let fs: FsLike;

const README = new URL("../README.md", import.meta.url);
/** 60 KiB. El límite real de npm es 64 KB (65536 bytes); el margen cubre el texto que se añada después. */
const LIMITE_BYTES = 61440;

describe("tamaño de README.md (0.5.1)", () => {
  beforeAll(async () => {
    seedRandom(0x5eed);
    const mod = "node:fs";
    fs = (await import(/* @vite-ignore */ mod)) as FsLike;
  });

  it("README.md pesa como mucho 61440 bytes", () => {
    const bytes = fs.statSync(README).size;
    expect(
      bytes,
      `README.md pesa ${bytes} bytes y el límite es ${LIMITE_BYTES} (60 KiB). npm guarda solo los primeros 64 KB ` +
        "del README y el resto no se ve en la página del paquete. Mueve el texto largo a docs/ y regenera con: " +
        "node specs/007-docs/tools/assemble-readme.mjs",
    ).toBeLessThanOrEqual(LIMITE_BYTES);
  });
});
