import { describe, it, expect, beforeAll } from "vitest";

// El proyecto no declara @types/node; se carga node:fs con un tipo minimo propio.
interface FsLike {
  existsSync(p: URL): boolean;
  readFileSync(p: URL, enc: "utf8"): string;
  readdirSync(p: URL): string[];
}
let fs: FsLike;

// Meta-test de cobertura del oraculo de gradientes (plan, decision 8).
// Lee src/ como texto; no importa ni modifica nada.

const SRC = new URL("../../src/", import.meta.url);
const ORACLE = new URL("./", import.meta.url);

const TRAIN_METHODS = [
  "train", "trainStep", "backward", "backprop", "fit",
  "fitGD", "trainBatch", "trainParams", "update",
];

/** Modelos entrenables cubiertos: clase exportada -> archivos de oraculo (en tests/oracle/). */
const COVERED: Record<string, string[]> = {
  Neuron: ["Neuron.oracle.test.ts"],
  NeuronN: ["NeuronN.oracle.test.ts"],
  Layer: ["Layer.oracle.test.ts"],
  Network: ["Network.oracle.test.ts"],
  NetworkN: ["NetworkN.oracle.test.ts"],
  LSTMLayer: ["LSTMLayer.oracle.test.ts"],
  GRULayer: ["GRU.oracle.test.ts"],
  Conv1D: ["Conv1D.oracle.test.ts"],
  Conv2D: ["Conv2D.oracle.test.ts"],
  BatchNorm: ["BatchNorm.oracle.test.ts"],
  LayerNorm: ["LayerNorm.oracle.test.ts"],
  AttentionHead: ["AttentionHead.oracle.test.ts"],
  MultiHeadAttention: ["MultiHeadAttention.oracle.test.ts"],
  TransformerBlock: ["TransformerBlock.oracle.test.ts"],
  NetworkLSTM: ["NetworkLSTM.oracle.test.ts"],
  Autoencoder: ["Autoencoder.oracle.test.ts"],
  NetworkTransformer: ["NetworkTransformer.oracle.test.ts"],
  NetworkTransformerRL: ["NetworkTransformerRL.oracle.test.ts"],
  RNN: ["RNN.oracle.test.ts"],
  CausalConv1D: ["TCN.oracle.test.ts"],
  TCN: ["TCN.oracle.test.ts"],
  Seq2Seq: ["Seq2Seq.oracle.test.ts"],
  GAN: ["GAN.oracle.test.ts"],
  VAE: ["VAE.oracle.test.ts"],
  Perceptron: ["Perceptron.oracle.test.ts"],
  LogisticRegression: ["LogisticRegression.oracle.test.ts"],
  SoftmaxRegression: ["LogisticRegression.oracle.test.ts"],
  LinearRegression: ["LinearRegression.oracle.test.ts"],
  Word2Vec: ["Word2Vec.oracle.test.ts"],
  SOM: ["SOM.oracle.test.ts"],
  ContrastiveLearning: ["ContrastiveLearning.oracle.test.ts"],
  Value: ["Tape.oracle.test.ts"],
  WeightMatrix: ["MatMul.oracle.test.ts"],
  BiasVector: ["MatMul.oracle.test.ts"],
  EmbeddingMatrix: ["MatMul.oracle.test.ts"],
  LearnedPositionalEncoding: ["PositionalEncoding.oracle.test.ts"],
};

/** Exclusiones: clase -> motivo por escrito (Constitucion II). */
const EXCLUDED: Record<string, string> = {
  DecisionTree: "Sin gradiente: fit hace particiones greedy; no hay descenso por gradiente. Se cubre en la feature 006.",
  KMeans: "Sin gradiente: fit itera asignacion/centroides (Lloyd); no hay descenso por gradiente. Feature 006.",
  PCA: "Sin gradiente: fit es una descomposicion espectral cerrada; no hay descenso por gradiente. Feature 006.",
  GaussianNaiveBayes: "Sin gradiente: fit estima medias y varianzas en forma cerrada. Feature 006.",
  TSNE: "Visualizacion no parametrica: fit optimiza coordenadas de salida, no pesos de un modelo; no expone actualizacion por -dL/dw. Feature 006.",
  Tokenizer: "No es un modelo: fit construye un vocabulario; sin parametros diferenciables.",
  EarlyStopping: "No es un modelo: update registra la metrica de validacion y decide parar; sin parametros diferenciables.",
  Trainer: "Orquestador: train delega en el modelo (TrainableNetwork), que ya tiene su oraculo; no posee pesos ni gradiente propios.",
  Dropout: "Sin parametros entrenables: backward solo aplica la mascara a dx; el oraculo de pesos (delta w) no aplica.",
  Flatten: "Sin parametros entrenables: backward solo remodela dx; el oraculo de pesos no aplica.",
  MaxPool2D: "Sin parametros entrenables: backward enruta dx al maximo; el oraculo de pesos no aplica.",
};

function listSrcFiles(): string[] {
  return fs.readdirSync(SRC).filter((f: string) => f.endsWith(".ts") && f !== "index.ts");
}

/** Nombres de valor (no `export type`) reexportados por src/index.ts. */
function exportedNames(): Set<string> {
  const text = fs.readFileSync(new URL("index.ts", SRC), "utf8").replace(/\/\/.*$/gm, "");
  const names = new Set<string>();
  const re = /export\s*(type\s*)?\{([^}]*)\}\s*from/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m[1]) continue;
    for (const part of m[2].split(",")) {
      const n = part.trim().split(/\s+as\s+/).pop()!.trim();
      if (n) names.add(n);
    }
  }
  return names;
}

/** Clases exportadas por src/*.ts con su cuerpo aproximado (hasta la siguiente clase del archivo). */
function classBodies(): Map<string, string> {
  const out = new Map<string, string>();
  for (const f of listSrcFiles()) {
    const text = fs.readFileSync(new URL(f, SRC), "utf8");
    const re = /^export\s+(?:abstract\s+)?class\s+(\w+)/gm;
    const hits: { name: string; start: number }[] = [];
    let m: RegExpExecArray | null;
    while ((m = re.exec(text))) hits.push({ name: m[1], start: m.index });
    hits.forEach((h, i) => {
      out.set(h.name, text.slice(h.start, i + 1 < hits.length ? hits[i + 1].start : text.length));
    });
  }
  return out;
}

function trainableExported(): string[] {
  const exported = exportedNames();
  const methodRe = new RegExp(`^[ \\t]+(?:public\\s+|async\\s+)?(?:${TRAIN_METHODS.join("|")})\\s*[(<]`, "m");
  const res: string[] = [];
  for (const [name, body] of classBodies()) {
    if (exported.has(name) && methodRe.test(body)) res.push(name);
  }
  return res.sort();
}

describe("oracle coverage meta-test", () => {
  let trainable: string[] = [];
  beforeAll(async () => {
    const mod = "node:fs";
    fs = (await import(/* @vite-ignore */ mod)) as FsLike;
    trainable = trainableExported();
  });

  it("detecta un conjunto no trivial de clases entrenables", () => {
    expect(trainable.length).toBeGreaterThan(30);
    expect(trainable).toContain("NetworkN");
    expect(trainable).toContain("GAN");
  });

  it("toda clase entrenable exportada esta cubierta o excluida con motivo", () => {
    const missing = trainable.filter((c) => !(c in COVERED) && !(c in EXCLUDED));
    expect(missing, `clases entrenables sin oraculo ni exclusion: ${missing.join(", ")}`).toEqual([]);
  });

  it("cada archivo de oraculo declarado existe en tests/oracle/", () => {
    const absent: string[] = [];
    for (const [cls, files] of Object.entries(COVERED)) {
      expect(files.length, cls).toBeGreaterThan(0);
      for (const f of files) if (!fs.existsSync(new URL(f, ORACLE))) absent.push(`${cls} -> ${f}`);
    }
    expect(absent).toEqual([]);
  });

  it("las entradas de las listas son clases exportadas (sin obsoletas)", () => {
    // Layer y LayerNorm no usan los nombres de metodo de la regla (usan otros, p. ej. backwardOne),
    // pero tienen oraculo; basta con que sigan siendo clases exportadas.
    const exported = exportedNames();
    const bodies = classBodies();
    const stale = [...Object.keys(COVERED), ...Object.keys(EXCLUDED)].filter(
      (c) => !exported.has(c) || !bodies.has(c),
    );
    expect(stale).toEqual([]);
    const excludedNotTrainable = Object.keys(EXCLUDED).filter((c) => !trainable.includes(c));
    expect(excludedNotTrainable).toEqual([]);
  });

  it("ninguna clase esta a la vez cubierta y excluida", () => {
    const both = Object.keys(COVERED).filter((c) => c in EXCLUDED);
    expect(both).toEqual([]);
  });

  it("toda exclusion tiene un motivo escrito", () => {
    for (const [cls, why] of Object.entries(EXCLUDED)) {
      expect(why.trim().length, `motivo de ${cls}`).toBeGreaterThanOrEqual(30);
    }
  });
});
