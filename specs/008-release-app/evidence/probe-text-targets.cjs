// Evidencia del «Hallazgo 1» de app-inventory.md (008 release-app). Solo lectura de ambos repositorios.
//
// Uso (desde la raíz de la librería, con `npm run build` hecho):
//   node specs/008-release-app/evidence/probe-text-targets.cjs
//
// Reproduce TextGeneratorSession de neuron_app (NetworkTransformer 20 tokens, 27 clases, d_model 16,
// 2 bloques) con dos formas de `targets`: índices de clase (como hace la app hoy,
// neuron_app/src/core/text/TextGeneratorSession.ts:163-169) y one-hot plano (lo que espera la
// librería: src/NetworkTransformer.ts:140). Compara la 0.2.5 que instala la app con el dist local.
//
// Variables de entorno opcionales: APP_LIB (ruta al dist/index.js de 0.2.5), LOCAL_LIB (dist/index.js local).
// Cifras de la sonda original (una ejecución sin semilla por fila): ver app-inventory.md.
const path = require('path')
const LIBS = {
  '0.2.5(app)': process.env.APP_LIB ||
    '/Users/daniel/Desktop/projects/experiments/neuron_app/node_modules/@dniskav/neuron/dist/index.js',
  'local': process.env.LOCAL_LIB || path.resolve(__dirname, '../../../dist/index.js'),
}
const CORPUS = 'the quick brown fox jumps over the lazy dog'
const VOCAB = 'abcdefghijklmnopqrstuvwxyz '
const idx = c => { const i = VOCAB.indexOf(c); return i === -1 ? 26 : i }
const SEQ = 20, V = 27

function run(libName, oneHot, lr, steps) {
  const { NetworkTransformer } = require(LIBS[libName])
  const net = new NetworkTransformer(SEQ, { vocabSize: V, d_model: 16, nHeads: 2, d_ff: 32, nBlocks: 2, nClasses: V })
  let ema = 0, firstNaN = -1, l0 = 0
  for (let i = 1; i <= steps; i++) {
    const s = Math.floor(Math.random() * (CORPUS.length - SEQ - 1))
    const tk = Array.from(CORPUS.slice(s, s + SEQ), idx)
    const ti = Array.from(CORPUS.slice(s + 1, s + SEQ + 1), idx)
    let tg = ti
    if (oneHot) { tg = new Array(SEQ * V).fill(0); ti.forEach((c, p) => { tg[p * V + c] = 1 }) }
    const l = net.train(tk, tg, lr)
    if (i === 1) l0 = l
    if (!Number.isFinite(l)) { firstNaN = i; break }
    ema = ema === 0 ? l : 0.95 * ema + 0.05 * l
  }
  console.log(libName, oneHot ? 'one-hot' : 'indices', 'lr', lr, 'loss1', l0.toFixed(3),
    'primer NaN en paso', firstNaN, 'EMA final', ema.toFixed(3))
}

run('0.2.5(app)', false, 0.001, 200)
run('local', false, 0.001, 200)
for (const lr of [0.003, 0.001, 0.0003]) run('local', true, lr, 2500)
run('0.2.5(app)', true, 0.001, 2500)
