import { defineConfig } from 'tsup'

// Build de la biblioteca (npm run build). Las entradas, formatos, dts y sourcemaps
// son los de siempre; el minificado de infer (dist/infer.min.mjs) va aparte, en el
// script build, con esbuild.
//
// sourcesContent = false: los mapas no llevan el código fuente dentro. El paquete
// publica src/ (campo "files"), así que los mapas apuntan a ../src/ y el contenido
// no hace falta. Sin esto, los mapas duplican todo el código (unos 1,9 MB).
export default defineConfig({
  entry: ['src/index.ts', 'src/infer.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  clean: true,
  sourcemap: true,
  esbuildOptions(options) {
    options.sourcesContent = false
  },
})
