import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.ts'],
    setupFiles: ['tests/setup.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/**'],
      exclude: ['src/index.ts'],
      // Medido 2026-11-11 (3 ejecuciones iguales, json-summary): lines 99.38, branches 96.50,
      // functions 99.55, statements 99.25. Umbral = medido - 2, redondeado hacia abajo.
      thresholds: { lines: 97, branches: 94, functions: 97, statements: 97 },
    },
  },
})
