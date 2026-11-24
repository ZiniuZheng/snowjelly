import { defineConfig, type Plugin } from 'vite';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = dirname(fileURLToPath(import.meta.url));

// The repo root index.html is the proposal page served by GitHub Pages, so the
// app lives in app/ and Vite uses that as its root.
export default defineConfig({
  root: 'app',
  base: './',
  plugins: [screenshotSaver()],
  server: { port: 5173 },
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rolldownOptions: {
      input: {
        main: resolve(repoRoot, 'app/index.html'),
        cpu2d: resolve(repoRoot, 'app/cpu2d.html'),
        gpu: resolve(repoRoot, 'app/gpu.html'),
      },
    },
  },
});

/**
 * Dev-only helper: POST a PNG to /__screenshot?name=foo and it is written to
 * screenshots/foo.png in the repo. Used by the "Save PNG" buttons so milestone
 * screenshots land on disk without a download dialog.
 */
function screenshotSaver(): Plugin {
  return {
    name: 'snowjelly-screenshot-saver',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__screenshot', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end();
          return;
        }
        const chunks: Buffer[] = [];
        req.on('data', (c: Buffer) => chunks.push(c));
        req.on('end', () => {
          const raw = new URL(req.url ?? '/', 'http://localhost').searchParams.get('name');
          const name = (raw || `shot-${Date.now()}`).replace(/[^\w.-]/g, '_');
          const dir = resolve(repoRoot, 'screenshots');
          mkdirSync(dir, { recursive: true });
          const file = resolve(dir, `${name}.png`);
          writeFileSync(file, Buffer.concat(chunks));
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: true, file }));
        });
      });
    },
  };
}
