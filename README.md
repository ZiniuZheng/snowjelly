# Snowjelly

Real-time snow and jelly simulation with MLS-MPM in the browser, on WebGPU.
CS 184 Fall 2026 final project. Proposal: https://ziniuzheng.github.io/snowjelly/

## Run it

Needs Node 20+ and a WebGPU browser (recent Chrome or Edge) for the GPU pages.

```sh
npm install
npm run dev        # http://localhost:5173
```

- `http://localhost:5173/cpu2d.html`: 2D CPU MLS-MPM reference with jelly. Drag on the canvas to push things; "save PNG" writes to `screenshots/`.
- `http://localhost:5173/gpu.html`: WebGPU skeleton (device setup, particle buffer, layout and 3x3 SVD self-tests).

Other scripts: `npm run typecheck`, `npm run build` (outputs `dist/`), `node scripts/check-mpm2d.mjs` (headless solver checks: stress sanity, momentum conservation, stability), `node scripts/check-svd3.mjs` (3x3 SVD prototype: reconstruction, orthogonality, det and sign convention).

## Layout

```
index.html                  proposal page (served by GitHub Pages, leave it at the root)
app/                        Vite root
  cpu2d.html, gpu.html      entry pages
  src/common/particleFormat.ts   shared particle buffer format (read docs/particle-format.md)
  src/gpu/device.ts         WebGPU init helpers
  src/gpu/main.ts           WebGPU page: add render/compute passes here
  src/sim2d/mpm2d.ts        2D CPU MLS-MPM solver (APIC, quadratic B-splines)
  src/sim2d/materials2d.ts  2D stress models (neo-Hookean, fixed corotated)
  src/sim3d/svd3.ts         3x3 SVD CPU prototype (two-sided Jacobi, McAdams et al. 2011)
  src/gpu/svd3.wgsl.ts      the same 3x3 SVD in WGSL, exercised by a self-test on gpu.html
docs/particle-format.md     particle buffer format spec
scripts/check-mpm2d.mjs     headless solver checks
```

## Working together

Each of us commits under our own GitHub account. Small, focused commits on a
branch, then merge to `main`. Don't change the particle format without posting
in the group first.
