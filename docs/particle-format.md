# Particle buffer format (v1)

Source of truth: [`app/src/common/particleFormat.ts`](../app/src/common/particleFormat.ts).
If you need a change, post in the group first so the solver and renderer stay in sync.

## Conventions

| | |
|---|---|
| Domain | unit cube `[0,1]^3`, world units (nominally meters) |
| Axes | y up, gravity along -y |
| Grid | `N` cells per axis (default 64), `dx = 1/N`; solver keeps particles at least 2 cells from the walls |
| Seeding | 2 particles per cell per axis, spacing `dx/2`, so a good render radius is about `dx/4` |
| Matrices | column-major, WGSL `mat3x3f` |
| Materials | `0` = jelly, `1` = snow (`MATERIAL.JELLY`, `MATERIAL.SNOW`) |

## Layout: one storage buffer, array of structs, 144 bytes per particle

```wgsl
struct Particle {
  x: vec3f,        // offset   0  position
  mass: f32,       // offset  12
  v: vec3f,        // offset  16  velocity
  vol0: f32,       // offset  28  rest volume
  C: mat3x3f,      // offset  32  APIC affine matrix
  F: mat3x3f,      // offset  80  deformation gradient (elastic part Fe for snow)
  Jp: f32,         // offset 128  plastic volume ratio det(Fp); 1.0 for jelly
  material: u32,   // offset 132  0 = jelly, 1 = snow
  _pad0: u32,      // offset 136  reserved
  _pad1: u32,      // offset 140  reserved
};                 // size 144
```

`vec3f` and each `mat3x3f` column are 16-byte aligned in storage buffers, so a
column `k` of `C` sits at byte `32 + 16k` (3 floats + 1 pad). In `f32` units:
`x` = 0..2, `mass` = 3, `v` = 4..6, `vol0` = 7, `C` col k = `8 + 4k`,
`F` col k = `20 + 4k`, `Jp` = 32, `material` = 33 (read as `u32`).

`gpu.html` writes particles from TypeScript, reads every field back through
this WGSL struct in a compute shader, and prints PASS/FAIL. It passes.

## Buffer

- Created with `STORAGE | COPY_DST | COPY_SRC` (`createParticleGPUBuffer`).
- Solver binds it `read_write` in compute passes.
- Renderer binds it `var<storage, read> particles: array<Particle>;` in the
  vertex shader and indexes it with `@builtin(instance_index)`. No separate
  vertex buffer or copy is needed: draw `6` vertices (a quad) per instance,
  `particleCount` instances.

## Helpers (TypeScript)

```ts
import {
  PARTICLE_WGSL,            // paste into shaders: `${PARTICLE_WGSL}`
  makeRandomParticles,      // fake data for renderer work
  createParticleGPUBuffer,  // upload to a GPU storage buffer
  createParticleData, writeParticle, readPosition,
} from './common/particleFormat';

const data = makeRandomParticles(20000);
const buf = createParticleGPUBuffer(device, data);
```

The 2D CPU solver can export real simulation states in this format with
`sim.toParticleData()` (z = 0.5), which is handy for testing the renderer on
real motion.

## Grid (solver-internal, may change)

Planned for the 3D GPU P2G: one node per `(N+1)^3` with
`struct GridNode { vx, vy, vz, m: atomic<i32> }` (16 bytes), values stored
in fixed point because WGSL atomics are integer-only. The renderer never
touches it.
