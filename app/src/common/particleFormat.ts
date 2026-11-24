/**
 * Shared particle buffer format (v1). Everyone builds on this.
 * Full write-up: docs/particle-format.md
 *
 * One GPU storage buffer, array-of-structs, 144 bytes per particle.
 * The layout follows WGSL storage-buffer alignment rules exactly
 * (vec3f and mat3x3f columns are 16-byte aligned), so the TS offsets
 * below and the WGSL struct describe the same bytes. gpu.html runs a
 * self-test that checks this on the GPU.
 *
 * Conventions:
 *  - Positions are world units inside the unit cube [0,1]^3, y up, gravity -y.
 *  - Grid has N cells per axis (default 64), dx = 1/N. The solver keeps
 *    particles at least 2 cells away from the walls.
 *  - Particles are seeded 2 per cell per axis (spacing dx/2), so a good
 *    render radius is about dx/4.
 *  - Matrices are column-major (WGSL mat3x3f): column k starts at
 *    byte offset base + 16*k and holds 3 floats + 1 pad float.
 */

export const MATERIAL = {
  JELLY: 0,
  SNOW: 1,
} as const;
export type MaterialId = (typeof MATERIAL)[keyof typeof MATERIAL];

export const PARTICLE_STRIDE_BYTES = 144;
export const PARTICLE_STRIDE_F32 = PARTICLE_STRIDE_BYTES / 4; // 36

/** Byte offsets of each field inside one particle. */
export const PARTICLE_OFFSET = {
  x: 0, //        vec3f   position (world, unit cube)
  mass: 12, //    f32     particle mass
  v: 16, //       vec3f   velocity (world units / s)
  vol0: 28, //    f32     initial (rest) volume
  C: 32, //       mat3x3f APIC affine velocity matrix
  F: 80, //       mat3x3f deformation gradient (elastic part Fe for snow)
  Jp: 128, //     f32     plastic volume ratio det(Fp), 1.0 for jelly
  material: 132, // u32   MATERIAL.* id
  // 136..143: two u32 pads (reserved; e.g. a color/group id later)
} as const;

/** Same offsets in f32 units, for writing through a Float32Array view. */
const F32 = {
  x: 0,
  mass: 3,
  v: 4,
  vol0: 7,
  C: 8, // col k at 8 + 4k
  F: 20, // col k at 20 + 4k
  Jp: 32,
  material: 33,
} as const;

/** Paste this into any shader that reads the particle buffer. */
export const PARTICLE_WGSL = /* wgsl */ `
struct Particle {
  x: vec3f,        // offset   0
  mass: f32,       // offset  12
  v: vec3f,        // offset  16
  vol0: f32,       // offset  28
  C: mat3x3f,      // offset  32 (APIC affine)
  F: mat3x3f,      // offset  80 (deformation gradient)
  Jp: f32,         // offset 128 (plastic volume ratio)
  material: u32,   // offset 132 (0 = jelly, 1 = snow)
  _pad0: u32,      // offset 136
  _pad1: u32,      // offset 140
};                 // size 144
`;

/**
 * Usage flags the particle buffer is created with:
 * STORAGE | COPY_DST | COPY_SRC. Written as literals so this module also
 * loads in browsers without WebGPU (the 2D CPU demo imports it).
 */
export const PARTICLE_BUFFER_USAGE = 0x80 | 0x08 | 0x04;

export interface ParticleData {
  count: number;
  buffer: ArrayBuffer;
  f32: Float32Array;
  u32: Uint32Array;
}

export interface ParticleInit {
  x: readonly [number, number, number];
  v?: readonly [number, number, number];
  mass: number;
  vol0: number;
  material: MaterialId;
  /** 9 floats, column-major. Defaults to identity. */
  F?: ArrayLike<number>;
  /** 9 floats, column-major. Defaults to zero. */
  C?: ArrayLike<number>;
  Jp?: number;
}

export function createParticleData(count: number): ParticleData {
  const buffer = new ArrayBuffer(count * PARTICLE_STRIDE_BYTES);
  return { count, buffer, f32: new Float32Array(buffer), u32: new Uint32Array(buffer) };
}

const IDENTITY3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

export function writeParticle(d: ParticleData, i: number, p: ParticleInit): void {
  const b = i * PARTICLE_STRIDE_F32;
  const f = d.f32;
  f[b + F32.x] = p.x[0];
  f[b + F32.x + 1] = p.x[1];
  f[b + F32.x + 2] = p.x[2];
  f[b + F32.mass] = p.mass;
  const v = p.v ?? [0, 0, 0];
  f[b + F32.v] = v[0];
  f[b + F32.v + 1] = v[1];
  f[b + F32.v + 2] = v[2];
  f[b + F32.vol0] = p.vol0;
  writeMat3(f, b + F32.C, p.C);
  writeMat3(f, b + F32.F, p.F ?? IDENTITY3);
  f[b + F32.Jp] = p.Jp ?? 1;
  d.u32[b + F32.material] = p.material;
  d.u32[b + 34] = 0;
  d.u32[b + 35] = 0;
}

/** Writes a column-major 3x3 into the padded mat3x3f slot (zero if m is undefined). */
function writeMat3(f: Float32Array, base: number, m?: ArrayLike<number>): void {
  for (let col = 0; col < 3; col++) {
    for (let row = 0; row < 3; row++) {
      f[base + 4 * col + row] = m ? m[3 * col + row] : 0;
    }
    f[base + 4 * col + 3] = 0;
  }
}

export function readPosition(d: ParticleData, i: number): [number, number, number] {
  const b = i * PARTICLE_STRIDE_F32;
  return [d.f32[b], d.f32[b + 1], d.f32[b + 2]];
}

/**
 * Fake data for renderer work: a jittered block of particles, left half
 * jelly and right half snow, with small random velocities.
 */
export function makeRandomParticles(count: number, seed = 1): ParticleData {
  const rand = mulberry32(seed);
  const d = createParticleData(count);
  const dx = 1 / 64;
  const vol0 = (dx / 2) ** 3;
  for (let i = 0; i < count; i++) {
    const x: [number, number, number] = [
      0.2 + 0.6 * rand(),
      0.1 + 0.5 * rand(),
      0.2 + 0.6 * rand(),
    ];
    writeParticle(d, i, {
      x,
      v: [rand() - 0.5, rand() - 0.5, rand() - 0.5],
      mass: vol0, // density 1
      vol0,
      material: x[0] < 0.5 ? MATERIAL.JELLY : MATERIAL.SNOW,
    });
  }
  return d;
}

/** Creates the storage buffer and uploads the data. */
export function createParticleGPUBuffer(device: GPUDevice, d: ParticleData, label = 'particles'): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(d.buffer.byteLength, PARTICLE_STRIDE_BYTES),
    usage: PARTICLE_BUFFER_USAGE,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(d.buffer));
  buffer.unmap();
  return buffer;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
