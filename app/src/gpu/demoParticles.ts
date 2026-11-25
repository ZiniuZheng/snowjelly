import {
  MATERIAL,
  PARTICLE_OFFSET,
  PARTICLE_STRIDE_F32,
  makeRandomParticles,
  readPosition,
  type ParticleData,
} from '../common/particleFormat';

/** Shape the seeded random preview data into a jelly block and a snowball. */
export function makeDemoParticles(count: number): ParticleData {
  const data = makeRandomParticles(count);
  const positionOffset = PARTICLE_OFFSET.x / Float32Array.BYTES_PER_ELEMENT;
  const materialOffset = PARTICLE_OFFSET.material / Uint32Array.BYTES_PER_ELEMENT;
  for (let i = 0; i < count; i++) {
    const base = i * PARTICLE_STRIDE_F32;
    const [x, y, z] = readPosition(data, i);
    const v = (y - 0.1) / 0.5;
    const w = (z - 0.2) / 0.6;
    const snow = data.u32[base + materialOffset] === MATERIAL.SNOW;
    // makeRandomParticles distributes each material over half of the x interval.
    const u = (x - (snow ? 0.5 : 0.2)) / 0.3;
    if (snow) {
      // Uniform volume sampling: cube-root radius, uniform azimuth and polar cosine.
      const radius = 0.176 * Math.cbrt(u);
      const polarCos = 2 * v - 1;
      const angle = 2 * Math.PI * w;
      const horizontal = radius * Math.sqrt(Math.max(0, 1 - polarCos * polarCos));
      data.f32.set([
        0.725 + horizontal * Math.cos(angle), 0.18 + radius * polarCos, 0.49 + horizontal * Math.sin(angle),
      ], base + positionOffset);
    } else {
      const px = 2 * u - 1, py = 2 * v - 1, pz = 2 * w - 1;
      const ex = Math.max(0, Math.abs(px) - 0.79);
      const ey = Math.max(0, Math.abs(py) - 0.79);
      const ez = Math.max(0, Math.abs(pz) - 0.79);
      // Project the corners of the cube onto a rounded boundary.
      const shrink = 1 - Math.max(ex, ey, ez) / (Math.hypot(ex, ey, ez) || 1);
      data.f32.set([
        0.275 + 0.156 * (px - Math.sign(px) * ex * shrink),
        0.153 + 0.149 * (py - Math.sign(py) * ey * shrink),
        0.51 + 0.155 * (pz - Math.sign(pz) * ez * shrink),
      ], base + positionOffset);
    }
  }
  return data;
}
