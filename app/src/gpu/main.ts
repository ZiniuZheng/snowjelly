/**
 * WebGPU skeleton page. Sets up the device, uploads a particle buffer in the
 * shared format, checks on the GPU that the TS layout matches the WGSL
 * struct, then runs a frame loop that just clears the screen.
 *
 * Renderer (Ian): add your pipeline in renderFrame() below and bind
 * `particleBuffer` as `var<storage, read> particles: array<Particle>`
 * in the vertex stage, indexed by instance_index.
 * Solver (Ziniu/Ruize): compute passes go in the same command encoder
 * before the render pass.
 */
import '../common/style.css';
import {
  MATERIAL,
  PARTICLE_STRIDE_BYTES,
  PARTICLE_WGSL,
  createParticleData,
  createParticleGPUBuffer,
  makeRandomParticles,
  writeParticle,
} from '../common/particleFormat';
import { describeAdapter, initWebGPU, resizeCanvasToDisplaySize } from './device';
import { SVD3_WGSL } from './svd3.wgsl';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const statusEl = document.querySelector<HTMLElement>('#status')!;
const PARTICLE_COUNT = 20000;

function log(html: string): void {
  statusEl.innerHTML += html + '\n';
}

async function main(): Promise<void> {
  const { adapter, device, context } = await initWebGPU(canvas);
  log(`adapter: ${describeAdapter(adapter)}`);
  log(`maxStorageBufferBindingSize: ${(device.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MB`);

  const ok = await selfTestParticleLayout(device);
  log(ok ? '<span class="ok">particle layout self-test: PASS (TS offsets match WGSL struct)</span>'
         : '<span class="bad">particle layout self-test: FAIL, see console</span>');

  const svd = await selfTestSvd3(device);
  log(svd.ok
    ? `<span class="ok">3x3 SVD self-test: PASS (${svd.cases} matrices, max |U S V^T - F| / |F| ${svd.maxRecon.toExponential(2)})</span>`
    : `<span class="bad">3x3 SVD self-test: FAIL (recon ${svd.maxRecon.toExponential(2)}, ` +
      `ortho ${svd.maxOrtho.toExponential(2)}, det ${svd.maxDet.toExponential(2)}), see console</span>`);

  const particles = makeRandomParticles(PARTICLE_COUNT);
  const particleBuffer = createParticleGPUBuffer(device, particles);
  log(`particle buffer: ${PARTICLE_COUNT} particles x ${PARTICLE_STRIDE_BYTES} B = ${(particleBuffer.size / 1024).toFixed(0)} KB`);
  log('renderer: not implemented yet (clear pass only)');

  function renderFrame(): void {
    resizeCanvasToDisplaySize(canvas);
    const encoder = device.createCommandEncoder();
    // --- compute passes (solver) go here ---
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.063, g: 0.086, b: 0.125, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    // --- particle draw goes here (uses particleBuffer) ---
    void particleBuffer;
    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(renderFrame);
  }
  requestAnimationFrame(renderFrame);
}

/**
 * Writes particles with distinct field values from TS, reads each field back
 * through the WGSL struct in a compute shader, and compares.
 */
async function selfTestParticleLayout(device: GPUDevice): Promise<boolean> {
  const n = 4;
  const data = createParticleData(n);
  const expected: number[][] = [];
  for (let i = 0; i < n; i++) {
    const k = i + 1;
    const C = Array.from({ length: 9 }, (_, j) => 100 * k + j); // col-major
    const F = Array.from({ length: 9 }, (_, j) => 200 * k + j);
    writeParticle(data, i, {
      x: [k + 0.1, k + 0.2, k + 0.3],
      mass: k + 0.4,
      v: [k + 0.5, k + 0.6, k + 0.7],
      vol0: k + 0.8,
      C,
      F,
      Jp: k + 0.9,
      material: i % 2 === 0 ? MATERIAL.JELLY : MATERIAL.SNOW,
    });
    // C[2][1] = column 2 row 1 = C[7]; F[1][2] = F[5]; F[2][2] = F[8]
    expected.push([k + 0.1, k + 0.2, k + 0.3, k + 0.4, k + 0.5, k + 0.6, k + 0.7, k + 0.8,
      C[0], C[7], F[0], F[5], F[8], k + 0.9, i % 2, 0]);
  }
  const src = device.createBuffer({ size: data.buffer.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(src, 0, data.buffer);
  const outSize = n * 16 * 4;
  const out = device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const module = device.createShaderModule({
    code: /* wgsl */ `
      ${PARTICLE_WGSL}
      @group(0) @binding(0) var<storage, read> particles: array<Particle>;
      @group(0) @binding(1) var<storage, read_write> outv: array<f32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= arrayLength(&particles)) { return; }
        let p = particles[i];
        let o = i * 16u;
        outv[o + 0u] = p.x.x;  outv[o + 1u] = p.x.y;  outv[o + 2u] = p.x.z;
        outv[o + 3u] = p.mass;
        outv[o + 4u] = p.v.x;  outv[o + 5u] = p.v.y;  outv[o + 6u] = p.v.z;
        outv[o + 7u] = p.vol0;
        outv[o + 8u] = p.C[0][0];  outv[o + 9u] = p.C[2][1];
        outv[o + 10u] = p.F[0][0]; outv[o + 11u] = p.F[1][2]; outv[o + 12u] = p.F[2][2];
        outv[o + 13u] = p.Jp;
        outv[o + 14u] = f32(p.material);
        outv[o + 15u] = f32(p._pad0 + p._pad1);
      }`,
  });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: out } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(out, 0, read, 0, outSize);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const got = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  [src, out, read].forEach((b) => b.destroy());

  let ok = true;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 16; j++) {
      const g = got[i * 16 + j], e = expected[i][j];
      if (Math.abs(g - e) > 1e-4 * Math.max(1, Math.abs(e))) {
        ok = false;
        console.error(`layout mismatch: particle ${i} slot ${j}: got ${g}, expected ${e}`);
      }
    }
  }
  return ok;
}

interface SvdTestResult {
  ok: boolean;
  cases: number;
  maxRecon: number;
  maxOrtho: number;
  maxDet: number;
}

/**
 * Runs the WGSL 3x3 SVD (app/src/gpu/svd3.wgsl.ts) on random, rotation,
 * det < 0 and near-singular matrices, reads U, S, V back, and checks
 * reconstruction, orthogonality, det = +1 and the sign convention on the CPU.
 * Same protocol as selfTestParticleLayout: TS writes, GPU computes, TS checks.
 */
async function selfTestSvd3(device: GPUDevice): Promise<SvdTestResult> {
  const n = 512;
  const IN_FLOATS = 12; // 3 vec4f per matrix
  const OUT_FLOATS = 28; // U (3 vec4f), S (1), V (3)
  const mats = new Float32Array(n * IN_FLOATS);

  let seed = 0x9e3779b9;
  const rand = (): number => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = (): number => {
    const u = Math.max(rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  };
  const randomRotation = (): number[] => {
    let x = gauss(), y = gauss(), z = gauss(), w = gauss();
    const inv = 1 / Math.hypot(x, y, z, w);
    x *= inv; y *= inv; z *= inv; w *= inv;
    // Column-major proper rotation.
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
      2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
      2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y),
    ];
  };

  const inputs: number[][] = [];
  for (let i = 0; i < n; i++) {
    let F: number[];
    const kind = i % 4;
    if (kind === 0) {
      F = Array.from({ length: 9 }, () => gauss());
      const s = 10 ** (-2 + 4 * rand());
      for (let k = 0; k < 9; k++) F[k] *= s;
    } else if (kind === 1) {
      F = randomRotation();
    } else if (kind === 2) {
      F = Array.from({ length: 9 }, () => gauss());
      for (let r = 0; r < 3; r++) F[r] = -F[r]; // det < 0
    } else {
      F = Array.from({ length: 9 }, () => gauss());
      const eps = 10 ** (-6 * rand());
      for (let r = 0; r < 3; r++) F[6 + r] = F[r] + eps * F[3 + r];
    }
    inputs.push(F);
    const base = i * IN_FLOATS;
    for (let c = 0; c < 3; c++) {
      mats[base + 4 * c] = F[3 * c];
      mats[base + 4 * c + 1] = F[3 * c + 1];
      mats[base + 4 * c + 2] = F[3 * c + 2];
    }
  }

  const src = device.createBuffer({ size: mats.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(src, 0, mats);
  const outSize = n * OUT_FLOATS * 4;
  const out = device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const module = device.createShaderModule({
    code: /* wgsl */ `
      ${SVD3_WGSL}
      @group(0) @binding(0) var<storage, read> mats: array<vec4f>;
      @group(0) @binding(1) var<storage, read_write> outv: array<vec4f>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= arrayLength(&mats) / 3u) { return; }
        let b = i * 3u;
        let F = mat3x3f(mats[b].xyz, mats[b + 1u].xyz, mats[b + 2u].xyz);
        let r = svd3(F);
        let o = i * 7u;
        outv[o + 0u] = vec4f(r.U[0].x, r.U[0].y, r.U[0].z, 0.0);
        outv[o + 1u] = vec4f(r.U[1].x, r.U[1].y, r.U[1].z, 0.0);
        outv[o + 2u] = vec4f(r.U[2].x, r.U[2].y, r.U[2].z, 0.0);
        outv[o + 3u] = vec4f(r.S.x, r.S.y, r.S.z, 0.0);
        outv[o + 4u] = vec4f(r.V[0].x, r.V[0].y, r.V[0].z, 0.0);
        outv[o + 5u] = vec4f(r.V[1].x, r.V[1].y, r.V[1].z, 0.0);
        outv[o + 6u] = vec4f(r.V[2].x, r.V[2].y, r.V[2].z, 0.0);
      }`,
  });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: out } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  enc.copyBufferToBuffer(out, 0, read, 0, outSize);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const got = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  [src, out, read].forEach((b) => b.destroy());

  const det3 = (m: ArrayLike<number>): number =>
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);

  let maxRecon = 0, maxOrtho = 0, maxDet = 0, badConvention = 0;
  for (let i = 0; i < n; i++) {
    const o = i * OUT_FLOATS;
    const U = [got[o], got[o + 1], got[o + 2], got[o + 4], got[o + 5], got[o + 6], got[o + 8], got[o + 9], got[o + 10]];
    const S = [got[o + 12], got[o + 13], got[o + 14]];
    const V = [got[o + 16], got[o + 17], got[o + 18], got[o + 20], got[o + 21], got[o + 22], got[o + 24], got[o + 25], got[o + 26]];
    const F = inputs[i];

    let err = 0, scale = 1e-30;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        let m = 0;
        for (let k = 0; k < 3; k++) m += U[3 * k + r] * S[k] * V[3 * k + c];
        err = Math.max(err, Math.abs(m - F[3 * c + r]));
        scale = Math.max(scale, Math.abs(F[3 * c + r]));
      }
    }
    maxRecon = Math.max(maxRecon, err / scale);

    const ortho = (m: number[]): number => {
      let e = 0;
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          let d = 0;
          for (let r = 0; r < 3; r++) d += m[3 * a + r] * m[3 * b + r];
          e = Math.max(e, Math.abs(d - (a === b ? 1 : 0)));
        }
      }
      return e;
    };
    maxOrtho = Math.max(maxOrtho, ortho(U), ortho(V));
    maxDet = Math.max(maxDet, Math.abs(det3(U) - 1), Math.abs(det3(V) - 1));

    const sMax = Math.max(Math.abs(S[0]), Math.abs(S[1]), Math.abs(S[2]), 1e-30);
    const sTol = 1e-4 * sMax;
    const detF = det3(F);
    const sorted = Math.abs(S[0]) + sTol >= Math.abs(S[1]) && Math.abs(S[1]) + sTol >= Math.abs(S[2]);
    const positive = S[0] >= -sTol && S[1] >= -sTol;
    const signOk = Math.abs(detF) < 1e-6 * scale * scale * scale
      || (detF > 0 ? S[2] >= -sTol : S[2] <= sTol);
    if (!sorted || !positive || !signOk) {
      badConvention++;
      if (badConvention <= 5) {
        console.error(`svd convention mismatch on case ${i}: detF=${detF}, S=[${S.join(', ')}]`);
      }
    }
  }

  // f32 rotates U/V by 18 Givens steps, and GPUs differ in sin/cos precision,
  // so 1e-3 is a "logic is right" bound, not a tight error budget.
  const ok = maxRecon < 1e-3 && maxOrtho < 1e-3 && maxDet < 1e-3 && badConvention === 0;
  return { ok, cases: n, maxRecon, maxOrtho, maxDet };
}

main().catch((err: unknown) => {
  log(`<span class="bad">${err instanceof Error ? err.message : String(err)}</span>`);
  console.error(err);
});
