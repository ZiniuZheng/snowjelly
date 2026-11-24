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

main().catch((err: unknown) => {
  log(`<span class="bad">${err instanceof Error ? err.message : String(err)}</span>`);
  console.error(err);
});
