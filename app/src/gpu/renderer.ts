import { MATERIAL, PARTICLE_WGSL } from '../common/particleFormat';

export const SCENE_BACKGROUND = { r: 0.043, g: 0.061, b: 0.087, a: 1 };

const DEPTH_FORMAT = 'depth24plus';
const DOMAIN_FRAMING = { centerY: 0.5, height: 1.5, minWidth: 1.3 };
const shader = /* wgsl */ `
  ${PARTICLE_WGSL}

  struct Camera {
    clipScale: vec2f,
    centerY: f32,
  };
  @group(0) @binding(0) var<uniform> camera: Camera;
  @group(0) @binding(1) var<storage, read> particles: array<Particle>;

  const RADIUS = 1.0 / 256.0;
  const DEPTH_SCALE = 0.5;
  const BACKGROUND = vec3f(${SCENE_BACKGROUND.r}, ${SCENE_BACKGROUND.g}, ${SCENE_BACKGROUND.b});
  const CORNERS = array<vec2f, 6>(
    vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
    vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1),
  );

  // Orthographic camera looking down at the y-up unit cube from +x and +z.
  fn project(world: vec3f) -> vec3f {
    let yaw = 0.34;
    let pitch = 0.48;
    let relative = world - vec3f(0.5, camera.centerY, 0.5);
    let right = vec3f(cos(yaw), 0, -sin(yaw));
    let up = vec3f(-sin(yaw) * sin(pitch), cos(pitch), -cos(yaw) * sin(pitch));
    let away = cross(up, right);
    return vec3f(
      vec2f(dot(relative, right), dot(relative, up)) * camera.clipScale,
      0.5 + dot(relative, away) * DEPTH_SCALE,
    );
  }

  struct ParticleVertex {
    @builtin(position) position: vec4f,
    @location(0) corner: vec2f,
    @location(1) @interpolate(flat) material: u32,
  };

  @vertex fn particleVertex(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
  ) -> ParticleVertex {
    let center = project(particles[instanceIndex].x);
    let corner = CORNERS[vertexIndex];
    let offset = corner * RADIUS * camera.clipScale;
    return ParticleVertex(vec4f(center.xy + offset, center.z, 1), corner, particles[instanceIndex].material);
  }

  struct ParticleFragment {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
  };

  @fragment fn particleFragment(input: ParticleVertex) -> ParticleFragment {
    let radiusSquared = dot(input.corner, input.corner);
    if (radiusSquared > 1.0) { discard; }

    let front = sqrt(1.0 - radiusSquared);
    let normal = vec3f(input.corner, front);
    let light = normalize(vec3f(-0.48, 0.66, 0.58));
    let halfway = normalize(light + vec3f(0, 0, 1));
    let diffuse = max(0.0, dot(normal, light));
    let highlight = max(0.0, dot(normal, halfway));
    let snow = input.material == ${MATERIAL.SNOW}u;
    let base = select(vec3f(0.94, 0.29, 0.43), vec3f(0.58, 0.80, 0.97), snow);
    let specular = select(0.28 * pow(highlight, 20), 0.09 * pow(highlight, 8), snow);
    let color = base * (0.27 + 0.68 * diffuse) + vec3f(specular);
    // Move depth from the billboard center to the visible sphere surface.
    return ParticleFragment(vec4f(color, 1), input.position.z - front * RADIUS * DEPTH_SCALE);
  }

  struct FloorVertex {
    @builtin(position) position: vec4f,
    @location(0) worldXZ: vec2f,
  };

  @vertex fn floorVertex(@builtin(vertex_index) vertexIndex: u32) -> FloorVertex {
    let xz = vec2f(0.5) + 0.85 * CORNERS[vertexIndex];
    return FloorVertex(vec4f(project(vec3f(xz.x, 0, xz.y)), 1), xz);
  }

  @fragment fn floorFragment(input: FloorVertex) -> @location(0) vec4f {
    let grid = input.worldXZ * 10.0;
    let lineDistance = abs(fract(grid + 0.5) - 0.5) / fwidth(grid);
    let line = 1.0 - smoothstep(0.15, 1.1, min(lineDistance.x, lineDistance.y));
    let fade = 1.0 - smoothstep(0.26, 0.83, distance(input.worldXZ, vec2f(0.5)));
    let tint = vec3f(0.042, 0.069, 0.11) + line * vec3f(0.065, 0.095, 0.14);
    return vec4f(BACKGROUND + fade * tint, 1);
  }
`;

export async function createParticleRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
  particleBuffer: GPUBuffer,
  framing = DOMAIN_FRAMING,
) {
  const module = device.createShaderModule({ label: 'Particle and floor shaders', code: shader });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  const depthStencil = { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' } as const;
  const [particlePipeline, floorPipeline] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'Particle sphere impostors',
      layout,
      vertex: { module, entryPoint: 'particleVertex' },
      fragment: { module, entryPoint: 'particleFragment', targets: [{ format }] },
      depthStencil,
    }),
    device.createRenderPipelineAsync({
      label: 'Ground grid',
      layout,
      vertex: { module, entryPoint: 'floorVertex' },
      fragment: { module, entryPoint: 'floorFragment', targets: [{ format }] },
      depthStencil,
    }),
  ]);

  const camera = device.createBuffer({
    label: 'Orthographic camera',
    size: 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: camera } },
      { binding: 1, resource: { buffer: particleBuffer } },
    ],
  });

  let depth: { texture: GPUTexture; view: GPUTextureView } | undefined;
  return {
    depthView(width: number, height: number): GPUTextureView {
      if (depth?.texture.width === width && depth.texture.height === height) return depth.view;

      const aspect = width / height;
      const heightInWorld = Math.max(framing.height, framing.minWidth / aspect);
      device.queue.writeBuffer(camera, 0, new Float32Array([
        2 / (heightInWorld * aspect), 2 / heightInWorld, framing.centerY, 0,
      ]));
      depth?.texture.destroy();
      const texture = device.createTexture({
        label: 'Scene depth',
        size: { width, height },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      depth = { texture, view: texture.createView() };
      return depth.view;
    },
    draw(pass: GPURenderPassEncoder, particleCount: number): void {
      pass.setPipeline(floorPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6);
      pass.setPipeline(particlePipeline);
      pass.draw(6, particleCount);
    },
  };
}
