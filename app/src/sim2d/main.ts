import GUI from 'lil-gui';
import '../common/style.css';
import { saveCanvasPNG } from '../common/screenshot';
import { MPM2D } from './mpm2d';
import { JELLY_MODELS } from './materials2d';
import { SCENES_2D } from './scenes2d';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const ctx = canvas.getContext('2d')!;
const SIZE = canvas.width; // square canvas, unit domain maps to SIZE px

const sim = new MPM2D();
const settings = {
  scene: 'Drop jelly',
  model: 'neo-Hookean',
  paused: false,
  stepsPerFrame: 80,
  colorBy: 'strain' as 'strain' | 'body' | 'speed',
  showGrid: false,
  dropBlock: () => dropBlock(),
  reset: () => reset(),
  savePNG: () => saveCanvasPNG(canvas, `cpu2d-${settings.scene.replace(/\s+/g, '-').toLowerCase()}`),
};

function reset(): void {
  sim.clear();
  SCENES_2D[settings.scene](sim);
}

let groupCounter = 10;
function dropBlock(): void {
  const w = 0.1 + Math.random() * 0.1;
  const cx = 0.2 + Math.random() * 0.6;
  if (Math.random() < 0.5) sim.addBox(cx, 0.82, w, w * (0.6 + Math.random() * 0.6), 0, -1, groupCounter++);
  else sim.addDisk(cx, 0.82, w * 0.5, 0, -1, groupCounter++);
}

// ---------------- GUI ----------------
const gui = new GUI({ title: 'Snowjelly · 2D CPU MLS-MPM' });
const fScene = gui.addFolder('Scene');
fScene.add(settings, 'scene', Object.keys(SCENES_2D)).onChange(reset);
fScene.add(settings, 'reset');
fScene.add(settings, 'dropBlock').name('drop a jelly');
const fSim = gui.addFolder('Simulation');
fSim.add(settings, 'paused');
fSim.add(settings, 'stepsPerFrame', 1, 300, 1).name('substeps / frame');
fSim.add(sim.params, 'dt', 2e-5, 3e-4, 1e-5).name('dt (s)');
fSim.add(sim.params, 'gravity', 0, 50, 0.1);
fSim.add(sim.params, 'floorFriction', 0, 1, 0.01).name('floor friction');
const fMat = gui.addFolder('Jelly');
fMat.add(settings, 'model', Object.keys(JELLY_MODELS)).onChange((k: string) => (sim.jellyModel = JELLY_MODELS[k]));
fMat.add(sim.params, 'E', 100, 20000, 10).name("Young's E");
fMat.add(sim.params, 'nu', 0, 0.45, 0.01).name('Poisson nu');
const fView = gui.addFolder('Display');
fView.add(settings, 'colorBy', ['strain', 'body', 'speed']).name('color by');
fView.add(settings, 'showGrid').name('show grid');
gui.add(settings, 'savePNG').name('save PNG');

// ---------------- mouse collider ----------------
let lastMouse = { x: 0, y: 0, t: 0 };
function toSim(e: PointerEvent): { x: number; y: number } {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: 1 - (e.clientY - r.top) / r.height };
}
canvas.addEventListener('pointerdown', (e) => {
  const p = toSim(e);
  Object.assign(sim.collider, { active: true, x: p.x, y: p.y, vx: 0, vy: 0 });
  lastMouse = { ...p, t: performance.now() };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove', (e) => {
  if (!sim.collider.active) return;
  const p = toSim(e);
  const now = performance.now();
  const dt = Math.max(1, now - lastMouse.t) / 1000;
  // Smooth the velocity a little so a jittery mouse does not inject huge impulses.
  sim.collider.vx = 0.5 * sim.collider.vx + 0.5 * ((p.x - lastMouse.x) / dt);
  sim.collider.vy = 0.5 * sim.collider.vy + 0.5 * ((p.y - lastMouse.y) / dt);
  sim.collider.x = p.x;
  sim.collider.y = p.y;
  lastMouse = { ...p, t: now };
});
const release = () => (sim.collider.active = false);
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);

// ---------------- rendering ----------------
const image = ctx.createImageData(SIZE, SIZE);
const pixels = new Uint32Array(image.data.buffer);
const BG = rgba(16, 22, 32);
const WALL = rgba(30, 40, 54);
const BODY_COLORS: [number, number, number][] = [
  [226, 80, 122], // jelly pink (proposal accent)
  [242, 140, 70], // orange
  [150, 96, 230], // violet
  [60, 175, 180], // teal
];

function rgba(r: number, g: number, b: number): number {
  return (255 << 24) | (b << 16) | (g << 8) | r; // little-endian ABGR
}

// Static background (domain walls), drawn once and copied every frame.
const background = new Uint32Array(SIZE * SIZE).fill(BG);
{
  const wallPx = Math.round(2 * sim.dx * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (x < wallPx || x >= SIZE - wallPx || y < wallPx || y >= SIZE - wallPx) background[y * SIZE + x] = WALL;
    }
  }
}

function draw(stepMs: number): void {
  pixels.set(background);

  const rad = Math.max(1, Math.round(sim.dx * 0.5 * SIZE * 0.5));
  for (let p = 0; p < sim.count; p++) {
    const cx = Math.round(sim.x[2 * p] * SIZE);
    const cy = Math.round((1 - sim.x[2 * p + 1]) * SIZE);
    const base = BODY_COLORS[sim.group[p] % BODY_COLORS.length];
    let k = 1;
    if (settings.colorBy === 'strain') {
      // Isochoric stretch measure: ||F||^2 / J - 2 (0 at rest, grows with shear).
      const o = 4 * p;
      const f00 = sim.F[o], f10 = sim.F[o + 1], f01 = sim.F[o + 2], f11 = sim.F[o + 3];
      const J = f00 * f11 - f01 * f10;
      const dev = (f00 * f00 + f10 * f10 + f01 * f01 + f11 * f11) / Math.max(J, 1e-3) - 2;
      const vol = Math.abs(J - 1);
      k = 0.55 + Math.min(1, 6 * dev + 4 * vol) * 0.9;
    } else if (settings.colorBy === 'speed') {
      const s = Math.hypot(sim.v[2 * p], sim.v[2 * p + 1]);
      k = 0.45 + Math.min(1, s / 3) * 1.0;
    }
    const r = Math.min(255, base[0] * k), g = Math.min(255, base[1] * k), b = Math.min(255, base[2] * k);
    const c = rgba(r | 0, g | 0, b | 0);
    for (let dy = -rad; dy <= rad; dy++) {
      const yy = cy + dy;
      if (yy < 0 || yy >= SIZE) continue;
      for (let dx = -rad; dx <= rad; dx++) {
        const xx = cx + dx;
        if (xx >= 0 && xx < SIZE) pixels[yy * SIZE + xx] = c;
      }
    }
  }
  ctx.putImageData(image, 0, 0);

  if (settings.showGrid) {
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= sim.N; i += 4) {
      const t = Math.round(i * sim.dx * SIZE) + 0.5;
      ctx.moveTo(t, 0); ctx.lineTo(t, SIZE);
      ctx.moveTo(0, t); ctx.lineTo(SIZE, t);
    }
    ctx.stroke();
  }

  if (sim.collider.active) {
    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sim.collider.x * SIZE, (1 - sim.collider.y) * SIZE, sim.collider.r * SIZE, 0, 2 * Math.PI);
    ctx.stroke();
  }

  // Caption baked into the canvas so screenshots are self-describing.
  ctx.fillStyle = 'rgba(235,240,245,0.92)';
  ctx.font = '600 15px system-ui, sans-serif';
  ctx.fillText('Snowjelly · 2D CPU MLS-MPM reference · jelly', 14, 26);
  ctx.font = '13px ui-monospace, Consolas, monospace';
  ctx.fillStyle = 'rgba(235,240,245,0.75)';
  const lines = [
    `${sim.count} particles · ${sim.N}x${sim.N} grid · model: ${settings.model}`,
    `E=${sim.params.E}  nu=${sim.params.nu}  dt=${sim.params.dt.toExponential(1)} s · ${settings.stepsPerFrame} substeps/frame`,
    `${stepMs.toFixed(2)} ms/substep (CPU) · sim t=${sim.time.toFixed(2)} s · ${fps.toFixed(0)} fps`,
  ];
  lines.forEach((l, i) => ctx.fillText(l, 14, 46 + i * 17));
}

// ---------------- main loop ----------------
let fps = 60;
let stepMsAvg = 0;
let lastFrame = performance.now();
function frame(now: number): void {
  const elapsed = now - lastFrame;
  lastFrame = now;
  fps = 0.95 * fps + 0.05 * (1000 / Math.max(1, elapsed));

  if (!settings.paused) {
    const t0 = performance.now();
    for (let s = 0; s < settings.stepsPerFrame; s++) sim.step();
    const ms = (performance.now() - t0) / settings.stepsPerFrame;
    stepMsAvg = stepMsAvg === 0 ? ms : 0.9 * stepMsAvg + 0.1 * ms;
  }
  draw(stepMsAvg);
  requestAnimationFrame(frame);
}

reset();
requestAnimationFrame(frame);

// Debug handle for the console.
Object.assign(window, { sim, settings });
