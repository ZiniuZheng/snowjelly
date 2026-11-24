// Headless sanity checks for the 2D CPU solver: `node scripts/check-mpm2d.mjs`
// 1. Closed box, no gravity: mass and linear momentum must be conserved.
// 2. Default "Drop jelly" scene: stays finite and inside the domain; reports ms/substep.
import { createServer } from 'vite';

const server = await createServer({ configFile: 'vite.config.ts', server: { middlewareMode: true }, appType: 'custom' });
try {
  const { MPM2D } = await server.ssrLoadModule('/src/sim2d/mpm2d.ts');
  const { SCENES_2D } = await server.ssrLoadModule('/src/sim2d/scenes2d.ts');

  // --- 1. conservation ---
  {
    const sim = new MPM2D({ gravity: 0 });
    sim.addBox(0.4, 0.5, 0.2, 0.2, 1.0, 0.3, 0);
    sim.addBox(0.62, 0.5, 0.12, 0.12, -0.8, 0.1, 1);
    const d0 = sim.diagnostics();
    // Stop before anything reaches a wall (walls remove momentum by design).
    for (let s = 0; s < 1000; s++) sim.step();
    const d1 = sim.diagnostics();
    const rel = (a, b) => Math.abs(a - b) / Math.max(1e-12, Math.abs(b));
    console.log(`conservation: mass rel err ${rel(d1.mass, d0.mass).toExponential(2)}, ` +
      `px ${d0.px.toExponential(4)} -> ${d1.px.toExponential(4)} (rel ${rel(d1.px, d0.px).toExponential(2)}), ` +
      `py ${d0.py.toExponential(4)} -> ${d1.py.toExponential(4)} (rel ${rel(d1.py, d0.py).toExponential(2)})`);
  }

  // --- 2. stability on every demo scene ---
  for (const name of Object.keys(SCENES_2D)) {
    const sim = new MPM2D();
    SCENES_2D[name](sim);
    const t0 = performance.now();
    const steps = Math.round(3 / sim.params.dt); // 3 s of simulated time
    for (let s = 0; s < steps; s++) sim.step();
    const ms = (performance.now() - t0) / steps;
    let finite = true, minY = 1, maxSpeed = 0, minJ = Infinity, maxJ = -Infinity;
    for (let p = 0; p < sim.count; p++) {
      const x = sim.x[2 * p], y = sim.x[2 * p + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) finite = false;
      minY = Math.min(minY, y);
      maxSpeed = Math.max(maxSpeed, Math.hypot(sim.v[2 * p], sim.v[2 * p + 1]));
      const o = 4 * p;
      const J = sim.F[o] * sim.F[o + 3] - sim.F[o + 2] * sim.F[o + 1];
      minJ = Math.min(minJ, J); maxJ = Math.max(maxJ, J);
    }
    console.log(`${name}: ${sim.count} particles, t=${sim.time.toFixed(2)} s, finite=${finite}, ` +
      `minY=${minY.toFixed(4)}, maxSpeed=${maxSpeed.toFixed(3)}, J in [${minJ.toFixed(3)}, ${maxJ.toFixed(3)}], ` +
      `${ms.toFixed(3)} ms/substep`);
  }
} finally {
  await server.close();
}
