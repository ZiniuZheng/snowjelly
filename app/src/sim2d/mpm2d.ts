/**
 * 2D CPU reference implementation of MLS-MPM (Hu et al. 2018) with APIC
 * transfers and quadratic B-spline weights. Used to validate the math
 * before the 3D WebGPU port, and as the "CPU baseline" in our evaluation.
 *
 * Field names and conventions mirror the GPU particle format
 * (app/src/common/particleFormat.ts): unit-square domain, y up,
 * column-major matrices, material ids from MATERIAL.
 */
import {
  MATERIAL,
  createParticleData,
  writeParticle,
  type MaterialId,
  type ParticleData,
} from '../common/particleFormat';
import { lame, neoHookean, type StressModel2D } from './materials2d';

export interface SimParams2D {
  /** Grid cells per axis. dx = 1 / gridRes. */
  gridRes: number;
  /** Substep size in seconds. */
  dt: number;
  /** Gravity magnitude, pointing -y. */
  gravity: number;
  /** Jelly Young's modulus and Poisson ratio. */
  E: number;
  nu: number;
  /** Rest density (mass per area). */
  rho: number;
  /** Coulomb friction coefficient on the floor (0 = frictionless slip). */
  floorFriction: number;
  maxParticles: number;
}

export const DEFAULT_PARAMS_2D: SimParams2D = {
  gridRes: 96,
  // E = 500 gives a visibly soft jelly; elastic wave speed ~26 m/s, so
  // dt = 2e-4 stays under the dx / c ~ 4e-4 CFL limit at gridRes 96.
  dt: 2e-4,
  gravity: 9.8,
  E: 500,
  nu: 0.3,
  rho: 1,
  floorFriction: 0.4,
  maxParticles: 40000,
};

/** Circular kinematic collider, e.g. driven by the mouse. */
export interface Collider2D {
  active: boolean;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

/** Number of boundary cells on each side where outward velocity is removed. */
const BOUNDARY_CELLS = 3;

export class MPM2D {
  readonly params: SimParams2D;
  readonly N: number;
  readonly dx: number;
  readonly invDx: number;
  jellyModel: StressModel2D = neoHookean;
  readonly collider: Collider2D = { active: false, x: 0.5, y: 0.5, vx: 0, vy: 0, r: 0.06 };

  count = 0;
  time = 0;
  // Particle state (struct-of-arrays). 2x2 matrices are column-major.
  readonly x: Float64Array;
  readonly v: Float64Array;
  readonly C: Float64Array;
  readonly F: Float64Array;
  readonly Jp: Float64Array;
  readonly mass: Float64Array;
  readonly vol0: Float64Array;
  readonly material: Uint32Array;
  /** Which seeded body a particle came from (for coloring only). */
  readonly group: Uint16Array;

  // Grid: (N+1)^2 nodes, momentum then velocity in gridV, mass in gridM.
  readonly G: number;
  readonly gridV: Float64Array;
  readonly gridM: Float64Array;

  private readonly P = new Float64Array(4);

  constructor(params: Partial<SimParams2D> = {}) {
    this.params = { ...DEFAULT_PARAMS_2D, ...params };
    this.N = this.params.gridRes;
    this.dx = 1 / this.N;
    this.invDx = this.N;
    const n = this.params.maxParticles;
    this.x = new Float64Array(2 * n);
    this.v = new Float64Array(2 * n);
    this.C = new Float64Array(4 * n);
    this.F = new Float64Array(4 * n);
    this.Jp = new Float64Array(n);
    this.mass = new Float64Array(n);
    this.vol0 = new Float64Array(n);
    this.material = new Uint32Array(n);
    this.group = new Uint16Array(n);
    this.G = this.N + 1;
    this.gridV = new Float64Array(2 * this.G * this.G);
    this.gridM = new Float64Array(this.G * this.G);
  }

  clear(): void {
    this.count = 0;
    this.time = 0;
  }

  addParticle(px: number, py: number, vx: number, vy: number, material: MaterialId, group: number): boolean {
    if (this.count >= this.params.maxParticles) return false;
    const p = this.count++;
    const vol = (this.dx * 0.5) ** 2;
    this.x[2 * p] = px;
    this.x[2 * p + 1] = py;
    this.v[2 * p] = vx;
    this.v[2 * p + 1] = vy;
    this.C.fill(0, 4 * p, 4 * p + 4);
    this.F[4 * p] = 1;
    this.F[4 * p + 1] = 0;
    this.F[4 * p + 2] = 0;
    this.F[4 * p + 3] = 1;
    this.Jp[p] = 1;
    this.vol0[p] = vol;
    this.mass[p] = vol * this.params.rho;
    this.material[p] = material;
    this.group[p] = group;
    return true;
  }

  /** Seeds a box with 2 particles per cell per axis (spacing dx/2). */
  addBox(cx: number, cy: number, w: number, h: number, vx: number, vy: number, group: number,
         material: MaterialId = MATERIAL.JELLY): void {
    const s = this.dx * 0.5;
    const nx = Math.max(1, Math.round(w / s));
    const ny = Math.max(1, Math.round(h / s));
    const x0 = cx - (nx * s) / 2 + s / 2;
    const y0 = cy - (ny * s) / 2 + s / 2;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) this.addParticle(x0 + i * s, y0 + j * s, vx, vy, material, group);
    }
  }

  addDisk(cx: number, cy: number, r: number, vx: number, vy: number, group: number,
          material: MaterialId = MATERIAL.JELLY): void {
    const s = this.dx * 0.5;
    const n = Math.ceil(r / s);
    for (let j = -n; j <= n; j++) {
      for (let i = -n; i <= n; i++) {
        const px = (i + 0.5) * s, py = (j + 0.5) * s;
        if (px * px + py * py <= r * r) this.addParticle(cx + px, cy + py, vx, vy, material, group);
      }
    }
  }

  /** Advances one substep of size params.dt: P2G, grid update, G2P. */
  step(): void {
    const { N, dx, invDx, G, gridV: gv, gridM: gm, x, v, C, F, P } = this;
    const { dt, gravity, E, nu, floorFriction } = this.params;
    const { mu, lambda } = lame(E, nu);
    const n = this.count;
    gv.fill(0);
    gm.fill(0);

    // ---- P2G: scatter mass, APIC momentum and MLS stress to the grid ----
    const stressScale = -dt * 4 * invDx * invDx;
    for (let p = 0; p < n; p++) {
      const gx = x[2 * p] * invDx, gy = x[2 * p + 1] * invDx;
      const bx = Math.floor(gx - 0.5), by = Math.floor(gy - 0.5);
      const fx = gx - bx, fy = gy - by;
      const wx0 = 0.5 * (1.5 - fx) ** 2, wx1 = 0.75 - (fx - 1) ** 2, wx2 = 0.5 * (fx - 0.5) ** 2;
      const wy0 = 0.5 * (1.5 - fy) ** 2, wy1 = 0.75 - (fy - 1) ** 2, wy2 = 0.5 * (fy - 0.5) ** 2;

      const o = 4 * p;
      // Jelly only for now; snow (plasticity) plugs in here later.
      this.jellyModel(F, o, mu, lambda, P);
      const f00 = F[o], f10 = F[o + 1], f01 = F[o + 2], f11 = F[o + 3];
      const p00 = P[0], p10 = P[1], p01 = P[2], p11 = P[3];
      // Kirchhoff stress tau = P F^T
      const t00 = p00 * f00 + p01 * f01;
      const t01 = p00 * f10 + p01 * f11;
      const t10 = p10 * f00 + p11 * f01;
      const t11 = p10 * f10 + p11 * f11;
      const s = stressScale * this.vol0[p];
      const m = this.mass[p];
      // affine = -dt * vol * 4/dx^2 * tau + m * C
      const a00 = s * t00 + m * C[o];
      const a10 = s * t10 + m * C[o + 1];
      const a01 = s * t01 + m * C[o + 2];
      const a11 = s * t11 + m * C[o + 3];
      const mvx = m * v[2 * p], mvy = m * v[2 * p + 1];

      for (let i = 0; i < 3; i++) {
        const wxi = i === 0 ? wx0 : i === 1 ? wx1 : wx2;
        const dxp = (i - fx) * dx;
        for (let j = 0; j < 3; j++) {
          const w = wxi * (j === 0 ? wy0 : j === 1 ? wy1 : wy2);
          const dyp = (j - fy) * dx;
          const idx = bx + i + (by + j) * G;
          gv[2 * idx] += w * (mvx + a00 * dxp + a01 * dyp);
          gv[2 * idx + 1] += w * (mvy + a10 * dxp + a11 * dyp);
          gm[idx] += w * m;
        }
      }
    }

    // ---- Grid update: momentum -> velocity, gravity, colliders, walls ----
    const col = this.collider;
    for (let j = 0; j < G; j++) {
      for (let i = 0; i < G; i++) {
        const idx = i + j * G;
        const m = gm[idx];
        if (m <= 0) continue;
        let vx = gv[2 * idx] / m;
        let vy = gv[2 * idx + 1] / m - dt * gravity;

        if (col.active) {
          const ox = i * dx - col.x, oy = j * dx - col.y;
          const d2 = ox * ox + oy * oy;
          if (d2 < col.r * col.r && d2 > 1e-12) {
            const d = Math.sqrt(d2);
            const nx = ox / d, ny = oy / d;
            const rvx = vx - col.vx, rvy = vy - col.vy;
            const vn = rvx * nx + rvy * ny;
            if (vn < 0) {
              vx = rvx - vn * nx + col.vx;
              vy = rvy - vn * ny + col.vy;
            }
          }
        }

        if (i < BOUNDARY_CELLS && vx < 0) vx = 0;
        if (i > N - BOUNDARY_CELLS && vx > 0) vx = 0;
        if (j > N - BOUNDARY_CELLS && vy > 0) vy = 0;
        if (j < BOUNDARY_CELLS && vy < 0) {
          // Coulomb friction on the floor: tangential speed drops by mu * |vn|.
          const vt = Math.abs(vx);
          const reduced = Math.max(0, vt + floorFriction * vy); // vy < 0
          vx = vt > 0 ? (vx / vt) * reduced : 0;
          vy = 0;
        }
        gv[2 * idx] = vx;
        gv[2 * idx + 1] = vy;
      }
    }

    // ---- G2P: gather velocity and APIC C, advect, update F ----
    const cScale = 4 * invDx * invDx;
    const lo = 2 * dx, hi = 1 - 2 * dx;
    for (let p = 0; p < n; p++) {
      const gx = x[2 * p] * invDx, gy = x[2 * p + 1] * invDx;
      const bx = Math.floor(gx - 0.5), by = Math.floor(gy - 0.5);
      const fx = gx - bx, fy = gy - by;
      const wx0 = 0.5 * (1.5 - fx) ** 2, wx1 = 0.75 - (fx - 1) ** 2, wx2 = 0.5 * (fx - 0.5) ** 2;
      const wy0 = 0.5 * (1.5 - fy) ** 2, wy1 = 0.75 - (fy - 1) ** 2, wy2 = 0.5 * (fy - 0.5) ** 2;

      let nvx = 0, nvy = 0, b00 = 0, b01 = 0, b10 = 0, b11 = 0;
      for (let i = 0; i < 3; i++) {
        const wxi = i === 0 ? wx0 : i === 1 ? wx1 : wx2;
        const dxp = (i - fx) * dx;
        for (let j = 0; j < 3; j++) {
          const w = wxi * (j === 0 ? wy0 : j === 1 ? wy1 : wy2);
          const dyp = (j - fy) * dx;
          const idx = bx + i + (by + j) * G;
          const gvx = gv[2 * idx], gvy = gv[2 * idx + 1];
          nvx += w * gvx;
          nvy += w * gvy;
          b00 += w * gvx * dxp;
          b01 += w * gvx * dyp;
          b10 += w * gvy * dxp;
          b11 += w * gvy * dyp;
        }
      }
      const o = 4 * p;
      const c00 = cScale * b00, c10 = cScale * b10, c01 = cScale * b01, c11 = cScale * b11;
      C[o] = c00;
      C[o + 1] = c10;
      C[o + 2] = c01;
      C[o + 3] = c11;
      v[2 * p] = nvx;
      v[2 * p + 1] = nvy;
      x[2 * p] = Math.min(hi, Math.max(lo, x[2 * p] + dt * nvx));
      x[2 * p + 1] = Math.min(hi, Math.max(lo, x[2 * p + 1] + dt * nvy));

      // F <- (I + dt C) F
      const f00 = F[o], f10 = F[o + 1], f01 = F[o + 2], f11 = F[o + 3];
      F[o] = (1 + dt * c00) * f00 + dt * c01 * f10;
      F[o + 1] = dt * c10 * f00 + (1 + dt * c11) * f10;
      F[o + 2] = (1 + dt * c00) * f01 + dt * c01 * f11;
      F[o + 3] = dt * c10 * f01 + (1 + dt * c11) * f11;
    }

    this.time += dt;
  }

  /** Total mass and linear momentum, for conservation checks. */
  diagnostics(): { mass: number; px: number; py: number } {
    let mass = 0, px = 0, py = 0;
    for (let p = 0; p < this.count; p++) {
      const m = this.mass[p];
      mass += m;
      px += m * this.v[2 * p];
      py += m * this.v[2 * p + 1];
    }
    return { mass, px, py };
  }

  /**
   * Exports the current state in the shared 3D GPU particle format
   * (z = 0.5, F and C embedded in the xy block). Handy for feeding real
   * simulation data into the WebGPU renderer.
   */
  toParticleData(): ParticleData {
    const d = createParticleData(this.count);
    const F3 = new Float64Array(9);
    const C3 = new Float64Array(9);
    for (let p = 0; p < this.count; p++) {
      const o = 4 * p;
      F3.set([this.F[o], this.F[o + 1], 0, this.F[o + 2], this.F[o + 3], 0, 0, 0, 1]);
      C3.set([this.C[o], this.C[o + 1], 0, this.C[o + 2], this.C[o + 3], 0, 0, 0, 0]);
      writeParticle(d, p, {
        x: [this.x[2 * p], this.x[2 * p + 1], 0.5],
        v: [this.v[2 * p], this.v[2 * p + 1], 0],
        mass: this.mass[p],
        vol0: this.vol0[p],
        material: this.material[p] as MaterialId,
        F: F3,
        C: C3,
        Jp: this.Jp[p],
      });
    }
    return d;
  }
}
