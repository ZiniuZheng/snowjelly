// Headless checks for the 3x3 SVD prototype: `node scripts/check-svd3.mjs`
// Verifies F = U diag(S) V^T, orthogonality and det(U) = det(V) = +1, the
// (-)-on-smallest convention, and accuracy on random / near-singular inputs.
import { createServer } from 'vite';

const server = await createServer({ configFile: 'vite.config.ts', server: { middlewareMode: true }, appType: 'custom' });
try {
  const { svd3 } = await server.ssrLoadModule('/src/sim3d/svd3.ts');

  const U = new Float64Array(9), S = new Float64Array(3), V = new Float64Array(9);
  const worst = { recon: 0, reconRel: 0, ortho: 0, det: 0 };
  let cases = 0, conventionFailures = 0;

  const det3 = (m) =>
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);

  const orthoErr = (m) => {
    let e = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let d = 0;
        for (let r = 0; r < 3; r++) d += m[3 * i + r] * m[3 * j + r];
        e = Math.max(e, Math.abs(d - (i === j ? 1 : 0)));
      }
    }
    return e;
  };

  function check(F, label) {
    svd3(F, U, S, V);
    cases++;
    let err = 0, scale = 1e-30;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        let m = 0;
        for (let k = 0; k < 3; k++) m += U[3 * k + r] * S[k] * V[3 * k + c];
        err = Math.max(err, Math.abs(m - F[3 * c + r]));
        scale = Math.max(scale, Math.abs(F[3 * c + r]));
      }
    }
    const rel = err / scale;
    worst.recon = Math.max(worst.recon, err);
    worst.reconRel = Math.max(worst.reconRel, rel);
    worst.ortho = Math.max(worst.ortho, orthoErr(U), orthoErr(V));
    worst.det = Math.max(worst.det, Math.abs(det3(U) - 1), Math.abs(det3(V) - 1));
    const detF = det3(F);
    const sorted = Math.abs(S[0]) + 1e-12 >= Math.abs(S[1]) && Math.abs(S[1]) + 1e-12 >= Math.abs(S[2]);
    const positive = S[0] >= -1e-12 && S[1] >= -1e-12;
    const signOk = Math.abs(detF) < 1e-12 || (detF > 0 ? S[2] >= -1e-12 : S[2] <= 1e-12);
    if (!sorted || !positive || !signOk) {
      conventionFailures++;
      if (conventionFailures <= 5) {
        console.log(`convention fail on ${label}: detF=${detF.toExponential(2)} S=[${S.join(', ')}]`);
      }
    }
    return { rel, err };
  }

  // Deterministic PRNG.
  let seed = 12345;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const rotation = () => {
    // Uniform-ish random rotation via quaternion.
    let q0 = gauss(), q1 = gauss(), q2 = gauss(), q3 = gauss();
    const n = Math.hypot(q0, q1, q2, q3);
    q0 /= n; q1 /= n; q2 /= n; q3 /= n;
    // Column-major rotation matrix.
    return new Float64Array([
      1 - 2 * (q1 * q1 + q2 * q2), 2 * (q0 * q1 + q2 * q3), 2 * (q0 * q2 - q1 * q3),
      2 * (q0 * q1 - q2 * q3), 1 - 2 * (q0 * q0 + q2 * q2), 2 * (q1 * q2 + q0 * q3),
      2 * (q0 * q2 + q1 * q3), 2 * (q1 * q2 - q0 * q3), 1 - 2 * (q0 * q0 + q1 * q1),
    ]);
  };

  // 1. random matrices at a few scales
  for (let i = 0; i < 2000; i++) {
    const scale = Math.pow(10, -3 + 6 * rand());
    const F = new Float64Array(9);
    for (let k = 0; k < 9; k++) F[k] = scale * gauss();
    check(F, `random #${i}`);
  }

  // 2. random rotations and rotated scalings (incl. tiny singular values)
  for (let i = 0; i < 500; i++) {
    const R0 = rotation(), R1 = rotation();
    const sv = [Math.pow(10, -3 * rand()), Math.pow(10, -6 * rand()), Math.pow(10, -6 * rand())];
    const F = new Float64Array(9);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        let m = 0;
        for (let k = 0; k < 3; k++) m += R0[3 * k + r] * sv[k] * R1[3 * k + c];
        F[3 * c + r] = m;
      }
    }
    check(F, `rotated scaling #${i}`);
  }

  // 3. pure rotations
  for (let i = 0; i < 200; i++) check(rotation(), `rotation #${i}`);

  // 4. negative determinant
  for (let i = 0; i < 500; i++) {
    const F = new Float64Array(9);
    for (let k = 0; k < 9; k++) F[k] = gauss();
    for (let r = 0; r < 3; r++) F[r] = -F[r]; // negate column 0
    check(F, `det<0 #${i}`);
  }

  // 5. rank-deficient / near-singular
  for (let i = 0; i < 500; i++) {
    const F = new Float64Array(9);
    for (let k = 0; k < 9; k++) F[k] = gauss();
    const eps = Math.pow(10, -12 * rand());
    for (let r = 0; r < 3; r++) F[6 + r] = F[r] + eps * F[3 + r];
    check(F, `near-singular #${i}`);
  }

  // 6. zero and identity
  check(new Float64Array(9), 'zero');
  check(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), 'identity');

  console.log(`cases: ${cases}`);
  console.log(`max reconstruction error: abs ${worst.recon.toExponential(2)}, rel ${worst.reconRel.toExponential(2)}`);
  console.log(`max orthogonality error: ${worst.ortho.toExponential(2)}`);
  console.log(`max |det - 1|: ${worst.det.toExponential(2)}`);
  console.log(`convention failures: ${conventionFailures}`);
  console.log(conventionFailures === 0 && worst.reconRel < 1e-10 ? 'PASS' : 'FAIL');
} finally {
  await server.close();
}
