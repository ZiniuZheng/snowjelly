/**
 * 3x3 SVD in WGSL, matching app/src/sim3d/svd3.ts.
 *
 * Two-sided (Kogbetliantz) Jacobi: fixed sweeps over the index pairs, each
 * step diagonalizing the 2x2 block in closed form (atan2 + sin/cos, never
 * forming F^T F). See app/src/sim3d/svd3.ts for how this differs from
 * McAdams et al. 2011. Matrices are column-major mat3x3f.
 *
 * U and V are rotations (det = +1), S is sorted by magnitude with at most one
 * negative entry on the smallest singular value (iff det F < 0), and
 * F = U diag(S) V^T. Snow clamping and fixed corotated rely on this.
 */
export const SVD3_WGSL = /* wgsl */ `
struct SVD3 {
  U: mat3x3f,
  S: vec3f,
  V: mat3x3f,
};

const SVD3_IDENTITY: mat3x3f = mat3x3f(
  vec3f(1.0, 0.0, 0.0),
  vec3f(0.0, 1.0, 0.0),
  vec3f(0.0, 0.0, 1.0),
);

// WGSL leaves atan2(0, 0) undefined: a backend may return any value or NaN.
// It is hit on every already-diagonal block (e.g. F = I for particles at
// rest, where G = F = 0), so pin it to 0 like Math.atan2 does on the CPU.
fn svd3SafeAtan2(y: f32, x: f32) -> f32 {
  return select(atan2(y, x), 0.0, abs(x) + abs(y) < 1e-30);
}

// cos/sin of the left and right rotations diagonalizing a 2x2 block.
fn svd3Givens(a: f32, b: f32, c: f32, d: f32) -> vec4f {
  let E = 0.5 * (a + d);
  let F = 0.5 * (a - d);
  let G = 0.5 * (c + b);
  let H = 0.5 * (c - b);
  let a1 = svd3SafeAtan2(G, F);
  let a2 = svd3SafeAtan2(H, E);
  let th = 0.5 * (a1 + a2);
  let ph = 0.5 * (a1 - a2);
  return vec4f(cos(th), sin(th), cos(ph), sin(ph));
}

fn svd3RotateCols(m: ptr<function, mat3x3f>, p: u32, q: u32, c: f32, s: f32) {
  for (var r = 0u; r < 3u; r = r + 1u) {
    let mp = (*m)[p][r];
    let mq = (*m)[q][r];
    (*m)[p][r] = c * mp + s * mq;
    (*m)[q][r] = -s * mp + c * mq;
  }
}

fn svd3Jacobi(A: ptr<function, mat3x3f>, U: ptr<function, mat3x3f>, V: ptr<function, mat3x3f>, p: u32, q: u32) {
  let r = svd3Givens((*A)[p][p], (*A)[q][p], (*A)[p][q], (*A)[q][q]);
  let cu = r.x;
  let su = r.y;
  let cv = r.z;
  let sv = r.w;
  // A <- R^T A: rotate rows p and q.
  for (var k = 0u; k < 3u; k = k + 1u) {
    let ap = (*A)[k][p];
    let aq = (*A)[k][q];
    (*A)[k][p] = cu * ap + su * aq;
    (*A)[k][q] = -su * ap + cu * aq;
  }
  // A <- A C: rotate columns p and q.
  for (var k = 0u; k < 3u; k = k + 1u) {
    let ap = (*A)[p][k];
    let aq = (*A)[q][k];
    (*A)[p][k] = cv * ap + sv * aq;
    (*A)[q][k] = -sv * ap + cv * aq;
  }
  svd3RotateCols(U, p, q, cu, su);
  svd3RotateCols(V, p, q, cv, sv);
}

fn svd3SwapCols(m: ptr<function, mat3x3f>, i: u32, j: u32) {
  let t = (*m)[i];
  (*m)[i] = (*m)[j];
  (*m)[j] = t;
}

fn svd3NegCol(m: ptr<function, mat3x3f>, i: u32) {
  (*m)[i] = -(*m)[i];
}

// Sorts |S| descending and keeps at most one negative entry, on S.z.
fn svd3Canonicalize(U: ptr<function, mat3x3f>, V: ptr<function, mat3x3f>, S0: vec3f) -> vec3f {
  var S = S0;
  var parity = 1.0;
  if (abs(S.x) < abs(S.y)) {
    S = vec3f(S.y, S.x, S.z);
    svd3SwapCols(U, 0u, 1u);
    svd3SwapCols(V, 0u, 1u);
    parity = -parity;
  }
  if (abs(S.y) < abs(S.z)) {
    S = vec3f(S.x, S.z, S.y);
    svd3SwapCols(U, 1u, 2u);
    svd3SwapCols(V, 1u, 2u);
    parity = -parity;
  }
  if (abs(S.x) < abs(S.y)) {
    S = vec3f(S.y, S.x, S.z);
    svd3SwapCols(U, 0u, 1u);
    svd3SwapCols(V, 0u, 1u);
    parity = -parity;
  }
  // An odd permutation left det U = det V = -1. Flipping column 0 of both
  // restores det = +1 and leaves U diag(S) V^T unchanged.
  if (parity < 0.0) {
    svd3NegCol(U, 0u);
    svd3NegCol(V, 0u);
  }

  var negative = 0;
  if (S.x < 0.0) { negative = negative + 1; }
  if (S.y < 0.0) { negative = negative + 1; }
  if (S.z < 0.0) { negative = negative + 1; }
  if (negative == 2) {
    if (S.x < 0.0) { S.x = -S.x; svd3NegCol(U, 0u); }
    if (S.y < 0.0) { S.y = -S.y; svd3NegCol(U, 1u); }
    if (S.z < 0.0) { S.z = -S.z; svd3NegCol(U, 2u); }
  } else if (negative == 3) {
    S.x = -S.x;
    svd3NegCol(U, 0u);
    S.y = -S.y;
    svd3NegCol(U, 1u);
  } else if (negative == 1) {
    if (S.x < 0.0) {
      S.x = -S.x;
      svd3NegCol(U, 0u);
      S.z = -S.z;
      svd3NegCol(U, 2u);
    } else if (S.y < 0.0) {
      S.y = -S.y;
      svd3NegCol(U, 1u);
      S.z = -S.z;
      svd3NegCol(U, 2u);
    }
  }
  return S;
}

fn svd3(F: mat3x3f) -> SVD3 {
  var A = F;
  var U = SVD3_IDENTITY;
  var V = SVD3_IDENTITY;
  for (var sweep = 0u; sweep < 6u; sweep = sweep + 1u) {
    svd3Jacobi(&A, &U, &V, 0u, 1u);
    svd3Jacobi(&A, &U, &V, 0u, 2u);
    svd3Jacobi(&A, &U, &V, 1u, 2u);
  }
  let S = svd3Canonicalize(&U, &V, vec3f(A[0][0], A[1][1], A[2][2]));
  return SVD3(U, S, V);
}
`;
