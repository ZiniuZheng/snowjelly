/**
 * 3x3 SVD, CPU reference for the WebGPU solver.
 *
 * Two-sided Jacobi in the style of McAdams et al. 2011 ("Computing the SVD of
 * 3x3 matrices with minimal branching and elementary floating point
 * operations"): a fixed number of sweeps over the index pairs (0,1), (0,2),
 * (1,2), each step solving the 2x2 block in closed form and applying the
 * rotations to the full matrix. Unlike eigen-decomposing F^T F, this never
 * squares the condition number, which matters when snow clamps tiny singular
 * values.
 *
 * Convention, relied on by snow clamping and fixed corotated:
 *  - U and V are rotations (det = +1), column-major like everything else.
 *  - S is sorted by magnitude, |S0| >= |S1| >= |S2|, with S0, S1 >= 0.
 *  - If det F < 0 the single negative sign is carried by the smallest
 *    singular value S2; if det F > 0 all three are non-negative.
 *  - F = U diag(S) V^T.
 */
export const SVD3_SWEEPS = 6;

/**
 * Computes the SVD of F (9 floats, column-major) into U, S, V.
 * Writes 9, 3 and 9 floats. F may alias an output only if it is U, S or V
 * alone; it is copied to a scratch matrix first.
 */
export function svd3(
  F: ArrayLike<number>,
  U: Float64Array,
  S: Float64Array,
  V: Float64Array,
  sweeps: number = SVD3_SWEEPS,
): void {
  const A = new Float64Array(9);
  for (let i = 0; i < 9; i++) A[i] = F[i];
  identity3(U);
  identity3(V);

  for (let sweep = 0; sweep < sweeps; sweep++) {
    jacobi2(A, U, V, 0, 1);
    jacobi2(A, U, V, 0, 2);
    jacobi2(A, U, V, 1, 2);
  }

  S[0] = A[0];
  S[1] = A[4];
  S[2] = A[8];
  canonicalize(U, S, V);
}

/** One 2x2 SVD of the (p,q) block, applied to A = U^T F V. */
function jacobi2(A: Float64Array, U: Float64Array, V: Float64Array, p: number, q: number): void {
  // Block B = [[a, b], [c, d]] in row-major order.
  const a = A[3 * p + p], b = A[3 * q + p], c = A[3 * p + q], d = A[3 * q + q];
  // Closed-form 2x2 SVD: B = R diag(s0, s1) C^T.
  const E = 0.5 * (a + d), F = 0.5 * (a - d);
  const G = 0.5 * (c + b), H = 0.5 * (c - b);
  const a1 = Math.atan2(G, F), a2 = Math.atan2(H, E);
  const th = 0.5 * (a1 + a2), ph = 0.5 * (a1 - a2);
  const cu = Math.cos(th), su = Math.sin(th);
  const cv = Math.cos(ph), sv = Math.sin(ph);

  // A <- R^T A, rotating rows p and q.
  for (let k = 0; k < 3; k++) {
    const ap = A[3 * k + p], aq = A[3 * k + q];
    A[3 * k + p] = cu * ap + su * aq;
    A[3 * k + q] = -su * ap + cu * aq;
  }
  // A <- A C, rotating columns p and q.
  for (let r = 0; r < 3; r++) {
    const ap = A[3 * p + r], aq = A[3 * q + r];
    A[3 * p + r] = cv * ap + sv * aq;
    A[3 * q + r] = -sv * ap + cv * aq;
  }
  // U <- U R, V <- V C.
  rotateCols(U, p, q, cu, su);
  rotateCols(V, p, q, cv, sv);
}

/** Replaces columns p,q by (c * colp + s * colq, -s * colp + c * colq). */
function rotateCols(M: Float64Array, p: number, q: number, c: number, s: number): void {
  for (let r = 0; r < 3; r++) {
    const mp = M[3 * p + r], mq = M[3 * q + r];
    M[3 * p + r] = c * mp + s * mq;
    M[3 * q + r] = -s * mp + c * mq;
  }
}

function identity3(M: Float64Array): void {
  M.fill(0);
  M[0] = 1;
  M[4] = 1;
  M[8] = 1;
}

function swapCols(M: Float64Array, i: number, j: number): void {
  for (let r = 0; r < 3; r++) {
    const t = M[3 * i + r];
    M[3 * i + r] = M[3 * j + r];
    M[3 * j + r] = t;
  }
}

/**
 * Sorts the singular values by magnitude and puts the (at most one) negative
 * sign on the smallest one, keeping U and V rotations.
 */
function canonicalize(U: Float64Array, S: Float64Array, V: Float64Array): void {
  let parity = 1;
  // Sort columns by |S|, descending. Three compare-exchanges suffice.
  for (const [i, j] of [[0, 1], [1, 2], [0, 1]] as const) {
    if (Math.abs(S[i]) < Math.abs(S[j])) {
      const t = S[i]; S[i] = S[j]; S[j] = t;
      swapCols(U, i, j);
      swapCols(V, i, j);
      parity = -parity;
    }
  }
  // An odd permutation left det U = det V = -1; flipping a column of both
  // restores det = +1 and leaves U diag(S) V^T unchanged.
  if (parity < 0) {
    for (let r = 0; r < 3; r++) U[r] = -U[r];
    for (let r = 0; r < 3; r++) V[r] = -V[r];
  }

  // negCount is even when det F > 0. Remove negatives in pairs by flipping
  // the sign of S_i and column i of U (which keeps F = U diag(S) V^T and
  // det U = +1). One negative remains iff det F < 0; move it to S[2].
  const neg: number[] = [];
  for (let i = 0; i < 3; i++) if (S[i] < 0) neg.push(i);
  if (neg.length === 2) {
    for (const i of neg) flipSign(U, S, i);
  } else if (neg.length === 3) {
    flipSign(U, S, 0);
    flipSign(U, S, 1);
  } else if (neg.length === 1 && neg[0] !== 2) {
    flipSign(U, S, neg[0]);
    flipSign(U, S, 2);
  }
}

function flipSign(U: Float64Array, S: Float64Array, i: number): void {
  S[i] = -S[i];
  for (let r = 0; r < 3; r++) U[3 * i + r] = -U[3 * i + r];
}
