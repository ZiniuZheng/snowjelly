/**
 * 2D constitutive models for the CPU reference solver.
 *
 * Every model computes the first Piola-Kirchhoff stress P(F).
 * 2x2 matrices are stored column-major: [m00, m10, m01, m11].
 * F is read from F[o..o+3]; P is written to out[0..3].
 */
export type StressModel2D = (
  F: Float64Array,
  o: number,
  mu: number,
  lambda: number,
  out: Float64Array,
) => void;

/**
 * Compressible neo-Hookean:
 *   P = mu (F - F^{-T}) + lambda log(J) F^{-T}
 * Placeholder jelly model so the solver runs end to end.
 */
export const neoHookean: StressModel2D = (F, o, mu, lambda, out) => {
  const f00 = F[o], f10 = F[o + 1], f01 = F[o + 2], f11 = F[o + 3];
  // Guard against inverted / degenerate elements (log J undefined).
  const J = Math.max(f00 * f11 - f01 * f10, 1e-4);
  const invJ = 1 / J;
  // F^{-T} = (1/J) [[f11, -f10], [-f01, f00]]
  const it00 = f11 * invJ, it10 = -f01 * invJ, it01 = -f10 * invJ, it11 = f00 * invJ;
  const lj = lambda * Math.log(J);
  out[0] = mu * (f00 - it00) + lj * it00;
  out[1] = mu * (f10 - it10) + lj * it10;
  out[2] = mu * (f01 - it01) + lj * it01;
  out[3] = mu * (f11 - it11) + lj * it11;
};

/**
 * Fixed corotated (Stomakhin et al. 2012):
 *   P = 2 mu (F - R) + lambda (J - 1) J F^{-T}
 * with F = R S the polar decomposition. In 2D the rotation is closed form:
 *   theta = atan2(f10 - f01, f00 + f11),  R = [[cos, -sin], [sin, cos]].
 * lambda (J - 1) J F^{-T} = lambda (J - 1) cofactor(F), which stays finite
 * as J -> 0, so unlike neoHookean this needs no clamp.
 */
export const fixedCorotated: StressModel2D = (F, o, mu, lambda, out) => {
  const f00 = F[o], f10 = F[o + 1], f01 = F[o + 2], f11 = F[o + 3];
  const theta = Math.atan2(f10 - f01, f00 + f11);
  const c = Math.cos(theta), s = Math.sin(theta);
  // R in column-major order is [c, s, -s, c].
  const J = f00 * f11 - f01 * f10;
  const lj = lambda * (J - 1);
  const twoMu = 2 * mu;
  // cofactor(F) in storage order is [f11, -f01, -f10, f00].
  out[0] = twoMu * (f00 - c) + lj * f11;
  out[1] = twoMu * (f10 - s) - lj * f01;
  out[2] = twoMu * (f01 + s) - lj * f10;
  out[3] = twoMu * (f11 - c) + lj * f00;
};

/** Models selectable for jelly in the demo GUI. */
export const JELLY_MODELS: Record<string, StressModel2D> = {
  'neo-Hookean': neoHookean,
  'fixed corotated': fixedCorotated,
};

/** Lame parameters from Young's modulus E and Poisson ratio nu. */
export function lame(E: number, nu: number): { mu: number; lambda: number } {
  return {
    mu: E / (2 * (1 + nu)),
    lambda: (E * nu) / ((1 + nu) * (1 - 2 * nu)),
  };
}
