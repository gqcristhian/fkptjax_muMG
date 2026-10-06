"""JAX-accelerated implementation of k-functions calculation.

This module provides a JAX/JIT-compiled version of the calculate() function that is
compatible with calculate_numpy.py but uses jax.numpy for automatic differentiation
and GPU acceleration.

Key differences from calculate_numpy.py:
- Uses jax.numpy instead of numpy
- All operations are functional (no in-place modifications)
- Critical functions are JIT-compiled for performance
- Accepts numpy arrays as input, converts internally, returns numpy arrays
"""

# Enable 64-bit precision in JAX to match NumPy
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, Array
import numpy as np
from functools import partial
from typing import Dict, Any, Tuple

from fkptjax.types import KFunctionsInitData, KFunctionsOut, Float64NDArray, AbsCalculator


def precompute_spline_matrix_factors(x: Array) -> Tuple[Array, Array, Array, int]:
    """Precompute matrix factorization for cubic spline second derivatives.

    The tridiagonal system structure depends only on the x-grid (knot positions),
    not on the y-values. By precomputing these factors once during initialization,
    we avoid redundant computation in every evaluate() call.

    Args:
        x: Knot positions (1D array, shape: (n_knots,))

    Returns:
        Tuple of (sig, inv_h, inv_h_span, n) where:
        - sig: h[i-1] / h_span[i-1] for i=1..n-2 (shape: (n_knots-2,))
        - inv_h: 1/h for all intervals (shape: (n_knots-1,))
        - inv_h_span: 1/h_span for all spans (shape: (n_knots-2,))
        - n: number of knots (scalar)
    """
    n = len(x)
    h = jnp.diff(x)  # x[i+1] - x[i] for all i
    h_span = x[2:] - x[:-2]  # x[i+1] - x[i-1] for i=1..n-2

    # Precompute sig = h[i-1] / h_span[i-1] for i=1..n-2
    sig = h[:-1] / h_span

    # Precompute reciprocals to replace division with multiplication
    inv_h = 1.0 / h
    inv_h_span = 1.0 / h_span

    return sig, inv_h, inv_h_span, n


@partial(jit, static_argnums=(4,))
def calc_2nd_derivs_jax_optimized(y: Array, sig: Array, inv_h: Array, inv_h_span: Array, n: int) -> Array:
    """Optimized cubic spline second derivative computation using precomputed factors.

    This version uses precomputed matrix factors (sig, inv_h, inv_h_span) to avoid
    redundant computation of grid-dependent quantities. This is especially beneficial
    when called repeatedly with the same x-grid but different y-values (e.g., in MCMC).

    Args:
        y: Function values at knots (shape: (n_features, n_knots))
        sig: Precomputed h[i-1]/h_span[i-1] values (shape: (n_knots-2,))
        inv_h: Precomputed 1/h values (shape: (n_knots-1,))
        inv_h_span: Precomputed 1/h_span values (shape: (n_knots-2,))
        n: Number of knots (static argument)

    Returns:
        y2: Second derivatives (shape: (n_features, n_knots))
    """
    from jax import lax

    # Initialize arrays
    y2 = jnp.zeros_like(y)
    u = jnp.zeros_like(y)

    # Forward sweep using precomputed factors
    def forward_sweep_scan(carry: Tuple[Array, Array], i: Array) -> Tuple[Tuple[Array, Array], None]:
        y2, u = carry
        sig_i = sig[i-1]
        p = sig_i * y2[:, i-1] + 2.0
        y2_i = (sig_i - 1.0) / p

        # Use precomputed reciprocals (multiplication is faster than division)
        udiff = (y[:, i+1] - y[:, i]) * inv_h[i] - (y[:, i] - y[:, i-1]) * inv_h[i-1]
        u_i = (6.0 * udiff * inv_h_span[i-1] - sig_i * u[:, i-1]) / p

        # Update y2 and u at index i
        y2 = y2.at[:, i].set(y2_i)
        u = u.at[:, i].set(u_i)
        return (y2, u), None

    (y2, u), _ = lax.scan(forward_sweep_scan, (y2, u), jnp.arange(1, n-1))

    # Back substitution using lax.scan
    def back_sub_scan(y2: Array, k: Array) -> Tuple[Array, None]:
        # k iterates from n-2 down to 0
        y2_k = y2[:, k] * y2[:, k+1] + u[:, k]
        return y2.at[:, k].set(y2_k), None

    y2, _ = lax.scan(back_sub_scan, y2, jnp.arange(n-2, -1, -1))

    return y2


def init_cubic_spline_jax(xa: Array, x: Array) -> Dict[str, Any]:
    """Pre-compute cubic spline interpolation coefficients for given xa and x.

    Args:
        xa: Knot positions (1D array, must be increasing)
        x: Evaluation points (any shape)

    Returns:
        Dictionary containing precomputed coefficients needed for evaluation
    """
    # Remember original shape as a concrete Python tuple (not a traced value)
    x_shape = tuple(int(s) for s in x.shape)
    x_flat = x.ravel()

    # Get flat indices
    idx_hi_flat = jnp.searchsorted(xa, x_flat, side='right')
    idx_hi_flat = jnp.clip(idx_hi_flat, 1, xa.size - 1)
    idx_lo_flat = idx_hi_flat - 1

    # Compute interpolation coefficients
    h_flat = xa[idx_hi_flat] - xa[idx_lo_flat]
    a_flat = (xa[idx_hi_flat] - x_flat) / h_flat
    b_flat = (x_flat - xa[idx_lo_flat]) / h_flat

    # Power operations
    a2_flat = a_flat * a_flat
    a3_flat = a2_flat * a_flat
    b2_flat = b_flat * b_flat
    b3_flat = b2_flat * b_flat
    h2_flat = h_flat * h_flat

    return {
        'x_shape': x_shape,
        'idx_lo_flat': idx_lo_flat,
        'idx_hi_flat': idx_hi_flat,
        'a_flat': a_flat,
        'b_flat': b_flat,
        'a3_flat': a3_flat,
        'b3_flat': b3_flat,
        'h2_flat': h2_flat
    }


def eval_cubic_spline_jax(ya: Array, y2a: Array, x_shape: Tuple[int, ...], idx_lo_flat: Array, idx_hi_flat: Array, a_flat: Array, b_flat: Array, a3_flat: Array, b3_flat: Array, h2_flat: Array) -> Array:
    """JAX-compatible cubic spline evaluation using precomputed coefficients.

    Args:
        ya: Function values at knots (shape: (n_features, len(xa)))
        y2a: Second derivatives at knots (shape: (n_features, len(xa)))
        x_shape: Original shape of evaluation points (as Python tuple)
        idx_lo_flat, idx_hi_flat: Precomputed indices
        a_flat, b_flat, a3_flat, b3_flat, h2_flat: Precomputed interpolation coefficients

    Returns:
        Interpolated values (shape: (*x_shape, n_features))
    """
    # Index into ya and y2a (ya has shape (n_features, n_knots))
    # We want result shape (n_features, n_eval_points)
    ya_lo = ya[:, idx_lo_flat]  # (n_features, n_eval_points)
    ya_hi = ya[:, idx_hi_flat]
    y2a_lo = y2a[:, idx_lo_flat]
    y2a_hi = y2a[:, idx_hi_flat]

    # Compute interpolated values (broadcasting a_flat etc to match)
    result_flat = (
        a_flat[None, :] * ya_lo + b_flat[None, :] * ya_hi +
        ((a3_flat[None, :] - a_flat[None, :]) * y2a_lo +
         (b3_flat[None, :] - b_flat[None, :]) * y2a_hi) * h2_flat[None, :] / 6.0
    )

    # Reshape to (*x_shape, n_features)
    n_features = ya.shape[0]
    result = result_flat.T.reshape(*x_shape, n_features)

    return result


@partial(jit, static_argnums=(4, 5, 13, 21, 68))
def _calculate_jax_core(
        Y: Array,
        # Spline matrix factors for computing Y2 (k_in grid structure)
        sig: Array, inv_h: Array, inv_h_span: Array, n: int,
        # Spline coefficients for logk_grid
        spline_logk_shape: Tuple[int, ...], spline_logk_idx_lo: Array, spline_logk_idx_hi: Array,
        spline_logk_a: Array, spline_logk_b: Array, spline_logk_a3: Array, spline_logk_b3: Array, spline_logk_h2: Array,
        # Spline coefficients for kk_grid
        spline_kk_shape: Tuple[int, ...], spline_kk_idx_lo: Array, spline_kk_idx_hi: Array,
        spline_kk_a: Array, spline_kk_b: Array, spline_kk_a3: Array, spline_kk_b3: Array, spline_kk_h2: Array,
        # Spline coefficients for y
        spline_y_shape: Tuple[int, ...], spline_y_idx_lo: Array, spline_y_idx_hi: Array,
        spline_y_a: Array, spline_y_b: Array, spline_y_a3: Array, spline_y_b3: Array, spline_y_h2: Array,
        logk_grid2: Array, dkk: Array, dkk_reshaped: Array, scale_Q: Array,
        r: Array, r2: Array, x: Array, w: Array, x2: Array, y2: Array, y: Array,
        r_r: Array, r2_r: Array, x_r: Array, w_r: Array, x2_r: Array, y2_r: Array, AngleEvR: Array, AngleEvR2: Array, dkk_r: Array,
        kk_grid: Array, logk_grid: Array,
        A: float, ApOverf0: float, CFD3: float, CFD3p: float, sigma2v: float,
        A_Q: Array, B_Q: Array, ApOverf0_Q: Array, BpOverf0_Q: Array,
        A_fused_R: Array = None, ApOverf0_fused_R: Array = None,
        CFD3_R: Array = None, CFD3p_R: Array = None,
        A_N: Array = None, B_N: Array = None,
        ApOverf0_N: Array = None, BpOverf0_N: Array = None,
        # Spline coefficients for interpolating P(k), f(k) at the R-grid's
        # "kminus" points -- needed unconditionally by Gamma2fevR (P13-loop)
        # and by the I1udd1-family's N-ordering piece.
        spline_y_r_shape: Tuple[int, ...] = None, spline_y_r_idx_lo: Array = None,
        spline_y_r_idx_hi: Array = None, spline_y_r_a: Array = None,
        spline_y_r_b: Array = None, spline_y_r_a3: Array = None,
        spline_y_r_b3: Array = None, spline_y_r_h2: Array = None,
        # "R ordering", evaluated on the Q-loop's clipped grid: grid-shaped
        # overrides for F2evR_Q/G2evR_Q, from
        # MG_kernels.A_B_grid(kminus_Q, k_ext, q_Q, ...). None (default) ->
        # broadcasts the scalar A/ApOverf0 into this ordering too, exactly
        # like A_Q's own default.
        A_RQ: Array = None, B_RQ: Array = None,
        ApOverf0_RQ: Array = None, BpOverf0_RQ: Array = None,
    ) -> Any:
    """Core JAX calculation - all arrays are JAX arrays.

    This is the JIT-compiled inner function that operates entirely on JAX arrays.
    All precomputed quantities are passed as parameters.

    This function now computes Y2 (spline second derivatives) internally using
    precomputed matrix factors, allowing XLA to fuse this computation with
    subsequent interpolations for better performance.
    """

    # Compute spline second derivatives using precomputed matrix factors
    # This computation is now inside the JIT boundary, allowing XLA to optimize
    # across the entire calculation and eliminate intermediate GPU synchronization
    Y2 = calc_2nd_derivs_jax_optimized(Y, sig, inv_h, inv_h_span, n)

    # Interpolate onto output grid using precomputed coefficients
    Pout, Pout_nw, fout = eval_cubic_spline_jax(
        Y, Y2, spline_logk_shape, spline_logk_idx_lo, spline_logk_idx_hi,
        spline_logk_a, spline_logk_b, spline_logk_a3, spline_logk_b3, spline_logk_h2
    ).T

    # f(k_ext) and P(k_ext) (wiggle/no-wiggle stacked) -- needed early by the
    # I1udd1-family's genuine three-ordering (Q+R+N) combination below, which
    # (unlike F2evQ/G2evQ) needs the EXTERNAL leg's own growth rate and power
    # spectrum, not just the loop-q/kminus ones.
    fk = fout
    Pout_stack = jnp.stack([Pout, Pout_nw], axis=0)  # shape (2, Nk)

    # Interpolate onto quadrature grid using precomputed coefficients
    Pkk, Pkk_nw, fkk = eval_cubic_spline_jax(
        Y, Y2, spline_kk_shape, spline_kk_idx_lo, spline_kk_idx_hi,
        spline_kk_a, spline_kk_b, spline_kk_a3, spline_kk_b3, spline_kk_h2
    ).T

    # ============================================================================
    # Q-FUNCTIONS: Vectorized over ALL dimensions
    # ============================================================================

    # Use precomputed Q-function quantities
    # (r, r2, x, w, x2, y2, y are passed as parameters)

    # Loop over quadrature k values (line 378 in C)
    fp = fkk[1:].reshape(-1, 1, 1)

    # Interpolate power spectra at (ki * y) points using precomputed coefficients
    interp_result = eval_cubic_spline_jax(
        Y, Y2, spline_y_shape, spline_y_idx_lo, spline_y_idx_hi,
        spline_y_a, spline_y_b, spline_y_a3, spline_y_b3, spline_y_h2
    )
    psl_w, psl_nw, fkmp = interp_result.T  # Unpack gives (120, 1, 299, 10) each

    # Transpose back to (NQ, nquadSteps-1, 1, Nk)
    psl_w = psl_w.T  # (10, 299, 1, 120)
    psl_nw = psl_nw.T  # (10, 299, 1, 120)
    fkmp = fkmp.T  # (10, 299, 1, 120) - already has the correct shape!

    # Concatenate wiggle and no-wiggle components (matches numpy version)
    psl = jnp.concatenate([psl_w, psl_nw], axis=2)  # shape (NQ, nquadSteps-1, 2, Nk) -> (10, 299, 2, 120)
    # fkmp already has shape (10, 299, 1, 120) which is what we need

    # Compute SPT kernels F2evQ and G2evQ
    AngleEvQ = (x - r) / y
    AngleEvQ2 = AngleEvQ * AngleEvQ
    fsum = fp + fkmp

    S2evQ = AngleEvQ ** 2 - 1.0/3.0
    # A_Q/B_Q/ApOverf0_Q/BpOverf0_Q generalize A/A/ApOverf0/ApOverf0: the fkPT
    # (large-scale) approximation sets caligraphic-A == caligraphic-B, so the
    # caller passes the same scalar A/ApOverf0 in both slots by default; the
    # full-kernel path (fkpt_approximation=False) passes genuinely different,
    # grid-shaped A(k,q) != B(k,q) here instead. See MG_kernels.A_B_grid.
    F2evQ = (1.0/2.0 + 3.0/14.0 * A_Q + (1.0/2.0 - 3.0/14.0 * B_Q) * AngleEvQ2 +
             AngleEvQ / 2.0 * (y/r + r/y))
    G2evQ = (3.0/14.0 * A_Q * fsum + 3.0/14.0 * ApOverf0_Q +
             (1.0/2.0 * fsum - 3.0/14.0 * B_Q * fsum - 3.0/14.0 * BpOverf0_Q) * AngleEvQ2 +
             AngleEvQ / 2.0 * (fkmp * y/r + fp * r/y))

    # Precompute some temporary expressions that are used multiple times
    wpsl = w * psl
    fkmpr2 = fkmp * r2
    rx = r * x
    y4 = y2 * y2

    # P22 kernels
    P22dd_B = jnp.sum(wpsl * (2.0 * r2 * F2evQ**2), axis=0)
    P22du_B = jnp.sum(wpsl * (2.0 * r2 * F2evQ * G2evQ), axis=0)
    P22uu_B = jnp.sum(wpsl * (2.0 * r2 * G2evQ**2), axis=0)

    # ========== 5 THREE-POINT CORRELATION FUNCTION KERNELS (Q-part) ==========

    # I1udd1tA
    I1udd1tA_B = jnp.sum(wpsl * (
        2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * F2evQ
        ), axis=0)

    # I2uud1tA
    I2uud1tA_B = jnp.sum(wpsl * (-fp * fkmpr2 * (1.0 - x2) / y2 * F2evQ), axis=0)

    # I2uud2tA
    I2uud2tA_B = jnp.sum(wpsl * (
        2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * G2evQ
        + fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * F2evQ
        ), axis=0)

    # I3uuu2tA
    I3uuu2tA_B = jnp.sum(wpsl * (fp * fkmpr2 * (x2 - 1.0) / y2 * G2evQ), axis=0)

    # I3uuu3tA
    I3uuu3tA_B = jnp.sum(wpsl * (
        fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * G2evQ
        ), axis=0)

    # ---- Genuine three-ordering (Q+R+N) combination -----------------------
    # sMGPT's own I1udd1-family (2_P22type.wl's computeOneKBothExact/
    # AKernelsT) sums THREE orderings -- Q ("tA", output leg k_ext, computed
    # above), R ("a", output leg kminus) and N ("A", output leg q) -- all
    # evaluated at this SAME (r,x) point on this SAME clipped Q-loop domain,
    # weighted as
    #   Kern_ij = A_ij(N)*P(k)*P(kminus) + tA_ij(Q)*P(q)*P(kminus) + a_ij(R)*P(k)*P(q)
    # before a SINGLE q-integral. This is now the ONLY route for the
    # I1udd1-family, for BOTH fkpt_approximation=True and False:
    # A_RQ/B_RQ/A_N/B_N default to the SAME scalar A/B used by A_Q (exactly
    # the pattern A_Q_eff already follows a few lines up) when the caller
    # doesn't supply grid-shaped arrays, so the squeezed path pays no extra
    # ODE-solve cost -- it just broadcasts the same scalar into all three
    # legs instead of running the old, separately-discretized "tA + 2a"
    # shortcut (R-ordering on the UNCLIPPED R-loop/P13-domain, see git
    # history for that formula). That old shortcut is an identity only in
    # the squeezed (scalar A=B everywhere) limit -- unifying the domain here
    # means fkpt_approximation=True/False now agree to solver tolerance for
    # any scale-independent model (previously only to a resolution-dependent
    # residual, since the two paths used different quadrature domains), and
    # fkpt_approximation=False is the genuine sMGPT-matching physics rather
    # than a second, undocumented approximation for scale-dependent models.
    A_RQ_eff = A if A_RQ is None else A_RQ
    B_RQ_eff = A if B_RQ is None else B_RQ
    ApOverf0_RQ_eff = ApOverf0 if ApOverf0_RQ is None else ApOverf0_RQ
    BpOverf0_RQ_eff = ApOverf0 if BpOverf0_RQ is None else BpOverf0_RQ
    A_N_eff = A if A_N is None else A_N
    B_N_eff = A if B_N is None else B_N
    ApOverf0_N_eff = ApOverf0 if ApOverf0_N is None else ApOverf0_N
    BpOverf0_N_eff = ApOverf0 if BpOverf0_N is None else BpOverf0_N

    # "R ordering" (a_ij), evaluated on the Q-loop's OWN (clipped) grid --
    # sMGPT's Ah[kminus,k_ext,q]: output leg kminus, input legs (k_ext,q).
    AngleEvR_Q = -x
    AngleEvR_Q2 = AngleEvR_Q * AngleEvR_Q
    F2evR_Q = (1.0/2.0 + 3.0/14.0 * A_RQ_eff + (1.0/2.0 - 3.0/14.0 * B_RQ_eff) * AngleEvR_Q2 +
               AngleEvR_Q / 2.0 * (1.0/r + r))
    G2evR_Q = (3.0/14.0 * A_RQ_eff * (fp + fk) + 3.0/14.0 * ApOverf0_RQ_eff +
               (1.0/2.0 * (fp + fk) - 3.0/14.0 * B_RQ_eff * (fp + fk) -
                3.0/14.0 * BpOverf0_RQ_eff) * AngleEvR_Q2 +
               AngleEvR_Q / 2.0 * (fk/r + fp*r))

    # "N ordering" (A_ij) -- sMGPT's Ah[q,k_ext,kminus]: output leg q,
    # input legs (k_ext,kminus).
    AngleEvN_Q = -((1.0 - rx) / y)
    AngleEvN_Q2 = AngleEvN_Q * AngleEvN_Q
    fksum = fk + fkmp
    F2evN_Q = (1.0/2.0 + 3.0/14.0 * A_N_eff + (1.0/2.0 - 3.0/14.0 * B_N_eff) * AngleEvN_Q2 +
               AngleEvN_Q / 2.0 * (y + 1.0/y))
    G2evN_Q = (3.0/14.0 * A_N_eff * fksum + 3.0/14.0 * ApOverf0_N_eff +
               (1.0/2.0 * fksum - 3.0/14.0 * B_N_eff * fksum - 3.0/14.0 * BpOverf0_N_eff) * AngleEvN_Q2 +
               AngleEvN_Q / 2.0 * (fkmp*y + fk/y))

    Pout_bcast = Pout_stack[None, None, :, :]  # -> (1,1,2,Nk), broadcasts vs (NQ,nquadSteps-1,{1,2},Nk)

    # a_ij (R-ordering): weight P(k_ext)*P(q) -- no P(kminus), so no
    # "psl" factor; P(q) is supplied later by trapsumQ's own PSLB.
    a11 = 2.0 * (F2evR_Q * rx * fp + G2evR_Q * r2 * (1.0 - rx) / y2)
    a12 = -(r2 * (1.0 - x2) / y2) * fp * G2evR_Q
    a22 = (((r2 * (1.0 - 3.0*x2) + 2.0*rx)/y2 * fp + (2.0*r2*(1.0 - rx))/y2 * fk) * G2evR_Q
           + 2.0*rx*fk*fp*F2evR_Q)
    a23 = r2 * (x2 - 1.0) / y2 * fp * fk * G2evR_Q
    a33 = (r2 * (1.0 - 3.0*x2) + 2.0*rx) / y2 * fp * fk * G2evR_Q

    # A_ij (N-ordering): weight P(k_ext)*P(kminus) -- carries NO P(q) at
    # all, unlike every other Q-loop kernel; integrated separately below
    # via trapsumQ_noPq (no PSLB multiply).
    A11 = 2.0 * (G2evN_Q * rx + F2evN_Q * r2 * (1.0 - rx) / y2 * fkmp)
    A12 = -(r2 * (1.0 - x2) / y2) * fkmp * G2evN_Q
    A22 = (((r2 * (1.0 - 3.0*x2) + 2.0*rx)/y2 * fkmp + 2.0*rx*fk) * G2evN_Q
           + (2.0*r2*(1.0 - rx))/y2 * fkmp*fk * F2evN_Q)
    A23 = r2 * (x2 - 1.0) / y2 * fkmp * fk * G2evN_Q
    A33 = (r2 * (1.0 - 3.0*x2) + 2.0*rx) / y2 * fkmp * fk * G2evN_Q

    # tA_ij (Q-ordering): the exact same raw kernels already summed into
    # I1udd1tA_B/.../I3uuu3tA_B above (weight P(q)*P(kminus), i.e. wpsl +
    # trapsumQ's own PSLB) -- repeated here, unsummed, so they can be
    # combined with a_ij/A_ij point-by-point before the single q-integral,
    # matching sMGPT's own KernI1udd1/.../KernI3uuu3 exactly.
    tA11 = 2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * F2evQ
    tA12 = -fp * fkmpr2 * (1.0 - x2) / y2 * F2evQ
    tA22 = (2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * G2evQ
            + fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * F2evQ)
    tA23 = fp * fkmpr2 * (x2 - 1.0) / y2 * G2evQ
    tA33 = fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * G2evQ

    # qr_B: the P(q)-dependent part (tA + a), goes through the ordinary
    # trapsumQ (below) which supplies P(q) automatically via PSLB.
    I1udd1_qrN_B = jnp.sum(wpsl * tA11 + w * a11 * Pout_bcast, axis=0)
    I2uud1_qrN_B = jnp.sum(wpsl * tA12 + w * a12 * Pout_bcast, axis=0)
    I2uud2_qrN_B = jnp.sum(wpsl * tA22 + w * a22 * Pout_bcast, axis=0)
    I3uuu2_qrN_B = jnp.sum(wpsl * tA23 + w * a23 * Pout_bcast, axis=0)
    I3uuu3_qrN_B = jnp.sum(wpsl * tA33 + w * a33 * Pout_bcast, axis=0)

    # n_B: the N-ordering part, carries NO P(q) at all -- integrated
    # separately via trapsumQ_noPq (below), which skips PSLB.
    I1udd1_n_B = jnp.sum(wpsl * A11 * Pout_bcast, axis=0)
    I2uud1_n_B = jnp.sum(wpsl * A12 * Pout_bcast, axis=0)
    I2uud2_n_B = jnp.sum(wpsl * A22 * Pout_bcast, axis=0)
    I3uuu2_n_B = jnp.sum(wpsl * A23 * Pout_bcast, axis=0)
    I3uuu3_n_B = jnp.sum(wpsl * A33 * Pout_bcast, axis=0)

    # ========== 7 BpC TERM KERNELS (Q-part, will become D-terms) ==========

    # I2uudd1BpC
    I2uudd1BpC_B = jnp.sum(wpsl * (
        1.0 / 4.0 * (1.0 - x2) * (fp * fp + fkmpr2 * fkmpr2 / y4)
        + fp * fkmpr2 * (-1.0 + x2) / y2 / 2.0
        ), axis=0)

    # I2uudd2BpC
    I2uudd2BpC_B = jnp.sum(wpsl * (
        (
            fp * fp * (-1.0 + 3.0 * x2)
            + 2.0 * fkmp * fp * r * (r + 2.0 * x - 3.0 * r * x2) / y2
            + fkmp * fkmpr2 * (2.0 - 4.0 * rx + r2 * (-1.0 + 3.0 * x2)) / y4
        )
        / 4.0
        ), axis=0)

    # I3uuud2BpC
    I3uuud2BpC_B = jnp.sum(wpsl * (
        -(
            fkmp * fp * (
                fkmp * (-2.0 + 3.0 * rx) * r2
                - fp * (-1.0 + 3.0 * rx) * (1.0 - 2.0 * rx + r2)
            )
            * (-1.0 + x2)
        )
        / (2.0 * y2 * y2)
        ), axis=0)

    # I3uuud3BpC
    I3uuud3BpC_B = jnp.sum(wpsl * (
        (
            fkmp * fp * (
                -(
                    fp
                    * (1.0 - 2.0 * rx + r2)
                    * (1.0 - 3.0 * x2 + rx * (-3.0 + 5.0 * x2))
                )
                + fkmp * r * (2.0 * x + r * (2.0 - 6.0 * x2 + rx * (-3.0 + 5.0 * x2)))
            )
        )
        / (2.0 * y4)
        ), axis=0)

    # I4uuuu2BpC
    I4uuuu2BpC_B = jnp.sum(wpsl * (
        3.0 * fkmp**2 * fp**2 * r2 * (-1.0 + x2) ** 2 / (16.0 * y4)
        ), axis=0)

    # I4uuuu3BpC
    I4uuuu3BpC_B = jnp.sum(wpsl * (
        -(
            fkmp**2 * fp**2 * (-1.0 + x2) * (2.0 + 3.0 * r * (-4.0 * x + r * (-1.0 + 5.0 * x2)))
        )
        / (8.0 * y2 * y2)
        ), axis=0)

    # I4uuuu4BpC
    I4uuuu4BpC_B = jnp.sum(wpsl * (
        (
            fkmp**2 * fp**2 * (
                -4.0
                + 8.0 * rx * (3.0 - 5.0 * x2)
                + 12.0 * x2
                + r2 * (3.0 - 30.0 * x2 + 35.0 * x2 ** 2)
            )
        )
        / (16.0 * y4)
        ), axis=0)

    # Left endpoints for power spectra
    PSLB = jnp.stack([Pkk[1:], Pkk_nw[1:]], axis=1)[:, :, None]  # shape (nQuadSteps-1, 2, 1)

    # Use precomputed dkk_reshaped and scale_Q

    # Bias terms
    Pb1b2_B = jnp.sum(wpsl * (r2 * F2evQ), axis=0)
    Pb1bs2_B = jnp.sum(wpsl * (r2 * F2evQ * S2evQ), axis=0)
    Pratio = PSLB / psl
    PratioInv = psl / PSLB
    Pb22_B = jnp.sum(wpsl * (
        1.0 / 2.0 * r2 * (1.0 / 2.0 * (1.0 - Pratio) + 1.0 / 2.0 * (1.0 - PratioInv))
        ), axis=0)
    Pb2s2_B = jnp.sum(wpsl * (
        1.0 / 2.0 * r2 * (
            1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * Pratio)
            + 1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * PratioInv)
        )
        ), axis=0)
    Ps22_B = jnp.sum(wpsl * (
        1.0 / 2.0 * r2
        * (
            1.0 / 2.0 * (S2evQ ** 2 - 4.0 / 9.0 * Pratio)
            + 1.0 / 2.0 * (S2evQ ** 2 - 4.0 / 9.0 * PratioInv)
        )
        ), axis=0)
    Pb2theta_B = jnp.sum(wpsl * (r2 * G2evQ), axis=0)
    Pbs2theta_B = jnp.sum(wpsl * (r2 * S2evQ * G2evQ), axis=0)

    # Apply trapezoidal rule (functional version for JAX)
    def trapsumQ(B: Array) -> Array:
        """Functional trapezoidal integration for Q-functions.

        Matches numpy version which computes:
        sum_i ( (B[i] + B[i-1]) * dk[i] ) * scale_Q * PSLB
        The first element uses dk[0] with B[0] only (no previous element).
        """
        B = B * scale_Q * PSLB
        # Compute trapezoidal sum: (B[i-1] + B[i]) * dk[i] for i >= 1, plus B[0] * dk[0]
        return jnp.sum((B[:-1] + B[1:]) * dkk_reshaped[1:], axis=0) + B[0] * dkk_reshaped[0]

    def trapsumQ_noPq(B: Array) -> Array:
        """Same q-integral as trapsumQ, WITHOUT the automatic P(q)=PSLB factor.

        Used only for the I1udd1-family's N-ordering piece, whose sMGPT
        weight (P(k_ext)*P(kminus)) carries no P(q) at all -- unlike every
        other Q-loop kernel here.
        """
        B = B * scale_Q
        return jnp.sum((B[:-1] + B[1:]) * dkk_reshaped[1:], axis=0) + B[0] * dkk_reshaped[0]

    P22dd = trapsumQ(P22dd_B)
    P22du = trapsumQ(P22du_B)
    P22uu = trapsumQ(P22uu_B)

    I1udd1tA = trapsumQ(I1udd1tA_B)
    I2uud1tA = trapsumQ(I2uud1tA_B)
    I2uud2tA = trapsumQ(I2uud2tA_B)
    I3uuu2tA = trapsumQ(I3uuu2tA_B)
    I3uuu3tA = trapsumQ(I3uuu3tA_B)

    I2uudd1BpC = trapsumQ(I2uudd1BpC_B)
    I2uudd2BpC = trapsumQ(I2uudd2BpC_B)
    I3uuud2BpC = trapsumQ(I3uuud2BpC_B)
    I3uuud3BpC = trapsumQ(I3uuud3BpC_B)
    I4uuuu2BpC = trapsumQ(I4uuuu2BpC_B)
    I4uuuu3BpC = trapsumQ(I4uuuu3BpC_B)
    I4uuuu4BpC = trapsumQ(I4uuuu4BpC_B)

    # Bias terms
    Pb1b2 = trapsumQ(Pb1b2_B)
    Pb1bs2 = trapsumQ(Pb1bs2_B)
    Pb22 = trapsumQ(Pb22_B)
    Pb2s2 = trapsumQ(Pb2s2_B)
    Ps22 = trapsumQ(Ps22_B)
    Pb2theta = trapsumQ(Pb2theta_B)
    Pbs2theta = trapsumQ(Pbs2theta_B)

    # ============================================================================
    # R-FUNCTIONS: Also fully vectorized
    # ============================================================================

    # f(k) at output k values already computed above (see "fk = fout" near the
    # top) -- needed early by the I1udd1-family combination.

    # Use precomputed R-function quantities
    # (r_r, r2_r, x_r, w_r, x2_r, y2_r, AngleEvR, AngleEvR2 are passed as parameters)

    # R-function uses fp from kk[1:-1] and psl from Pkk[1:-1]
    fp_r = fkk[1:-1].reshape(-1, 1, 1)
    psl_r = jnp.stack((Pkk[1:-1], Pkk_nw[1:-1]), axis=1)[:,:,None] # shape (nquadSteps-2, 2, 1)

    # P(kminus)/f(kminus) on the R grid -- needed unconditionally by Gamma2fevR
    # below (sMGPT's own formula uses the growth rate at the OUTPUT leg of that
    # sub-solve, i.e. at kminus, not at k_ext -- see the Gamma2fevR comment),
    # and reused by the N-ordering block further down when that's active.
    interp_result_rkm = eval_cubic_spline_jax(
        Y, Y2, spline_y_r_shape, spline_y_r_idx_lo, spline_y_r_idx_hi,
        spline_y_r_a, spline_y_r_b, spline_y_r_a3, spline_y_r_b3, spline_y_r_h2
    )
    psl_rkm_w, psl_rkm_nw, fkminusp_r = interp_result_rkm.T
    psl_rkm_w = psl_rkm_w.T
    psl_rkm_nw = psl_rkm_nw.T
    fkminusp_r = fkminusp_r.T
    psl_kminus_r = jnp.concatenate([psl_rkm_w, psl_rkm_nw], axis=2)

    # ========== R-part ==========
    wpsl_r = w_r * psl_r

    # Gamma2evR/Gamma2fevR use the FUSED (caligraphic-A - caligraphic-B x^2)
    # combination (sMGPT's "AminusBx2"/"gamma2evR"), not a separated A/B
    # pair -- see MG_kernels.D2_fused_grid.
    #
    # The scalar (fkPT) path's "A" is a bare large-scale-limit number (from
    # kernel_constants, evaluated at x=0 where B*x^2 vanishes so A and
    # AminusBx2(x=0) coincide); calc_jax reconstructs the angular shape by
    # hand, multiplying it by (1-x_r^2). A_fused_R (from
    # MG_kernels.D2_fused_grid), by contrast, IS ALREADY AminusBx2 evaluated
    # at the genuine x_r -- sMGPT uses it directly, with no extra (1-x_r^2)
    # multiplication (3_P13type.wl: "gamma2evRv = KAminusBx2"). A from-scratch
    # re-derivation against that source, cross-checked numerically against an
    # independent Wolfram evaluation, found the array-valued branch below
    # previously re-applied that scalar-only reconstruction by mistake.
    if A_fused_R is None:
        Gamma2evR = A * (1.0 - x2_r)
    else:
        Gamma2evR = A_fused_R
    if ApOverf0_fused_R is None:
        Gamma2evR_prime_term = ApOverf0 * (1.0 - x2_r)
    else:
        Gamma2evR_prime_term = ApOverf0_fused_R
    # Gamma2fevR: sMGPT's own formula (3_P13type.wl) is
    # "gamma2evR*(fkminusp+fp)/(2f0) + gamma2evRprime/(2f0)" -- the growth
    # rate at the OUTPUT leg of the underlying D2 solve (kminus, since that
    # solve is called at x=-x_r,k=k_ext,p=q so its output mode is kminus) plus
    # at the loop q. A from-scratch re-derivation against that source (cross
    # -checked numerically against an independent Wolfram evaluation) found
    # this used to read fk (external-k growth) instead of fkminusp_r -- a
    # pre-existing bug predating this module's fkpt_approximation=False work,
    # present in the fkPT-approximated path too (masked there since this is a
    # small, near-constant LS correction).
    Gamma2fevR = Gamma2evR * (fkminusp_r + fp_r) / 2.0 + Gamma2evR_prime_term / 2.0
    # C3Gamma3/C3Gamma3f: the scalar (fkPT) path reconstructs the LCDM-shaped
    # x-dependence by hand (the "(1-x^2)^2/y^2" factor), because CFD3/CFD3p
    # are single, x-independent large-scale-limit numbers -- that
    # reconstruction is specific to the LS-approximated shape (sMGPT's own
    # "C3gamma3LCDM"/"C3gamma3fLCDM" helpers, 3_P13type.wl) and must NOT be
    # applied to CFD3_R/CFD3p_R (from MG_kernels.D3_fused_grid), which are
    # already evaluated at the genuine x_r -- their x-dependence (and, for
    # C3Gamma3, the factor-of-2 normalization difference between the
    # LS-shape helper and the raw quantity) is already baked into the ODE
    # solve itself. A from-scratch re-derivation against sMGPT's raw
    # (non-approximated) F3K/G3K formulas, cross-checked numerically against
    # an independent Wolfram evaluation, found the array-valued branch below
    # previously kept the LS-shape-specific factor-of-2 by mistake.
    #
    # C3Gamma3f previously also carried an extra "(fk+2*fp_r)/3" growth-rate
    # weighting that has NO counterpart in sMGPT's C3gamma3fLCDM (verified by
    # directly comparing against 3_P13type.wl: C3gamma3fLCDM is just
    # 2*(5/21)*lcdmCFprime*(1-x^2)^2/y^2, no growth-rate factor at all).
    # Removing it was confirmed against sMGPT's own reference output (their
    # BGS_F5/fkKernels AllFunctions table): P13dt/P13tt went from ~4-6% off
    # to <1% agreement, with P13dd (which never had this factor) unaffected.
    if CFD3_R is None:
        C3Gamma3 = 2.0 * 5.0/21.0 * CFD3 * (1.0 - x2_r) * (1.0 - x2_r) / y2_r
    else:
        C3Gamma3 = 5.0/21.0 * CFD3_R
    if CFD3p_R is None:
        C3Gamma3f = 2.0 * 5.0/21.0 * CFD3p * (1.0 - x2_r) * (1.0 - x2_r) / y2_r
    else:
        C3Gamma3f = 5.0/21.0 * CFD3p_R
    # NOTE: the middle term below uses fk (external-k growth), matching sMGPT's
    # G3K term "1/3*C2*(fk/f0)*(k^2-kpx)/(k^2+p^2-2pkx)*gamma2evR" (3_P13type.wl)
    # exactly -- a from-scratch re-derivation against that source found this used
    # to read fp_r (loop-q growth) here, a pre-existing bug predating this
    # module's fkpt_approximation=False work (present in the fkPT-approximated
    # path too, just masked there because Gamma2evR/Gamma2fevR are small,
    # near-constant LS corrections). The OTHER fp_r below (last term) is
    # correct as-is -- sMGPT's own formula calls for fp (loop q) there.
    G3K = (
        C3Gamma3f / 2.0 + (2.0 * Gamma2fevR * x_r) / (7.0 * r_r) - (fk * x2_r) / (6.0 * r2_r)
        + fk * Gamma2evR * (1.0 - r_r * x_r) / (7.0 * y2_r)
        - 1.0/7.0 * (fp_r * Gamma2evR + 2.0 * Gamma2fevR) * (1.0 - x2_r) / y2_r)
    F3K = C3Gamma3 / 6.0 - x2_r / (6.0 * r2_r) + (Gamma2evR * x_r * (1.0 - r_r * x_r)) / (7.0 * r_r * y2_r)

    P13dd_B = jnp.sum(wpsl_r * (6.0 * r2_r * F3K), axis=0)
    P13du_B = jnp.sum(wpsl_r * (3.0 * r2_r * G3K + 3.0 * r2_r * F3K * fk), axis=0)
    P13uu_B = jnp.sum(wpsl_r * (6.0 * r2_r * G3K * fk), axis=0)

    sigma32PSL_B = jnp.sum(wpsl_r * (
        (5.0 * r2_r * (7.0 - 2.0*r2_r + 4.0*r_r*x_r + 6.0*(-2.0 + r2_r)*x2_r - 12.0*r_r*x2_r*x_r + 9.0*x2_r*x2_r))
        / (24.0 * y2_r)
    ), axis=0)

    # psl_kminus_r/fkminusp_r above are still needed unconditionally by
    # Gamma2fevR. The old R-loop-based I1udd1-family formulas (I1udd1a etc.,
    # evaluated on this unclipped R-grid) were removed once the I1udd1-family
    # kernels were unified onto the Q-grid's genuine three-ordering (Q+R+N)
    # sum below -- see the "Combine Q and R functions" comment.

    # Calculate scaling for R-functions
    pkl_k = jnp.stack([Pout, Pout_nw], axis=0)  # shape (2, Nk)
    scale_R = logk_grid2 / (8.0 * jnp.pi ** 2) * pkl_k

    # Use precomputed dkk_r for trapezoidal integration

    def trapsumR(B: Array) -> Array:
        """Functional trapezoidal integration for R-functions.

        Matches numpy version computation pattern.
        """
        B = B * scale_R
        # Compute trapezoidal sum: (B[i-1] + B[i]) * dk[i] for i >= 1, plus B[0] * dk[0]
        return jnp.sum((B[:-1] + B[1:]) * dkk_r[1:], axis=0) + B[0] * dkk_r[0]

    P13uu = trapsumR(P13uu_B)
    P13du = trapsumR(P13du_B)
    P13dd = trapsumR(P13dd_B)

    sigma32PSL = trapsumR(sigma32PSL_B)

    # ============================================================================
    # Combine Q and R functions
    # ============================================================================
    # The I1udd1-family kernels are the genuine three-ordering (Q+R+N) sum,
    # unconditionally, for both fkpt_approximation=True and False -- see the
    # "Genuine three-ordering (Q+R+N) combination" comment above for the
    # formula and why the domain is now unified between the two paths (this
    # replaced an older "tA + 2x a" shortcut, evaluated on the separate
    # unclipped R-loop/P13-domain, that was only exact in the squeezed
    # limit). Two earlier attempts to add a genuine N-ordering term on that
    # separate R-grid failed the GR-limit exact-match check; the fix that
    # stuck evaluates R and N on the Q-loop's OWN clipped grid, matching
    # sMGPT's own computeOneKBothExact/AKernelsT architecture exactly -- see
    # git history and .claude/plans/quizzical-mapping-catmull.md for that
    # evidence trail.
    I1udd1A = trapsumQ(I1udd1_qrN_B) + trapsumQ_noPq(I1udd1_n_B)
    I2uud1A = trapsumQ(I2uud1_qrN_B) + trapsumQ_noPq(I2uud1_n_B)
    I2uud2A = trapsumQ(I2uud2_qrN_B) + trapsumQ_noPq(I2uud2_n_B)
    I3uuu2A = trapsumQ(I3uuu2_qrN_B) + trapsumQ_noPq(I3uuu2_n_B)
    I3uuu3A = trapsumQ(I3uuu3_qrN_B) + trapsumQ_noPq(I3uuu3_n_B)

    # ============================================================================
    # D-TERMS (B + C - G corrections)
    # ============================================================================
    fk_grid = fk

    # I2uudd1D: subtract k^2 * sigma2v * P_L(k)
    I2uudd1BpC = I2uudd1BpC - logk_grid2 * sigma2v * pkl_k  

    # I3uuud2D: subtract 2 * k^2 * sigma2v * f(k) * P_L(k)
    I3uuud2BpC = I3uuud2BpC - 2.0 * logk_grid2 * sigma2v * fk_grid * pkl_k  

    # I4uuuu3D: subtract k^2 * sigma2v * f(k)^2 * P_L(k)
    I4uuuu3BpC = I4uuuu3BpC - logk_grid2 * sigma2v * fk_grid ** 2 * pkl_k  

    return (
        P22dd, P22du, P22uu,
        I1udd1A, I2uud1A, I2uud2A,
        I3uuu2A, I3uuu3A,
        I2uudd1BpC, I2uudd2BpC,
        I3uuud2BpC, I3uuud3BpC,
        I4uuuu2BpC, I4uuuu3BpC, I4uuuu4BpC,
        Pb1b2, Pb1bs2, Pb22, Pb2s2, Ps22,
        Pb2theta, Pbs2theta,
        P13dd, P13du, P13uu,
        sigma32PSL,
        pkl_k
    )


class JaxCalculator(AbsCalculator):
    """JAX-accelerated k-functions calculator implementing the AbsCalculator interface.

    This calculator uses JAX for JIT compilation and automatic differentiation.
    Grid data is converted to JAX arrays once during initialization and reused.
    """

    def __init__(self) -> None:
        """Initialize an empty calculator. Call initialize() before evaluate()."""
        self.k_in_jax = None
        self.logk_grid_jax = None
        self.kk_grid_jax = None
        self.xxQ_jax = None
        self.wwQ_jax = None
        self.xxR_jax = None
        self.wwR_jax = None
        # Precomputed spline interpolation coefficients
        self.spline_logk = None
        self.spline_kk = None
        self.spline_y = None
        self.spline_y_r = None
        # Precomputed spline matrix factors for k_in grid (optimization for MCMC)
        self.spline_sig_jax = None
        self.spline_inv_h_jax = None
        self.spline_inv_h_span_jax = None
        self.spline_n = None
        # Precomputed Q-function quantities
        self.logk_grid2_jax = None
        self.dkk_jax = None
        self.dkk_reshaped_jax = None
        self.scale_Q_jax = None
        self.r_jax = None
        self.r2_jax = None
        self.x_jax = None
        self.w_jax = None
        self.x2_jax = None
        self.y2_jax = None
        self.y_jax = None
        # Precomputed R-function quantities
        self.r_r_jax = None
        self.r2_r_jax = None
        self.x_r_jax = None
        self.w_r_jax = None
        self.x2_r_jax = None
        self.y2_r_jax = None
        self.AngleEvR_jax = None
        self.AngleEvR2_jax = None
        self.dkk_r_jax = None

    def initialize(self, data: KFunctionsInitData) -> None:
        """Initialize the calculator with grid data and quadrature points.

        Converts numpy arrays to JAX arrays and stores them for reuse.
        Pre-computes all fixed quantities that don't depend on input power spectra.

        Args:
            data: Initialization data containing k-grid, quadrature points, etc.
        """
        self.k_in_jax = jnp.asarray(data.k_in, dtype=jnp.float64)
        self.logk_grid_jax = jnp.asarray(data.logk_grid, dtype=jnp.float64)
        self.kk_grid_jax = jnp.asarray(data.kk_grid, dtype=jnp.float64)
        self.xxQ_jax = jnp.asarray(data.xxQ, dtype=jnp.float64)
        self.wwQ_jax = jnp.asarray(data.wwQ, dtype=jnp.float64)
        self.xxR_jax = jnp.asarray(data.xxR, dtype=jnp.float64)
        self.wwR_jax = jnp.asarray(data.wwR, dtype=jnp.float64)

        # Pre-compute commonly used grid quantities
        self.logk_grid2_jax = self.logk_grid_jax * self.logk_grid_jax
        self.dkk_jax = jnp.diff(self.kk_grid_jax)
        self.dkk_reshaped_jax = self.dkk_jax.reshape(-1, 1, 1)
        self.scale_Q_jax = 0.25 * self.logk_grid2_jax / jnp.pi ** 2

        # Pre-compute spline coefficients for fixed interpolation grids
        # 1. Coefficients for interpolating onto logk_grid
        self.spline_logk = init_cubic_spline_jax(self.k_in_jax, self.logk_grid_jax)

        # 2. Coefficients for interpolating onto kk_grid
        self.spline_kk = init_cubic_spline_jax(self.k_in_jax, self.kk_grid_jax)

        # 3. Pre-compute spline matrix factors for k_in grid (optimization for MCMC)
        # These factors depend only on the k_in grid structure, not on Y values,
        # so we compute them once during initialization to speed up evaluate()
        sig, inv_h, inv_h_span, n = precompute_spline_matrix_factors(self.k_in_jax)
        self.spline_sig_jax = sig
        self.spline_inv_h_jax = inv_h
        self.spline_inv_h_span_jax = inv_h_span
        self.spline_n = int(n)  # Convert to Python int for static argument

        # 4. Pre-compute Q-function quantities and spline coefficients
        # Compute variable integration limits for mu (local variables only needed here)
        rmax = self.k_in_jax[-1] / self.logk_grid_jax
        rmin = self.k_in_jax[0] / self.logk_grid_jax
        rmax2 = rmax * rmax
        rmin2 = rmin * rmin

        self.r_jax = self.kk_grid_jax[1:].reshape(-1, 1, 1) / self.logk_grid_jax
        self.r2_jax = self.r_jax * self.r_jax

        mumin = jnp.maximum(-1.0, (1.0 + self.r2_jax - rmax2) / (2.0 * self.r_jax))
        mumax = jnp.minimum(1.0, (1.0 + self.r2_jax - rmin2) / (2.0 * self.r_jax))
        mumax = jnp.where(self.r_jax >= 0.5, 0.5 / self.r_jax, mumax)

        # Scale Gauss-Legendre nodes and weights to [mumin, mumax]
        dmu = mumax - mumin
        xGL = 0.5 * (dmu * self.xxQ_jax.reshape(-1, 1, 1, 1) + (mumax + mumin))
        wGL = 0.5 * dmu * self.wwQ_jax.reshape(-1, 1, 1, 1)

        # Compute x, w, x2, y2, y values for Q-function integration
        self.x_jax = xGL
        self.w_jax = wGL
        self.x2_jax = self.x_jax * self.x_jax
        self.y2_jax = 1.0 + self.r2_jax - 2.0 * self.r_jax * self.x_jax
        self.y_jax = jnp.sqrt(self.y2_jax)

        # Pre-compute coefficients for interpolating at logk_grid * y
        self.spline_y = init_cubic_spline_jax(self.k_in_jax, self.logk_grid_jax * self.y_jax)

        # Pre-compute R-function quantities
        # R-function uses r from kk[1:-1] (indices 1 to nquadSteps-2)
        self.r_r_jax = self.kk_grid_jax[1:-1].reshape(-1, 1, 1) / self.logk_grid_jax
        self.r2_r_jax = self.r_r_jax * self.r_r_jax

        # Gauss-Legendre points in [-1, 1] (fixed limits for R-functions)
        self.x_r_jax = self.xxR_jax.reshape(-1, 1, 1, 1)
        self.w_r_jax = self.wwR_jax.reshape(-1, 1, 1, 1)
        self.x2_r_jax = self.x_r_jax * self.x_r_jax
        self.y2_r_jax = 1.0 + self.r2_r_jax - 2.0 * self.r_r_jax * self.x_r_jax

        # R-function angles (independent of input parameters)
        self.AngleEvR_jax = -self.x_r_jax
        self.AngleEvR2_jax = self.AngleEvR_jax * self.AngleEvR_jax

        # R-function trapezoidal integration spacing
        self.dkk_r_jax = self.dkk_reshaped_jax[:-1].reshape(-1, 1, 1)

        # Pre-compute coefficients for interpolating P(k), f(k) at the R-grid's
        # "kminus" = k*sqrt(1+r_r^2-2 r_r x_r) points. Needed only by the
        # non-fkPT-approximated ("N ordering") bias/RSD kernels -- see
        # fkptjax.MG_kernels and the AngleEvN/F2evN/G2evN block in
        # _calculate_jax_core.
        y_r_jax = jnp.sqrt(self.y2_r_jax)
        self.spline_y_r = init_cubic_spline_jax(self.k_in_jax, self.logk_grid_jax * y_r_jax)

    def evaluate(self, Pk_in: Float64NDArray, Pk_nw_in: Float64NDArray,
                 fk_in: Float64NDArray, A: float, ApOverf0: float, CFD3: float,
                 CFD3p: float, sigma2v: float, f0: float,
                 A_Q: Any = None, B_Q: Any = None,
                 ApOverf0_Q: Any = None, BpOverf0_Q: Any = None,
                 A_fused_R: Any = None, ApOverf0_fused_R: Any = None,
                 CFD3_R: Any = None, CFD3p_R: Any = None,
                 A_N: Any = None, B_N: Any = None,
                 ApOverf0_N: Any = None, BpOverf0_N: Any = None,
                 A_RQ: Any = None, B_RQ: Any = None,
                 ApOverf0_RQ: Any = None, BpOverf0_RQ: Any = None) -> KFunctionsOut:
        """Evaluate k-functions given input power spectra.

        Args:
            Pk_in: Linear power spectrum values at k_in grid points
            Pk_nw_in: No-wiggle linear power spectrum values at k_in grid points
            fk_in: Growth rate f(k) values at k_in grid points
            A: Cosmological parameter A
            ApOverf0: Cosmological parameter Ap/f0
            CFD3: Cosmological parameter CFD3
            CFD3p: Cosmological parameter CFD3p
            sigma2v: Velocity dispersion parameter
            f0: Reference growth rate
            A_Q, B_Q, ApOverf0_Q, BpOverf0_Q: optional grid-shaped (Q-loop,
                "Q ordering") overrides for the F2evQ/G2evQ caligraphic-A/-B
                pieces, e.g. from ``MG_kernels.A_B_grid`` for the
                non-fkPT-approximated path. Default (``None``) reproduces the
                fkPT-approximated behavior exactly: ``A_Q=B_Q=A``,
                ``ApOverf0_Q=BpOverf0_Q=ApOverf0``.
            A_fused_R, ApOverf0_fused_R: grid-shaped overrides for the FUSED
                (caligraphic-A - caligraphic-B x^2) combination used by
                Gamma2evR/Gamma2fevR, from ``MG_kernels.D2_fused_grid``.
                Default reproduces ``A_fused_R -> A``.
            CFD3_R, CFD3p_R: grid-shaped overrides for the third-order
                C3Gamma3/C3Gamma3f pieces, from ``MG_kernels.D3_fused_grid``.
                Unlike the others, these REPLACE (not multiply into) the
                scalar-path geometric reconstruction -- see the code comment
                at their use site.
            A_N, B_N, ApOverf0_N, BpOverf0_N, A_RQ, B_RQ, ApOverf0_RQ,
                BpOverf0_RQ: grid-shaped "N ordering" and "R ordering,
                evaluated on the Q-loop's grid" pieces -- the second and
                third legs of the I1udd1-family bias/RSD kernels' genuine
                three-ordering (Q+R+N) sum (mirrors sMGPT's A11..A33), which
                is now used unconditionally for this kernel family. Default
                (``None``) broadcasts the scalar ``A``/``ApOverf0`` into
                these two orderings too, exactly like ``A_Q``'s own default
                -- i.e. the squeezed (``fkpt_approximation=True``) path pays
                no extra cost, it just reuses the same scalar for all three
                legs instead of running a separately-discretized shortcut.

        Returns:
            KFunctionsOut containing all computed k-functions (as numpy arrays)
        """
        A_Q = A if A_Q is None else A_Q
        B_Q = A if B_Q is None else B_Q
        ApOverf0_Q = ApOverf0 if ApOverf0_Q is None else ApOverf0_Q
        BpOverf0_Q = ApOverf0 if BpOverf0_Q is None else BpOverf0_Q

        # Stack input power spectra and normalize fk by f0. jnp.stack (not np.stack), so this
        # stays traceable when Pk_in/Pk_nw_in/fk_in/f0 are jax tracers (e.g. under jit with a
        # beyond_eds=True, fkpt_approximation=False caller supplying both fk/f0 and
        # ingredients_fn -- see Kfuncs_to_tables's own static_ctx/lazy-derivs handling).
        Y_jax = jnp.stack([Pk_in, Pk_nw_in, fk_in / f0], axis=0).astype(jnp.float64)

        # Run JIT-compiled calculation with all precomputed values
        # Y2 (spline second derivatives) is now computed inside _calculate_jax_core
        # using precomputed matrix factors. This eliminates an extra kernel launch
        # and allows XLA to fuse the Y2 computation with subsequent operations.
        # Unpack spline coefficient dictionaries
        results = _calculate_jax_core(
            Y_jax,
            # Spline matrix factors for computing Y2 inside the JIT boundary
            self.spline_sig_jax, self.spline_inv_h_jax, self.spline_inv_h_span_jax, self.spline_n,
            # Spline coefficients for logk_grid
            self.spline_logk['x_shape'], self.spline_logk['idx_lo_flat'], self.spline_logk['idx_hi_flat'],
            self.spline_logk['a_flat'], self.spline_logk['b_flat'], self.spline_logk['a3_flat'],
            self.spline_logk['b3_flat'], self.spline_logk['h2_flat'],
            # Spline coefficients for kk_grid
            self.spline_kk['x_shape'], self.spline_kk['idx_lo_flat'], self.spline_kk['idx_hi_flat'],
            self.spline_kk['a_flat'], self.spline_kk['b_flat'], self.spline_kk['a3_flat'],
            self.spline_kk['b3_flat'], self.spline_kk['h2_flat'],
            # Spline coefficients for y
            self.spline_y['x_shape'], self.spline_y['idx_lo_flat'], self.spline_y['idx_hi_flat'],
            self.spline_y['a_flat'], self.spline_y['b_flat'], self.spline_y['a3_flat'],
            self.spline_y['b3_flat'], self.spline_y['h2_flat'],
            self.logk_grid2_jax, self.dkk_jax, self.dkk_reshaped_jax, self.scale_Q_jax,
            self.r_jax, self.r2_jax, self.x_jax, self.w_jax, self.x2_jax, self.y2_jax, self.y_jax,
            self.r_r_jax, self.r2_r_jax, self.x_r_jax, self.w_r_jax, self.x2_r_jax, self.y2_r_jax,
            self.AngleEvR_jax, self.AngleEvR2_jax, self.dkk_r_jax,
            self.kk_grid_jax, self.logk_grid_jax,
            A, ApOverf0, CFD3, CFD3p, sigma2v,
            A_Q, B_Q, ApOverf0_Q, BpOverf0_Q,
            A_fused_R, ApOverf0_fused_R, CFD3_R, CFD3p_R,
            A_N, B_N, ApOverf0_N, BpOverf0_N,
            self.spline_y_r['x_shape'], self.spline_y_r['idx_lo_flat'], self.spline_y_r['idx_hi_flat'],
            self.spline_y_r['a_flat'], self.spline_y_r['b_flat'], self.spline_y_r['a3_flat'],
            self.spline_y_r['b3_flat'], self.spline_y_r['h2_flat'],
            A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ,
        )

        # NOT converted to numpy (was: tuple(np.asarray(r) for r in results)) -- this is the
        # last line of evaluate(), so returning the jax arrays as-is keeps this traceable
        # under jit/vmap when Pk_in/A_Q/etc are tracers; Kfuncs_to_tables's own table assembly
        # already re-wraps every field in jnp.asarray() regardless (see its `_arr` helper), so
        # nothing downstream needed these to be numpy specifically.
        return KFunctionsOut(*results)

    def evaluate_jax(self, Pk_in, Pk_nw_in, fk_in, A, ApOverf0, CFD3,
                     CFD3p, sigma2v, f0,
                     A_Q=None, B_Q=None, ApOverf0_Q=None, BpOverf0_Q=None,
                     A_fused_R=None, ApOverf0_fused_R=None,
                     CFD3_R=None, CFD3p_R=None,
                     A_N=None, B_N=None, ApOverf0_N=None, BpOverf0_N=None,
                     A_RQ=None, B_RQ=None, ApOverf0_RQ=None, BpOverf0_RQ=None):
        """Fully jax-traceable variant of :meth:`evaluate`.

        Identical to ``evaluate`` but keeps everything in ``jax.numpy`` (no
        ``np.stack`` on inputs, no ``np.asarray`` on outputs), so it can be
        ``jax.jit`` / ``jax.vmap``'d.  Returns a ``KFunctionsOut`` of JAX arrays.

        See :meth:`evaluate` for all the ``*_Q``/``*_R``/``*_fused_R``/``*_N``
        overrides.
        """
        A_Q = A if A_Q is None else A_Q
        B_Q = A if B_Q is None else B_Q
        ApOverf0_Q = ApOverf0 if ApOverf0_Q is None else ApOverf0_Q
        BpOverf0_Q = ApOverf0 if BpOverf0_Q is None else BpOverf0_Q

        Y_jax = jnp.stack([Pk_in, Pk_nw_in, fk_in / f0], axis=0).astype(jnp.float64)
        results = _calculate_jax_core(
            Y_jax,
            self.spline_sig_jax, self.spline_inv_h_jax, self.spline_inv_h_span_jax, self.spline_n,
            self.spline_logk['x_shape'], self.spline_logk['idx_lo_flat'], self.spline_logk['idx_hi_flat'],
            self.spline_logk['a_flat'], self.spline_logk['b_flat'], self.spline_logk['a3_flat'],
            self.spline_logk['b3_flat'], self.spline_logk['h2_flat'],
            self.spline_kk['x_shape'], self.spline_kk['idx_lo_flat'], self.spline_kk['idx_hi_flat'],
            self.spline_kk['a_flat'], self.spline_kk['b_flat'], self.spline_kk['a3_flat'],
            self.spline_kk['b3_flat'], self.spline_kk['h2_flat'],
            self.spline_y['x_shape'], self.spline_y['idx_lo_flat'], self.spline_y['idx_hi_flat'],
            self.spline_y['a_flat'], self.spline_y['b_flat'], self.spline_y['a3_flat'],
            self.spline_y['b3_flat'], self.spline_y['h2_flat'],
            self.logk_grid2_jax, self.dkk_jax, self.dkk_reshaped_jax, self.scale_Q_jax,
            self.r_jax, self.r2_jax, self.x_jax, self.w_jax, self.x2_jax, self.y2_jax, self.y_jax,
            self.r_r_jax, self.r2_r_jax, self.x_r_jax, self.w_r_jax, self.x2_r_jax, self.y2_r_jax,
            self.AngleEvR_jax, self.AngleEvR2_jax, self.dkk_r_jax,
            self.kk_grid_jax, self.logk_grid_jax,
            A, ApOverf0, CFD3, CFD3p, sigma2v,
            A_Q, B_Q, ApOverf0_Q, BpOverf0_Q,
            A_fused_R, ApOverf0_fused_R, CFD3_R, CFD3p_R,
            A_N, B_N, ApOverf0_N, BpOverf0_N,
            self.spline_y_r['x_shape'], self.spline_y_r['idx_lo_flat'], self.spline_y_r['idx_hi_flat'],
            self.spline_y_r['a_flat'], self.spline_y_r['b_flat'], self.spline_y_r['a3_flat'],
            self.spline_y_r['b3_flat'], self.spline_y_r['h2_flat'],
            A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ,
        )
        return KFunctionsOut(*results)
