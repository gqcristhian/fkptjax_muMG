"""
MG_kernels_Wronskian.py
------------------------
Drop-in-compatible, Green's-function/Wronskian reformulation of
``fkptjax.MG_kernels``'s full (non-fkPT-approximated) beyond-EdS kernels.

``MG_kernels.A_B_grid``/``D2_fused_grid``/``D3_fused_grid`` solve one adaptive
ODE per quadrature grid point (up to ~3.6e5 points at production
resolution) -- correct, but the dominant cost turns out to be the *number*
of adaptive solver steps, not raw FLOPs (both CPU and GPU wall time stay
nearly flat across an 81x change in grid-point count; see the module-level
performance note in ``MG_kernels.py``).

This module exploits a structural fact visible directly in
``mg_jax.secondOrder``/``thirdOrder``: for a FIXED output-leg momentum, the
second- and third-order growth kernels' homogeneous (LHS) operator is the
same regardless of the other two legs -- only the *source* (RHS) depends on
the full ``(q,x)`` triple. So instead of solving the full driven ODE
independently at every grid point, this module:

1. Solves the *homogeneous* equation (two independent solutions) ONCE on a
   small interpolation grid over ``k`` (``n_ktable`` nodes, batched via
   ``jax.vmap`` exactly like ``MG_kernels`` already batches its per-point
   solves -- just over a much smaller batch), with dense output on a shared
   ``eta`` grid (``n_eta`` nodes).
2. For every point in the *actual* quadrature grid, gets the needed
   homogeneous-solution values by interpolating that small table (cheap,
   vectorized array ops), evaluates the source there, and combines them via
   the standard variation-of-parameters formula (a dot product against the
   precomputed table, not a new ODE solve).

Validated (2026-09-02, see the accompanying
``examples/check_MG_kernels_wronskian.py``) against ``MG_kernels.py``'s
adaptive per-point solve for the HDKI/BZ_Mass model, both well-separated and
deliberately near-collinear (``kminus/kf`` as small as ~0.02) configurations,
for: the second-order ``A``/``B`` pair, the nested third-order ``D3``
kernel, and all three of the I1udd1-family's leg orderings (Q/R/N). Every
case converges cleanly (no sign flip, no divergence) as ``n_eta`` increases
-- unlike ``MG_kernels.py``'s own documented ``solver='rk4'`` fixed-step
alternative, which is unsafe specifically near collinear configurations.

This module intentionally does not touch ``MG_kernels.py``, ``mg_jax.py``,
``calculate_jax.py``, or ``kfuncs_to_tables.py`` -- it is a standalone
sibling, developed and validated independently before any decision to wire
it into the production pipeline.
"""

import jax
import jax.numpy as jnp
import diffrax

from . import mg_jax as bj

# Matches MG_kernels.py's defaults for the (still-adaptive) homogeneous-basis
# solves -- these ODEs are smooth away from singularities, well converged at
# this tolerance there, and the same holds for the basis solves here.
_RTOL = 1e-8
_ATOL = 1e-11
_MAXSTEPS = 100000
_N_KTABLE = 300
_N_ETA = 256


def _x_from_triple(kf, k1, k2):
    """cos(angle) between legs k1,k2 such that |k1+k2| = kf.

    Identical formula to ``MG_kernels._x_from_triple`` (duplicated here
    rather than imported, to keep this module fully self-contained from
    ``MG_kernels.py``).
    """
    return (kf * kf - k1 * k1 - k2 * k2) / (2.0 * k1 * k2)


def _safe_k_range(kmin, kmax, margin=0.05, floor=1e-5):
    """Widen (kmin,kmax) by a small margin and floor away from zero, so
    ``jnp.geomspace`` stays well-defined under jit even if the actual
    momentum arguments include a very small value."""
    lo = jnp.maximum(kmin * (1.0 - margin), floor)
    hi = kmax * (1.0 + margin) + floor
    return lo, hi


def _homogeneous_rhs(kf, P):
    """RHS of D'' + fr*D' - f2*mu(eta,kf)*D = 0 (diffrax 3-arg convention)."""
    def rhs(eta, y, args):
        f2 = bj.f1(eta, P)
        fr = 2.0 - f2
        return jnp.stack([y[1], f2 * bj.mu(eta, kf, P) * y[0] - fr * y[1]])
    return rhs


def _solve_homogeneous_dense(kf, y0, eta_grid, P, xnow, xstop, rtol, atol):
    """One homogeneous solve, densely sampled at every node of eta_grid."""
    term = diffrax.ODETerm(_homogeneous_rhs(kf, P))
    solver = diffrax.Tsit5()
    ctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term, solver, t0=xnow, t1=xstop, dt0=0.01, y0=y0,
        stepsize_controller=ctrl, saveat=diffrax.SaveAt(ts=eta_grid),
        max_steps=_MAXSTEPS,
    )
    return sol.ys[:, 0], sol.ys[:, 1]


def _basis_table(k_table, y0, eta_grid, P, xnow, xstop, rtol, atol):
    """Batch the homogeneous solve over the whole k-interpolation table.

    Returns (D, dD), each shape (n_ktable, n_eta) -- this is the ONLY place
    an adaptive ODE solve happens; everything downstream is array algebra.
    """
    solve = jax.vmap(
        lambda kf_: _solve_homogeneous_dense(kf_, y0, eta_grid, P, xnow, xstop, rtol, atol)
    )
    return solve(k_table)


def _interp_over_eta(kq, k_table, table):
    """Interpolate ``table`` (shape (n_ktable, n_eta)) along the k-axis at
    query points ``kq`` (shape (Npoints,)), for every eta node at once.
    Returns shape (Npoints, n_eta).
    """
    return jax.vmap(lambda col: jnp.interp(kq, k_table, col), in_axes=1, out_axes=1)(table)


def _cumtrapz(y, x):
    """Cumulative trapezoidal integral of y(x) along the last axis, batched
    over any leading axes. y[..., 0] of the result is always 0."""
    dx = jnp.diff(x)
    avg = 0.5 * (y[..., 1:] + y[..., :-1])
    inc = avg * dx
    zeros = jnp.zeros(y.shape[:-1] + (1,), dtype=y.dtype)
    return jnp.concatenate([zeros, jnp.cumsum(inc, axis=-1)], axis=-1)


# ---------------------------------------------------------------------------
# Second order (separately-sourced A/B pair) -- P22-loop / I1udd1-family
# ---------------------------------------------------------------------------

def A_B_grid(kf, k1, k2, P, xnow, xstop, solver='wronskian',
             n_ktable=_N_KTABLE, n_eta=_N_ETA, rtol=_RTOL, atol=_ATOL):
    """Batched A(kf,k1,k2), B(kf,k1,k2) and their eta-derivatives.

    Drop-in equivalent of ``MG_kernels.A_B_grid`` (same call signature,
    return shape, and normalization convention), computed via the
    Green's-function/Wronskian method described in the module docstring
    instead of one ODE solve per grid point. ``kf``, ``k1``, ``k2`` are
    broadcastable arrays. ``n_ktable``/``n_eta`` replace ``MG_kernels``'s
    ``n_steps`` as the accuracy knobs -- ``n_eta=256`` (default) gave
    ~3e-5 relative error against the adaptive reference in every case
    tested (well-separated and near-collinear, all three I1udd1-family leg
    orderings); ``n_eta=512-1024`` buys another 1-2 orders of magnitude at
    negligible extra cost, since the one-time basis-table build (not the
    per-point evaluation) dominates the runtime either way.
    """
    kf_b, k1_b, k2_b = jnp.broadcast_arrays(
        jnp.asarray(kf, dtype=jnp.float64),
        jnp.asarray(k1, dtype=jnp.float64),
        jnp.asarray(k2, dtype=jnp.float64),
    )
    shape = kf_b.shape
    flat = lambda a: a.reshape(-1)  # noqa: E731
    kf_f, k1_f, k2_f = flat(kf_b), flat(k1_b), flat(k2_b)

    eta_grid = jnp.linspace(xnow, xstop, n_eta)
    kmin = jnp.min(jnp.concatenate([kf_f, k1_f, k2_f]))
    kmax = jnp.max(jnp.concatenate([kf_f, k1_f, k2_f]))
    lo, hi = _safe_k_range(kmin, kmax)
    k_table = jnp.geomspace(lo, hi, n_ktable)

    e0 = jnp.exp(xnow)
    y1_0 = jnp.array([e0, e0])
    y2_0 = jnp.array([e0, -2.0 * e0])
    y1_tab, dy1_tab = _basis_table(k_table, y1_0, eta_grid, P, xnow, xstop, rtol, atol)
    y2_tab, dy2_tab = _basis_table(k_table, y2_0, eta_grid, P, xnow, xstop, rtol, atol)

    y1_kf = _interp_over_eta(kf_f, k_table, y1_tab)
    dy1_kf = _interp_over_eta(kf_f, k_table, dy1_tab)
    y2_kf = _interp_over_eta(kf_f, k_table, y2_tab)
    dy2_kf = _interp_over_eta(kf_f, k_table, dy2_tab)
    W_kf = y1_kf * dy2_kf - y2_kf * dy1_kf

    Dk1 = _interp_over_eta(k1_f, k_table, y1_tab)  # growing-mode leg growth factor
    Dk2 = _interp_over_eta(k2_f, k_table, y1_tab)
    dDk1_end = jnp.interp(k1_f, k_table, dy1_tab[:, -1])
    dDk2_end = jnp.interp(k2_f, k_table, dy1_tab[:, -1])

    x = _x_from_triple(kf_f, k1_f, k2_f)
    eta_row = eta_grid[None, :]
    x_col = x[:, None]; k1_col = k1_f[:, None]; k2_col = k2_f[:, None]
    src_A = bj.S2a(eta_row, x_col, k1_col, k2_col, P) + bj.S2FL(eta_row, x_col, k1_col, k2_col, P)
    src_B = bj.S2b(eta_row, x_col, k1_col, k2_col, P)
    gA = src_A * Dk1 * Dk2
    gB = src_B * Dk1 * Dk2

    IA1 = jnp.trapezoid(y1_kf * gA / W_kf, eta_grid, axis=-1)
    IA2 = jnp.trapezoid(y2_kf * gA / W_kf, eta_grid, axis=-1)
    IB1 = jnp.trapezoid(y1_kf * gB / W_kf, eta_grid, axis=-1)
    IB2 = jnp.trapezoid(y2_kf * gB / W_kf, eta_grid, axis=-1)

    y1_end = y1_kf[:, -1]; dy1_end = dy1_kf[:, -1]
    y2_end = y2_kf[:, -1]; dy2_end = dy2_kf[:, -1]
    uA = y2_end * IA1 - y1_end * IA2
    duA = dy2_end * IA1 - dy1_end * IA2
    uB = y2_end * IB1 - y1_end * IB2
    duB = dy2_end * IB1 - dy1_end * IB2

    # Physical initial conditions (matches MG_kernels._solve_AB's y0) -- a
    # TRUE constant, independent of (kf,k1,k2), so the 2x2 homogeneous-
    # correction system is solved once, not per point.
    D2plusi = 3.0 * jnp.exp(2.0 * xnow) / 7.0
    dD2plusi = 6.0 * jnp.exp(2.0 * xnow) / 7.0
    Mmat = jnp.array([[y1_0[0], y2_0[0]], [y1_0[1], y2_0[1]]])
    c = jnp.linalg.solve(Mmat, jnp.array([D2plusi, dD2plusi]))
    c0, c1 = c[0], c[1]

    DA = c0 * y1_end + c1 * y2_end + uA
    dDA = c0 * dy1_end + c1 * dy2_end + duA
    DB = c0 * y1_end + c1 * y2_end + uB
    dDB = c0 * dy1_end + c1 * dy2_end + duB

    Dk1_end = Dk1[:, -1]; Dk2_end = Dk2[:, -1]
    C = (3.0 / 7.0) * Dk1_end * Dk2_end
    Cp = (3.0 / 7.0) * (dDk1_end * Dk2_end + Dk1_end * dDk2_end)
    A = DA / C
    B = DB / C
    Ap = dDA / C - DA * Cp / (C * C)
    Bp = dDB / C - DB * Cp / (C * C)

    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return reshape(A), reshape(B), reshape(Ap), reshape(Bp)


# ---------------------------------------------------------------------------
# Fused (combined) second-order kernel -- P13-loop's Gamma2/Gamma2f
# ---------------------------------------------------------------------------

def D2_fused_grid(x, k, p, P, xnow, xstop, solver='wronskian',
                   n_ktable=_N_KTABLE, n_eta=_N_ETA, rtol=_RTOL, atol=_ATOL):
    """Batched, fused A - B*x^2 combination (sMGPT's ``AminusBx2``).

    Drop-in equivalent of ``MG_kernels.D2_fused_grid``. ``x, k, p``
    broadcastable arrays; see ``A_B_grid``'s docstring for ``n_ktable``/
    ``n_eta``.
    """
    xb, kb, pb = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(p, dtype=jnp.float64),
    )
    shape = xb.shape
    flat = lambda a: a.reshape(-1)  # noqa: E731
    x_f, k_f, p_f = flat(xb), flat(kb), flat(pb)
    kf_f = bj.kpp(x_f, k_f, p_f)

    eta_grid = jnp.linspace(xnow, xstop, n_eta)
    kmin = jnp.min(jnp.concatenate([kf_f, k_f, p_f]))
    kmax = jnp.max(jnp.concatenate([kf_f, k_f, p_f]))
    lo, hi = _safe_k_range(kmin, kmax)
    k_table = jnp.geomspace(lo, hi, n_ktable)

    e1 = jnp.exp(xnow)
    y1_0 = jnp.array([e1, e1])
    y2_0 = jnp.array([e1, -2.0 * e1])
    y1_tab, dy1_tab = _basis_table(k_table, y1_0, eta_grid, P, xnow, xstop, rtol, atol)
    y2_tab, dy2_tab = _basis_table(k_table, y2_0, eta_grid, P, xnow, xstop, rtol, atol)

    y1_kf = _interp_over_eta(kf_f, k_table, y1_tab)
    dy1_kf = _interp_over_eta(kf_f, k_table, dy1_tab)
    y2_kf = _interp_over_eta(kf_f, k_table, y2_tab)
    dy2_kf = _interp_over_eta(kf_f, k_table, dy2_tab)
    W_kf = y1_kf * dy2_kf - y2_kf * dy1_kf

    Dpk = _interp_over_eta(k_f, k_table, y1_tab)
    Dpp = _interp_over_eta(p_f, k_table, y1_tab)
    dDpk_end = jnp.interp(k_f, k_table, dy1_tab[:, -1])
    dDpp_end = jnp.interp(p_f, k_table, dy1_tab[:, -1])

    eta_row = eta_grid[None, :]
    src = bj.SD2(eta_row, x_f[:, None], k_f[:, None], p_f[:, None], P) * Dpk * Dpp
    I1 = jnp.trapezoid(y1_kf * src / W_kf, eta_grid, axis=-1)
    I2 = jnp.trapezoid(y2_kf * src / W_kf, eta_grid, axis=-1)

    y1_end = y1_kf[:, -1]; dy1_end = dy1_kf[:, -1]
    y2_end = y2_kf[:, -1]; dy2_end = dy2_kf[:, -1]
    u = y2_end * I1 - y1_end * I2
    du = dy2_end * I1 - dy1_end * I2

    one_m_x2 = 1.0 - x_f * x_f
    e2 = jnp.exp(2.0 * xnow)
    D2i = 3.0 * e2 / 7.0 * one_m_x2
    dD2i = 6.0 * e2 / 7.0 * one_m_x2

    # D2i/dD2i depend on x (per point), unlike A_B_grid's constant IC -- so
    # the homogeneous correction is solved per point, via a fixed 2x2
    # inverse applied to the whole batch at once (no per-point ODE work).
    Mmat = jnp.array([[y1_0[0], y2_0[0]], [y1_0[1], y2_0[1]]])
    Minv = jnp.linalg.inv(Mmat)
    c0 = Minv[0, 0] * D2i + Minv[0, 1] * dD2i
    c1 = Minv[1, 0] * D2i + Minv[1, 1] * dD2i

    D2_raw = c0 * y1_end + c1 * y2_end + u
    dD2_raw = c0 * dy1_end + c1 * dy2_end + du

    # Normalize by the leg-growth product, matching MG_kernels.D2_fused_grid
    # exactly (this is the same C/Cp convention as A_B_grid, applied here to
    # the single fused D2 quantity instead of separate A/B).
    Dk_end = Dpk[:, -1]; Dp_end = Dpp[:, -1]
    C = (3.0 / 7.0) * Dk_end * Dp_end
    Cp = (3.0 / 7.0) * (dDpk_end * Dp_end + Dk_end * dDpp_end)
    A = D2_raw / C
    Ap = dD2_raw / C - D2_raw * Cp / (C * C)

    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return reshape(A), reshape(Ap)


# ---------------------------------------------------------------------------
# Fused (combined) third-order kernel -- P13-loop's C3Gamma3/C3Gamma3f
# ---------------------------------------------------------------------------

def D3_fused_grid(x, k, p, P, xnow, xstop, f0, solver='wronskian',
                   n_ktable=_N_KTABLE, n_eta=_N_ETA, rtol=_RTOL, atol=_ATOL):
    """Batched, fused third-order kernel (sMGPT's ``C3gamma3``/``C3gamma3f``).

    Drop-in equivalent of ``MG_kernels.D3_fused_grid``. This is a NESTED
    Green's-function solve: the third-order source needs the full
    eta-trajectory of two second-order-type quantities (D2f at output leg
    ``kpp(x,k,p)``, D2mf at ``kpp(-x,k,p)``, via cumulative variation of
    parameters), before a third homogeneous layer -- for the SAME output
    leg ``k`` as the leg's own growth factor, since ``mg_jax.thirdOrder``'s
    D3 equation and the leg-k equation share the identical homogeneous
    operator -- produces D3 itself. ``x, k, p`` broadcastable arrays;
    ``f0`` a scalar. See ``A_B_grid``'s docstring for ``n_ktable``/``n_eta``.
    """
    xb, kb, pb = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(p, dtype=jnp.float64),
    )
    shape = xb.shape
    flat = lambda a: a.reshape(-1)  # noqa: E731
    x_f, k_f, p_f = flat(xb), flat(kb), flat(pb)
    kplusp_f = bj.kpp(x_f, k_f, p_f)
    kpluspm_f = bj.kpp(-x_f, k_f, p_f)

    eta_grid = jnp.linspace(xnow, xstop, n_eta)
    kmin = jnp.min(jnp.concatenate([k_f, p_f, kplusp_f, kpluspm_f]))
    kmax = jnp.max(jnp.concatenate([k_f, p_f, kplusp_f, kpluspm_f]))
    lo, hi = _safe_k_range(kmin, kmax)
    k_table = jnp.geomspace(lo, hi, n_ktable)

    e1 = jnp.exp(xnow)
    y1_0 = jnp.array([e1, e1])
    y2_0 = jnp.array([e1, -2.0 * e1])
    y1_tab, dy1_tab = _basis_table(k_table, y1_0, eta_grid, P, xnow, xstop, rtol, atol)
    y2_tab, dy2_tab = _basis_table(k_table, y2_0, eta_grid, P, xnow, xstop, rtol, atol)

    Mmat = jnp.array([[y1_0[0], y2_0[0]], [y1_0[1], y2_0[1]]])
    Minv = jnp.linalg.inv(Mmat)

    one_m_x2 = 1.0 - x_f * x_f
    e2 = jnp.exp(2.0 * xnow)
    D2i = 3.0 * e2 / 7.0 * one_m_x2
    dD2i = 6.0 * e2 / 7.0 * one_m_x2
    c0_2 = Minv[0, 0] * D2i + Minv[0, 1] * dD2i
    c1_2 = Minv[1, 0] * D2i + Minv[1, 1] * dD2i

    Dpk = _interp_over_eta(k_f, k_table, y1_tab)  # full trajectory
    Dpp = _interp_over_eta(p_f, k_table, y1_tab)
    eta_row = eta_grid[None, :]

    def second_order_traj(kf_leg, x_local):
        y1_l = _interp_over_eta(kf_leg, k_table, y1_tab)
        y2_l = _interp_over_eta(kf_leg, k_table, y2_tab)
        W_l = y1_l * (_interp_over_eta(kf_leg, k_table, dy2_tab)) - y2_l * (_interp_over_eta(kf_leg, k_table, dy1_tab))
        src = bj.SD2(eta_row, x_local[:, None], k_f[:, None], p_f[:, None], P) * Dpk * Dpp
        I1 = _cumtrapz(y1_l * src / W_l, eta_grid)
        I2 = _cumtrapz(y2_l * src / W_l, eta_grid)
        u = y2_l * I1 - y1_l * I2
        return c0_2[:, None] * y1_l + c1_2[:, None] * y2_l + u

    D2f = second_order_traj(kplusp_f, x_f)
    D2mf = second_order_traj(kpluspm_f, -x_f)

    src3 = (bj.S3I(eta_row, x_f[:, None], k_f[:, None], p_f[:, None], Dpk, Dpp, D2f, D2mf, P)
            + bj.S3II(eta_row, x_f[:, None], k_f[:, None], p_f[:, None], Dpk, Dpp, D2f, D2mf, P)
            + bj.S3FL(eta_row, x_f[:, None], k_f[:, None], p_f[:, None], Dpk, Dpp, D2f, D2mf, P))

    y1_k = _interp_over_eta(k_f, k_table, y1_tab)
    dy1_k = _interp_over_eta(k_f, k_table, dy1_tab)
    y2_k = _interp_over_eta(k_f, k_table, y2_tab)
    dy2_k = _interp_over_eta(k_f, k_table, dy2_tab)
    W_k = y1_k * dy2_k - y2_k * dy1_k

    I1_3 = jnp.trapezoid(y1_k * src3 / W_k, eta_grid, axis=-1)
    I2_3 = jnp.trapezoid(y2_k * src3 / W_k, eta_grid, axis=-1)
    y1_end = y1_k[:, -1]; dy1_end = dy1_k[:, -1]
    y2_end = y2_k[:, -1]; dy2_end = dy2_k[:, -1]
    u3 = y2_end * I1_3 - y1_end * I2_3
    du3 = dy2_end * I1_3 - dy1_end * I2_3

    e3 = jnp.exp(3.0 * xnow)
    pk = p_f / k_f
    ang = 1.0 / (1.0 + pk * pk + 2.0 * pk * x_f) + 1.0 / (1.0 + pk * pk - 2.0 * pk * x_f)
    D3i = (5.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang
    dD3i = (15.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang
    c0_3 = Minv[0, 0] * D3i + Minv[0, 1] * dD3i
    c1_3 = Minv[1, 0] * D3i + Minv[1, 1] * dD3i

    D3 = c0_3 * y1_end + c1_3 * y2_end + u3
    dD3 = c0_3 * dy1_end + c1_3 * dy2_end + du3

    Dk_end = Dpk[:, -1]; Dp_end = Dpp[:, -1]
    denom = Dk_end * Dp_end * Dp_end
    CFD3 = (21.0 / 5.0) * D3 / denom
    CFD3p = (21.0 / 5.0) * dD3 / denom / (3.0 * f0)

    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return reshape(CFD3), reshape(CFD3p)
