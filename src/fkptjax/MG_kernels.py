"""
MG_kernels.py
-------------
Full (non-fkPT-approximated) beyond-EdS kernel building blocks.

``fkptjax.ode.kernel_constants`` / ``fkptjax.jax_ode.kernel_constants_jax``
implement the **fkPT approximation** (Aviles et al., arXiv:2312.10510 S4.1):
the second/third-order growth kernels A(k1,k2,t)/B(k1,k2,t) are evaluated
once in the large-scale limit (k1,k2 -> 0) and then reused as global scalar
constants for every mode in the one-loop momentum integral.

This module instead evaluates the *same*, already-validated growth ODEs
(``fkptjax.mg_jax.secondOrder`` / ``.thirdOrder``, and a new paired-source
variant for the second-order A/B split) at the genuine (not squeezed)
momentum triples needed by the loop integral, batched over the whole
quadrature grid via ``jax.vmap`` + a diffrax adaptive integrator by default
(mirrors ``fkptjax.jax_ode``'s ``solver='adaptive'`` path; a vmap-friendly
fixed-step RK4, mirroring ``jax_ode``'s ``solver='rk4'``, is available via
``solver='rk4'`` but is NOT the default here -- see "Solver choice" below).

Solver choice
-------------
Grid points near a collinear configuration (``q`` close to ``k_ext``, i.e.
``y = |k_ext-q|/k_ext -> 0``) drive a genuine, well-known SPT kernel
singularity in the RHS (``mg_jax.S3FLplus`` and friends divide by ``p*k`` and
``|k+p|``-type combinations that shrink towards zero there). The closed-form
fkPT-approximated kernels never hit this because they only ever evaluate the
ODEs at the squeezed (k,p) -> 0 point; evaluating at the genuine, grid-wide
(k,p,x) triples needed here means some grid points land arbitrarily close to
it. A **fixed**-step integrator (``solver='rk4'``) takes same-size steps
straight through that region regardless of how fast the RHS is varying there,
which was found (empirically, comparing against a real sMGPT reference table)
to produce wrong -- sometimes wrong-*sign* -- kernel values nearby. The
**adaptive** diffrax solver (default) shrinks its step size there instead,
which is why it is the default despite being slower and less
vmap-uniform (see ``jax_ode``'s own docstring on the padding-overhead
tradeoff). Use ``solver='rk4'`` only once you've checked your (k,p,x) grid
stays well clear of collinear configurations.

Physics reference / validation source
--------------------------------------
The exact (non-approximated) kernel construction implemented here was cross
-checked term-by-term against the collaborator reference implementation
``github.com/alejandroaviles/sMGPT`` (Mathematica, Sc=0 / "MG no Screenings"
branch -- fkptjax does not implement chameleon screening, matching that
branch, not sMGPT's Sc=1 "FullKernels" branch):

- ``sMGPT/src/2_P22type.wl:86-103`` defines the coupled ODE pair

      Af'' + f1 Af' - f2 mu(kf) Af == sourceA(kf,k1,k2) Dpk1 Dpk2
      Bf'' + f1 Bf' - f2 mu(kf) Bf == sourceb(kf,k1,k2) Dpk1 Dpk2

  with ``sourceA = S2a + S2FL`` and ``sourceb = S2b`` (verified algebraically
  against ``sMGPT/src/3_P13type.wl:70-72``, which are themselves identical to
  ``fkptjax.ode.ModelDerivatives.S2a/S2b/S2FL``).  This is ``secondOrderAB``
  below.
- ``sMGPT/src/3_P13type.wl:151,154`` shows the *fused* combination
  ``A - B*x^2`` (used for the P13-loop's Gamma2/Gamma3 pieces) is exactly
  ``D2v2``'s existing combined output, i.e. ``fkptjax.mg_jax.secondOrder`` /
  ``.thirdOrder`` called at a genuine (not squeezed) momentum triple already
  give the right answer with **no new source terms** -- ``D2_fused_grid`` and
  ``D3_fused_grid`` below just batch that over a grid.

Model coverage
---------------
Whatever ``fkptjax.mg_jax.mu(eta, k, P)`` supports -- the ODE machinery here
is already model-agnostic; only ``mu()`` is per-model.  As of this writing
that is every model :mod:`fkptjax.ode`'s ``ModelDerivatives`` implements:
LCDM/GR, HS (Hu-Sawicki f(R)), NDGP, HDKI (mu_OmDE, BZ, BZ_Mass, EFT_DE), and
PHENOM (binning, growth_index, growth_index_yukawa).  Extending to a new
model means adding its case to ``mg_jax.mu`` first.

Performance note
-----------------
The Q-loop grid used by ``calculate_jax.JaxCalculator`` has shape
``(NQ, nquadSteps-1, 1, Nk_kernel)`` -- e.g. 10*299*120 ~ 3.6e5 points with
the library defaults.  ``A_B_grid`` solves one 8-state ODE per grid point
(vmapped); this is the actual cost of *not* approximating, and is expected to
be much slower than the fkPT path.  Reduce ``Nk_kernel``/``nquadSteps``/``NQ``
for quick checks.
"""

from functools import partial

import jax
import jax.numpy as jnp
import diffrax

from . import mg_jax as bj

# Matches jax_ode.py's defaults -- validated there to be well converged for
# these smooth-away-from-singularities growth/kernel ODEs.
_RTOL = 1e-8
_ATOL = 1e-11
_MAXSTEPS = 100000
_N_STEPS = 128


def _solve_adaptive(rhs, y0, xnow, xstop, rtol=_RTOL, atol=_ATOL):
    """Adaptive diffrax solve of ``rhs(t, y)`` from ``xnow`` to ``xstop``.

    Mirrors ``jax_ode._solve``; shrinks its step size near sharp features
    (e.g. the collinear-kernel singularity below) instead of stepping
    through them at fixed size -- see the module docstring.
    """
    term = diffrax.ODETerm(lambda t, y, args: rhs(t, y))
    solver = diffrax.Tsit5()
    ctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term, solver, t0=xnow, t1=xstop, dt0=0.01, y0=y0,
        stepsize_controller=ctrl, saveat=diffrax.SaveAt(t1=True),
        max_steps=_MAXSTEPS,
    )
    return sol.ys[-1]


def _run_solver(rhs, y0, xnow, xstop, solver, n_steps, rtol, atol):
    if solver == 'rk4':
        return _solve_rk4_generic(rhs, y0, xnow, xstop, n_steps)
    if solver == 'adaptive':
        return _solve_adaptive(rhs, y0, xnow, xstop, rtol, atol)
    raise ValueError(f"solver must be 'adaptive' or 'rk4', got {solver!r}")


def _solve_rk4_generic(rhs, y0, xnow, xstop, n_steps):
    h = (xstop - xnow) / n_steps

    def step(carry, _):
        t, y = carry
        a = rhs(t, y)
        b = rhs(t + 0.5 * h, y + 0.5 * h * a)
        c = rhs(t + 0.5 * h, y + 0.5 * h * b)
        d = rhs(t + h, y + h * c)
        y = y + (h / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        return (t + h, y), None

    (_, yf), _ = jax.lax.scan(step, (xnow, y0), None, length=n_steps)
    return yf


def _x_from_triple(kf, k1, k2):
    """cos(angle) between legs k1,k2 such that |k1+k2| = kf.

    Matches sMGPT's magnitude-only ``(kf,k1,k2)`` parametrization
    (``2_P22type.wl``), which avoids re-deriving kf from (x,k1,k2) --  it is
    the exact algebraic inverse of ``mg_jax.kpp``.
    """
    return (kf * kf - k1 * k1 - k2 * k2) / (2.0 * k1 * k2)


# ---------------------------------------------------------------------------
# Separately-sourced A/B pair (needed for F2evQ/G2evQ-type, P22-loop kernels)
# ---------------------------------------------------------------------------

def secondOrderAB(eta, y, kf, k1, k2, P):
    """RHS for the 8-state system ``y = [Dk1,Dk1',Dk2,Dk2',DA,DA',DB,DB']``.

    ``DA`` is sourced by ``S2a + S2FL`` (the paper's caligraphic-A piece of
    the second-order growth kernel); ``DB`` by ``S2b`` alone (the caligraphic
    -B piece).  Both are driven by ``mu(eta, kf)`` on the LHS.  ``kf`` is the
    THIRD/output leg magnitude; ``k1,k2`` are the two input legs -- see the
    module docstring.
    """
    x = _x_from_triple(kf, k1, k2)
    f2 = bj.f1(eta, P)
    fr = 2.0 - f2
    mu_kf = bj.mu(eta, kf, P)
    src_A = bj.S2a(eta, x, k1, k2, P) + bj.S2FL(eta, x, k1, k2, P)
    src_B = bj.S2b(eta, x, k1, k2, P)
    return jnp.stack([
        y[1], f2 * bj.mu(eta, k1, P) * y[0] - fr * y[1],
        y[3], f2 * bj.mu(eta, k2, P) * y[2] - fr * y[3],
        y[5], f2 * mu_kf * y[4] - fr * y[5] + src_A * y[0] * y[2],
        y[7], f2 * mu_kf * y[6] - fr * y[7] + src_B * y[0] * y[2],
    ])


def _solve_AB(kf, k1, k2, P, xnow, xstop, solver, n_steps, rtol, atol):
    """Solve ``secondOrderAB`` for one (kf,k1,k2) triple."""
    e0 = jnp.exp(xnow)
    D2plusi = 3.0 * jnp.exp(2.0 * xnow) / 7.0
    dD2plusi = 6.0 * jnp.exp(2.0 * xnow) / 7.0
    y0 = jnp.stack([e0, e0, e0, e0, D2plusi, dD2plusi, D2plusi, dD2plusi])
    rhs = lambda t, y: secondOrderAB(t, y, kf, k1, k2, P)  # noqa: E731
    return _run_solver(rhs, y0, xnow, xstop, solver, n_steps, rtol, atol)


def A_B_grid_raw(kf, k1, k2, P, xnow, xstop, solver='adaptive', n_steps=_N_STEPS,
                  rtol=_RTOL, atol=_ATOL):
    """The raw 8-state ODE solution ``A_B_grid`` combines into ``(A, B, Ap, Bp)``,
    exposed separately: ``(Dk1, dDk1, Dk2, dDk2, DA, dDA, DB, dDB)``, each of the
    broadcast ``(kf, k1, k2)`` shape.

    Callers that need to EMULATE these quantities (e.g. ``fkptjax.ab_ingredients``)
    should fit these raw states, not ``Ap``/``Bp`` directly: ``Ap = dDA/C - DA*Cp/C^2``
    is a DIFFERENCE of two comparably-sized terms (a genuine quotient-rule derivative),
    so it can be -- and, measured for HDKI/BZ_Mass, is -- close to zero over much of
    parameter space even where every one of ``Dk1``, ``dDk1``, ``Dk2``, ``dDk2``, ``DA``,
    ``dDA`` is individually smooth and well-behaved; a Chebyshev/polynomial fit direct on
    ``Ap`` inherits that cancellation as apparent noise, producing enormous RELATIVE error
    (measured: 80%+ mean, thousands-of-% max) even though the fit is not meaningfully
    worse in absolute terms. Reconstructing ``Ap``/``Bp`` from independently-fit raw states
    via the same formula, at predict time, is mathematically identical (same formula, same
    inputs) but conditions the FIT itself on quantities that do not cancel.
    """
    kf_b, k1_b, k2_b = jnp.broadcast_arrays(
        jnp.asarray(kf, dtype=jnp.float64),
        jnp.asarray(k1, dtype=jnp.float64),
        jnp.asarray(k2, dtype=jnp.float64),
    )
    shape = kf_b.shape
    flat = lambda a: a.reshape(-1)  # noqa: E731

    solve = jax.vmap(
        lambda kf_, k1_, k2_: _solve_AB(kf_, k1_, k2_, P, xnow, xstop, solver, n_steps, rtol, atol)
    )
    Y = solve(flat(kf_b), flat(k1_b), flat(k2_b))  # (N, 8)
    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return tuple(reshape(Y[:, i]) for i in range(8))


def A_B_grid(kf, k1, k2, P, xnow, xstop, solver='adaptive', n_steps=_N_STEPS,
             rtol=_RTOL, atol=_ATOL):
    """Batched A(kf,k1,k2), B(kf,k1,k2) and their eta-derivatives.

    ``kf``, ``k1``, ``k2`` are broadcastable arrays (any common shape, e.g.
    the Q-loop's ``(NQ, nquadSteps-1, 1, Nk_kernel)``).  Returns four arrays
    of that broadcast shape: ``(A, B, Aprime, Bprime)``.

    Normalization matches ``ode.kernel_constants``'s ``ALS = D2p/C`` exactly
    (``C = (3/7) Dk1 Dk2``), just evaluated at the real triple instead of the
    squeezed (KMIN, KMIN, 0) point. ``solver``: see the module docstring
    ("Solver choice") -- default ``'adaptive'`` is the numerically robust
    choice; ``'rk4'`` is faster but unreliable near collinear (kf,k1,k2)
    triples. See :func:`A_B_grid_raw` for the underlying raw states, exposed
    separately for emulation callers.
    """
    Dk1, dDk1, Dk2, dDk2, DA, dDA, DB, dDB = A_B_grid_raw(
        kf, k1, k2, P, xnow, xstop, solver=solver, n_steps=n_steps, rtol=rtol, atol=atol)

    C = (3.0 / 7.0) * Dk1 * Dk2
    Cp = (3.0 / 7.0) * (dDk1 * Dk2 + Dk1 * dDk2)
    A = DA / C
    B = DB / C
    Ap = dDA / C - DA * Cp / (C * C)
    Bp = dDB / C - DB * Cp / (C * C)
    return A, B, Ap, Bp


# ---------------------------------------------------------------------------
# Fused (combined) second/third order kernels -- reuse the EXISTING,
# already-validated mg_jax.secondOrder / mg_jax.thirdOrder RHS unchanged;
# needed for the P13-loop's Gamma2/Gamma3-type pieces.
# ---------------------------------------------------------------------------

def _solve_D2(x, k, p, P, xnow, xstop, solver, n_steps, rtol, atol):
    e0 = jnp.exp(xnow)
    one_m_x2 = 1.0 - x * x
    D2i = 3.0 * jnp.exp(2.0 * xnow) / 7.0 * one_m_x2
    dD2i = 6.0 * jnp.exp(2.0 * xnow) / 7.0 * one_m_x2
    y0 = jnp.stack([e0, e0, e0, e0, D2i, dD2i])
    rhs = lambda t, y: bj.secondOrder(t, y, x, k, p, P)  # noqa: E731
    return _run_solver(rhs, y0, xnow, xstop, solver, n_steps, rtol, atol)


def D2_fused_grid(x, k, p, P, xnow, xstop, solver='adaptive', n_steps=_N_STEPS,
                   rtol=_RTOL, atol=_ATOL):
    """Batched, fused ``A - B*x^2`` combination (sMGPT's ``AminusBx2``).

    ``x, k, p`` broadcastable arrays.  Returns ``(A_minus_Bx2, Aprime_minus_Bprimex2)``
    of the broadcast shape -- the exact generalization of ``ode.kernel_constants``'s
    ``ALS, AprimeLS`` away from the squeezed point.  Used for the P13-loop's
    ``Gamma2evR``/``Gamma2fevR``. ``solver``: see the module docstring
    ("Solver choice").
    """
    xb, kb, pb = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(p, dtype=jnp.float64),
    )
    shape = xb.shape
    flat = lambda a: a.reshape(-1)  # noqa: E731

    solve = jax.vmap(
        lambda x_, k_, p_: _solve_D2(x_, k_, p_, P, xnow, xstop, solver, n_steps, rtol, atol)
    )
    Y = solve(flat(xb), flat(kb), flat(pb))  # (N, 6)
    Dk, dDk, Dp, dDp, D2, dD2 = (Y[:, i] for i in range(6))

    C = (3.0 / 7.0) * Dk * Dp
    Cp = (3.0 / 7.0) * (dDk * Dp + Dk * dDp)
    A = D2 / C
    Ap = dD2 / C - D2 * Cp / (C * C)

    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return reshape(A), reshape(Ap)


def _solve_D3(x, k, p, P, xnow, xstop, solver, n_steps, rtol, atol):
    e1 = jnp.exp(xnow)
    e2 = jnp.exp(2.0 * xnow)
    e3 = jnp.exp(3.0 * xnow)
    one_m_x2 = 1.0 - x * x
    pk = p / k
    ang = 1.0 / (1.0 + pk * pk + 2.0 * pk * x) + 1.0 / (1.0 + pk * pk - 2.0 * pk * x)
    y0 = jnp.stack([
        e1, e1, e1, e1,
        3.0 * e2 / 7.0 * one_m_x2, 6.0 * e2 / 7.0 * one_m_x2,
        3.0 * e2 / 7.0 * one_m_x2, 6.0 * e2 / 7.0 * one_m_x2,
        (5.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang,
        (15.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang,
    ])
    rhs = lambda t, y: bj.thirdOrder(t, y, x, k, p, P)  # noqa: E731
    return _run_solver(rhs, y0, xnow, xstop, solver, n_steps, rtol, atol)


def D3_fused_grid(x, k, p, P, xnow, xstop, f0, solver='adaptive', n_steps=_N_STEPS,
                   rtol=_RTOL, atol=_ATOL):
    """Batched, fused third-order kernel (sMGPT's ``C3gamma3``/``C3gamma3f``).

    ``x, k, p`` broadcastable arrays; ``f0`` a scalar.  Returns
    ``(CFD3, CFD3p)`` of the broadcast shape -- the exact generalization of
    ``ode.kernel_constants``'s ``KR1LS, KR1pLS`` away from the squeezed
    point.  Used for the P13-loop's ``C3Gamma3``/``C3Gamma3f``. ``solver``:
    see the module docstring ("Solver choice") -- the third-order RHS is the
    one with the sharpest collinear behavior, so this is where the adaptive
    default matters most.
    """
    xb, kb, pb = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(p, dtype=jnp.float64),
    )
    shape = xb.shape
    flat = lambda a: a.reshape(-1)  # noqa: E731

    solve = jax.vmap(
        lambda x_, k_, p_: _solve_D3(x_, k_, p_, P, xnow, xstop, solver, n_steps, rtol, atol)
    )
    Y = solve(flat(xb), flat(kb), flat(pb))  # (N, 10)
    Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3 = (Y[:, i] for i in range(10))

    denom = Dk * Dp * Dp
    CFD3 = (21.0 / 5.0) * D3 / denom
    CFD3p = (21.0 / 5.0) * dD3 / denom / (3.0 * f0)

    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return reshape(CFD3), reshape(CFD3p)


def D2_D3_fused_grid(x, k, p, P, xnow, xstop, f0, solver='adaptive', n_steps=_N_STEPS,
                      rtol=_RTOL, atol=_ATOL):
    """Combined D2 (at output leg ``kpp(-x,k,p)``) and D3 (fused third-order)
    kernels from a SINGLE 10-state ODE solve, instead of two separate ones.

    ``mg_jax.thirdOrder``'s own state already contains the "D2mf" quantity
    (the second-order fused kernel at output leg ``kpp(-x,k,p)``) as an
    intermediate on the way to D3 -- ``D3_fused_grid`` computes it but
    discards it. Verified (2026-09-02) that extracting and normalizing it
    the same way ``D2_fused_grid`` does reproduces
    ``D2_fused_grid(-x,k,p,P,xnow,xstop)``'s own output to ~1e-7-1e-10
    (floating-point-level agreement, not a real difference), so calling
    this function once is equivalent to calling ``D2_fused_grid(-x,...)``
    and ``D3_fused_grid(x,...,f0)`` separately, at roughly the cost of the
    D3 solve alone -- eliminating one whole redundant ODE solve for
    callers (like ``kfuncs_to_tables.py``) that need both.

    ``x, k, p`` broadcastable arrays; ``f0`` a scalar. Returns
    ``(A_fused, Aprime_fused, CFD3, CFD3p)`` -- the first two matching
    ``D2_fused_grid(-x,k,p,...)``, the last two matching
    ``D3_fused_grid(x,k,p,...,f0)`` exactly (unchanged formulas). See
    :func:`D2_D3_fused_grid_raw` for the underlying raw states, exposed
    separately for emulation callers (``Ap_fused``, like ``A_B_grid``'s own
    ``Ap``/``Bp``, is a cancellation-prone quotient-rule difference -- see
    that function's docstring).
    """
    (Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3) = D2_D3_fused_grid_raw(
        x, k, p, P, xnow, xstop, solver=solver, n_steps=n_steps, rtol=rtol, atol=atol)

    C = (3.0 / 7.0) * Dk * Dp
    Cp = (3.0 / 7.0) * (dDk * Dp + Dk * dDp)
    A_fused = D2m / C
    Ap_fused = dD2m / C - D2m * Cp / (C * C)

    denom = Dk * Dp * Dp
    CFD3 = (21.0 / 5.0) * D3 / denom
    CFD3p = (21.0 / 5.0) * dD3 / denom / (3.0 * f0)
    return A_fused, Ap_fused, CFD3, CFD3p


def D2_D3_fused_grid_raw(x, k, p, P, xnow, xstop, solver='adaptive', n_steps=_N_STEPS,
                          rtol=_RTOL, atol=_ATOL):
    """The raw 10-state ODE solution ``D2_D3_fused_grid`` combines into
    ``(A_fused, Ap_fused, CFD3, CFD3p)``, exposed separately:
    ``(Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3)``, each of the
    broadcast ``(x, k, p)`` shape. See :func:`A_B_grid_raw`'s docstring for
    why emulation callers should fit these raw states rather than
    ``Ap_fused`` directly.
    """
    xb, kb, pb = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(k, dtype=jnp.float64),
        jnp.asarray(p, dtype=jnp.float64),
    )
    shape = xb.shape
    flat = lambda a: a.reshape(-1)  # noqa: E731

    solve = jax.vmap(
        lambda x_, k_, p_: _solve_D3(x_, k_, p_, P, xnow, xstop, solver, n_steps, rtol, atol)
    )
    Y = solve(flat(xb), flat(kb), flat(pb))  # (N, 10)
    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return tuple(reshape(Y[:, i]) for i in range(10))


# ---------------------------------------------------------------------------
# Fully combined I1udd1-family (Q/N/R orderings) + P13-loop (D2/D3) grid --
# ONE vmap/diffrax dispatch instead of two separate ones (A_B_grid's merged
# Q/N/R call, and D2_D3_fused_grid). secondOrderAB (8-state) and thirdOrder
# (10-state) are genuinely different ODEs, so this pads the 8-state system
# to 10 (two unused, zero-driven dummy states) and selects the right
# RHS/initial-condition pair per point via jnp.where on a "mode" flag (both
# branches are evaluated for every point under vmap/jit -- there is no
# actual control-flow branching -- so this trades some wasted per-step FLOPs
# for one fewer top-level dispatch; validated (2026-09-02) this trade is a
# net win here since dispatch overhead, not FLOPs, dominates the cost).
# ---------------------------------------------------------------------------

def _combined_rhs(eta, y, kf, k1, k2, x, k, p, mode, P):
    y8 = y[:8]
    rhs8 = secondOrderAB(eta, y8, kf, k1, k2, P)
    rhs_A_padded = jnp.concatenate([rhs8, jnp.zeros(2, dtype=y.dtype)])
    rhs_D3 = bj.thirdOrder(eta, y, x, k, p, P)
    return jnp.where(mode < 0.5, rhs_A_padded, rhs_D3)


def _combined_y0(x, k, p, mode, xnow):
    e0 = jnp.exp(xnow)
    e2 = jnp.exp(2.0 * xnow)
    e3 = jnp.exp(3.0 * xnow)
    D2plusi = 3.0 * e2 / 7.0
    dD2plusi = 6.0 * e2 / 7.0
    y0_A = jnp.stack([e0, e0, e0, e0, D2plusi, dD2plusi, D2plusi, dD2plusi, 0.0 * e0, 0.0 * e0])

    one_m_x2 = 1.0 - x * x
    pk = p / k
    ang = 1.0 / (1.0 + pk * pk + 2.0 * pk * x) + 1.0 / (1.0 + pk * pk - 2.0 * pk * x)
    D2i = 3.0 * e2 / 7.0 * one_m_x2
    dD2i = 6.0 * e2 / 7.0 * one_m_x2
    D3i = (5.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang
    dD3i = (15.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang
    y0_D3 = jnp.stack([e0, e0, e0, e0, D2i, dD2i, D2i, dD2i, D3i, dD3i])

    return jnp.where(mode < 0.5, y0_A, y0_D3)


def _solve_combined(kf, k1, k2, x, k, p, mode, P, xnow, xstop, solver, n_steps, rtol, atol):
    y0 = _combined_y0(x, k, p, mode, xnow)
    rhs = lambda t, y: _combined_rhs(t, y, kf, k1, k2, x, k, p, mode, P)  # noqa: E731
    return _run_solver(rhs, y0, xnow, xstop, solver, n_steps, rtol, atol)


@partial(jax.jit, static_argnames=('solver', 'n_steps', 'rtol', 'atol'))
def I1udd1_and_P13_grid(k_ext_q, q_loop_Q, kminus_Q, x_r, k_r, p_r, P, xnow, xstop, f0,
                         solver='adaptive', n_steps=_N_STEPS, rtol=_RTOL, atol=_ATOL):
    """Everything ``kfuncs_to_tables.py``'s full-kernel path needs from this
    module, in ONE combined ODE batch: the I1udd1-family's three orderings
    (Q/N/R, ``A_B_grid``'s formula, on the {k_ext_q, q_loop_Q, kminus_Q}
    Q-loop triple) AND the P13-loop's fused D2/D3 kernel
    (``D2_D3_fused_grid``'s formula, on the {x_r, k_r, p_r} R-loop triple).

    Verified (2026-09-02) to reproduce calling ``A_B_grid`` three times (for
    the Q/N/R orderings) and ``D2_D3_fused_grid`` once separately, to
    ~1e-10, for both a scale-independent (HDKI/mu_OmDE) and a
    scale-dependent (Hu-Sawicki f(R)) model, at ~1.8-2x the speed on GPU.

    ``@jax.jit``: an additional ~2.6x on top of that (measured at production
    resolution) once ``P`` (an ``mg_jax.MGConstants``) could be jit-traced
    directly, made possible by its pytree registration in mg_jax.py --
    ``kind``/``scale_bins``/``eftde_scale_dependent`` are static aux data
    (one compile per model choice), every numeric field is a traced child
    (reused across calls that vary only the MG parameters, e.g. across an
    MCMC chain at fixed model choice).

    ``k_ext_q, q_loop_Q, kminus_Q`` and ``x_r, k_r, p_r`` are broadcastable
    arrays (need not share a shape with each other). ``f0`` a scalar.

    Returns ``(A_Q, B_Q, ApOverf0_Q, BpOverf0_Q, A_N, B_N, ApOverf0_N,
    BpOverf0_N, A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ, A_fused, ApOverf0_fused,
    CFD3, CFD3p)`` -- NOTE: the ``Ap``/``Bp`` outputs here are already
    divided by ``f0`` (i.e. ``ApOverf0``/``BpOverf0``, matching what
    ``kfuncs_to_tables.py`` needs directly), unlike ``A_B_grid``'s own raw
    ``Ap``/``Bp``.
    """
    kf_Q_b, q_Q_b, kminus_Q_b = jnp.broadcast_arrays(
        jnp.asarray(k_ext_q, dtype=jnp.float64),
        jnp.asarray(q_loop_Q, dtype=jnp.float64),
        jnp.asarray(kminus_Q, dtype=jnp.float64),
    )
    shape_Q = kf_Q_b.shape
    x_r_b, k_r_b, p_r_b = jnp.broadcast_arrays(
        jnp.asarray(x_r, dtype=jnp.float64),
        jnp.asarray(k_r, dtype=jnp.float64),
        jnp.asarray(p_r, dtype=jnp.float64),
    )
    shape_R = x_r_b.shape

    flat = lambda a: a.reshape(-1)  # noqa: E731
    npts_Q = kf_Q_b.size
    npts_R = x_r_b.size
    zeros_Q3 = jnp.zeros(3 * npts_Q, dtype=jnp.float64)
    zeros_R = jnp.zeros(npts_R, dtype=jnp.float64)
    ones_R = jnp.ones(npts_R, dtype=jnp.float64)

    kf_qnr = jnp.concatenate([flat(kf_Q_b), flat(q_Q_b), flat(kminus_Q_b)])
    k1_qnr = jnp.concatenate([flat(q_Q_b), flat(kf_Q_b), flat(kf_Q_b)])
    k2_qnr = jnp.concatenate([flat(kminus_Q_b), flat(kminus_Q_b), flat(q_Q_b)])

    kf_all = jnp.concatenate([kf_qnr, flat(k_r_b)])
    k1_all = jnp.concatenate([k1_qnr, zeros_R])
    k2_all = jnp.concatenate([k2_qnr, zeros_R])
    x_all = jnp.concatenate([zeros_Q3, flat(x_r_b)])
    k_all = jnp.concatenate([zeros_Q3, flat(k_r_b)])
    p_all = jnp.concatenate([zeros_Q3, flat(p_r_b)])
    mode_all = jnp.concatenate([zeros_Q3, ones_R])

    solve = jax.vmap(
        lambda kf_, k1_, k2_, x_, k_, p_, m_: _solve_combined(
            kf_, k1_, k2_, x_, k_, p_, m_, P, xnow, xstop, solver, n_steps, rtol, atol)
    )
    Y = solve(kf_all, k1_all, k2_all, x_all, k_all, p_all, mode_all)

    def _AB(Yb, shape):
        Dk1, dDk1, Dk2, dDk2, DA, dDA, DB, dDB = (Yb[:, i] for i in range(8))
        C = (3.0 / 7.0) * Dk1 * Dk2
        Cp = (3.0 / 7.0) * (dDk1 * Dk2 + Dk1 * dDk2)
        A = DA / C
        B = DB / C
        Ap = dDA / C - DA * Cp / (C * C)
        Bp = dDB / C - DB * Cp / (C * C)
        reshape = lambda a: a.reshape(shape)  # noqa: E731
        return reshape(A), reshape(B), reshape(Ap / f0), reshape(Bp / f0)

    A_Q, B_Q, ApOverf0_Q, BpOverf0_Q = _AB(Y[:npts_Q], shape_Q)
    A_N, B_N, ApOverf0_N, BpOverf0_N = _AB(Y[npts_Q:2 * npts_Q], shape_Q)
    A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ = _AB(Y[2 * npts_Q:3 * npts_Q], shape_Q)

    Y_R = Y[3 * npts_Q:]
    Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3 = (Y_R[:, i] for i in range(10))
    Cr = (3.0 / 7.0) * Dk * Dp
    Cpr = (3.0 / 7.0) * (dDk * Dp + Dk * dDp)
    A_fused = D2m / Cr
    Ap_fused = dD2m / Cr - D2m * Cpr / (Cr * Cr)
    denom = Dk * Dp * Dp
    CFD3 = (21.0 / 5.0) * D3 / denom
    CFD3p = (21.0 / 5.0) * dD3 / denom / (3.0 * f0)
    reshape_R = lambda a: a.reshape(shape_R)  # noqa: E731

    return (A_Q, B_Q, ApOverf0_Q, BpOverf0_Q,
            A_N, B_N, ApOverf0_N, BpOverf0_N,
            A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ,
            reshape_R(A_fused), reshape_R(Ap_fused / f0), reshape_R(CFD3), reshape_R(CFD3p))
