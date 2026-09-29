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
    triples.
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
    Dk1, dDk1, Dk2, dDk2, DA, dDA, DB, dDB = (Y[:, i] for i in range(8))

    C = (3.0 / 7.0) * Dk1 * Dk2
    Cp = (3.0 / 7.0) * (dDk1 * Dk2 + Dk1 * dDk2)
    A = DA / C
    B = DB / C
    Ap = dDA / C - DA * Cp / (C * C)
    Bp = dDB / C - DB * Cp / (C * C)

    reshape = lambda a: a.reshape(shape)  # noqa: E731
    return reshape(A), reshape(B), reshape(Ap), reshape(Bp)


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
