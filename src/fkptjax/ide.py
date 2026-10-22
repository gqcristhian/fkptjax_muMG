"""
ide.py
------
Interacting-dark-energy (IDE) beyond-EdS one-loop kernel coefficients.

For an IDE model whose growth rescaling factor g depends on time only, the nonlinear
kernels are exactly scale independent. Everything the k-functions need is
therefore a handful of scalars at the output redshift, collected in
:class:`IDEKernelCoeffs`.

Notation and equations: Kamble & Ishak, IDE full-shape notes (in prep.), cited
below as "notes" (Sec. 1 model and fluid equations, Sec. 3 growth equations in
N = ln a, Sec. 4 kernels, Sec. 6 IR resummation, Sec. 10 dictionaries):

- time variable N = ln a, primes are d/dN;
- velocity divergence normalised by the velocity growth rate,
  theta = -div(v)/(a H fQ), fQ = f + g, so F1 = G1 = 1 and fQ plays the role of f0
  (every f(k)/f0 ratio is 1);
- displacement amplitudes Gamma1, L, F, S and their mass-weight partners
  W1 = 1 - Gamma1, WL, WF, WS.

The background enters only through three functions of N (notes Sec. 1, eq. 3):

    E(N)  = 2 + dlnH/dN                (friction)
    A_Phi(N) = A(t)/H^2 = (3/2) Omega_cb(N)   (Poisson coefficient in N-time)
    g(N)  = Qbar/(H rhobar_cb) (1 - dQ/(Qbar delta_cb))   (dilution: delta' + g delta + ... = 0)

supplied by any background object exposing ``__call__(N) -> (E, A_Phi, g)`` that
is also a JAX pytree. :class:`TabulatedBackground` wraps tables from an IDE
Boltzmann run; :class:`TanhDilutionLCDM` is an analytic tanh-dilution benchmark
(the collaborator's fkpt-DMdilution test model). The growth system is integrated
with fixed-step RK4 or diffrax, and :func:`ide_kernel_coeffs` is jit-compiled and
differentiable with respect to the background parameters and ``z_out``.

Requires ``jax_enable_x64=True`` (fkptjax is float64).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx

jax.config.update("jax_enable_x64", True)

# Adaptive (diffrax) reference solve: tighter than jax_ode's defaults, since the
# coefficients are a handful of scalars. atol is far below a_i**3 ~ 1e-5.
_RTOL = 1e-10
_ATOL = 1e-14
_MAXSTEPS = 100000

# Fixed-step RK4 (the default, as in jax_ode/MG_kernels): uniform steps in N,
# one lax.scan, identical work for every call, so it runs well on GPU and vmaps
# cleanly. The adaptive solve's per-step control flow is ~400x slower on GPU.
# Step count from a convergence test against the adaptive solve: 256 steps from
# z_ini = 50 give < 4e-7 absolute error in every coefficient for the tanh-dilution
# benchmark (dz = 0.2) and ~4e-8 for ISiTIDE backgrounds; a much sharper g(z)
# needs more (dz = 0.05: 384+). See tests/test_ide.py.
_N_STEPS = 256
# Scan unroll: ~1.6x faster on GPU (fewer loop iterations), ~same on CPU.
_UNROLL = 8

# Default starting redshift. The EdS initial data assume Omega_cb ~ 1 and
# |g| ~ 0 there; check both for your background (or use ic='growing').
Z_INI = 50.0

# Indices of the growth state vector (notes Sec. 3.1-3.3).
(_D, _DW, _DTHETA, _DL, _DLP, _DWL, _DF, _DFP, _DWF, _DS, _DSP, _DWS) = range(12)
STATE_NAMES = ('D', 'Dw', 'DTheta', 'DL', 'DLp', 'DWL', 'DF', 'DFp', 'DWF', 'DS', 'DSp', 'DWS')


# ----------------------------------------------------------------------------
# Backgrounds
# ----------------------------------------------------------------------------

class TabulatedBackground(NamedTuple):
    """E, A_Phi and g tabulated on a uniform, increasing grid in N = ln a.

    Built from an IDE Boltzmann solution. Values between nodes use four-point
    (cubic Lagrange) interpolation located by index arithmetic, so evaluation
    is a gather rather than a search (fast on GPU), with error O(dN^4); a
    few thousand points over N in [ln a_ini, 0] is ample. The grid must be
    uniform and cover [ln a_ini, ln a_out]; :func:`ide_growth` raises
    otherwise (also under jit). ``E`` must be ``2 + dlnH/dN`` and ``A_Phi`` must
    be ``1.5 * Omega_cb(N)``, both from the IDE background (not LCDM).
    """
    N: jax.Array
    E: jax.Array
    A_Phi: jax.Array
    g: jax.Array

    def __call__(self, N):
        n = self.N.shape[0]
        u = (N - self.N[0]) / (self.N[1] - self.N[0])
        i = jnp.clip(jnp.floor(u).astype(jnp.int32), 1, n - 3)
        t = u - i
        w = ((-t * (t - 1.0) * (t - 2.0) / 6.0),
             ((t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0),
             (-(t + 1.0) * t * (t - 2.0) / 2.0),
             ((t + 1.0) * t * (t - 1.0) / 6.0))

        def interp(arr):
            return w[0] * arr[i - 1] + w[1] * arr[i] + w[2] * arr[i + 1] + w[3] * arr[i + 2]

        return interp(self.E), interp(self.A_Phi), interp(self.g)

    def check_range(self, N_lo, N_hi, x):
        """Return ``x``, raising if the grid is not uniform or does not cover
        [N_lo, N_hi]."""
        dN = self.N[1] - self.N[0]
        eps = 1e-10
        outside = (N_lo < self.N[0] - eps) | (N_hi > self.N[-1] + eps)
        x = eqx.error_if(
            x, outside, "TabulatedBackground grid does not cover [ln a_ini, ln a_out]")
        uneven = jnp.max(jnp.abs(jnp.diff(self.N) - dN)) > 1e-6 * jnp.abs(dN)
        return eqx.error_if(x, uneven, "TabulatedBackground grid in N must be uniform")


class TanhDilutionLCDM(NamedTuple):
    """Analytic benchmark (Alejandro's fkpt-DMdilution test model): flat
    LCDM background (no radiation) with g(z) = g0/2 [1 - tanh((z - zt)/dz)]
    acting on the matter contrast only.

    ``g0 = 0`` gives plain flat LCDM.
    """
    Om0: float
    g0: float
    zt: float
    dz: float

    def __call__(self, N):
        Om = self.Om0 / (self.Om0 + (1.0 - self.Om0) * jnp.exp(3.0 * N))
        z = jnp.exp(-N) - 1.0
        g = 0.5 * self.g0 * (1.0 - jnp.tanh((z - self.zt) / self.dz))
        return 2.0 - 1.5 * Om, 1.5 * Om, g


# ----------------------------------------------------------------------------
# Growth system
# ----------------------------------------------------------------------------

def growth_rhs(N, y, background):
    """d/dN of the growth state (notes Sec. 3.1-3.3, eqs. 17, 19-20, 22-25).

    State: [D, Dw, DTheta, DL, DL', DWL, DF, DF', DWF, DS, DS', DWS], with
    Gamma1 = 1 + Dw/D:

        D'      = DTheta - g D,   Dw' = g D,   DTheta' = -E DTheta + A_Phi D
        DL'' + E DL' - A_Phi DL = A_Phi (D^2 - Dw^2 + DWL)
        DWL' + g DWL         = g (Gamma1^2 D^2 - DL)
        DF'' + E DF' - A_Phi DF = A_Phi [D DL + Gamma1 D (D^2 - Dw^2 + DWL) + DWF]
        DWF' + g DWF         = g (Gamma1 D DL - DF)
        DS'' + E DS' - A_Phi DS = A_Phi [(Gamma1^3 - 3 Gamma1^2) D^3 + DWS]
        DWS' + g DWS         = -g (Gamma1^3 D^3 + DS)
    """
    E, A_Phi, g = background(N)
    D, Dw, DTheta, DL, DLp, DWL, DF, DFp, DWF, DS, DSp, DWS = y
    G1D = D + Dw                      # Gamma1 D
    D2 = D * D
    D3 = D2 * D
    src2 = D2 - Dw * Dw + DWL         # D^2 - Dw^2 + DWL
    return jnp.stack([
        DTheta - g * D,
        g * D,
        -E * DTheta + A_Phi * D,
        DLp,
        -E * DLp + A_Phi * DL + A_Phi * src2,
        -g * DWL + g * (G1D * G1D - DL),
        DFp,
        -E * DFp + A_Phi * DF + A_Phi * (D * DL + G1D * src2 + DWF),
        -g * DWF + g * (G1D * DL - DF),
        DSp,
        -E * DSp + A_Phi * DS + A_Phi * (G1D * G1D * G1D - 3.0 * D * G1D * G1D + DWS),
        -g * DWS - g * (G1D * G1D * G1D + DS),
    ])


def eds_initial_state(a_ini):
    """EdS growing mode at a_ini (notes Sec. 3.4, eq. 27): Gamma1 = L = F = S = 1, WL = WF = WS = 0."""
    a2 = a_ini * a_ini
    a3 = a2 * a_ini
    zero = 0.0 * a_ini
    return jnp.stack([
        a_ini, zero, a_ini,
        3.0 / 7.0 * a2, 6.0 / 7.0 * a2, zero,
        5.0 / 21.0 * a3, 5.0 / 7.0 * a3, zero,
        -a3 / 3.0, -a3, zero,
    ])


def growing_mode_initial_state(background, N_ini):
    """Growing mode of the full system at N_ini, with E, A_Phi and g frozen there.

    Each order takes its power-law particular solution, D ~ a^n, second order ~ a^2n,
    third order ~ a^3n:

        (n + g)(n + E) = A_Phi,   DTheta = (n + g) D,   Gamma1 = r = (n + g)/n,
        DL = l D^2, DWL = w D^2:
            l [(2n)^2 + 2nE - A_Phi] - A_Phi w = A_Phi [1 - (r - 1)^2],   g l + (2n + g) w = g r^2,
        DF = q D^3, DWF = u D^3:
            q [(3n)^2 + 3nE - A_Phi] - A_Phi u = A_Phi [l + r (1 - (r - 1)^2 + w)],   g q + (3n + g) u = g r l,
        DS = s D^3, DWS = v D^3:
            s [(3n)^2 + 3nE - A_Phi] - A_Phi v = A_Phi (r^3 - 3 r^2),   g s + (3n + g) v = -g r^3.

    For g = 0 this is the power-law mode of notes Sec. 3.4, eq. 28 (Model 2); in EdS (n = 1,
    g = 0, E = 1/2, A_Phi = 3/2) it reduces to :func:`eds_initial_state`. Unlike the EdS
    data it carries no decaying-mode transient when the coupling is already on at z_ini
    (Q proportional to rho_c), or when Omega_cb(z_ini) < 1. D is normalised to a_ini,
    as in :func:`eds_initial_state`.
    """
    E, A_Phi, g = background(N_ini)
    s_ = g + E
    n = 0.5 * (-s_ + jnp.sqrt(s_ * s_ - 4.0 * (g * E - A_Phi)))
    r = (n + g) / n
    d = jnp.exp(N_ini)

    def solve2(a, b, c1, c2):
        # a X - A_Phi Y = c1,  g X + b Y = c2
        det = a * b + A_Phi * g
        return (c1 * b + A_Phi * c2) / det, (a * c2 - g * c1) / det

    a2, b2 = (2.0 * n) ** 2 + 2.0 * n * E - A_Phi, 2.0 * n + g
    l, w = solve2(a2, b2, A_Phi * (1.0 - (r - 1.0) ** 2), g * r * r)

    a3, b3 = (3.0 * n) ** 2 + 3.0 * n * E - A_Phi, 3.0 * n + g
    q, u = solve2(a3, b3, A_Phi * (l + r * (1.0 - (r - 1.0) ** 2 + w)), g * r * l)
    s, v = solve2(a3, b3, A_Phi * (r ** 3 - 3.0 * r * r), -g * r ** 3)

    d2, d3 = d * d, d * d * d
    return jnp.stack([d, (r - 1.0) * d, (n + g) * d,
                      l * d2, 2.0 * n * l * d2, w * d2,
                      q * d3, 3.0 * n * q * d3, u * d3,
                      s * d3, 3.0 * n * s * d3, v * d3])


def _initial_state(ic, background, N_ini):
    if ic == 'eds':
        return eds_initial_state(jnp.exp(N_ini))
    if ic == 'growing':
        return growing_mode_initial_state(background, N_ini)
    raise ValueError(f"ic must be 'eds' or 'growing', got {ic!r}")


def _solve_rk4_dense(rhs, y0, N_ini, N_end, N_out, n_steps, unroll=1):
    """Classical RK4 on uniform steps in one lax.scan (cf. jax_ode._solve_rk4),
    read off at every N_out by cubic Hermite dense output from the node values
    and slopes (O(h^4), like RK4 itself)."""
    h = (N_end - N_ini) / n_steps

    def step(carry, _):
        N, y = carry
        k1 = rhs(N, y)
        k2 = rhs(N + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(N + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(N + h, y + h * k3)
        return (N + h, y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)), (y, k1)

    (_, y_end), (ys, fs) = jax.lax.scan(step, (N_ini, y0), None, length=n_steps, unroll=unroll)
    ys = jnp.concatenate([ys, y_end[None]], axis=0)
    fs = jnp.concatenate([fs, rhs(N_end, y_end)[None]], axis=0)

    u = (N_out - N_ini) / h
    i = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_steps - 1)
    t = (u - i)[:, None]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * fs[i]
            + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * fs[i + 1])


def _solve_adaptive(rhs, y0, N_ini, N_end, N_out, rtol, atol):
    """Reference: adaptive diffrax Tsit5, saved at every N_out."""
    # lax.sort with an explicit int32 index rather than jnp.argsort: under x64 the
    # latter trips an XLA GPU verifier bug (s32 vs s64 in the sort rewrite).
    n = N_out.shape[0]
    ts, order = jax.lax.sort((N_out, jnp.arange(n, dtype=jnp.int32)), num_keys=1)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda N, y, args: rhs(N, y)), diffrax.Tsit5(),
        t0=N_ini, t1=N_end, dt0=0.01, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=ts), max_steps=_MAXSTEPS,
    )
    return jnp.zeros_like(sol.ys).at[order].set(sol.ys)


@partial(jax.jit, static_argnames=('solver', 'n_steps', 'rtol', 'atol', 'ic'))
def ide_growth(z_out, background, z_ini=Z_INI, solver='rk4', n_steps=_N_STEPS,
               rtol=_RTOL, atol=_ATOL, ic='eds'):
    """Integrate the growth system from z_ini to every z_out in one pass.

    ``z_out`` is a scalar or an array. Returns ``(y, dy)`` with shape
    ``z_out.shape + (12,)``: the state (order :data:`STATE_NAMES`) and its
    N-derivative (from the RHS, never by finite differences) at each z_out.
    D is normalised to a_ini at z_ini, so ratios of D between two backgrounds
    sharing z_ini are primordial-matched.

    ``solver='rk4'`` (default): fixed-step RK4 with ``n_steps`` uniform steps
    from z_ini to min(z_out). ``solver='adaptive'``: diffrax Tsit5 with
    ``rtol``/``atol``, the reference.

    ``ic='eds'`` (default): EdS growing mode at z_ini (notes Sec. 3.4).
    ``ic='growing'``: growing mode of the full system with E, A_Phi, g frozen at
    z_ini (:func:`growing_mode_initial_state`); use it for Boltzmann
    backgrounds, and always when the coupling is on at z_ini (Q proportional to
    rho_c: the EdS data then leave ~0.25% (beta = 0.1) to ~0.7% (beta = 0.3)
    errors in the second- and third-order amplitudes that decay only slowly
    with z_ini).
    """
    z_out = jnp.asarray(z_out, dtype=jnp.float64)
    N_out = -jnp.log1p(z_out).ravel()
    N_ini = -jnp.log1p(jnp.asarray(z_ini, dtype=jnp.float64))
    N_end = jnp.max(N_out)
    y0 = _initial_state(ic, background, N_ini)
    if hasattr(background, 'check_range'):
        y0 = background.check_range(N_ini, N_end, y0)
    rhs = lambda N, y: growth_rhs(N, y, background)
    if solver == 'rk4':
        y = _solve_rk4_dense(rhs, y0, N_ini, N_end, N_out, n_steps, _UNROLL)
    elif solver == 'adaptive':
        y = _solve_adaptive(rhs, y0, N_ini, N_end, N_out, rtol, atol)
    else:
        raise ValueError(f"solver must be 'rk4' or 'adaptive', got {solver!r}")
    dy = jax.vmap(rhs)(N_out, y)
    shape = z_out.shape + (y.shape[-1],)
    return y.reshape(shape), dy.reshape(shape)


# ----------------------------------------------------------------------------
# Kernel coefficients
# ----------------------------------------------------------------------------

class IDEKernelCoeffs(NamedTuple):
    """Scale-independent IDE kernel coefficients at one output redshift.

    A JAX pytree of scalars, so it can cross jit boundaries and carry
    gradients. It is the single carrier of everything IDE-specific downstream:
    the loop kernels (Gamma1, L, WL, F, WF and primes, f, fQ), the table
    normalisation (fQ, which must equal the builders' f0) and the IR damping
    (Gamma1, fPsi). ``z_out`` records the redshift the coefficients belong to.

    Second order (notes Sec. 4.2, eqs. 32, 34), for momenta (k1, k2) with cosine mu,
    alpha_s = 1 + mu/2 (k1/k2 + k2/k1) and gamma12 = 1 - mu^2:

        F2 = Gamma1 alpha_s + [3/14 (L + WL) - Gamma1^2/2] gamma12
        G2 = Gamma1 alpha_s + r_G gamma12,   r_G = 3/7 ((f/fQ) L + L'/(2 fQ)) - Gamma1

    Third order, collapsed (P13): :func:`F3K_G3K` (notes Sec. 4.3, eqs. 43-44). The
    Sigma_3 amplitudes S, WS do not enter P13; they are kept for the one-loop
    bispectrum. In LCDM (g = 0): Gamma1 = 1, fQ = f, WL = WF = WS = 0.
    """
    z_out: jax.Array
    D: jax.Array        # linear growth, normalised to a_ini at z_ini
    f: jax.Array        # D'/D: density growth rate
    fQ: jax.Array       # DTheta/D = f + g: velocity growth rate; plays f0
    Gamma1: jax.Array   # 1 + Dw/D: linear displacement per unit linear density
    fPsi: jax.Array     # fQ/Gamma1 = dln(Gamma1 D)/dN: displacement growth rate (IR damping)
    L: jax.Array        # 7 DL/(3 D^2): second-order displacement amplitude
    Lp: jax.Array       # L' = 7/3 (DL'/D^2 - 2 f DL/D^2)
    WL: jax.Array       # 7 DWL/(3 D^2): second-order mass weight
    F: jax.Array        # 21 DF/(5 D^3): third-order displacement, gamma-gamma sector
    Fp: jax.Array       # F' = 21/5 (DF'/D^3 - 3 f DF/D^3)
    WF: jax.Array       # 21 DWF/(5 D^3): third-order mass weight, gamma-gamma sector
    S: jax.Array        # -3 DS/D^3: third-order displacement, Sigma_3 sector
    WS: jax.Array       # -3 DWS/D^3: third-order mass weight, Sigma_3 sector


def coeffs_from_growth(z_out, y, dy):
    """Amplitudes of notes Sec. 3.2-3.3 (eqs. 21, 26) from the growth state at z_out.

    ``dy`` is the N-derivative of the state from the RHS (never finite differences).
    """
    D, Dw, DTheta, DL, DLp, DWL, DF, DFp, DWF, DS, DSp, DWS = jnp.moveaxis(y, -1, 0)
    Dp = dy[..., _D]

    f = Dp / D
    fQ = DTheta / D
    Gamma1 = 1.0 + Dw / D
    fPsi = fQ / Gamma1

    D2 = D * D
    D3 = D2 * D
    L = 7.0 * DL / (3.0 * D2)
    WL = 7.0 * DWL / (3.0 * D2)
    Lp = 7.0 / 3.0 * (DLp / D2 - 2.0 * f * DL / D2)
    F = 21.0 * DF / (5.0 * D3)
    WF = 21.0 * DWF / (5.0 * D3)
    Fp = 21.0 / 5.0 * (DFp / D3 - 3.0 * f * DF / D3)
    S = -3.0 * DS / D3
    WS = -3.0 * DWS / D3

    return IDEKernelCoeffs(
        z_out=jnp.asarray(z_out, dtype=jnp.float64), D=D, f=f, fQ=fQ, Gamma1=Gamma1,
        fPsi=fPsi, L=L, Lp=Lp, WL=WL, F=F, Fp=Fp, WF=WF, S=S, WS=WS)


@partial(jax.jit, static_argnames=('solver', 'n_steps', 'rtol', 'atol', 'ic'))
def ide_kernel_coeffs(z_out, background, z_ini=Z_INI, solver='rk4', n_steps=_N_STEPS,
                      rtol=_RTOL, atol=_ATOL, ic='eds'):
    """IDE kernel coefficients at ``z_out`` (scalar or array) for ``background``.

    ``background`` is any pytree with ``__call__(N) -> (E, A_Phi, g)``, e.g.
    :class:`TabulatedBackground` or :class:`TanhDilutionLCDM`. For an array
    ``z_out`` every field of the result has ``z_out``'s shape; all redshifts
    come from one integration. Jit-compiled and differentiable with respect
    to ``z_out`` and the background's leaves. See :func:`ide_growth` for
    ``solver``/``n_steps``/``rtol``/``atol``/``ic``.
    """
    z_out = jnp.asarray(z_out, dtype=jnp.float64)
    y, dy = ide_growth(z_out, background, z_ini=z_ini, solver=solver, n_steps=n_steps,
                       rtol=rtol, atol=atol, ic=ic)
    return coeffs_from_growth(z_out, y, dy)


def to_DMdilution_notation(c):
    """Translate :class:`IDEKernelCoeffs` into the DM-dilution notation (notes Sec. 10).

    For cross-checks only (benchmark tables, Eulerian time basis); nothing in
    fkptjax uses the result. Dictionary:

        R = Gamma1, fv = fQ, Nw = WF,
        DL = 2 D2Psi, DWL = 2 DM2, DF = 2 D3A, DWF = 2 DM3A, DS = -6 D3B, DWS = 6 DM3B,
        C = 7/3 (Gamma1 - 1),  A = L + WL - 7/3 (Gamma1 - 1)^2,  B = A - 2C,
        u0 = Gamma1 + r_G,  u2 = -r_G,  r_G = 3/7 ((f/fQ) L + L'/(2 fQ)) - Gamma1.

    A', B', C' follow from u0, u2 and Gamma1' = fQ - f Gamma1 through the
    DM-dilution's definitions u0 = [3/14 (A (2f + g) + A') - g/2]/fv and
    u2 = [(2f + g)/2 - 3/14 (B (2f + g) + B')]/fv, with g = fQ - f.

    Returns a dict of arrays with the shape of ``c``'s fields.
    """
    D, f, fQ, G1 = c.D, c.f, c.fQ, c.Gamma1
    g = fQ - f
    D2 = D * D
    D3 = D2 * D
    C = 7.0 / 3.0 * (G1 - 1.0)
    A = c.L + c.WL - 7.0 / 3.0 * (G1 - 1.0) ** 2
    B = A - 2.0 * C
    rG = 3.0 / 7.0 * (f / fQ * c.L + c.Lp / (2.0 * fQ)) - G1
    u0 = G1 + rG
    u2 = -rG
    t = 2.0 * f + g
    return dict(
        D=D, f=f, R=G1, fv=fQ, fPsi=c.fPsi,
        D2Psi=3.0 / 14.0 * c.L * D2, DM2=3.0 / 14.0 * c.WL * D2,
        D3A=5.0 / 42.0 * c.F * D3, DM3A=5.0 / 42.0 * c.WF * D3,
        D3B=c.S * D3 / 18.0, DM3B=-c.WS * D3 / 18.0,
        A=A, B=B, C=C,
        Ap=14.0 / 3.0 * (fQ * u0 + 0.5 * g) - A * t,
        Bp=14.0 / 3.0 * (0.5 * t - fQ * u2) - B * t,
        Cp=7.0 / 3.0 * (fQ - f * G1),
        u0=u0, u2=u2, L=c.L, Lp=c.Lp, F=c.F, Fp=c.Fp, Nw=c.WF)


# ----------------------------------------------------------------------------
# Kernels (notes Sec. 4.2-4.3)
# ----------------------------------------------------------------------------

def eds_kernel_coefficients(c):
    """The EdS limit of the IDE kernel coefficients ``c``, for EdS-kernel loops with everything else IDE.

    Gamma1 = L = F = S = 1, L' = F' = 0, WL = WF = WS = 0 and f = fQ (so f/fQ = 1): with these, :func:`F2_G2` and
    :func:`F3K_G3K` reduce to the EdS kernels (F2 = alpha_s - 2/7 gamma12, G2 = alpha_s - 4/7 gamma12, ...). ``fQ``,
    ``fPsi``, ``D`` and ``z_out`` are kept, so the linear growth rate (Kaiser term, table f0) is unchanged; pass the
    original ``c`` wherever the IR damping is computed (``tables_to_pkmu(ide=...)``, the bispectrum's
    ``damping_ide``), so only the second- and third-order kernels differ from the full IDE pipeline.
    """
    one, zero = jnp.ones_like(c.fQ), jnp.zeros_like(c.fQ)
    return c._replace(f=c.fQ, Gamma1=one, L=one, Lp=zero, WL=zero, F=one, Fp=zero, WF=zero, S=one, WS=zero)


def F2_G2(mu, ratio, c):
    """Second-order kernels F2 and G2 for a pair (k1, k2) (notes Sec. 4.2, eqs. 32, 34).

    ``mu`` is the cosine between k1 and k2 and ``ratio = k1/k2 + k2/k1``. G2 is the
    kernel of theta = -div(v)/(a H fQ):

        F2 = Gamma1 alpha_s + [3/14 (L + WL) - Gamma1^2/2] (1 - mu^2)
        G2 = Gamma1 alpha_s + r_G (1 - mu^2),   r_G = 3/7 ((f/fQ) L + L'/(2 fQ)) - Gamma1

    with alpha_s = 1 + mu/2 (k1/k2 + k2/k1).
    """
    G1 = c.Gamma1
    alpha_s = 1.0 + 0.5 * mu * ratio
    gamma12 = 1.0 - mu * mu
    rG = 3.0 / 7.0 * (c.f / c.fQ * c.L + c.Lp / (2.0 * c.fQ)) - G1
    F2 = G1 * alpha_s + (3.0 / 14.0 * (c.L + c.WL) - 0.5 * G1 * G1) * gamma12
    G2 = G1 * alpha_s + rG * gamma12
    return F2, G2


def F3K_G3K(r, x, c):
    """Collapsed third-order kernels F3(k, -p, p), G3(k, -p, p) in fkpt's
    R-type variables (k = 1, r = p/k, x = khat.phat, y^2 = 1 + r^2 - 2 r x);
    notes Sec. 4.3, eqs. 43-44:

        F3K = 5/63 (F + WF) (1-x^2)^2/y^2 + Gamma2 (1 - r x)/(7 y^2) (Gamma1 x/r + W1)
              - Gamma1^2 x^2/(6 r^2)
        G3K = GammaQ3/2 + 2/7 Gamma1 (x/r) GammaQ2 + Gamma2 (1 - r x)/(7 y^2)
              - (2 Gamma1 GammaQ2 + Gamma2)(1 - x^2)/(7 y^2) - Gamma1^2 x^2/(6 r^2)

    with W1 = 1 - Gamma1, Gamma2 = L (1-x^2), GammaQ2 = (1-x^2)((f/fQ) L + L'/(2 fQ)),
    GammaQ3 = 10/21 (1-x^2)^2/y^2 ((f/fQ) F + F'/(3 fQ)).

    Not symmetrised under p -> -p (x -> -x); the angular integral does that.
    Reduce to the stock fkpt F3K/G3K for g = 0.
    """
    r2 = r * r
    x2 = x * x
    one_m_x2 = 1.0 - x2
    y2 = 1.0 + r2 - 2.0 * r * x
    f_fQ = c.f / c.fQ
    G1 = c.Gamma1
    W1 = 1.0 - G1

    Gam2 = c.L * one_m_x2
    GamQ2 = one_m_x2 * (f_fQ * c.L + c.Lp / (2.0 * c.fQ))
    GamQ3 = 10.0 / 21.0 * one_m_x2 * one_m_x2 / y2 * (f_fQ * c.F + c.Fp / (3.0 * c.fQ))
    ir = G1 * G1 * x2 / (6.0 * r2)
    gam2_term = Gam2 * (1.0 - r * x) / (7.0 * y2)

    F3K = (5.0 / 63.0 * (c.F + c.WF) * one_m_x2 * one_m_x2 / y2
           + gam2_term * (G1 * x / r + W1) - ir)
    G3K = (0.5 * GamQ3 + 2.0 / 7.0 * G1 * x / r * GamQ2 + gam2_term
           - (2.0 * G1 * GamQ2 + Gam2) * one_m_x2 / (7.0 * y2) - ir)
    return F3K, G3K


def ir_sigma2_tot(mu, sigma2, delta_sigma2, ide):
    """BAO damping Sigma^2_tot(mu) with the IDE displacement (notes Sec. 6, eq. 53):

        Gamma1^2 [ (1 + fPsi mu^2 (2 + fPsi)) Sigma^2 + fPsi^2 mu^2 (mu^2 - 1) dSigma^2 ]

    Gamma1 = 1, fPsi = f recovers the standard FOLPS expression.
    """
    G1, fPsi = ide.Gamma1, ide.fPsi
    mu2 = mu * mu
    return G1 * G1 * ((1.0 + fPsi * mu2 * (2.0 + fPsi)) * sigma2
                      + fPsi * fPsi * mu2 * (mu2 - 1.0) * delta_sigma2)
