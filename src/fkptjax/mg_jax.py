"""
mg_jax.py
---------
JAX (jit/vmap-traceable) implementation of the modified-gravity ODE right-hand
sides -- the JAX counterpart of ``binning_numba.py``, generalised beyond
PHENOM/binning.

Same arithmetic as the numpy ``ModelDerivatives``, but with ``jax.numpy`` so the
growth / kernel-constant ODEs can be integrated by a JAX solver and the whole
fkpt loop becomes ``jax.jit`` / ``jax.vmap``-able ("Wall 2").

Supported models (``MGConstants.kind``), mirroring ``ode.ModelDerivatives.mu_mg``
term for term:

``'lcdm'``
    LCDM / GR: mu == 1 identically.

``'hs'``
    Hu-Sawicki f(R)::

        mu(a,k) = 1 + 2 beta2 k^2 / (k^2 + (a m(a))^2)

    with the same chameleon mass ``m(a)`` as ``ode.py`` (``fR0_HS``, ``beta2``,
    ``n_HS``, background ``w0``/``wa``).  ``mu -> 1`` as ``fR0_HS -> 0``.

``'ndgp'``
    nDGP braneworld, scale-independent::

        mu(a) = 1 + 1/(3 beta(a)),  beta(a) = 1 + 2 E(a) r_c (1 - 0.5 Om(a))

``'mu_omde'``
    HDKI/mu_OmDE, scale-independent::

        mu(a) = 1 + mu0 * Omega_DE(a) / Omega_Lambda

``'bz'``
    HDKI/BZ, the (non mass-scale) Bertschinger-Zukin form::

        mu(a,k) = (1 + beta_1 x) / (1 + x),  x = lambda_1^2 k^2 a^exp_s

``'bz_mass'``
    HDKI/BZ_Mass, the mass-scale Bertschinger-Zukin form (arXiv 2208.10508)::

        mu(a,k) = (1 + mu_kinf X) / (1 + X)
        X       = ( k lambda_a lambda_dS / (D(a) a) )^2
        D(a)    = lambda_dS a^-3 + lambda_a

    Sigma == 1 identically for this model, so there is no Sigma sector.

``'eft_de'``
    HDKI/EFT_DE, EFTCAMB Horndeski mu from tabulated h1/h3/h5(eta)::

        mu(k,eta) = h1(eta) * (1 + k^2 h5(eta)) / (1 + k^2 h3(eta))

    The h1/h3/h5(eta) callables ``ode.py`` accepts are not JAX-traceable in
    general, so this module instead takes them pre-sampled on an eta grid
    (``eftde_eta_grid``/``eftde_h1_grid``/``eftde_h3_grid``/``eftde_h5_grid``)
    and interpolates with ``jnp.interp``.  ``eftde_scale_dependent`` (STATIC,
    mirrors ``ode.py``'s ``is_EFTDE_scale_dependent``) picks between the exact
    k-dependent expression and the fixed-pivot (``eftde_k2_pivot``, default
    ``k=0.1``) scale-independent proxy used for effectively-scale-independent
    parameter points.  ``None`` grids (the default) fall back to GR (mu=1),
    matching ``ode.py``'s missing-interpolator fallback.

``'binning'``
    PHENOM/binning: four binned ``mu_i``, in two redshift bins x two k bins
    (``scale_bins=True``) or four redshift bins (``scale_bins=False``).

``'growth_index'``
    PHENOM/growth_index, scale-independent ISiTGR growth-index mu(a).

``'growth_index_yukawa'``
    PHENOM/growth_index_yukawa: growth-index mu(a) gated by a Yukawa-like
    k-dependent window ``F_k(a,k) = (k^2/(k^2+kc(a)^2))^d_s``.

WHY THE MODEL IS A STATIC FIELD, NOT AN ARRAY ENTRY
---------------------------------------------------
``kind`` (and ``scale_bins``) are configuration, not parameters: they come from a
theory-option string and are never sampled.  Selecting on them with ``jnp.where``
-- as this module previously did for ``scale_bins`` -- would evaluate BOTH
branches on every call and then throw one away.  That is not merely wasted work:
each branch is then evaluated at the *other* model's parameter values, where it
need not be finite.  BZ_Mass at the binning defaults has ``lambda_a = lambda_dS =
0``, which makes its ``X`` a genuine ``0/0``; a NaN there would propagate through
``jnp.where`` into the gradient of every binning chain, because ``where`` takes
the derivative of both arms.

Holding them as Python attributes on a frozen dataclass makes the dispatch a
Python ``if``, so exactly one branch is ever traced and the question does not
arise.  ``P`` is only ever CAPTURED in a closure (``jax_ode`` builds ``rhs``
around it) and never crosses a JAX API boundary as data, so it needs no pytree
registration; its numeric fields are ordinary tracers under ``jit``/``vmap``.

Data-dependent guards in ``S3FLplus`` use ``jnp.where`` (trace-safe) instead of
Python ``if`` -- those depend on ``k``/``p``, which ARE traced.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp


LCDM = 'lcdm'
HS = 'hs'
NDGP = 'ndgp'
MU_OMDE = 'mu_omde'
BZ = 'bz'
BINNING = 'binning'
BZ_MASS = 'bz_mass'
EFT_DE = 'eft_de'
GROWTH_INDEX = 'growth_index'
GROWTH_INDEX_YUKAWA = 'growth_index_yukawa'
KINDS = (LCDM, HS, NDGP, MU_OMDE, BZ, BINNING, BZ_MASS, EFT_DE,
          GROWTH_INDEX, GROWTH_INDEX_YUKAWA)

# c / H0 in Mpc/h, matching ode.ModelDerivatives.invH0.
INV_H0 = 2997.92458


@dataclasses.dataclass(frozen=True)
class MGConstants:
    """Background + MG constants for the JAX RHS.

    ``kind`` and the other config flags (``scale_bins``, ``eftde_scale_dependent``)
    are STATIC (Python) -- see the module docstring.  Every other field may be a
    JAX tracer.
    """

    # static configuration
    kind: str = BINNING
    scale_bins: bool = False
    eftde_scale_dependent: bool = True

    # background
    om: Any = 0.3
    ol: Any = 0.7
    w0: Any = -1.0
    wa: Any = 0.0

    # HS / f(R)
    fR0_HS: Any = 0.0
    beta2: Any = 1.0 / 6.0
    n_HS: Any = 1.0

    # nDGP
    r_c: Any = 1.0e30

    # HDKI/mu_OmDE
    mu0: Any = 0.0

    # HDKI/BZ
    beta_1: Any = 1.0
    lambda_1: Any = 1.0
    exp_s: Any = 1.0

    # HDKI/BZ_Mass
    mu_kinf: Any = 1.0
    lambda_a: Any = 0.0
    lambda_dS: Any = 0.0

    # HDKI/EFT_DE: h1/h3/h5(eta), pre-sampled on a common eta grid.
    eftde_eta_grid: Any = None
    eftde_h1_grid: Any = None
    eftde_h3_grid: Any = None
    eftde_h5_grid: Any = None
    eftde_k2_pivot: Any = 0.01

    # PHENOM/binning
    mu1: Any = 1.0
    mu2: Any = 1.0
    mu3: Any = 1.0
    mu4: Any = 1.0
    z_div: Any = 1.0
    z_TGR: Any = 2.0
    z_tw: Any = 0.05
    k_TGR: Any = 0.01
    k_c: Any = 0.1
    k_S: Any = 0.2
    k_tw: Any = 0.001

    # PHENOM/growth_index[_yukawa]
    gamma_0: Any = 0.54545
    gamma_a: Any = 0.0
    t_k: Any = 1000.0
    d_s: Any = 0.0001

    # Massive-neutrino Poisson-source correction: mu_nu(k, eta) = delta_tot/delta_nonu,
    # pre-sampled on a (eta, log k) grid -- see fkptjax.neutrinos.NeutrinoTransferCorrection
    # (the eager-NumPy class this mirrors) and pack_constants_jnp's neutrino_correction=
    # kwarg below. Defaults (filled in __post_init__) give mu_nu == 1 identically, i.e. a
    # genuine no-op when no correction is requested.
    nu_eta_grid: Any = None
    nu_logk_grid: Any = None
    nu_mu_grid: Any = None
    nu_k_min: Any = 1.0e-8
    nu_k_max: Any = 1.0e8

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, got {self.kind!r}')
        if self.kind == EFT_DE and self.eftde_eta_grid is None:
            # GR fallback, matching ode.py's missing-interpolator behaviour.
            object.__setattr__(self, 'eftde_eta_grid', jnp.asarray([0.0, 1.0]))
            object.__setattr__(self, 'eftde_h1_grid', jnp.asarray([1.0, 1.0]))
            object.__setattr__(self, 'eftde_h3_grid', jnp.asarray([0.0, 0.0]))
            object.__setattr__(self, 'eftde_h5_grid', jnp.asarray([0.0, 0.0]))
        if self.nu_eta_grid is None:
            # No neutrino correction requested (or none available): a trivial 2x2 grid of
            # all-ones makes _mu_neutrino_jax return exactly 1.0 everywhere, a genuine no-op.
            object.__setattr__(self, 'nu_eta_grid', jnp.asarray([0.0, 1.0]))
            object.__setattr__(self, 'nu_logk_grid', jnp.asarray([-20.0, 20.0]))
            object.__setattr__(self, 'nu_mu_grid', jnp.ones((2, 2)))


# Pytree registration: lets `P` cross a jax.jit/vmap boundary as a direct
# traced ARGUMENT (needed to jit-wrap call sites that don't already
# closure-capture it, e.g. jax_ode.kernel_constants_jax -- previously
# un-jitted and re-traced from scratch on every call, ~450x slower than
# necessary; see the module-level "WHY THE MODEL IS A STATIC FIELD" note
# above). This does NOT change the existing closure-capture usage
# elsewhere, nor the Python-level `if`/`kind` dispatch that note describes:
# `kind`/`scale_bins`/`eftde_scale_dependent` are registered as pytree
# "aux_data" (static, hashable, triggers a recompile only if one of them
# actually changes -- exactly mirroring the existing Python-`if` semantics),
# while every numeric field is a traced "child" leaf. Purely additive.
_MG_STATIC_FIELDS = ('kind', 'scale_bins', 'eftde_scale_dependent')


def _mgconstants_flatten(p):
    names = [f.name for f in dataclasses.fields(p) if f.name not in _MG_STATIC_FIELDS]
    children = tuple(getattr(p, n) for n in names)
    aux_data = (tuple(getattr(p, n) for n in _MG_STATIC_FIELDS), tuple(names))
    return children, aux_data


def _mgconstants_unflatten(aux_data, children):
    static_vals, names = aux_data
    kwargs = dict(zip(_MG_STATIC_FIELDS, static_vals))
    kwargs.update(zip(names, children))
    return MGConstants(**kwargs)


jax.tree_util.register_pytree_node(MGConstants, _mgconstants_flatten, _mgconstants_unflatten)


def pack_constants_jnp(om, ol, mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
                       z_div=1.0, z_TGR=2.0, z_tw=0.05,
                       scale_bins=False, k_TGR=0.01, k_c=0.1, k_S=0.2, k_tw=0.001,
                       kind=BINNING, mu_kinf=1.0, lambda_a=0.0, lambda_dS=0.0,
                       w0=-1.0, wa=0.0,
                       fR0_HS=0.0, beta2=1.0 / 6.0, n_HS=1.0,
                       r_c=1.0e30, mu0=0.0,
                       beta_1=1.0, lambda_1=1.0, exp_s=1.0,
                       eftde_eta_grid=None, eftde_h1_grid=None,
                       eftde_h3_grid=None, eftde_h5_grid=None,
                       eftde_scale_dependent=True, eftde_k2_pivot=0.01,
                       gamma_0=0.54545, gamma_a=0.0, t_k=1000.0, d_s=0.0001,
                       neutrino_correction=None):
    """Build :class:`MGConstants`; numeric arguments may be JAX tracers.

    Signature is backwards-compatible with the binning-only version: the
    positional/keyword binning arguments are unchanged and ``kind`` defaults to
    ``'binning'``.

    ``neutrino_correction``: an optional ``fkptjax.neutrinos.NeutrinoTransferCorrection``
    (the same eager-NumPy object used by the ``ode.py``/squeezed-limit path). Its
    ``.k``/``.eta``/``.mu_nu`` tables are copied into ``MGConstants``' ``nu_*`` fields here
    (a one-time, eager NumPy->JAX-array conversion), so that ``mu()`` below can evaluate the
    JAX-native bilinear (eta, log k) interpolation (:func:`_mu_neutrino_jax`) at the traced
    eta values inside the full kernel's own RK4 ODE integration -- unlike the original
    class's ``__call__``, which is pure eager NumPy and cannot be called under jit/vmap.
    ``None`` (the default) leaves ``MGConstants``' own trivial all-ones grid in place, a
    genuine no-op.
    """
    f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)      # noqa: E731
    f64_opt = lambda v: None if v is None else f64(v)      # noqa: E731
    if neutrino_correction is None:
        nu_kwargs = {}
    else:
        nu_k = f64(neutrino_correction.k)
        nu_kwargs = dict(
            nu_eta_grid=f64(neutrino_correction.eta),
            nu_logk_grid=jnp.log(nu_k),
            nu_mu_grid=f64(neutrino_correction.mu_nu),
            nu_k_min=nu_k[0],
            nu_k_max=nu_k[-1],
        )
    return MGConstants(
        kind=str(kind), scale_bins=bool(scale_bins),
        eftde_scale_dependent=bool(eftde_scale_dependent),
        om=f64(om), ol=f64(ol), w0=f64(w0), wa=f64(wa),
        fR0_HS=f64(fR0_HS), beta2=f64(beta2), n_HS=f64(n_HS),
        r_c=f64(r_c), mu0=f64(mu0),
        beta_1=f64(beta_1), lambda_1=f64(lambda_1), exp_s=f64(exp_s),
        mu1=f64(mu1), mu2=f64(mu2), mu3=f64(mu3), mu4=f64(mu4),
        z_div=f64(z_div), z_TGR=f64(z_TGR), z_tw=f64(z_tw),
        k_TGR=f64(k_TGR), k_c=f64(k_c), k_S=f64(k_S), k_tw=f64(k_tw),
        mu_kinf=f64(mu_kinf), lambda_a=f64(lambda_a), lambda_dS=f64(lambda_dS),
        eftde_eta_grid=f64_opt(eftde_eta_grid), eftde_h1_grid=f64_opt(eftde_h1_grid),
        eftde_h3_grid=f64_opt(eftde_h3_grid), eftde_h5_grid=f64_opt(eftde_h5_grid),
        eftde_k2_pivot=f64(eftde_k2_pivot),
        gamma_0=f64(gamma_0), gamma_a=f64(gamma_a), t_k=f64(t_k), d_s=f64(d_s),
        **nu_kwargs)


def _Ea2_de_scaling(eta, P):
    """Generalized w0wa (CPL) dark energy background, shared by every model.

    ``de_scaling(a) = rho_DE(a) / rho_DE(a=1)``, ``Ea2(a) = H(a)^2 / H0^2``. Matches
    ISiTGR's own DarkEnergyInterface.f90 exactly: ``w_de(a) = w0 + wa*(1-a)`` and
    ``rho_de(a) propto a**(-3*(1+w0+wa)) * exp(-3*wa*(1-a))`` (TDarkEnergyEqnOfState_w_de /
    the ``grho_de`` assignment there -- ``exp(3*wa*(a-1))`` below is the same factor written
    the other way). Reduces to plain LCDM (``de_scaling=1``, ``Ea2=om*a^-3+ol``) exactly at
    ``w0=-1, wa=0``. ``ok0`` (curvature) is kept for generality even though this pipeline
    otherwise assumes flat (``ol = 1-om``).
    """
    a = jnp.exp(eta)
    ok0 = 1.0 - P.om - P.ol
    de_scaling = a ** (-3.0 * (1.0 + P.w0 + P.wa)) * jnp.exp(3.0 * P.wa * (a - 1.0))
    Ea2 = P.om * a ** (-3.0) + ok0 * a ** (-2.0) + P.ol * de_scaling
    return Ea2, de_scaling


def f1(eta, P):
    """(3/2) Omega_m(a), generalized to a w0wa (CPL) dark energy background instead of the
    pure-LCDM special case -- see _Ea2_de_scaling. Used as the friction/source coefficient in
    every beyond-EdS kernel-constant equation (S2a/S2b/S3IIplus/... below) and the growth ODE,
    for every model, regardless of that model's own mu(a,k) formula."""
    a = jnp.exp(eta)
    Ea2, _ = _Ea2_de_scaling(eta, P)
    Om = P.om * a ** (-3.0) / Ea2
    return 1.5 * Om


def kpp(x, k, p):
    return jnp.sqrt(k * k + p * p + 2.0 * k * p * x)


def _mu_binning(eta, k, P):
    """PHENOM/binning mu(k, eta); works on scalar or array k. Mirrors ode.py."""
    a = jnp.exp(eta)
    z = 1.0 / a - 1.0
    ztw = P.z_tw

    if P.scale_bins:
        # scale-dependent (ISiTGR k-windows)
        Tz_div = jnp.tanh((z - P.z_div) / ztw)
        Tz_TGR = jnp.tanh((z - P.z_TGR) / ztw)
        ktw = P.k_tw
        t1 = jnp.tanh((k - P.k_TGR) / ktw)
        t2 = jnp.tanh((k - P.k_c) / ktw)
        t3 = jnp.tanh((k - P.k_S) / ktw)
        W1 = 0.5 * (1.0 - t1)
        W2 = 0.5 * (t1 - t2)
        W3 = 0.5 * (t2 - t3)
        W4 = 0.5 * (1.0 + t3)
        mu_z1 = W1 + P.mu1 * W2 + P.mu2 * W3 + W4
        mu_z2 = W1 + P.mu3 * W2 + P.mu4 * W3 + W4
        return 0.5 * (1.0 + mu_z1 + (mu_z2 - mu_z1) * Tz_div + (1.0 - mu_z2) * Tz_TGR)

    # redshift-only 4-bin
    zTGR = P.z_TGR
    T1 = jnp.tanh((z - zTGR / 4.0) / ztw)
    T2 = jnp.tanh((z - 2.0 * zTGR / 4.0) / ztw)
    T3 = jnp.tanh((z - 3.0 * zTGR / 4.0) / ztw)
    T4 = jnp.tanh((z - zTGR) / ztw)
    mu_z = (0.5 * (1.0 + P.mu1) + 0.5 * (P.mu2 - P.mu1) * T1 + 0.5 * (P.mu3 - P.mu2) * T2
            + 0.5 * (P.mu4 - P.mu3) * T3 + 0.5 * (1.0 - P.mu4) * T4)
    # This branch is k-INDEPENDENT, but the result is still broadcast to k's shape. The
    # previous implementation selected between the two branches with jnp.where against the
    # k-shaped scale_bins branch, so it always returned a k-shaped array; a bare scalar here
    # would broadcast correctly in the RHS but would silently change the shape contract.
    return mu_z + jnp.zeros_like(k)


def _mu_bz_mass(eta, k, P):
    """HDKI/BZ_Mass mu(k, eta) = (1 + mu_kinf X)/(1 + X), X = (k lam_a lam_dS/(D a))^2.

    Written MULTIPLIED THROUGH by (D a)^2 rather than as the literal ratio::

        mu = ( (D a)^2 + mu_kinf (k lam_a lam_dS)^2 ) / ( (D a)^2 + (k lam_a lam_dS)^2 )

    which matters twice.

    ``D(a) = lambda_dS a^-3 + lambda_a`` VANISHES inside the sampled box whenever
    the two lambdas have opposite signs and ``|lambda_dS| <= lambda_a`` (about a
    third of the box).  There ``X -> inf`` and ``mu -> mu_kinf`` on all scales at
    once -- physical, and a feature the kernels must represent.  The literal form
    computes ``inf/inf = NaN`` at that point; this one evaluates to exactly
    ``mu_kinf``, with no branch.

    The remaining singular corner is ``lambda_a = lambda_dS = 0`` (the GR default),
    where numerator and denominator are both zero.  That is GR, so the guarded
    value is 1.  The guard is the double-``where`` idiom: the divisor is replaced
    by 1 INSIDE the division as well, because ``jnp.where`` differentiates both
    arms and a NaN in the discarded arm would still poison the gradient.
    """
    a = jnp.exp(eta)
    D = P.lambda_dS * a ** (-3.0) + P.lambda_a
    num = (k * P.lambda_a * P.lambda_dS) ** 2      # X numerator, times (D a)^2
    den = (D * a) ** 2
    denom = den + num
    ok = denom > 0.0
    safe = jnp.where(ok, denom, 1.0)
    return jnp.where(ok, (den + P.mu_kinf * num) / safe, jnp.ones_like(denom))


def _mu_lcdm(eta, k, P):
    return 1.0 + jnp.zeros_like(k)


def _mu_hs(eta, k, P):
    """Hu-Sawicki f(R); mirrors ``ode.ModelDerivatives.mu_mg`` model='HS'.

    ``mu -> 1`` as ``fR0_HS -> 0`` is the genuine physical limit (the chameleon
    mass ``m -> infinity``), so no guard would be needed for a *fixed* nonzero
    ``fR0_HS``.  The guard below only protects the ``fR0_HS == 0`` (GR) point
    itself, where ``1/|fR0_HS|`` is a literal division by zero; it uses the
    same grad-safe double-``where`` idiom as :func:`_mu_bz_mass` so a traced
    ``fR0_HS`` that is swept through/near 0 keeps a finite gradient.
    """
    a = jnp.exp(eta)
    k2 = k * k
    omega_de0 = P.ol
    Y_a = omega_de0 * a ** (-3.0 * (1.0 + P.w0 + P.wa)) * jnp.exp(3.0 * P.wa * (a - 1.0))
    Y_0 = omega_de0
    fR0_abs = jnp.abs(P.fR0_HS)
    ok = fR0_abs > 1.0e-30
    fR0_safe = jnp.where(ok, fR0_abs, 1.0)
    m = (
        (1.0 / INV_H0)
        * jnp.sqrt(1.0 / ((1.0 + P.n_HS) * fR0_safe))
        * (P.om * a ** (-3.0) + 4.0 * Y_a) ** ((2.0 + P.n_HS) / 2.0)
        / (P.om + 4.0 * Y_0) ** ((1.0 + P.n_HS) / 2.0)
    )
    mu_full = 1.0 + 2.0 * P.beta2 * k2 / (k2 + (a * m) ** 2)
    return jnp.where(ok, mu_full, jnp.ones_like(mu_full))


def _mu_ndgp(eta, k, P):
    """nDGP braneworld; scale-independent, mirrors ``ode.py`` model='NDGP'."""
    a = jnp.exp(eta)
    Ea2 = P.om * a ** (-3.0) + P.ol
    E = jnp.sqrt(Ea2)
    Om = (P.om * a ** (-3.0)) / Ea2
    beta = 1.0 + 2.0 * E * P.r_c * (1.0 - 0.5 * Om)
    mu_a = 1.0 + 1.0 / (3.0 * beta)
    return mu_a + jnp.zeros_like(k)


def _mu_mu_omde(eta, k, P):
    """HDKI/mu_OmDE; scale-independent, mirrors ``ode.py``'s ``'mu_OmDE'`` variant.

    Generalized to a w0wa (CPL) dark energy background -- see _Ea2_de_scaling.
    OmDE_over_OmL = Omega_DE(a)/Omega_Lambda = (ol*de_scaling/Ea2)/ol = de_scaling/Ea2,
    which reduces to the pure-LCDM 1/(ol+om*a^-3) exactly at w0=-1, wa=0 (de_scaling=1)."""
    Ea2, de_scaling = _Ea2_de_scaling(eta, P)
    OmDE_over_OmL = de_scaling / Ea2
    mu_a = 1.0 + P.mu0 * OmDE_over_OmL
    return mu_a + jnp.zeros_like(k)


def _mu_bz(eta, k, P):
    """HDKI/BZ; mirrors ``ode.py``'s ``'BZ'`` variant."""
    a = jnp.exp(eta)
    x = (P.lambda_1 ** 2) * (k * k) * (a ** P.exp_s)
    return (1.0 + P.beta_1 * x) / (1.0 + x)


def _mu_eft_de(eta, k, P):
    """HDKI/EFT_DE; mirrors ``ode.py``'s ``'EFT_DE'`` variant, but takes
    h1/h3/h5(eta) as pre-sampled grids (see the module docstring) instead of
    Python callables.
    """
    h1 = jnp.interp(eta, P.eftde_eta_grid, P.eftde_h1_grid)
    h3 = jnp.interp(eta, P.eftde_eta_grid, P.eftde_h3_grid)
    h5 = jnp.interp(eta, P.eftde_eta_grid, P.eftde_h5_grid)
    if P.eftde_scale_dependent:
        k2 = k * k
        return h1 * (1.0 + k2 * h5) / (1.0 + k2 * h3)
    k2p = P.eftde_k2_pivot
    mu_pivot = h1 * (1.0 + k2p * h5) / (1.0 + k2p * h3)
    return mu_pivot + jnp.zeros_like(k)


def _growth_index_mu(eta, P):
    """Scale-independent ISiTGR growth-index mu(a); shared by growth_index
    and growth_index_yukawa (which then gates it with a k-dependent window).
    """
    a = jnp.exp(eta)
    ok0 = 1.0 - P.om - P.ol
    Ea2, de_scaling = _Ea2_de_scaling(eta, P)
    Om = P.om * a ** (-3.0) / Ea2
    Ok = ok0 * a ** (-2.0) / Ea2
    Ode = P.ol * de_scaling / Ea2
    gamma = P.gamma_0 + P.gamma_a * (1.0 - a)
    gamma_star = -P.gamma_a * a
    w_de = P.w0 + P.wa * (1.0 - a)
    mu_gi = (2.0 / 3.0) * Om ** (gamma - 1.0) * (
        Om ** gamma
        + (2.0 - 3.0 * gamma)
        + 3.0 * (gamma - 0.5) * Om
        + (2.0 * gamma - 1.0) * Ok
        + 3.0 * (gamma - 0.5) * (1.0 + w_de) * Ode
        + gamma_star * jnp.log(Om)
    )
    return mu_gi, Ea2


def _mu_growth_index(eta, k, P):
    mu_gi, _ = _growth_index_mu(eta, P)
    return mu_gi + jnp.zeros_like(k)


def _mu_growth_index_yukawa(eta, k, P):
    mu_gi, Ea2 = _growth_index_mu(eta, P)
    a = jnp.exp(eta)
    Hconf = a * jnp.sqrt(Ea2) / INV_H0
    k2 = k * k
    kc2 = (P.t_k * Hconf) ** 2
    Fk = (k2 / (k2 + kc2)) ** P.d_s
    return 1.0 + (mu_gi - 1.0) * Fk


_MU = {
    LCDM: _mu_lcdm,
    HS: _mu_hs,
    NDGP: _mu_ndgp,
    MU_OMDE: _mu_mu_omde,
    BZ: _mu_bz,
    BINNING: _mu_binning,
    BZ_MASS: _mu_bz_mass,
    EFT_DE: _mu_eft_de,
    GROWTH_INDEX: _mu_growth_index,
    GROWTH_INDEX_YUKAWA: _mu_growth_index_yukawa,
}


def _mu_neutrino_jax(eta, k, P):
    """Massive-neutrino Poisson-source correction mu_nu(k, eta) = delta_tot/delta_nonu.

    JAX-native bilinear (eta, log k) interpolation of ``P.nu_eta_grid``/``nu_logk_grid``/
    ``nu_mu_grid`` (filled by :func:`pack_constants_jnp` from a
    ``fkptjax.neutrinos.NeutrinoTransferCorrection`` table) -- mirrors that class's own
    ``__call__`` (which does the same bilinear interpolation in pure eager NumPy) closely
    enough to reproduce it, but works under jit/vmap/grad, and at a traced scalar ``eta``.
    Returns exactly 1.0 everywhere when no correction was supplied (see MGConstants'
    trivial all-ones default grid) -- a genuine no-op in that case.
    """
    logk = jnp.log(jnp.clip(k, P.nu_k_min, P.nu_k_max))
    neta = P.nu_eta_grid.shape[0]
    j = jnp.clip(jnp.searchsorted(P.nu_eta_grid, eta, side='right') - 1, 0, neta - 2)
    eta0 = P.nu_eta_grid[j]
    eta1 = P.nu_eta_grid[j + 1]
    t = jnp.where(eta1 == eta0, 0.0,
                  (jnp.clip(eta, P.nu_eta_grid[0], P.nu_eta_grid[-1]) - eta0) / (eta1 - eta0))
    y0 = jnp.interp(logk, P.nu_logk_grid, P.nu_mu_grid[j])
    y1 = jnp.interp(logk, P.nu_logk_grid, P.nu_mu_grid[j + 1])
    return y0 + t * (y1 - y0)


def mu(eta, k, P):
    """Effective Poisson modification mu_eff(k, eta) = mu_MG(k, eta) * mu_nu(k, eta) for
    the configured model (arXiv:2012.05077 eq. 2.1) -- mu_nu defaults to 1 identically
    (see MGConstants.__post_init__) when no neutrino correction was requested, so this
    is a no-op unless one is actually supplied via pack_constants_jnp's
    neutrino_correction= kwarg.

    Dispatch is a Python ``if`` on the static ``P.kind``, so only one model's
    expression is ever traced -- see the module docstring.
    """
    return _MU[P.kind](eta, k, P) * _mu_neutrino_jax(eta, k, P)


# ---- second-order source terms ----

def S2a(eta, x, k, p, P):
    return f1(eta, P) * mu(eta, kpp(x, k, p), P)


def S2b(eta, x, k, p, P):
    return f1(eta, P) * (mu(eta, k, P) + mu(eta, p, P) - mu(eta, kpp(x, k, p), P))


def S2FL(eta, x, k, p, P):
    kp = kpp(x, k, p)
    f1v = f1(eta, P)
    mu_k = mu(eta, k, P); mu_p = mu(eta, p, P); mu_kp = mu(eta, kp, P)
    r = p / k; ri = k / p
    return f1v * (mu_kp * (r + ri) * x - ri * x * mu_k - r * x * mu_p)


def SD2(eta, x, k, p, P):
    return S2a(eta, x, k, p, P) - S2b(eta, x, k, p, P) * (x * x) + S2FL(eta, x, k, p, P)


# ---- third-order source terms ----

def S3IIplus(eta, x, k, p, Dpk, Dpp, D2f, P):
    kplusp = kpp(x, k, p)
    f1v = f1(eta, P)
    mu_k = mu(eta, k, P); mu_p = mu(eta, p, P); mu_kp = mu(eta, kplusp, P)
    return (
        -f1v * (mu_p + mu_kp - 2.0 * mu_k) * Dpp * (D2f + Dpk * Dpp * x * x)
        - f1v * (mu_kp - mu_k + mu_kp * (p / k + k / p) * x
                 - k * x / p * mu_k - p * x / k * mu_p) * Dpk * Dpp * Dpp
    )


def S3FLplus(eta, x, k, p, Dpk, Dpp, D2f, P):
    k2 = k * k; p2 = p * p; pk = p * k
    denom = k2 + p2 + 2.0 * pk * x
    kplusp = kpp(x, k, p)
    mu_k = mu(eta, k, P); mu_p = mu(eta, p, P); mu_kp = mu(eta, kplusp, P)
    c1 = (p2 + pk * x) / denom
    c2 = (p2 + pk * x) / p2
    c3 = (p2 + k2) * (x * x / p2 + x / pk)
    term1 = c1 * (mu_p - mu_k) * (D2f * Dpp)
    term2 = c2 * (mu_kp - mu_k) * ((D2f * Dpp) + (1.0 + x * x) * (Dpk * Dpp * Dpp))
    term3 = c3 * (mu_kp - mu_k) * (Dpk * Dpp * Dpp)
    val = f1(eta, P) * (term1 + term2 + term3)
    # trace-safe guards (physical k,p>0 so these never trigger)
    return jnp.where((p2 == 0.0) | (pk == 0.0) | (denom == 0.0), 0.0, val)


def S3I(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P):
    kplusp = kpp(x, k, p); kpluspm = kpp(-x, k, p); pk = p / k
    return (
        (f1(eta, P) * (mu(eta, p, P) + mu(eta, kplusp, P) - mu(eta, k, P)) * D2f * Dpp
         + SD2(eta, x, k, p, P) * Dpk * Dpp * Dpp) * (1.0 - x * x) / (1.0 + pk * pk + 2.0 * pk * x)
        + (f1(eta, P) * (mu(eta, p, P) + mu(eta, kpluspm, P) - mu(eta, k, P)) * D2mf * Dpp
           + SD2(eta, -x, k, p, P) * Dpk * Dpp * Dpp) * (1.0 - x * x) / (1.0 + pk * pk - 2.0 * pk * x)
    )


def S3II(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P):
    return (S3IIplus(eta, x, k, p, Dpk, Dpp, D2f, P)
            + S3IIplus(eta, -x, k, p, Dpk, Dpp, D2mf, P))


def S3FL(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P):
    return (S3FLplus(eta, x, k, p, Dpk, Dpp, D2f, P)
            + S3FLplus(eta, -x, k, p, Dpk, Dpp, D2mf, P))


# ---- RHS functions ----

def firstOrder(x, Y, k_arr, P):
    """Y shape (2, nk); returns (2, nk).  Mirrors ModelDerivatives.firstOrder."""
    f1x = f1(x, P)
    mu_arr = mu(x, k_arr, P)
    return jnp.stack([Y[1], f1x * mu_arr * Y[0] - (2.0 - f1x) * Y[1]])


def secondOrder(eta, y, x, k, p, P):
    f2 = f1(eta, P); fr = 2.0 - f2
    kf = kpp(x, k, p)
    src = SD2(eta, x, k, p, P)
    return jnp.stack([
        y[1], f2 * mu(eta, k, P) * y[0] - fr * y[1],
        y[3], f2 * mu(eta, p, P) * y[2] - fr * y[3],
        y[5], f2 * mu(eta, kf, P) * y[4] - fr * y[5] + src * y[0] * y[2],
    ])


def thirdOrder(eta, y, x, k, p, P):
    f1eta = f1(eta, P); f2eta = 2.0 - f1eta
    kplusp = kpp(x, k, p); kpluspm = kpp(-x, k, p)
    Dpk = y[0]; Dpp = y[2]; D2f = y[4]; D2mf = y[6]
    return jnp.stack([
        y[1], f1eta * mu(eta, k, P) * y[0] - f2eta * y[1],
        y[3], f1eta * mu(eta, p, P) * y[2] - f2eta * y[3],
        y[5], f1eta * mu(eta, kplusp, P) * y[4] - f2eta * y[5] + SD2(eta, x, k, p, P) * y[0] * y[2],
        y[7], f1eta * mu(eta, kpluspm, P) * y[6] - f2eta * y[7] + SD2(eta, -x, k, p, P) * y[0] * y[2],
        y[9], f1eta * mu(eta, k, P) * y[8] - f2eta * y[9]
        + S3I(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P)
        + S3II(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P)
        + S3FL(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P),
    ])
