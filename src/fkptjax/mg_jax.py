"""
mg_jax.py
---------
JAX (jit/vmap-traceable) implementation of the modified-gravity ODE right-hand
sides -- the JAX counterpart of ``binning_numba.py``, generalised beyond
PHENOM/binning.

Same arithmetic as the numpy ``ModelDerivatives``, but with ``jax.numpy`` so the
growth / kernel-constant ODEs can be integrated by a JAX solver and the whole
fkpt loop becomes ``jax.jit`` / ``jax.vmap``-able ("Wall 2").

Supported models (``MGConstants.kind``):

``'binning'``
    PHENOM/binning: four binned ``mu_i``, in two redshift bins x two k bins
    (``scale_bins=True``) or four redshift bins (``scale_bins=False``).

``'bz_mass'``
    HDKI/BZ_Mass, the mass-scale Bertschinger-Zukin form (arXiv 2208.10508)::

        mu(a,k) = (1 + mu_kinf X) / (1 + X)
        X       = ( k lambda_a lambda_dS / (D(a) a) )^2
        D(a)    = lambda_dS a^-3 + lambda_a

    Sigma == 1 identically for this model, so there is no Sigma sector.

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

import jax.numpy as jnp


BINNING = 'binning'
BZ_MASS = 'bz_mass'
KINDS = (BINNING, BZ_MASS)


@dataclasses.dataclass(frozen=True)
class MGConstants:
    """Background + MG constants for the JAX RHS.

    ``kind`` and ``scale_bins`` are STATIC (Python) -- see the module docstring.
    Every other field may be a JAX tracer.
    """

    # static configuration
    kind: str = BINNING
    scale_bins: bool = False

    # background
    om: Any = 0.3
    ol: Any = 0.7

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

    # HDKI/BZ_Mass
    mu_kinf: Any = 1.0
    lambda_a: Any = 0.0
    lambda_dS: Any = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, got {self.kind!r}')


def pack_constants_jnp(om, ol, mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
                       z_div=1.0, z_TGR=2.0, z_tw=0.05,
                       scale_bins=False, k_TGR=0.01, k_c=0.1, k_S=0.2, k_tw=0.001,
                       kind=BINNING, mu_kinf=1.0, lambda_a=0.0, lambda_dS=0.0):
    """Build :class:`MGConstants`; numeric arguments may be JAX tracers.

    Signature is backwards-compatible with the binning-only version: the
    positional/keyword binning arguments are unchanged and ``kind`` defaults to
    ``'binning'``.
    """
    f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)      # noqa: E731
    return MGConstants(
        kind=str(kind), scale_bins=bool(scale_bins),
        om=f64(om), ol=f64(ol),
        mu1=f64(mu1), mu2=f64(mu2), mu3=f64(mu3), mu4=f64(mu4),
        z_div=f64(z_div), z_TGR=f64(z_TGR), z_tw=f64(z_tw),
        k_TGR=f64(k_TGR), k_c=f64(k_c), k_S=f64(k_S), k_tw=f64(k_tw),
        mu_kinf=f64(mu_kinf), lambda_a=f64(lambda_a), lambda_dS=f64(lambda_dS))


def f1(eta, P):
    return 3.0 / (2.0 * (1.0 + P.ol / P.om * jnp.exp(3.0 * eta)))


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


_MU = {BINNING: _mu_binning, BZ_MASS: _mu_bz_mass}


def mu(eta, k, P):
    """Effective Poisson modification mu(k, eta) for the configured model.

    Dispatch is a Python ``if`` on the static ``P.kind``, so only one model's
    expression is ever traced -- see the module docstring.
    """
    return _MU[P.kind](eta, k, P)


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
