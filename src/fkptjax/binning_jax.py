"""
binning_jax.py
--------------
JAX (jit/vmap-traceable) implementation of the PHENOM/binning modified-gravity
ODE right-hand sides -- the JAX counterpart of ``binning_numba.py``.

Same arithmetic as the numpy ``ModelDerivatives`` (binning branch), but with
``jax.numpy`` so the growth / kernel-constant ODEs can be integrated by a JAX
solver (diffrax) and the whole fkpt loop becomes ``jax.jit`` / ``jax.vmap``-able
("Wall 2").  Validated bit-for-bit against the numpy reference.

Constants are passed as a single float64 array ``P`` (no ``self``); identical
layout to ``binning_numba.pack_constants``:

    P = [om, ol, mu1, mu2, mu3, mu4, z_div, z_TGR, z_tw,
         k_TGR, k_c, k_S, k_tw, scale_bins]   (scale_bins as 0.0/1.0)

Data-dependent guards in ``_S3FLplus`` use ``jnp.where`` (trace-safe) instead of
Python ``if`` (the physical k, p > 0 so they never actually trigger).
"""

import numpy as np
import jax.numpy as jnp

# Indices into P (kept in sync with binning_numba).
_I_OM, _I_OL = 0, 1
_I_MU1, _I_MU2, _I_MU3, _I_MU4 = 2, 3, 4, 5
_I_ZDIV, _I_ZTGR, _I_ZTW = 6, 7, 8
_I_KTGR, _I_KC, _I_KS, _I_KTW = 9, 10, 11, 12
_I_SCALE = 13
N_P = 14


def pack_constants(om, ol, mu1, mu2, mu3, mu4, z_div, z_TGR, z_tw,
                   scale_bins, k_TGR, k_c, k_S, k_tw):
    """Build the float64 constants array (numpy; converted to jnp at use)."""
    P = np.empty(N_P, dtype=np.float64)
    P[_I_OM] = om; P[_I_OL] = ol
    P[_I_MU1] = mu1; P[_I_MU2] = mu2; P[_I_MU3] = mu3; P[_I_MU4] = mu4
    P[_I_ZDIV] = z_div; P[_I_ZTGR] = z_TGR; P[_I_ZTW] = z_tw
    P[_I_KTGR] = k_TGR; P[_I_KC] = k_c; P[_I_KS] = k_S; P[_I_KTW] = k_tw
    P[_I_SCALE] = 1.0 if scale_bins else 0.0
    return P


def pack_constants_jnp(om, ol, mu1, mu2, mu3, mu4, z_div, z_TGR, z_tw,
                       scale_bins, k_TGR, k_c, k_S, k_tw):
    """jnp constants array; ``mu*`` may be JAX tracers (jit/vmap-safe)."""
    sb = 1.0 if scale_bins else 0.0
    return jnp.stack([
        jnp.asarray(om, dtype=jnp.float64), jnp.asarray(ol, dtype=jnp.float64),
        jnp.asarray(mu1, dtype=jnp.float64), jnp.asarray(mu2, dtype=jnp.float64),
        jnp.asarray(mu3, dtype=jnp.float64), jnp.asarray(mu4, dtype=jnp.float64),
        jnp.asarray(z_div, dtype=jnp.float64), jnp.asarray(z_TGR, dtype=jnp.float64),
        jnp.asarray(z_tw, dtype=jnp.float64), jnp.asarray(k_TGR, dtype=jnp.float64),
        jnp.asarray(k_c, dtype=jnp.float64), jnp.asarray(k_S, dtype=jnp.float64),
        jnp.asarray(k_tw, dtype=jnp.float64), jnp.asarray(sb, dtype=jnp.float64),
    ])


def f1(eta, P):
    return 3.0 / (2.0 * (1.0 + P[_I_OL] / P[_I_OM] * jnp.exp(3.0 * eta)))


def kpp(x, k, p):
    return jnp.sqrt(k * k + p * p + 2.0 * k * p * x)


def mu(eta, k, P):
    """PHENOM/binning mu(k, eta); works on scalar or array k. Mirrors ode.py."""
    a = jnp.exp(eta)
    z = 1.0 / a - 1.0
    ztw = P[_I_ZTW]
    Tz_div = jnp.tanh((z - P[_I_ZDIV]) / ztw)
    Tz_TGR = jnp.tanh((z - P[_I_ZTGR]) / ztw)

    # scale-dependent (ISiTGR k-windows)
    ktw = P[_I_KTW]
    t1 = jnp.tanh((k - P[_I_KTGR]) / ktw)
    t2 = jnp.tanh((k - P[_I_KC]) / ktw)
    t3 = jnp.tanh((k - P[_I_KS]) / ktw)
    W1 = 0.5 * (1.0 - t1)
    W2 = 0.5 * (t1 - t2)
    W3 = 0.5 * (t2 - t3)
    W4 = 0.5 * (1.0 + t3)
    mu1, mu2, mu3, mu4 = P[_I_MU1], P[_I_MU2], P[_I_MU3], P[_I_MU4]
    mu_z1 = W1 + mu1 * W2 + mu2 * W3 + W4
    mu_z2 = W1 + mu3 * W2 + mu4 * W3 + W4
    mu_scale = 0.5 * (1.0 + mu_z1 + (mu_z2 - mu_z1) * Tz_div + (1.0 - mu_z2) * Tz_TGR)

    # redshift-only 4-bin (scale_bins == 0)
    zTGR = P[_I_ZTGR]
    T1 = jnp.tanh((z - zTGR / 4.0) / ztw)
    T2 = jnp.tanh((z - 2.0 * zTGR / 4.0) / ztw)
    T3 = jnp.tanh((z - 3.0 * zTGR / 4.0) / ztw)
    T4 = jnp.tanh((z - zTGR) / ztw)
    mu_z = (0.5 * (1.0 + mu1) + 0.5 * (mu2 - mu1) * T1 + 0.5 * (mu3 - mu2) * T2
            + 0.5 * (mu4 - mu3) * T3 + 0.5 * (1.0 - mu4) * T4)

    return jnp.where(P[_I_SCALE] != 0.0, mu_scale, mu_z)


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
