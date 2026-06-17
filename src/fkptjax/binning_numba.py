"""
binning_numba.py
----------------
Numba-JIT'd implementation of the PHENOM/binning modified-gravity ODE right-hand
sides used by :class:`fkptjax.ode.ModelDerivatives`.

This is an *opt-in fast path* (selected via ``ModelDerivatives(..., use_numba=True)``
for ``model='PHENOM'``, ``mg_variant='binning'``).  The existing pure-numpy methods
in ``ode.py`` remain the default and the reference; these njit functions reproduce
their arithmetic exactly so results are unchanged (validated bit-for-bit on the RHS).

Why this exists: the beyond-EdS ``kernel_constants`` integration is a single *scalar*
ODE whose RHS (``thirdOrder``) is evaluated thousands of times per power-spectrum
call, each evaluating the binned-mu(k, eta) windows tens of times -- all in eager
Python.  Compiling the whole RHS removes that per-call overhead.

Constants are passed as a single float64 array ``P`` (no ``self``):

    P = [om, ol, mu1, mu2, mu3, mu4, z_div, z_TGR, z_tw,
         k_TGR, k_c, k_S, k_tw, scale_bins]
    idx: 0   1   2    3    4    5    6      7      8
         9      10   11   12    13 (scale_bins as 0.0/1.0)

If numba is unavailable, ``HAVE_NUMBA`` is False and the symbols are not defined;
callers must guard on ``HAVE_NUMBA`` (``ModelDerivatives`` does).
"""

import math
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba optional
    HAVE_NUMBA = False


# Indices into the packed constant array P (kept in sync with ModelDerivatives).
_I_OM, _I_OL = 0, 1
_I_MU1, _I_MU2, _I_MU3, _I_MU4 = 2, 3, 4, 5
_I_ZDIV, _I_ZTGR, _I_ZTW = 6, 7, 8
_I_KTGR, _I_KC, _I_KS, _I_KTW = 9, 10, 11, 12
_I_SCALE = 13
N_P = 14


def pack_constants(om, ol, mu1, mu2, mu3, mu4, z_div, z_TGR, z_tw,
                   scale_bins, k_TGR, k_c, k_S, k_tw):
    """Build the float64 constants array consumed by the njit RHS functions."""
    P = np.empty(N_P, dtype=np.float64)
    P[_I_OM] = om; P[_I_OL] = ol
    P[_I_MU1] = mu1; P[_I_MU2] = mu2; P[_I_MU3] = mu3; P[_I_MU4] = mu4
    P[_I_ZDIV] = z_div; P[_I_ZTGR] = z_TGR; P[_I_ZTW] = z_tw
    P[_I_KTGR] = k_TGR; P[_I_KC] = k_c; P[_I_KS] = k_S; P[_I_KTW] = k_tw
    P[_I_SCALE] = 1.0 if scale_bins else 0.0
    return P


if HAVE_NUMBA:

    # ---- leaf functions ------------------------------------------------

    @njit
    def _f1(eta, P):
        # ModelDerivatives.f1: 3 / (2 (1 + ol/om e^{3 eta}))
        return 3.0 / (2.0 * (1.0 + P[_I_OL] / P[_I_OM] * math.exp(3.0 * eta)))

    @njit
    def _kpp(x, k, p):
        return math.sqrt(k * k + p * p + 2.0 * k * p * x)

    @njit
    def _mu_scalar(eta, k, P):
        # PHENOM/binning mu(k, eta); mirrors ode.py mu() binning branch.
        a = math.exp(eta)
        z = 1.0 / a - 1.0
        ztw = P[_I_ZTW]
        Tz_div = math.tanh((z - P[_I_ZDIV]) / ztw)
        Tz_TGR = math.tanh((z - P[_I_ZTGR]) / ztw)

        if P[_I_SCALE] != 0.0:
            # scale-dependent bins: ISiTGR k-windows
            ktw = P[_I_KTW]
            t1 = math.tanh((k - P[_I_KTGR]) / ktw)
            t2 = math.tanh((k - P[_I_KC]) / ktw)
            t3 = math.tanh((k - P[_I_KS]) / ktw)
            W1 = 0.5 * (1.0 - t1)
            W2 = 0.5 * (t1 - t2)
            W3 = 0.5 * (t2 - t3)
            W4 = 0.5 * (1.0 + t3)
            mu1 = P[_I_MU1]; mu2 = P[_I_MU2]; mu3 = P[_I_MU3]; mu4 = P[_I_MU4]
            mu_z1 = 1.0 * W1 + mu1 * W2 + mu2 * W3 + 1.0 * W4
            mu_z2 = 1.0 * W1 + mu3 * W2 + mu4 * W3 + 1.0 * W4
            return 0.5 * (
                1.0 + mu_z1
                + (mu_z2 - mu_z1) * Tz_div
                + (1.0 - mu_z2) * Tz_TGR
            )

        # redshift-only 4-bin version
        zTGR = P[_I_ZTGR]
        z1 = 1.0 * zTGR / 4.0
        z2 = 2.0 * zTGR / 4.0
        z3 = 3.0 * zTGR / 4.0
        z4 = 4.0 * zTGR / 4.0
        T1 = math.tanh((z - z1) / ztw)
        T2 = math.tanh((z - z2) / ztw)
        T3 = math.tanh((z - z3) / ztw)
        T4 = math.tanh((z - z4) / ztw)
        mu1 = P[_I_MU1]; mu2 = P[_I_MU2]; mu3 = P[_I_MU3]; mu4 = P[_I_MU4]
        return (
            0.5 * (1.0 + mu1)
            + 0.5 * (mu2 - mu1) * T1
            + 0.5 * (mu3 - mu2) * T2
            + 0.5 * (mu4 - mu3) * T3
            + 0.5 * (1.0 - mu4) * T4
        )

    @njit
    def _mu_arr(eta, k, P):
        out = np.empty(k.size, dtype=np.float64)
        for i in range(k.size):
            out[i] = _mu_scalar(eta, k[i], P)
        return out

    # ---- second-order source terms ------------------------------------

    @njit
    def _S2a(eta, x, k, p, P):
        kplusp = _kpp(x, k, p)
        return _f1(eta, P) * _mu_scalar(eta, kplusp, P)

    @njit
    def _S2b(eta, x, k, p, P):
        kplusp = _kpp(x, k, p)
        return _f1(eta, P) * (
            _mu_scalar(eta, k, P) + _mu_scalar(eta, p, P) - _mu_scalar(eta, kplusp, P))

    @njit
    def _S2FL(eta, x, k, p, P):
        kp = _kpp(x, k, p)
        f1v = _f1(eta, P)
        mu_k = _mu_scalar(eta, k, P)
        mu_p = _mu_scalar(eta, p, P)
        mu_kp = _mu_scalar(eta, kp, P)
        r = p / k
        ri = k / p
        return f1v * (mu_kp * (r + ri) * x - ri * x * mu_k - r * x * mu_p)

    @njit
    def _SD2(eta, x, k, p, P):
        return _S2a(eta, x, k, p, P) - _S2b(eta, x, k, p, P) * (x * x) + _S2FL(eta, x, k, p, P)

    # ---- third-order source terms -------------------------------------

    @njit
    def _S3IIplus(eta, x, k, p, Dpk, Dpp, D2f, P):
        kplusp = _kpp(x, k, p)
        f1 = _f1(eta, P)
        mu_k = _mu_scalar(eta, k, P)
        mu_p = _mu_scalar(eta, p, P)
        mu_kp = _mu_scalar(eta, kplusp, P)
        return (
            -f1 * (mu_p + mu_kp - 2.0 * mu_k) * Dpp * (D2f + Dpk * Dpp * x * x)
            - f1 * (mu_kp - mu_k + mu_kp * (p / k + k / p) * x
                    - k * x / p * mu_k - p * x / k * mu_p) * Dpk * Dpp * Dpp
        )

    @njit
    def _S3FLplus(eta, x, k, p, Dpk, Dpp, D2f, P):
        k2 = k * k
        p2 = p * p
        pk = p * k
        if p2 == 0.0 or pk == 0.0:
            return 0.0
        denom = k2 + p2 + 2.0 * pk * x
        if denom == 0.0:
            return 0.0
        kplusp = _kpp(x, k, p)
        mu_k = _mu_scalar(eta, k, P)
        mu_p = _mu_scalar(eta, p, P)
        mu_kp = _mu_scalar(eta, kplusp, P)
        c1 = (p2 + pk * x) / denom
        c2 = (p2 + pk * x) / p2
        c3 = (p2 + k2) * (x * x / p2 + x / pk)
        term1 = c1 * (mu_p - mu_k) * (D2f * Dpp)
        term2 = c2 * (mu_kp - mu_k) * ((D2f * Dpp) + (1.0 + x * x) * (Dpk * Dpp * Dpp))
        term3 = c3 * (mu_kp - mu_k) * (Dpk * Dpp * Dpp)
        return _f1(eta, P) * (term1 + term2 + term3)

    @njit
    def _S3I(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P):
        kplusp = _kpp(x, k, p)
        kpluspm = _kpp(-x, k, p)
        pk = p / k
        return (
            (
                _f1(eta, P) * (_mu_scalar(eta, p, P) + _mu_scalar(eta, kplusp, P)
                               - _mu_scalar(eta, k, P)) * D2f * Dpp
                + _SD2(eta, x, k, p, P) * Dpk * Dpp * Dpp
            ) * (1.0 - x * x) / (1.0 + pk * pk + 2.0 * pk * x)
            + (
                _f1(eta, P) * (_mu_scalar(eta, p, P) + _mu_scalar(eta, kpluspm, P)
                               - _mu_scalar(eta, k, P)) * D2mf * Dpp
                + _SD2(eta, -x, k, p, P) * Dpk * Dpp * Dpp
            ) * (1.0 - x * x) / (1.0 + pk * pk - 2.0 * pk * x)
        )

    @njit
    def _S3II(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P):
        return (_S3IIplus(eta, x, k, p, Dpk, Dpp, D2f, P)
                + _S3IIplus(eta, -x, k, p, Dpk, Dpp, D2mf, P))

    @njit
    def _S3FL(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P):
        return (_S3FLplus(eta, x, k, p, Dpk, Dpp, D2f, P)
                + _S3FLplus(eta, -x, k, p, Dpk, Dpp, D2mf, P))

    # ---- RHS functions (the dispatch targets) -------------------------

    @njit
    def nb_firstOrder(x, Y, k_arr, P):
        # Y has shape (2, nk); returns (2, nk).  Mirrors ModelDerivatives.firstOrder.
        f1x = _f1(x, P)
        nk = k_arr.size
        mu_arr = _mu_arr(x, k_arr, P)
        out = np.empty((2, nk), dtype=np.float64)
        for i in range(nk):
            out[0, i] = Y[1, i]
            out[1, i] = f1x * mu_arr[i] * Y[0, i] - (2.0 - f1x) * Y[1, i]
        return out

    @njit
    def nb_secondOrder(eta, y, x, k, p, P):
        f2 = _f1(eta, P)
        f1 = 2.0 - f2
        kf = _kpp(x, k, p)
        src = _SD2(eta, x, k, p, P)
        out = np.empty(6, dtype=np.float64)
        out[0] = y[1]
        out[1] = f2 * _mu_scalar(eta, k, P) * y[0] - f1 * y[1]
        out[2] = y[3]
        out[3] = f2 * _mu_scalar(eta, p, P) * y[2] - f1 * y[3]
        out[4] = y[5]
        out[5] = f2 * _mu_scalar(eta, kf, P) * y[4] - f1 * y[5] + src * y[0] * y[2]
        return out

    @njit
    def nb_thirdOrder(eta, y, x, k, p, P):
        f1eta = _f1(eta, P)
        f2eta = 2.0 - f1eta
        kplusp = _kpp(x, k, p)
        kpluspm = _kpp(-x, k, p)
        Dpk = y[0]; Dpp = y[2]; D2f = y[4]; D2mf = y[6]
        out = np.empty(10, dtype=np.float64)
        out[0] = y[1]
        out[1] = f1eta * _mu_scalar(eta, k, P) * y[0] - f2eta * y[1]
        out[2] = y[3]
        out[3] = f1eta * _mu_scalar(eta, p, P) * y[2] - f2eta * y[3]
        out[4] = y[5]
        out[5] = (f1eta * _mu_scalar(eta, kplusp, P) * y[4] - f2eta * y[5]
                  + _SD2(eta, x, k, p, P) * y[0] * y[2])
        out[6] = y[7]
        out[7] = (f1eta * _mu_scalar(eta, kpluspm, P) * y[6] - f2eta * y[7]
                  + _SD2(eta, -x, k, p, P) * y[0] * y[2])
        out[8] = y[9]
        out[9] = (f1eta * _mu_scalar(eta, k, P) * y[8] - f2eta * y[9]
                  + _S3I(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P)
                  + _S3II(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P)
                  + _S3FL(eta, x, k, p, Dpk, Dpp, D2f, D2mf, P))
        return out

    @njit
    def nb_mu(eta, k, P):
        """Scalar mu(k, eta) -- exposed for validation/inspection."""
        return _mu_scalar(eta, k, P)
