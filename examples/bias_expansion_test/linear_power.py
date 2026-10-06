"""GR and Hu-Sawicki f(R) linear matter power spectra via CAMB-ISiTGR.

Same route as examples/compute_mg_multipoles.ipynb (GR/muSigma branch) and the
production DR2 synthetic-data script (HS/mueta branch): ISiTGR natively
supports Hu-Sawicki f(R), so no manual growth-rescaling of an input P(k) is
needed here (unlike e.g. BZ_Mass).
"""

import os
import numpy as np
from scipy.signal import savgol_filter

import isitgr
from isitgr import model as isitgr_model

import config


def smooth_now_savgol(k, pk, window=41, polyorder=3):
    """No-wiggle proxy for pk (peak-averaging is not needed for this diagnostic)."""
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    logpk = np.log(pk)
    n = len(k)
    window = min(int(window), n - 1 if (n - 1) % 2 == 1 else n - 2)
    if window < polyorder + 2:
        window = polyorder + 3
    if window % 2 == 0:
        window += 1
    return np.exp(savgol_filter(logpk, window_length=window, polyorder=polyorder))


def _get_isitgr_spectrum(z, nk=300, minkh=config.KERNEL_KMIN, maxkh=config.KERNEL_KMAX,
                          mg_cosmology=None):
    pars = isitgr.CAMBparams()
    pars.set_cosmology(
        H0=config.H0,
        ombh2=config.OMBH2,
        omch2=config.OMCH2,
        mnu=config.MNU,
        omk=0.0,
        tau=config.TAU_REIO,
        nnu=3.046,
        **(mg_cosmology or {}),
    )
    pars.InitPower.set_params(As=config.AS, ns=config.NS, r=0.0)
    pars.set_accuracy(AccuracyBoost=2)
    pars.NonLinear = isitgr_model.NonLinear_none
    pars.set_matter_power(redshifts=[z], kmax=maxkh)

    results = isitgr.get_results(pars)
    k, _, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=nk)
    pk = pk[0]
    pk_now = smooth_now_savgol(k, pk)
    return k, pk, pk_now


def linear_power_gr(z=config.Z_EFF, nk=300):
    """GR/LCDM linear P_L(k,z) (muSigma with mu0=0 is effectively GR)."""
    return _get_isitgr_spectrum(z, nk=nk, mg_cosmology=dict(MG_parameterization="muSigma", mu0=0.0))


def linear_power_hs(fR0_HS, z=config.Z_EFF, n_HS=config.N_HS, nk=300):
    """Hu-Sawicki f(R) linear P_L(k,z)."""
    return _get_isitgr_spectrum(
        z, nk=nk,
        mg_cosmology=dict(MG_parameterization="mueta", use_HS_form=True,
                           fR0_HS=float(fR0_HS), n_HS=float(n_HS)),
    )


def interp_to(k_target, k_src, pk_src):
    """Log-log interpolation of pk_src(k_src) onto k_target."""
    return np.exp(np.interp(np.log(k_target), np.log(k_src), np.log(pk_src)))


if __name__ == "__main__":
    k_gr, pk_gr, pk_gr_now = linear_power_gr()
    print("GR:", k_gr.shape, k_gr[0], k_gr[-1], pk_gr[:3])

    for name, pt in config.MG_POINTS.items():
        k_mg, pk_mg, _ = linear_power_hs(pt["fR0_HS"])
        pk_gr_i = interp_to(k_mg, k_gr, pk_gr)
        ratio = pk_mg / pk_gr_i
        low_k_ratio = float(np.interp(0.005, k_mg, ratio))
        high_k_ratio = float(np.interp(0.2, k_mg, ratio))
        print(f"{name}: P_MG/P_GR at k=0.005 -> {low_k_ratio:.4f} (expect ~1), "
              f"at k=0.2 -> {high_k_ratio:.4f}")
