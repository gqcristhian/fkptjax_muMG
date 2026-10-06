"""GR/LCDM linear P(k) via CAMB-ISiTGR, parametrized on cosmology.

The MG signal (whichever parameter ``config.MG_PARAM_NAME`` points to -- mu0
for HDKI/mu_OmDE, fR0_HS for Hu-Sawicki f(R)) enters entirely through
fkptjax's kernel machinery (growth ODE + beyond-EdS kernels), not through the
input linear spectrum -- so this always calls ISiTGR with
``MG_parameterization="muSigma", mu0=0.0`` (effectively GR), exactly as in
``bias_expansion_test/linear_power.py`` and ``compute_mg_multipoles.ipynb``.
Parametrized on (H0, ombh2, omch2, As, ns) so a fresh spectrum can be built
for every cosmological finite-difference step.
"""

import numpy as np
from scipy.signal import savgol_filter

import isitgr
from isitgr import model as isitgr_model

import config


def smooth_now_savgol(k, pk, window=41, polyorder=3):
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


def linear_power_gr(H0=config.H0_FID, ombh2=config.OMBH2_FID, omch2=config.OMCH2_FID,
                     As=config.AS_FID, ns=config.NS_FID, mnu=config.MNU,
                     z=config.Z_EFF, nk=300, minkh=config.KERNEL_KMIN, maxkh=config.KERNEL_KMAX):
    """GR/LCDM linear P_L(k,z) at an arbitrary cosmological parameter point."""
    pars = isitgr.CAMBparams()
    pars.set_cosmology(
        H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0.0,
        tau=config.TAU_REIO, nnu=3.046,
        MG_parameterization="muSigma", mu0=0.0,
    )
    pars.InitPower.set_params(As=As, ns=ns, r=0.0)
    pars.set_accuracy(AccuracyBoost=2)
    pars.NonLinear = isitgr_model.NonLinear_none
    pars.set_matter_power(redshifts=[z], kmax=maxkh)

    results = isitgr.get_results(pars)
    k, _, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=nk)
    pk = pk[0]
    pk_now = smooth_now_savgol(k, pk)
    return k, pk, pk_now


if __name__ == "__main__":
    k, pk, pk_now = linear_power_gr()
    print("fiducial GR P(k):", k.shape, k[0], k[-1], pk[:3])
    print("Om_fid:", config.OM_FID)
