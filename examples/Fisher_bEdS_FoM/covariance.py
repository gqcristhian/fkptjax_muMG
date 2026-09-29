"""Analytic Gaussian P_ell covariance for BGS, shared across all 3 theory cases.

Same recipe as ``bias_expansion_test/covariance.py`` (itself matching
``thecov``-based ``compute_covariance()`` in
``/n/home12/cgarciaquintero/DESI/synthetic/scripts/generate_noiseless_synthetic_data.py``).
Built ONCE from the fiducial bEdS-full P_ell (the reference/correct theory)
and then reused identically for EdS and bEdS-fkPT, per the requirement that
everything except the nonlinear-kernel treatment is held fixed between the
three cases.
"""

import numpy as np

from thecov import covariance, geometry

import config


def compute_bgs_covariance(poles, k=None, ells=config.ELLS,
                            nbar=config.NBAR_BGS, veff=config.VEFF_BGS):
    """Full (n_ell*nk, n_ell*nk) analytic Gaussian covariance, block order = ells."""
    if k is None:
        k = config.K_FIT
    k = np.asarray(k)
    dk = float(np.mean(np.diff(k)))
    kmin_bin = float(k[0] - dk / 2.0)
    kmax_bin = float(k[-1] + dk / 2.0)

    geom = geometry.BoxGeometry(volume=float(veff), nbar=float(nbar))
    cov = covariance.GaussianCovariance(geom)
    cov.set_kbins(kmin=kmin_bin, kmax=kmax_bin, dk=dk)

    for ell in ells:
        cov.set_galaxy_pk_multipole(poles[ell], ell=ell)
    cov.compute_covariance()
    return np.asarray(cov.cov, dtype="f8")


def diag_sigma(cov_full, nk, ells=config.ELLS):
    diag = np.sqrt(np.diag(cov_full))
    return {ell: diag[i * nk:(i + 1) * nk] for i, ell in enumerate(ells)}


if __name__ == "__main__":
    import kernel_pipeline

    poles_fid = kernel_pipeline.compute_poles("full")
    cov_full = compute_bgs_covariance(poles_fid)
    print("cov shape:", cov_full.shape, "expect", (len(config.ELLS) * len(config.K_FIT),) * 2)
    sig = diag_sigma(cov_full, nk=len(config.K_FIT))
    for ell in config.ELLS:
        print(f"ell={ell}: sigma range [{sig[ell].min():.3g}, {sig[ell].max():.3g}], "
              f"P_ell/sigma range [{np.min(poles_fid[ell]/sig[ell]):.2f}, "
              f"{np.max(poles_fid[ell]/sig[ell]):.2f}]")
