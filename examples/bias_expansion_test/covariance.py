"""Analytic Gaussian P_ell covariance for BGS.

Same recipe as ``compute_covariance()`` in
``/n/home12/cgarciaquintero/DESI/synthetic/scripts/generate_noiseless_synthetic_data.py``
(``thecov.covariance.GaussianCovariance`` + ``thecov.geometry.BoxGeometry``),
using the real BGS ``nbar``/``veff``. Used only to put error bars on Plot 5 and
report a raw (no-fit) detection significance -- the nuisance/MG-parameter
refits (spec Sec. 9-10) are explicitly deferred, see README.md.
"""

import numpy as np

from thecov import covariance, geometry

import config


def compute_bgs_covariance(poles, k=None, ells=(0, 2, 4),
                            nbar=config.NBAR_BGS, veff=config.VEFF_BGS):
    """Full (n_ell*nk, n_ell*nk) analytic Gaussian covariance, block order = ells.

    Parameters
    ----------
    poles : dict {ell: array(nk,)}
        Galaxy P_ell(k) (shot-noise-included) used to build the covariance.
    k : array(nk,), default config.K_FIT (must be a uniform grid).
    """
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


def diag_sigma(cov_full, nk, ells=(0, 2, 4)):
    """Extract per-ell diagonal 1-sigma error arrays from the full covariance."""
    diag = np.sqrt(np.diag(cov_full))
    return {ell: diag[i * nk:(i + 1) * nk] for i, ell in enumerate(ells)}


if __name__ == "__main__":
    k = config.K_FIT
    toy_poles = {
        0: 1.0e4 * (k / 0.1) ** -1.2,
        2: 3.0e3 * (k / 0.1) ** -1.2,
        4: 5.0e2 * (k / 0.1) ** -1.2,
    }
    cov_full = compute_bgs_covariance(toy_poles, k=k)
    print("cov shape:", cov_full.shape, "expect", (3 * len(k), 3 * len(k)))
    sig = diag_sigma(cov_full, nk=len(k))
    for ell in (0, 2, 4):
        print(f"ell={ell}: sigma range [{sig[ell].min():.3g}, {sig[ell].max():.3g}], "
              f"P_ell/sigma range [{np.min(toy_poles[ell]/sig[ell]):.2f}, "
              f"{np.max(toy_poles[ell]/sig[ell]):.2f}]")
