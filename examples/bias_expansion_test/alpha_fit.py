"""Can the EFT counterterms (alpha0, alpha2) alone absorb the missing
scale-dependent-bias correction, to within 1%?

This is a deliberately narrow slice of the deferred Sec. 9 nuisance refit: it
holds b1 and every other nuisance parameter fixed at their ordinary BGS
values, and asks only whether re-tuning the monopole/quadrupole EFT
counterterms (alpha0 controls the ell=0 k^2 counterterm, alpha2 the ell=2
one) can mimic P_ell,toy(k;z*) well enough. alpha4 is left at 0 and ell=4 is
not fit: since Delta_b1(k) does not depend on mu, the exact Schmidt
correction only has mu^0 and mu^2 pieces (mu^2 itself has no Legendre-4
component either), so ell=4 is essentially untouched by construction (see
make_plots.plot4_delta_pbias) and there is nothing there to absorb.
"""

import numpy as np
from scipy.optimize import least_squares

import config
import full_kernel_pipeline

FIT_ELLS = (0, 2)
ALL_ELLS = (0, 2, 4)
ALPHA_BOUNDS = ([-500.0, -500.0], [500.0, 500.0])


def _residuals(x, state, k, P_truth):
    alpha0, alpha2 = x
    model = full_kernel_pipeline.poles_for_bias(
        state, {"alpha0": alpha0, "alpha2": alpha2}, k=k, ells=FIT_ELLS,
    )
    return np.concatenate([(P_truth[ell] - model[ell]) / P_truth[ell] for ell in FIT_ELLS])


def fit_alpha(k, fR0_HS, P_truth, x0=(0.0, 0.0)):
    """Least-squares fit of (alpha0, alpha2) to match P_truth (dict {ell: array}).

    Returns a dict with the best-fit alpha0/alpha2, the resulting model at
    ALL_ELLS (including the un-fit ell=4), and fractional residuals.
    """
    state = full_kernel_pipeline.build_state(fR0_HS)
    result = least_squares(
        _residuals, x0=list(x0), args=(state, k, P_truth), bounds=ALPHA_BOUNDS,
    )
    alpha0, alpha2 = result.x
    P_bestfit = full_kernel_pipeline.poles_for_bias(
        state, {"alpha0": alpha0, "alpha2": alpha2}, k=k, ells=ALL_ELLS,
    )
    residual_frac = {ell: (P_truth[ell] - P_bestfit[ell]) / P_truth[ell] for ell in ALL_ELLS}
    max_residual = {ell: float(np.max(np.abs(residual_frac[ell]))) for ell in ALL_ELLS}
    return dict(
        alpha0=float(alpha0), alpha2=float(alpha2),
        P_bestfit=P_bestfit, residual_frac=residual_frac,
        max_residual_frac=max_residual, success=bool(result.success),
    )


def fit_all_z_star(out, z_star_list=config.Z_STAR):
    """Run fit_alpha for every z* in one MG point's `toy_spectrum.build_all` output."""
    k = out["k"]
    fR0_HS = out["fR0_HS"]
    fits = {}
    for z_star in z_star_list:
        fits[z_star] = fit_alpha(k, fR0_HS, out["P_ell_toy"][z_star])
    return fits


if __name__ == "__main__":
    import toy_spectrum

    out = toy_spectrum.build_all("F5")
    fits = fit_all_z_star(out)
    for z_star, fit in fits.items():
        print(f"z*={z_star}: alpha0={fit['alpha0']:.3f}  alpha2={fit['alpha2']:.3f}  "
              f"max|resid| ell=0,2,4 = "
              f"{fit['max_residual_frac'][0]*100:.3f}%, "
              f"{fit['max_residual_frac'][2]*100:.3f}%, "
              f"{fit['max_residual_frac'][4]*100:.3f}%  "
              f"(<1%? {all(v < 0.01 for v in fit['max_residual_frac'].values())})")
