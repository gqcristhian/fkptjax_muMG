"""Orchestrates the whole Fisher_bEdS_FoM test: EdS vs bEdS-fkPT vs bEdS-full.

    python run_all.py

Writes outputs/summary.npz (+ printed report) and figures/*.png.
"""

import time

import numpy as np

import config
import kernel_pipeline
import derivatives
import fisher as fisher_mod
import param_bias
import covariance
import make_plots


def main():
    t_start = time.time()

    print("=" * 70)
    print("Step 1/4: fiducial P_ell for all 3 theories + shared covariance")
    print("=" * 70)
    poles_fid = {}
    for theory in config.THEORIES:
        t0 = time.time()
        poles_fid[theory] = derivatives.fiducial_poles(theory)
        print(f"  {theory:8s} fiducial P_ell built in {time.time() - t0:.1f}s  "
              f"finite={all(np.all(np.isfinite(poles_fid[theory][ell])) for ell in config.ELLS)}")

    cov_full = covariance.compute_bgs_covariance(poles_fid["full"])
    cov_inv = np.linalg.inv(cov_full)
    print(f"  covariance shape: {cov_full.shape}")

    print()
    print("=" * 70)
    print("Step 2/4: numerical derivatives dP/dtheta for all parameters, all 3 theories")
    print("=" * 70)
    derivs_by_theory = {}
    for theory in config.THEORIES:
        t0 = time.time()
        derivs_by_theory[theory] = derivatives.all_derivatives(theory)
        print(f"  {theory:8s}: {len(config.ALL_PARAMS)} parameter derivatives in "
              f"{time.time() - t0:.1f}s")

    print()
    print("=" * 70)
    print("Step 3/4: staged-marginalization Fisher matrices")
    print("=" * 70)
    pname = config.MG_PARAM_NAME
    staged_by_theory = {}
    for theory in config.THEORIES:
        staged_by_theory[theory] = fisher_mod.staged_results(derivs_by_theory[theory], cov_inv)
        for stage_name, res in staged_by_theory[theory].items():
            print(f"  {theory:8s} {stage_name:28s} sigma({pname}) = {res['sigma_mu0']:.5g}")

    sigma_full = staged_by_theory["full"]["(e) full nuisance model"]["sigma_mu0"]
    for theory in ("eds", "approx"):
        sigma_theory = staged_by_theory[theory]["(e) full nuisance model"]["sigma_mu0"]
        print(f"  sigma_{theory}({pname}) / sigma_full({pname}) = {sigma_theory / sigma_full:.4f}")

    print()
    print("=" * 70)
    print("Step 4/4: parameter-bias test (EdS, bEdS-fkPT vs bEdS-full truth)")
    print("=" * 70)
    bias_by_theory = {}
    for theory in ("eds", "approx"):
        bias_by_theory[theory] = param_bias.parameter_bias(theory, cov_inv)
        b = bias_by_theory[theory]
        print(f"  {theory:8s}: Delta_{pname}/sigma({pname}) = {b['delta_mu0_over_sigma']:.4f}")
        for p in ("alpha0", "alpha2", "alpha4", "b1", "b2"):
            if p in b["delta_theta"]:
                print(f"      Delta_{p:10s} = {b['delta_theta'][p]:.4f}")

    print()
    print("Building plots and summary table...")
    make_plots.plot1_dP_dmu0(derivs_by_theory, f"{config.FIGDIR}/plot1_dP_dmu0.png")
    make_plots.plot2_staged_sigma(staged_by_theory, f"{config.FIGDIR}/plot2_staged_sigma.png")
    make_plots.plot3_correlations(staged_by_theory, f"{config.FIGDIR}/plot3_correlations.png")
    make_plots.plot4_ellipses(staged_by_theory, f"{config.FIGDIR}/plot4_ellipse_alpha0.png",
                               other_param="alpha0")
    make_plots.plot4_ellipses(staged_by_theory, f"{config.FIGDIR}/plot4_ellipse_alpha2.png",
                               other_param="alpha2")
    rows = make_plots.summary_table(staged_by_theory, bias_by_theory)
    make_plots.plot5_summary_figure(rows, f"{config.FIGDIR}/plot5_summary.png")

    print()
    print("Summary table:")
    for row in rows:
        print(f"  {row}")

    np.savez(
        f"{config.OUTDIR}/summary.npz",
        poles_fid=poles_fid, derivs_by_theory=derivs_by_theory,
        staged_by_theory=staged_by_theory, bias_by_theory=bias_by_theory,
        rows=rows, cov_full=cov_full,
    )
    print(f"\nSaved {config.OUTDIR}/summary.npz. Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
