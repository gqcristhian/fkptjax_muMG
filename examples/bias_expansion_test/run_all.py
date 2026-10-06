"""Run the full BGS scale-dependent bias-expansion diagnostic (Sec. 1-8 of the
spec; Sec. 9-10, the nuisance/MG-parameter refits, are deferred -- see
README.md). Loops over MG points F4/F5/F6, saves numeric outputs (.npz) and
figures (.png).

Usage (from this directory, inside the cosmodesi_new conda env):
    python run_all.py
"""

import os

os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import time

import numpy as np

import config
import growth
import covariance
import toy_spectrum
import make_plots


def main():
    os.makedirs(config.OUTDIR, exist_ok=True)
    os.makedirs(config.FIGDIR, exist_ok=True)

    summary = {}

    for mg_name, mg_pt in config.MG_POINTS.items():
        t0 = time.time()
        print(f"\n=== {mg_name} (fR0_HS={mg_pt['fR0_HS']:.0e}) ===", flush=True)
        k_mg = growth.k_MG_estimate(mg_pt["fR0_HS"])
        print(f"  k_MG(z_eff) ~ {k_mg:.4f} h/Mpc", flush=True)

        out = toy_spectrum.build_all(mg_name)
        print(f"  built growth/bias/linear-P/full-kernel/toy in {time.time() - t0:.1f}s", flush=True)

        cov_full = covariance.compute_bgs_covariance(out["P_ell_standard"], k=out["k"])
        sigma = covariance.diag_sigma(cov_full, nk=out["k"].size)

        np.savez(
            os.path.join(config.OUTDIR, f"{mg_name}_results.npz"),
            k=out["k"], fR0_HS=out["fR0_HS"], k_MG=k_mg,
            PL_gr=out["PL_gr"], PL_mg=out["PL_mg"],
            **{f"D_z{z}": out["D"][z] for z in out["D"]},
            **{f"f_z{z}": out["f"][z] for z in out["f"]},
            **{f"D0_z{z}": out["D0"][z] for z in out["D0"]},
            **{f"b1_zstar{z}": out["b1"][z] for z in out["b1"]},
            **{f"deltab1_zstar{z}": out["delta_b1"][z] for z in out["delta_b1"]},
            **{f"deltaP_ell{ell}_zstar{z}": out["delta_P_ell"][z][ell]
               for z in config.Z_STAR for ell in (0, 2, 4)},
            **{f"Pstd_ell{ell}": out["P_ell_standard"][ell] for ell in (0, 2, 4)},
            **{f"Ptoy_ell{ell}_zstar{z}": out["P_ell_toy"][z][ell]
               for z in config.Z_STAR for ell in (0, 2, 4)},
            **{f"sigma_ell{ell}": sigma[ell] for ell in (0, 2, 4)},
            cov_full=cov_full,
        )

        make_plots.plot1_growth(mg_name, out)
        make_plots.plot2_bias(mg_name, out)
        make_plots.plot3_linear_power(mg_name, out)
        make_plots.plot4_delta_pbias(mg_name, out)
        _, max_sig = make_plots.plot5_standard_vs_toy(mg_name, out, sigma)

        summary[mg_name] = dict(
            fR0_HS=mg_pt["fR0_HS"], k_MG=k_mg,
            max_delta_b1_over_b1={
                float(z): float(np.max(np.abs(out["delta_b1"][z])) / config.B1_BGS)
                for z in config.Z_STAR
            },
            max_frac_delta_P_ell={
                float(z): {
                    int(ell): float(np.max(np.abs(out["delta_P_ell"][z][ell]))
                                     / np.max(np.abs(out["P_ell_standard"][ell])))
                    for ell in (0, 2, 4)
                }
                for z in config.Z_STAR
            },
            max_raw_significance_sigma=max_sig,
            elapsed_seconds=time.time() - t0,
        )
        print(f"  done in {time.time() - t0:.1f}s", flush=True)

    with open(os.path.join(config.OUTDIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved numeric outputs to {config.OUTDIR}/ and figures to {config.FIGDIR}/")
    print("NOTE: nuisance-only and MG-parameter refits (spec Sec. 9-10) are "
          "deferred -- see README.md.")


if __name__ == "__main__":
    main()
