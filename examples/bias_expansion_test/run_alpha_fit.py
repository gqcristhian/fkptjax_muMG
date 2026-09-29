"""Can (alpha0, alpha2) alone absorb the missing scale-dependent-bias effect?

For each MG point (F4/F5/F6) and each z*, holds b1 and every other nuisance
parameter fixed at the ordinary BGS values and re-fits only the monopole/
quadrupole EFT counterterms (alpha0, alpha2) to match P_ell,toy(k;z*). Reuses
the cached full-kernel table state (see full_kernel_pipeline.build_state), so
each fit itself is essentially free once P_ell,toy is available -- this
script re-loads P_ell,toy from outputs/{mg}_results.npz produced by
run_all.py rather than recomputing it.

Usage (after running run_all.py at least once):
    python run_alpha_fit.py
"""

import os
import json

import numpy as np

import config
import alpha_fit
import make_plots


def _load_out(mg_name):
    """Rebuild the `out` dict (toy_spectrum.build_all shape) from the saved npz."""
    d = np.load(os.path.join(config.OUTDIR, f"{mg_name}_results.npz"))
    k = d["k"]
    out = dict(k=k, fR0_HS=float(d["fR0_HS"]))
    out["P_ell_standard"] = {ell: d[f"Pstd_ell{ell}"] for ell in (0, 2, 4)}
    out["P_ell_toy"] = {
        z: {ell: d[f"Ptoy_ell{ell}_zstar{z}"] for ell in (0, 2, 4)}
        for z in config.Z_STAR
    }
    return out


def main():
    summary = {}
    for mg_name in config.MG_POINTS:
        print(f"\n=== {mg_name} ===", flush=True)
        out = _load_out(mg_name)
        fits = alpha_fit.fit_all_z_star(out)

        mg_summary = {}
        for z_star, fit in fits.items():
            ok_1pct = all(v < 0.01 for v in fit["max_residual_frac"].values())
            print(f"  z*={z_star}: alpha0={fit['alpha0']:.2f}  alpha2={fit['alpha2']:.2f}  "
                  f"max|resid| ell=0,2,4 = "
                  f"{fit['max_residual_frac'][0]*100:.3f}%, "
                  f"{fit['max_residual_frac'][2]*100:.3f}%, "
                  f"{fit['max_residual_frac'][4]*100:.3f}%  "
                  f"(<1% everywhere? {ok_1pct})", flush=True)
            mg_summary[float(z_star)] = dict(
                alpha0=fit["alpha0"], alpha2=fit["alpha2"],
                max_residual_frac_pct={int(e): fit["max_residual_frac"][e] * 100.0 for e in (0, 2, 4)},
                absorbed_below_1pct=ok_1pct,
            )

        make_plots.plot6_alpha_fit(mg_name, out, fits)
        summary[mg_name] = mg_summary

    with open(os.path.join(config.OUTDIR, "alpha_fit_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved outputs/alpha_fit_summary.json and figures/plot6_alpha_fit_*.png")


if __name__ == "__main__":
    main()
