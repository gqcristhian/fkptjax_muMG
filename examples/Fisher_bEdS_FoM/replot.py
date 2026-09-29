"""Regenerate figures from an already-saved outputs/summary.npz, without
redoing any of the expensive Fisher/derivative computation. Useful after a
make_plots.py label/style fix.
"""

import numpy as np

import config
import make_plots


def main():
    data = np.load(f"{config.OUTDIR}/summary.npz", allow_pickle=True)
    derivs_by_theory = data["derivs_by_theory"].item()
    staged_by_theory = data["staged_by_theory"].item()
    bias_by_theory = data["bias_by_theory"].item()

    make_plots.plot1_dP_dmu0(derivs_by_theory, f"{config.FIGDIR}/plot1_dP_dmu0.png")
    make_plots.plot2_staged_sigma(staged_by_theory, f"{config.FIGDIR}/plot2_staged_sigma.png")
    make_plots.plot3_correlations(staged_by_theory, f"{config.FIGDIR}/plot3_correlations.png")
    make_plots.plot4_ellipses(staged_by_theory, f"{config.FIGDIR}/plot4_ellipse_alpha0.png",
                               other_param="alpha0")
    make_plots.plot4_ellipses(staged_by_theory, f"{config.FIGDIR}/plot4_ellipse_alpha2.png",
                               other_param="alpha2")
    rows = make_plots.summary_table(staged_by_theory, bias_by_theory)
    make_plots.plot5_summary_figure(rows, f"{config.FIGDIR}/plot5_summary.png")
    print(f"Replotted into {config.FIGDIR}/")


if __name__ == "__main__":
    main()
