"""Diagnostics 1-4 + the final compact summary figure/table.

Reads ``config.OUTDIR/summary.npz`` (written by ``run_all.py``) and writes
``config.FIGDIR/*.png`` -- both suffixed by ``config.MODEL_CHOICE``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import config

THEORY_LABELS = {"eds": "EdS", "approx": "bEdS-fkPT", "full": "bEdS-full"}
THEORY_COLORS = {"eds": "tab:red", "approx": "tab:blue", "full": "k"}
THEORY_STYLES = {"eds": "--", "approx": ":", "full": "-"}

# LaTeX for whichever MG parameter config.MG_PARAM_NAME points to.
_MG_LATEX = {"mu0": r"\mu_0", "fR0_HS": r"f_{R0}"}
MG_LATEX = _MG_LATEX.get(config.MG_PARAM_NAME, config.MG_PARAM_NAME.replace("_", r"\_"))


def plot1_dP_dmu0(derivs_by_theory, outpath):
    """dP_ell/d(MG param) for EdS, bEdS-fkPT, bEdS-full."""
    nk = len(config.K_FIT)
    mgparam_label = config.MG_PARAM_NAME.replace("_", r"\_")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for iell, ell in enumerate(config.ELLS):
        ax = axes[iell]
        for theory in config.THEORIES:
            d = derivs_by_theory[theory][config.MG_PARAM_NAME][iell * nk:(iell + 1) * nk]
            ax.plot(config.K_FIT, d, color=THEORY_COLORS[theory], ls=THEORY_STYLES[theory],
                    label=THEORY_LABELS[theory])
        ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
        ax.set_ylabel(fr"$dP_{ell}/d{{\rm {mgparam_label}}}$")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle(fr"$dP_\ell/d{{\rm {mgparam_label}}}$: EdS vs bEdS-fkPT vs bEdS-full")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot2_staged_sigma(staged_by_theory, outpath):
    """sigma(mu0) vs staged-marginalization stage, one line per theory."""
    stage_names = list(config.STAGES.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for theory in config.THEORIES:
        sig = [staged_by_theory[theory][s]["sigma_mu0"] for s in stage_names]
        ax.plot(range(len(stage_names)), sig, marker="o",
                color=THEORY_COLORS[theory], ls=THEORY_STYLES[theory], label=THEORY_LABELS[theory])
    ax.set_xticks(range(len(stage_names)))
    ax.set_xticklabels(stage_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(fr"$\sigma({MG_LATEX})$")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(fr"Staged marginalization: $\sigma({MG_LATEX})$ vs. nuisance freedom")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot3_correlations(staged_by_theory, outpath, stage="(e) full nuisance model",
                        counterterms=("alpha0", "alpha2", "alpha4")):
    """rho(mu0, alpha_i) at the full-nuisance-model stage, one bar group per theory."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    x = np.arange(len(counterterms))
    for i, theory in enumerate(config.THEORIES):
        corr = staged_by_theory[theory][stage]["corr"]
        vals = [corr.get(c, np.nan) for c in counterterms]
        ax.bar(x + (i - 1) * width, vals, width, label=THEORY_LABELS[theory], color=THEORY_COLORS[theory])
    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\rho({MG_LATEX},\alpha_{{{c[-1] if c[-1].isdigit() else c}}})$" for c in counterterms])
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(fr"${MG_LATEX}$-counterterm correlations at {stage}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot4_ellipses(staged_by_theory, outpath, stage="(e) full nuisance model",
                    other_param="alpha0", nsigma=(1, 2)):
    """Fisher ellipses: MG param vs other_param, one ellipse per theory."""
    import fisher as fisher_mod

    fig, ax = plt.subplots(figsize=(6, 6))
    for theory in config.THEORIES:
        res = staged_by_theory[theory][stage]
        params = res["params"]
        ell = fisher_mod.ellipse_sub(res["Finv"], params, config.MG_PARAM_NAME, other_param)
        for ns in nsigma:
            e = Ellipse(
                xy=(config.MG_PARAM_FID, config.PARAMS[other_param]["fid"]),
                width=2 * ns * ell["semi_major"], height=2 * ns * ell["semi_minor"],
                angle=ell["angle_deg"], fill=False,
                edgecolor=THEORY_COLORS[theory], ls=THEORY_STYLES[theory],
                lw=1.5 if ns == 1 else 1.0, alpha=1.0 if ns == 1 else 0.5,
                label=THEORY_LABELS[theory] if ns == 1 else None,
            )
            ax.add_patch(e)
    ax.set_xlabel(config.MG_PARAM_NAME.replace("_", r"\_"))
    ax.set_ylabel(rf"${other_param}$")
    ax.relim()
    ax.autoscale_view()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(fr"Fisher ellipses 1$\sigma$/2$\sigma$: ${MG_LATEX}$ vs {other_param}, {stage}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_table(staged_by_theory, bias_by_theory, stage="(e) full nuisance model"):
    """Compact summary: rows = theories, columns = the requested quantities."""
    sigma_full = staged_by_theory["full"][stage]["sigma_mu0"]
    rows = []
    for theory in config.THEORIES:
        sigma = staged_by_theory[theory][stage]["sigma_mu0"]
        corr = staged_by_theory[theory][stage]["corr"]
        row = dict(
            theory=THEORY_LABELS[theory],
            sigma_mu0=sigma,
            sigma_ratio_to_full=sigma / sigma_full,
            rho_mu0_alpha0=corr.get("alpha0", np.nan),
            rho_mu0_alpha2=corr.get("alpha2", np.nan),
            rho_mu0_alpha4=corr.get("alpha4", np.nan),
        )
        if theory in bias_by_theory:
            b = bias_by_theory[theory]
            row["delta_mu0_over_sigma"] = b["delta_mu0_over_sigma"]
            row["delta_alpha0"] = b["delta_theta"].get("alpha0", np.nan)
            row["delta_alpha2"] = b["delta_theta"].get("alpha2", np.nan)
            row["delta_alpha4"] = b["delta_theta"].get("alpha4", np.nan)
            row["delta_b1"] = b["delta_theta"].get("b1", np.nan)
        else:
            row["delta_mu0_over_sigma"] = np.nan  # not applicable to "full" (its own truth)
            row["delta_alpha0"] = row["delta_alpha2"] = row["delta_alpha4"] = row["delta_b1"] = np.nan
        rows.append(row)
    return rows


def plot5_summary_figure(rows, outpath):
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    col_labels = ["theory", fr"$\sigma({MG_LATEX})$", r"$\sigma/\sigma_{full}$",
                  fr"$\Delta {MG_LATEX}/\sigma$", fr"$\rho({MG_LATEX},\alpha_0)$",
                  fr"$\rho({MG_LATEX},\alpha_2)$", fr"$\rho({MG_LATEX},\alpha_4)$",
                  r"$\Delta\alpha_0$", r"$\Delta\alpha_2$", r"$\Delta b_1$"]
    cell_text = []
    for row in rows:
        cell_text.append([
            row["theory"], f"{row['sigma_mu0']:.4g}", f"{row['sigma_ratio_to_full']:.3f}",
            "n/a" if np.isnan(row["delta_mu0_over_sigma"]) else f"{row['delta_mu0_over_sigma']:.3f}",
            f"{row['rho_mu0_alpha0']:.3f}", f"{row['rho_mu0_alpha2']:.3f}", f"{row['rho_mu0_alpha4']:.3f}",
            "n/a" if np.isnan(row["delta_alpha0"]) else f"{row['delta_alpha0']:.3f}",
            "n/a" if np.isnan(row["delta_alpha2"]) else f"{row['delta_alpha2']:.3f}",
            "n/a" if np.isnan(row["delta_b1"]) else f"{row['delta_b1']:.4f}",
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    fig.suptitle("Fisher_bEdS_FoM summary: EdS / bEdS-fkPT / bEdS-full", y=0.95)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
