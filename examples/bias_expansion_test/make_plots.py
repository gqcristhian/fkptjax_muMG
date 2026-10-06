"""Plots 1-5 for the BGS scale-dependent bias-expansion test.

Each function takes the dict returned by ``toy_spectrum.build_all`` (plus,
for plot 5, the analytic covariance) and saves one PNG under config.FIGDIR.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import growth

Z_STAR_COLORS = {1.0: "C0", 2.0: "C1", 3.0: "C2", 5.0: "C3"}
ELL_COLORS = {0: "C0", 2: "C1", 4: "C2"}


def _mark_scales(ax, k_mg, k_max=config.KMAX):
    ax.axvline(k_mg, color="0.4", ls="--", lw=1.0, label=fr"$k_{{\rm MG}}\approx{k_mg:.3f}$")
    ax.axvline(k_max, color="0.6", ls=":", lw=1.0, label=fr"$k_{{\max}}={k_max:.2f}$")


def plot1_growth(mg_name, out, outdir=config.FIGDIR):
    k_mg = growth.k_MG_estimate(out["fR0_HS"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    D0_eff = out["D0"][config.Z_EFF]
    ax1.plot(out["k"], out["D"][config.Z_EFF], color="k", lw=2, label=fr"$z_{{\rm eff}}={config.Z_EFF}$")
    ax2.plot(out["k"], out["D"][config.Z_EFF] / D0_eff, color="k", lw=2, label=fr"$z_{{\rm eff}}={config.Z_EFF}$")
    for z_star in config.Z_STAR:
        c = Z_STAR_COLORS[z_star]
        ax1.plot(out["k"], out["D"][z_star], color=c, lw=1.5, label=fr"$z_*={z_star:.0f}$")
        ax2.plot(out["k"], out["D"][z_star] / out["D0"][z_star], color=c, lw=1.5, label=fr"$z_*={z_star:.0f}$")

    ax1.set_xscale("log")
    ax1.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); ax1.set_ylabel(r"$D_+(k,z)$")
    ax1.set_title(f"{mg_name}: linear growth")
    _mark_scales(ax1, k_mg); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); ax2.set_ylabel(r"$D_+(k,z)/D_+(k\to0,z)$")
    ax2.set_title(f"{mg_name}: growth, normalized to $k\\to0$")
    _mark_scales(ax2, k_mg); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"plot1_growth_{mg_name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot2_bias(mg_name, out, outdir=config.FIGDIR):
    k_mg = growth.k_MG_estimate(out["fR0_HS"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.axhline(config.B1_BGS, color="k", lw=2, ls="-", label=fr"const. $b_1^{{\rm BGS}}={config.B1_BGS}$")
    for z_star in config.Z_STAR:
        c = Z_STAR_COLORS[z_star]
        ax1.plot(out["k"], out["b1"][z_star], color=c, lw=1.5, label=fr"$z_*={z_star:.0f}$")
        ax2.plot(out["k"], out["delta_b1"][z_star] / config.B1_BGS, color=c, lw=1.5, label=fr"$z_*={z_star:.0f}$")

    ax1.set_xscale("log")
    ax1.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); ax1.set_ylabel(r"$b_1(k)$")
    ax1.set_title(f"{mg_name}: Schmidt scale-dependent BGS bias")
    _mark_scales(ax1, k_mg); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.axhline(0.0, color="0.5", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); ax2.set_ylabel(r"$\Delta b_1(k)/b_1^{\rm BGS}$")
    ax2.set_title(f"{mg_name}: fractional bias correction")
    _mark_scales(ax2, k_mg); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"plot2_bias_{mg_name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot3_linear_power(mg_name, out, outdir=config.FIGDIR):
    k_mg = growth.k_MG_estimate(out["fR0_HS"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                    gridspec_kw=dict(height_ratios=(2, 1), hspace=0.08))

    ax1.loglog(out["k"], out["PL_gr"], color="k", lw=1.8, label=r"$P_L^{\rm GR}$")
    ax1.loglog(out["k"], out["PL_mg"], color="C3", lw=1.8, ls="--", label=fr"$P_L^{{\rm MG}}$ ({mg_name})")
    ax1.set_ylabel(r"$P_L(k)\ [({\rm Mpc}/h)^3]$")
    ax1.set_title(fr"{mg_name}: linear matter power at $z_{{\rm eff}}={config.Z_EFF}$")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, which="both")

    ax2.plot(out["k"], out["PL_mg"] / out["PL_gr"], color="C3", lw=1.8)
    ax2.axhline(1.0, color="0.5", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$k\ [h/{\rm Mpc}]$"); ax2.set_ylabel(r"$P_L^{\rm MG}/P_L^{\rm GR}$")
    for ax in (ax1, ax2):
        _mark_scales(ax, k_mg)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"plot3_linear_power_{mg_name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot4_delta_pbias(mg_name, out, outdir=config.FIGDIR):
    k_mg = growth.k_MG_estimate(out["fR0_HS"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)

    for j, ell in enumerate((0, 2, 4)):
        ax_top, ax_bot = axes[0, j], axes[1, j]
        for z_star in config.Z_STAR:
            c = Z_STAR_COLORS[z_star]
            dP = out["delta_P_ell"][z_star][ell]
            Pstd = out["P_ell_standard"][ell]
            ax_top.plot(out["k"], dP, color=c, lw=1.5, label=fr"$z_*={z_star:.0f}$")
            ax_bot.plot(out["k"], dP / Pstd, color=c, lw=1.5)
        ax_top.set_title(fr"$\ell={ell}$")
        ax_top.axhline(0.0, color="0.5", lw=0.8)
        ax_bot.axhline(0.0, color="0.5", lw=0.8)
        _mark_scales(ax_top, k_mg); _mark_scales(ax_bot, k_mg)
        ax_bot.set_xscale("log")
        ax_bot.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
        ax_top.grid(alpha=0.3); ax_bot.grid(alpha=0.3)
        if j == 0:
            ax_top.set_ylabel(r"$\Delta P_{\ell,{\rm bias}}(k)$")
            ax_bot.set_ylabel(r"$\Delta P_{\ell,{\rm bias}}/P_{\ell,{\rm full}}^{\rm standard}$")
            ax_top.legend(fontsize=8)

    fig.suptitle(f"{mg_name}: missing scale-dependent-bias correction")
    fig.tight_layout()
    path = os.path.join(outdir, f"plot4_delta_pbias_{mg_name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def plot5_standard_vs_toy(mg_name, out, sigma, outdir=config.FIGDIR):
    """sigma: dict {ell: array(nk,)} of 1-sigma analytic errors (covariance.py)."""
    k_mg = growth.k_MG_estimate(out["fR0_HS"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    max_sig = {}

    for j, ell in enumerate((0, 2, 4)):
        ax_top, ax_bot = axes[0, j], axes[1, j]
        Pstd = out["P_ell_standard"][ell]
        ax_top.errorbar(out["k"], Pstd, yerr=sigma[ell], fmt="-", color="k", lw=1.5,
                         ecolor="0.8", elinewidth=1, label="standard")
        for z_star in config.Z_STAR:
            c = Z_STAR_COLORS[z_star]
            Ptoy = out["P_ell_toy"][z_star][ell]
            ax_top.plot(out["k"], Ptoy, color=c, lw=1.3, ls="--", label=fr"toy, $z_*={z_star:.0f}$")
            ax_bot.plot(out["k"], (Ptoy - Pstd) / Pstd, color=c, lw=1.3)
        sig_ratio = np.abs(out["delta_P_ell"][config.Z_STAR[-1]][ell]) / sigma[ell]
        max_sig[ell] = float(np.max(sig_ratio))
        ax_top.set_title(fr"$\ell={ell}$   max $|\Delta P|/\sigma={max_sig[ell]:.2f}$ ($z_*={config.Z_STAR[-1]:.0f}$)")
        ax_bot.axhline(0.0, color="0.5", lw=0.8)
        _mark_scales(ax_top, k_mg); _mark_scales(ax_bot, k_mg)
        ax_bot.set_xscale("log")
        ax_bot.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
        ax_top.grid(alpha=0.3); ax_bot.grid(alpha=0.3)
        if j == 0:
            ax_top.set_ylabel(r"$P_\ell(k)$")
            ax_bot.set_ylabel(r"$(P_{\ell,{\rm toy}}-P_{\ell,{\rm full}}^{\rm standard})/P_{\ell,{\rm full}}^{\rm standard}$")
            ax_top.legend(fontsize=7, ncol=2)

    fig.suptitle(f"{mg_name}: full-kernel standard vs. toy (with analytic BGS errors)")
    fig.tight_layout()
    path = os.path.join(outdir, f"plot5_standard_vs_toy_{mg_name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path, max_sig


def plot6_alpha_fit(mg_name, out, fits, outdir=config.FIGDIR):
    """Residual before/after fitting the EFT counterterms (alpha0, alpha2).

    `fits`: dict {z_star: alpha_fit.fit_alpha(...) result}. Dotted = raw
    (P_toy - P_standard)/P_toy (alpha0=alpha2=0); solid = after re-fitting
    alpha0, alpha2 to match P_toy. A shaded +/-1% band marks the target.
    """
    k_mg = growth.k_MG_estimate(out["fR0_HS"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for j, ell in enumerate((0, 2, 4)):
        ax = axes[j]
        ax.axhspan(-1.0, 1.0, color="0.85", zorder=0, label=r"$\pm1\%$")
        Pstd = out["P_ell_standard"][ell]
        for z_star in config.Z_STAR:
            c = Z_STAR_COLORS[z_star]
            Ptruth = out["P_ell_toy"][z_star][ell]
            before = 100.0 * (Ptruth - Pstd) / Ptruth
            after = 100.0 * fits[z_star]["residual_frac"][ell]
            ax.plot(out["k"], before, color=c, lw=1.0, ls=":", alpha=0.6)
            ax.plot(out["k"], after, color=c, lw=1.8, ls="-", label=fr"$z_*={z_star:.0f}$")
        ax.axhline(0.0, color="0.4", lw=0.8)
        _mark_scales(ax, k_mg)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k\ [h/{\rm Mpc}]$")
        ax.set_title(fr"$\ell={ell}$")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("% residual  (dotted: before fit, solid: after $(\\alpha_0,\\alpha_2)$ fit)")
            ax.legend(fontsize=8)

    alpha_str = "; ".join(
        fr"$z_*={z:.0f}$: $\alpha_0={fits[z]['alpha0']:.1f}, \alpha_2={fits[z]['alpha2']:.1f}$"
        for z in config.Z_STAR
    )
    fig.suptitle(f"{mg_name}: residual after refitting EFT counterterms\n{alpha_str}", fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, f"plot6_alpha_fit_{mg_name}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    return path
