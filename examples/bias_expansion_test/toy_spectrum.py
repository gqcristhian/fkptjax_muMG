"""Build the toy 'true' BGS power spectrum: full-kernel standard + missing bias.

    P_ell,toy(k; z*) = P_ell,full^standard(k) + Delta_P_ell,bias(k; z*)

Per the spec: this adds the linear missing-bias correction on top of the
otherwise-complete full-kernel prediction; it does NOT insert b1(k) into the
one-loop bias integrals themselves.
"""

import numpy as np

import config
import growth
import linear_power
import schmidt_bias
import full_kernel_pipeline


def build_all(mg_name, ells=(0, 2, 4), z_star_list=config.Z_STAR, k=None):
    """Compute everything needed for one MG point (e.g. 'F5').

    Returns a dict with keys:
      k, fR0_HS, P_mg,
      PL_gr, PL_mg (on k),
      D, f (dict z -> array; z_eff and each z*),
      D0 (dict z -> scalar),
      b1 (dict z* -> array), delta_b1 (dict z* -> array),
      delta_P_ell (dict z* -> {ell: array}),
      P_ell_standard ({ell: array}),
      P_ell_toy (dict z* -> {ell: array}),
    """
    if k is None:
        k = config.K_FIT
    fR0_HS = config.MG_POINTS[mg_name]["fR0_HS"]
    P_mg = growth.hs_constants(fR0_HS)

    # --- linear spectra (Plot 3) ---
    k_gr, pk_gr, _ = linear_power.linear_power_gr()
    k_mg, pk_mg, _ = linear_power.linear_power_hs(fR0_HS)
    PL_gr = linear_power.interp_to(k, k_gr, pk_gr)
    PL_mg = linear_power.interp_to(k, k_mg, pk_mg)

    # --- growth (Plot 1) ---
    D = {}
    f = {}
    D0 = {}
    all_z = [config.Z_EFF] + list(z_star_list)
    for z in all_z:
        Dz, fz = growth.growth_D_f(k, P_mg, z)
        D[z] = Dz
        f[z] = fz
        D0[z] = growth.growth_D0(P_mg, z)

    # --- Schmidt bias + Delta_P_ell,bias (Plots 2, 4) ---
    b1 = {}
    delta_b1 = {}
    delta_P_ell = {}
    f_eff = f[config.Z_EFF]
    for z_star in z_star_list:
        b1k, poles = schmidt_bias.delta_P_bias_poles(
            k, P_mg, PL_mg, f_eff, z_star=z_star, ells=ells,
        )
        b1[z_star] = b1k
        delta_b1[z_star] = b1k - config.B1_BGS
        delta_P_ell[z_star] = {ell: poles[i] for i, ell in enumerate(ells)}

    # --- full-kernel standard P_ell (Plots 5, 6) ---
    P_ell_standard = full_kernel_pipeline.compute_standard_poles(fR0_HS, k=k, ells=ells)

    # --- toy = standard + missing bias correction ---
    P_ell_toy = {}
    for z_star in z_star_list:
        P_ell_toy[z_star] = {
            ell: P_ell_standard[ell] + delta_P_ell[z_star][ell] for ell in ells
        }

    return dict(
        k=np.asarray(k), fR0_HS=fR0_HS, PL_gr=PL_gr, PL_mg=PL_mg,
        D=D, f=f, D0=D0, b1=b1, delta_b1=delta_b1, delta_P_ell=delta_P_ell,
        P_ell_standard=P_ell_standard, P_ell_toy=P_ell_toy,
    )


if __name__ == "__main__":
    out = build_all("F5")
    print("k:", out["k"].shape)
    for z_star in config.Z_STAR:
        maxfrac = max(
            np.max(np.abs(out["delta_P_ell"][z_star][ell]) / np.max(np.abs(out["P_ell_standard"][ell])))
            for ell in (0, 2, 4)
        )
        print(f"z*={z_star}: max |Delta_P_ell| / max|P_ell_standard| = {maxfrac:.4f}")
