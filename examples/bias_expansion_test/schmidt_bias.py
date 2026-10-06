"""Schmidt conserved-tracer scale-dependent linear bias (diagnostic-only).

Implements Sec. 8.3 of Desjacques, Jeong & Schmidt (arXiv:1611.09787):

    b1(k, z_BGS; z*) = 1 + (b1_BGS - 1) * [D(0,z_BGS)/D(0,z*)] * [D(k,z*)/D(k,z_BGS)]

eliminating the unknown formation-time bias b1* by matching b1(k->0) to the
ordinary fiducial b1_BGS. Kept entirely separate from ``fkptjax``'s production
kernels/bias code: this module only combines the exact scale-dependent linear
growth (``growth.py``) with an externally supplied linear P_L(k) and growth
rate f(k), and reuses ``fkptjax.rsd.project_to_poles`` (the same
Gauss-Legendre projector production RSD code uses) to go from
Delta_P_bias^s(k,mu) to Delta_P_ell,bias(k).
"""

import numpy as np
import jax.numpy as jnp

from fkptjax.rsd import project_to_poles

import config
import growth


def make_mu_grid(nmu=12):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)


def schmidt_b1(k, P_mg, z_star, z_eff=config.Z_EFF, b1_eff=config.B1_BGS):
    """b1(k, z_eff; z_star) for MGConstants P_mg, normalized so b1(k->0)=b1_eff."""
    k = jnp.asarray(k)
    D0_eff = growth.growth_D0(P_mg, z_eff)
    D0_star = growth.growth_D0(P_mg, z_star)
    Dk_star, _ = growth.growth_D_f(k, P_mg, z_star)
    Dk_eff, _ = growth.growth_D_f(k, P_mg, z_eff)
    ratio = (D0_eff / D0_star) * (Dk_star / Dk_eff)
    return 1.0 + (b1_eff - 1.0) * ratio


def delta_b1(k, P_mg, z_star, z_eff=config.Z_EFF, b1_eff=config.B1_BGS):
    return schmidt_b1(k, P_mg, z_star, z_eff=z_eff, b1_eff=b1_eff) - b1_eff


def delta_P_bias_poles(k, P_mg, PL_k, f_k, z_star, z_eff=config.Z_EFF,
                        b1_eff=config.B1_BGS, ells=(0, 2, 4), nmu=12):
    """Exact (non-linearized) Delta_P_ell,bias(k; z_star), ell in `ells`.

    PL_k, f_k are the exact MG linear P_L(k,z_eff) and growth rate f(k,z_eff)
    (both evaluated at the BGS observation redshift, not the formation z_star).

    Returns
    -------
    b1k : array (nk,)
    poles : array (len(ells), nk)
    """
    k = jnp.asarray(k)
    PL_k = jnp.asarray(PL_k)
    f_k = jnp.asarray(f_k)
    b1k = schmidt_b1(k, P_mg, z_star, z_eff=z_eff, b1_eff=b1_eff)

    mu, wmu = make_mu_grid(nmu)
    mu2 = mu ** 2

    P_std = (b1_eff + f_k[:, None] * mu2[None, :]) ** 2 * PL_k[:, None]
    P_sdbias = (b1k[:, None] + f_k[:, None] * mu2[None, :]) ** 2 * PL_k[:, None]
    dP = P_sdbias - P_std

    poles = project_to_poles(dP, mu, wmu, ells=ells)
    return np.asarray(b1k), np.asarray(poles)


if __name__ == "__main__":
    P_mg = growth.hs_constants(config.MG_POINTS["F5"]["fR0_HS"])
    k = config.K_PLOT

    # b1(k->0) == b1_BGS check.
    b1_small_k = float(schmidt_b1(jnp.asarray([config.K_ZERO_PROXY]), P_mg, z_star=2.0)[0])
    print(f"b1(k->0; z*=2) = {b1_small_k:.6f}  vs b1_BGS = {config.B1_BGS:.6f}")
    assert abs(b1_small_k - config.B1_BGS) < 1e-6

    # Delta_b1 -> 0 as z* -> z_eff.
    db1_same_z = float(delta_b1(jnp.asarray([0.1]), P_mg, z_star=config.Z_EFF)[0])
    print(f"Delta_b1(k=0.1; z*=z_eff) = {db1_same_z:.3e} (expect ~0)")
    assert abs(db1_same_z) < 1e-6

    for z_star in config.Z_STAR:
        b1k = np.asarray(schmidt_b1(k, P_mg, z_star))
        print(f"F5, z*={z_star}: b1(k) range [{b1k.min():.4f}, {b1k.max():.4f}]  "
              f"Delta_b1/b1 max = {np.max(np.abs(b1k - config.B1_BGS)) / config.B1_BGS:.4f}")

    # Delta_P_bias_poles smoke test with a toy PL/f (real ones come from linear_power.py).
    PL_toy = 1.0e4 * (k / 0.1) ** -1.5
    _, f_toy = growth.growth_D_f(k, P_mg, config.Z_EFF)
    b1k, poles = delta_P_bias_poles(k, P_mg, PL_toy, f_toy, z_star=2.0)
    print("Delta_P_ell,bias poles shape:", poles.shape, "finite:", np.all(np.isfinite(poles)))
