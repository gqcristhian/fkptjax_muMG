"""Exact scale-dependent linear growth D_+(k,z) and growth rate f(k,z).

Thin wrapper around fkptjax's own linear-growth ODE solver
(``fkptjax.jax_ode.DP_jax`` / ``fkptjax.mg_jax``); no production code is
modified. ``DP_jax`` returns ``[D, D']`` at ``xstop = ln(a)``, with
``D' = dD/d(ln a)``, so ``f(k,z) = D'/D`` requires no finite differencing.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax import mg_jax as bj
from fkptjax.jax_ode import DP_jax

import config


def z_to_xstop(z):
    return float(np.log(1.0 / (1.0 + z)))


def hs_constants(fR0_HS, n_HS=config.N_HS, beta2=config.BETA2_HS):
    """MGConstants for Hu-Sawicki f(R); fR0_HS=0 reduces to GR (mu=1)."""
    return bj.pack_constants_jnp(
        om=config.OM, ol=config.OL, kind=bj.HS,
        fR0_HS=fR0_HS, beta2=beta2, n_HS=n_HS,
    )


def gr_constants():
    return bj.pack_constants_jnp(om=config.OM, ol=config.OL, kind=bj.LCDM)


def growth_D_f(k, P, z):
    """Return (D(k,z), f(k,z)) for MGConstants P, on array k."""
    k = jnp.asarray(k)
    xstop = z_to_xstop(z)
    Y = DP_jax(k, P, config.XNOW, xstop)
    D, Dp = Y[0], Y[1]
    f = Dp / D
    return np.asarray(D), np.asarray(f)


def growth_D0(P, z):
    """D(k->0, z): growth evaluated at a tiny k, the model's scale-independent limit."""
    D0, _ = growth_D_f(jnp.asarray([config.K_ZERO_PROXY]), P, z)
    return float(D0[0])


def k_MG_estimate(fR0_HS, z=config.Z_EFF, n_HS=config.N_HS, beta2=config.BETA2_HS):
    """Analytic Hu-Sawicki chameleon transition scale a*m(a) at redshift z.

    mu(a,k) = 1 + 2*beta2*k^2/(k^2+(a*m(a))^2); the transition (mu halfway
    between its k->0 and k->inf limits) sits at k = a*m(a). Used only as a
    human-readable label/vertical-line marker, not in any physics computation.
    """
    a = 1.0 / (1.0 + z)
    om, ol = config.OM, config.OL
    Y_a = ol  # w0=-1, wa=0 background, so Y(a) = Y0 = ol identically.
    Y_0 = ol
    m = (
        (1.0 / bj.INV_H0)
        * np.sqrt(1.0 / ((1.0 + n_HS) * abs(fR0_HS)))
        * (om * a ** -3.0 + 4.0 * Y_a) ** ((2.0 + n_HS) / 2.0)
        / (om + 4.0 * Y_0) ** ((1.0 + n_HS) / 2.0)
    )
    return float(a * m)


if __name__ == "__main__":
    for name, pt in config.MG_POINTS.items():
        kmg = k_MG_estimate(pt["fR0_HS"])
        print(f"{name}: fR0_HS={pt['fR0_HS']:.0e}  k_MG(z={config.Z_EFF})~{kmg:.4f} h/Mpc")

    P = hs_constants(config.MG_POINTS["F5"]["fR0_HS"])
    D, f = growth_D_f(config.K_PLOT, P, config.Z_EFF)
    print("F5 D(k,z_eff) range:", D.min(), D.max())
    print("F5 f(k,z_eff) range:", f.min(), f.max())

    Pgr = gr_constants()
    Dgr, fgr = growth_D_f(config.K_PLOT, Pgr, config.Z_EFF)
    print("GR D(k,z_eff) is k-independent:", np.allclose(Dgr, Dgr[0]))
    print("GR f(k,z_eff) is k-independent:", np.allclose(fgr, fgr[0]))
