import os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "src")
from fkptjax import mg_jax as bj
from fkptjax.jax_ode import DP_jax

DATA_DIR = "examples/sMGPT_reference_data"
Om = 0.31519
z_target = 0.295
xnow = -3.912023
xstop = float(np.log(1.0 / (1.0 + z_target)))

k_ab, pk_ab_z0 = np.loadtxt(os.path.join(DATA_DIR, "Abacus_pklin_z0.dat"), unpack=True)

raw = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen.dat"))
k_tab, PSL_smgpt = raw[:, 0], raw[:, 1]

# sMGPT's IMPLIED growth ratio^2(k) = PSL(k, zev) / inPk(k, z=0), via log-log
# interpolation of the raw Abacus input onto the output k-grid.
inPk_interp = np.exp(np.interp(np.log(k_tab), np.log(k_ab), np.log(pk_ab_z0)))
ratio2_smgpt_implied = PSL_smgpt / inPk_interp

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)
k_tab_j = jnp.asarray(k_tab)
Y0 = DP_jax(k_tab_j, P, xnow, 0.0)
Yz = DP_jax(k_tab_j, P, xnow, xstop)
ratio2_ours = np.asarray((Yz[0] / Y0[0]) ** 2)

print(f"{'k':>10s} {'ratio2_ours':>14s} {'ratio2_smgpt':>14s} {'pct_diff':>10s}")
for i in range(0, len(k_tab), 10):
    r_o, r_s = ratio2_ours[i], ratio2_smgpt_implied[i]
    print(f"{k_tab[i]:10.4g} {r_o:14.6f} {r_s:14.6f} {100*(r_o-r_s)/r_s:9.3f}%")

pct = 100.0 * (ratio2_ours - ratio2_smgpt_implied) / ratio2_smgpt_implied
mask = (k_tab >= 0.01) & (k_tab <= 0.5)
print(f"\nmax|pct diff| ratio2 over k in [0.01,0.5]: {np.max(np.abs(pct[mask])):.3f}%   mean: {np.mean(np.abs(pct[mask])):.3f}%")
print("ratio2_ours at k->0.001:", ratio2_ours[0], " ratio2_smgpt at k->0.001:", ratio2_smgpt_implied[0])
print("pure-GR expectation (Om=0.31519):", end=" ")
Pgr = bj.pack_constants_jnp(om=Om, ol=1.0-Om, kind=bj.MU_OMDE, mu0=0.0)
Y0g = DP_jax(k_tab_j, Pgr, xnow, 0.0)
Yzg = DP_jax(k_tab_j, Pgr, xnow, xstop)
print(float((Yzg[0][0]/Y0g[0][0])**2))
