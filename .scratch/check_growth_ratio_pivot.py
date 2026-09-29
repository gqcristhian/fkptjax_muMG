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
k_pivot = k_ab[1]  # kGrid[[2]] in Mathematica's 1-indexing = second row of the RAW input file
print("k_pivot =", k_pivot)

raw = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen.dat"))
k_tab, PSL_smgpt = raw[:, 0], raw[:, 1]
inPk_interp = np.exp(np.interp(np.log(k_tab), np.log(k_ab), np.log(pk_ab_z0)))
ratio2_smgpt_implied = PSL_smgpt / inPk_interp

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

k_tab_j = jnp.asarray(k_tab)
Yz = DP_jax(k_tab_j, P, xnow, xstop)          # D(k, zev)   -- per-k, numerator
D_pivot0 = DP_jax(jnp.asarray([k_pivot]), P, xnow, 0.0)[0][0]  # D(k_pivot, z=0) -- fixed scalar denominator

ratio2_pivot = np.asarray((Yz[0] / D_pivot0) ** 2)

pct = 100.0 * (ratio2_pivot - ratio2_smgpt_implied) / ratio2_smgpt_implied
mask = (k_tab >= 0.01) & (k_tab <= 0.5)
print(f"{'k':>10s} {'ratio2_pivot':>14s} {'ratio2_smgpt':>14s} {'pct_diff':>10s}")
for i in range(0, len(k_tab), 10):
    print(f"{k_tab[i]:10.4g} {ratio2_pivot[i]:14.6f} {ratio2_smgpt_implied[i]:14.6f} {pct[i]:9.4f}%")
print(f"\nmax|pct diff| over k in [0.01,0.5]: {np.max(np.abs(pct[mask])):.4f}%   mean: {np.mean(np.abs(pct[mask])):.4f}%")
