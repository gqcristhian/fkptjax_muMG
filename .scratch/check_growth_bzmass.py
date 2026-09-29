import os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "src")
from fkptjax import mg_jax as bj
from fkptjax.jax_ode import DP_jax

DATA_DIR = "examples/sMGPT_reference_data"
tab = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen.dat"))
k_tab, fk_smgpt, f0_smgpt = tab[:, 0], tab[:, 2], tab[0, 34]

Om = 0.31519
xnow = -3.912023
z = 0.295
xstop = float(np.log(1.0/(1.0+z)))

P = bj.pack_constants_jnp(om=Om, ol=1.0-Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

k_j = jnp.asarray(k_tab)
Y = DP_jax(k_j, P, xnow, xstop)
D, Dp = Y[0], Y[1]
fk_ours = np.asarray(Dp / D)

pct = 100.0*(fk_ours - fk_smgpt)/fk_smgpt
print("xnow(fkptjax)=", xnow, " vs sMGPT etaini=-6")
print("f0: fkptjax f(k->0) approx=", fk_ours[0], " sMGPT f0=", f0_smgpt)
print("max |pct diff| f(k) over all k:", np.max(np.abs(pct)))
print("mean |pct diff| f(k):", np.mean(np.abs(pct)))
for i in [0, 20, 40, 60, 80, 100, 120]:
    print(f"k={k_tab[i]:.4g}  fkptjax={fk_ours[i]:.6f}  sMGPT={fk_smgpt[i]:.6f}  pct={pct[i]:.3f}%")
