"""
Companion to wolfram_I1udd1_crosscheck_snippet.txt: prints fkptjax's own
production I1udd1-family values (all 5 kernels) at k_ext=0.18205642, genuine
HDKI/BZ_Mass (mu_kinf=1.2), real Abacus P(k) rescaled to z=0.295 -- for
direct comparison against the Wolfram Cloud output.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fkptjax.ode import ModelDerivatives, DP, ODESolver
from fkptjax.kfuncs_to_tables import Kfuncs_to_tables_jax

DATA_DIR = os.path.join(os.path.dirname(__file__), "sMGPT_reference_data")
Om = 0.315192
z_target = 0.295
xnow_gr = -3.912023

k_abacus, pk_abacus_z0 = np.loadtxt(os.path.join(DATA_DIR, "Abacus_pklin_z0.dat"), unpack=True)

gr_derivs = ModelDerivatives(om=Om, ol=1.0 - Om, model="HDKI", mg_variant="mu_OmDE", mu0=0.0)
D_gr_0 = DP(1e-3, gr_derivs, ODESolver(zout=0.0, xnow=xnow_gr, method="RKQS"))[0]
D_gr_target = DP(1e-3, gr_derivs, ODESolver(zout=z_target, xnow=xnow_gr, method="RKQS"))[0]
gr_growth_ratio2 = (D_gr_target / D_gr_0) ** 2
pk_target = pk_abacus_z0 * gr_growth_ratio2

common = dict(
    k=k_abacus, pk=pk_target, pk_now=pk_target, z=z_target, Om=Om,
    beyond_eds=True, kmin=0.005, kmax=2.0,
    Nk_kernel=6, nquadSteps=40, NQ=10, NR=10,
    xnow=xnow_gr, model="HDKI", mg_variant="BZ_Mass",
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,
    return_raw_kfuncs=True, return_kernel_constants=False,
)

table_w, table_nw, kfuncs = Kfuncs_to_tables_jax(fkpt_approximation=False, **common)
kout = np.asarray(table_w[0])
idx = 3  # k_ext = 0.18205642
print("k_ext =", kout[idx])
for name in ["I1udd1A", "I2uud1A", "I2uud2A", "I3uuu2A", "I3uuu3A"]:
    val = np.asarray(getattr(kfuncs, name))[0, idx]
    print(f"{name} (fkptjax, full-kernel fix) = {val!r}")
