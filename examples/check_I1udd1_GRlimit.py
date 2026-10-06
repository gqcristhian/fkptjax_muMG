"""
GR-limit sanity check for the new I1udd1-family three-ordering (Q+R+N) fix:
with mu_kinf=1.0 (BZ_Mass reduces exactly to mu==1 identically, i.e. GR),
fkpt_approximation=False should reproduce fkpt_approximation=True's I1udd1A
almost exactly (to ODE-solver/quadrature tolerance), since A=B=1 everywhere
in both paths.
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
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,  # genuine BZ_Mass, mu_kinf=1.2
    return_raw_kfuncs=True, return_kernel_constants=False,
)

print("Running fkpt_approximation=True (GR limit)...")
table_w_true, table_nw_true, kfuncs_true = Kfuncs_to_tables_jax(fkpt_approximation=True, **common)
kout = np.asarray(table_w_true[0])
I1udd1A_true = np.asarray(kfuncs_true.I1udd1A)[0]

print("Running fkpt_approximation=False (GR limit, NEW three-ordering fix)...")
table_w_full, table_nw_full, kfuncs_full = Kfuncs_to_tables_jax(fkpt_approximation=False, **common)
I1udd1A_full = np.asarray(kfuncs_full.I1udd1A)[0]

print()
print("k grid:            ", kout)
print("I1udd1A (squeezed): ", I1udd1A_true)
print("I1udd1A (full, GR): ", I1udd1A_full)
print("abs diff:           ", I1udd1A_full - I1udd1A_true)
print("rel diff:           ", (I1udd1A_full - I1udd1A_true) / I1udd1A_true)
