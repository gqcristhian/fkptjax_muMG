"""
Smoke test for the new genuine three-ordering (Q+R+N) I1udd1-family fix in
calculate_jax.py / kfuncs_to_tables.py -- runs the REAL Kfuncs_to_tables_jax
pipeline (not a hand reimplementation) with a small grid for speed, using the
real Abacus P(k) (GR-evolved to z=0.295, same convention as
check_vs_sMGPT_reference.ipynb), for the HDKI/BZ_Mass model.

Checks:
1. fkpt_approximation=True still runs and gives the old, validated numbers
   (regression safety -- A_N stays None on this path, so the "have_N" branch
   in calculate_jax.py is never taken here).
2. fkpt_approximation=False now runs (previously would silently keep using
   the broken "2x R-ordering" shortcut) and gives I1udd1A of a SENSIBLE
   order of magnitude (comparable to P22dd/fkpt_approximation=True's
   I1udd1A at the same k -- tens to hundreds, not the ~1e5 blowups seen in
   the hand-derived reimplementation attempts).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

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
pk_target = k_abacus_pk = pk_abacus_z0 * gr_growth_ratio2
print(f"pure-GR (D(z={z_target})/D(z=0))^2 = {gr_growth_ratio2:.6f}")

# Small grid for a fast smoke test (MG_kernels.py's own perf note: reduce
# Nk_kernel/nquadSteps/NQ/NR for quick checks -- production defaults are
# Nk_kernel=120, nquadSteps=300, NQ=NR=10).
common = dict(
    k=k_abacus, pk=pk_target, pk_now=pk_target, z=z_target, Om=Om,
    beyond_eds=True, kmin=0.005, kmax=2.0,
    Nk_kernel=6, nquadSteps=16, NQ=4, NR=4,
    xnow=xnow_gr, model="HDKI", mg_variant="BZ_Mass",
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,
    return_raw_kfuncs=True, return_kernel_constants=False,
)

print("Running fkpt_approximation=True (squeezed, old/validated path)...")
table_w_true, table_nw_true, kfuncs_true = Kfuncs_to_tables_jax(
    fkpt_approximation=True, **common)
kout = np.asarray(table_w_true[0])
I1udd1A_true = np.asarray(kfuncs_true.I1udd1A)
P22dd_true = np.asarray(kfuncs_true.P22dd)
print("k grid:", kout)
print("I1udd1A (fkpt_approximation=True):", I1udd1A_true)
print("P22dd   (fkpt_approximation=True):", P22dd_true)

print()
print("Running fkpt_approximation=False (full kernels, NEW three-ordering fix)...")
table_w_full, table_nw_full, kfuncs_full = Kfuncs_to_tables_jax(
    fkpt_approximation=False, **common)
I1udd1A_full = np.asarray(kfuncs_full.I1udd1A)
P22dd_full = np.asarray(kfuncs_full.P22dd)
print("I1udd1A (fkpt_approximation=False):", I1udd1A_full)
print("P22dd   (fkpt_approximation=False):", P22dd_full)

print()
print("ratio I1udd1A(full)/I1udd1A(squeezed):", I1udd1A_full / I1udd1A_true)
print("ratio P22dd(full)/P22dd(squeezed):    ", P22dd_full / P22dd_true)
