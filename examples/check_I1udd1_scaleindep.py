"""
Scale-independent MG sanity check for the new I1udd1-family fix -- STRONGER
than the pure-GR check: HDKI/mu_OmDE has mu(a) genuinely != 1 (real MG
effect, A/B != 1), but mu has NO k-dependence, so A(k1,k2,k3) is still
identical regardless of which leg plays which role in the (Q,R,N) triple.
fkpt_approximation=True (squeezed) and False (new genuine three-ordering
sum) MUST therefore still agree closely -- if they don't, that's a clean,
internal (no external reference needed) sign of a bug in the fix, since
there is no real angular/momentum "shape" here for the new formula to get
right or wrong.
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

# Real P(k), GR-evolved z=0 -> z=0.295 (same convention as before) -- this is
# just the INPUT spectrum shape; the mu_OmDE model's own (non-GR) growth is
# handled internally by Kfuncs_to_tables_jax's beyond_eds/kernel machinery.
gr_derivs = ModelDerivatives(om=Om, ol=1.0 - Om, model="HDKI", mg_variant="mu_OmDE", mu0=0.0)
D_gr_0 = DP(1e-3, gr_derivs, ODESolver(zout=0.0, xnow=xnow_gr, method="RKQS"))[0]
D_gr_target = DP(1e-3, gr_derivs, ODESolver(zout=z_target, xnow=xnow_gr, method="RKQS"))[0]
gr_growth_ratio2 = (D_gr_target / D_gr_0) ** 2
pk_target = pk_abacus_z0 * gr_growth_ratio2

common = dict(
    k=k_abacus, pk=pk_target, pk_now=pk_target, z=z_target, Om=Om,
    beyond_eds=True, kmin=0.005, kmax=2.0,
    Nk_kernel=6, nquadSteps=40, NQ=10, NR=10,
    xnow=xnow_gr, model="HDKI", mg_variant="mu_OmDE",
    mu0=0.3,  # genuinely non-trivial, but scale-independent, MG effect
    return_raw_kfuncs=True, return_kernel_constants=False,
)

print("Running fkpt_approximation=True (squeezed)...")
table_w_true, table_nw_true, kfuncs_true = Kfuncs_to_tables_jax(fkpt_approximation=True, **common)
kout = np.asarray(table_w_true[0])

print("Running fkpt_approximation=False (full, NEW three-ordering fix)...")
table_w_full, table_nw_full, kfuncs_full = Kfuncs_to_tables_jax(fkpt_approximation=False, **common)

print()
print("k grid:", kout)
for name in ["P22dd", "I1udd1A", "I2uud1A", "I2uud2A", "I3uuu2A", "I3uuu3A"]:
    v_true = np.asarray(getattr(kfuncs_true, name))[0]
    v_full = np.asarray(getattr(kfuncs_full, name))[0]
    rel = (v_full - v_true) / v_true
    print(f"{name}:")
    print(f"  squeezed: {v_true}")
    print(f"  full:     {v_full}")
    print(f"  rel diff: {rel}")
