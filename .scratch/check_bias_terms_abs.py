import os, sys
import numpy as np
from scipy.signal import savgol_filter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "src")
from fkptjax import mg_jax as bj
from fkptjax.jax_ode import DP_jax
from fkptjax.kfuncs_to_tables import Kfuncs_to_tables_jax

DATA_DIR = "examples/sMGPT_reference_data"
Om = 0.31519
z_target = 0.295
xnow = -3.912023
xstop = float(np.log(1.0 / (1.0 + z_target)))

k_ab, pk_ab_z0 = np.loadtxt(os.path.join(DATA_DIR, "Abacus_pklin_z0.dat"), unpack=True)
logpk = np.log(pk_ab_z0)
pk_ab_z0_nw = np.exp(savgol_filter(logpk, window_length=41, polyorder=3))
k_pivot = k_ab[1]
P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)
k_ab_j = jnp.asarray(k_ab)
Yz = DP_jax(k_ab_j, P, xnow, xstop)
D_pivot0 = DP_jax(jnp.asarray([k_pivot]), P, xnow, 0.0)[0][0]
growth_ratio2 = np.asarray((Yz[0] / D_pivot0) ** 2)
pk_target = pk_ab_z0 * growth_ratio2
pk_target_nw = pk_ab_z0_nw * growth_ratio2

raw = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen.dat"))
k_tab = raw[:, 0]

table_w, table_now, kernel_constants, kfuncs = Kfuncs_to_tables_jax(
    k=k_ab_j, pk=jnp.asarray(pk_target), pk_now=jnp.asarray(pk_target_nw),
    z=z_target, Om=Om, beyond_eds=True, fkpt_approximation=False,
    kmin=0.001, kmax=1.0, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    xnow=xnow, f0_kmax=0.001,
    model="HDKI", mg_variant="BZ_Mass",
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,
    return_kernel_constants=True, return_raw_kfuncs=True,
)
kout = np.asarray(table_w[0])

def interp_ours(field):
    return np.interp(k_tab, kout, np.asarray(field))

fields = [
    ("Pb1b2", kfuncs.Pb1b2[0], raw[:, 35]),
    ("Pb1bs2", kfuncs.Pb1bs2[0], raw[:, 36]),
    ("Pb22", kfuncs.Pb22[0], raw[:, 37]),
    ("Pb2s2", kfuncs.Pb2s2[0], raw[:, 38]),
    ("Ps22", kfuncs.Ps22[0], raw[:, 39]),
    ("Pb2theta", kfuncs.Pb2theta[0], raw[:, 40]),
    ("Pbs2theta", kfuncs.Pbs2theta[0], raw[:, 41]),
    ("sigma32PSL", kfuncs.sigma32PSL[0], raw[:, 42]),
]
idxs = [0, 20, 40, 60, 80, 100, 120]
for name, ours, smgpt in fields:
    ours_i = interp_ours(ours)
    print(f"\n=== {name} ===")
    for i in idxs:
        r = ours_i[i] / smgpt[i] if smgpt[i] != 0 else float("nan")
        print(f"  k={k_tab[i]:8.4g}  ours={ours_i[i]: .6e}  sMGPT={smgpt[i]: .6e}  ratio={r: .4f}")
