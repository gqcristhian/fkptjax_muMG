import os, sys, time
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

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

k_ab_j = jnp.asarray(k_ab)
k_pivot = k_ab[1]
Yz = DP_jax(k_ab_j, P, xnow, xstop)
D_pivot0 = DP_jax(jnp.asarray([k_pivot]), P, xnow, 0.0)[0][0]
growth_ratio2 = np.asarray((Yz[0] / D_pivot0) ** 2)
pk_target = pk_ab_z0 * growth_ratio2
pk_target_nw = pk_ab_z0_nw * growth_ratio2

raw = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen.dat"))
k_tab = raw[:, 0]
PSL_smgpt, fk_smgpt = raw[:, 1], raw[:, 2]

# --- compare our rescaled input PSL(k) against sMGPT's own PSL(k) column ---
PSL_ours_i = np.interp(k_tab, k_ab, pk_target)
pct_psl = 100.0 * (PSL_ours_i - PSL_smgpt) / PSL_smgpt
mask = (k_tab >= 0.01) & (k_tab <= 0.5)
print(f"PSL(k) input: max|pct diff| over k in [0.01,0.5] = {np.max(np.abs(pct_psl[mask])):.3f}%   mean = {np.mean(np.abs(pct_psl[mask])):.3f}%")

t0 = time.time()
table_w, table_now, kernel_constants, kfuncs = Kfuncs_to_tables_jax(
    k=k_ab_j, pk=jnp.asarray(pk_target), pk_now=jnp.asarray(pk_target_nw),
    z=z_target, Om=Om, beyond_eds=True, fkpt_approximation=False,
    kmin=0.001, kmax=1.0, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    xnow=xnow, f0_kmax=0.001,
    model="HDKI", mg_variant="BZ_Mass",
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,
    return_kernel_constants=True, return_raw_kfuncs=True,
)
print(f"Kfuncs_to_tables_jax: {time.time()-t0:.2f}s")
kout = np.asarray(table_w[0])

for name, ours, smgpt_col in [
    ("f(k)", kfuncs.fk if hasattr(kfuncs, "fk") else None, fk_smgpt),
    ("P22dd", kfuncs.P22dd[0], raw[:, 3]), ("P22dt", kfuncs.P22du[0], raw[:, 4]), ("P22tt", kfuncs.P22uu[0], raw[:, 5]),
    ("P13dd", kfuncs.P13dd[0], raw[:, 6]), ("P13dt", kfuncs.P13du[0], raw[:, 7]), ("P13tt", kfuncs.P13uu[0], raw[:, 8]),
    ("I1udd1A", kfuncs.I1udd1A[0], raw[:, 9]), ("I2uud1A", kfuncs.I2uud1A[0], raw[:, 10]),
    ("I2uud2A", kfuncs.I2uud2A[0], raw[:, 11]), ("I3uuu2A", kfuncs.I3uuu2A[0], raw[:, 12]),
    ("I3uuu3A", kfuncs.I3uuu3A[0], raw[:, 13]),
    ("I2uudd1BpC", kfuncs.I2uudd1BpC[0], raw[:, 14] + raw[:, 23]),
    ("I2uudd2BpC", kfuncs.I2uudd2BpC[0], raw[:, 15] + raw[:, 24]),
    ("I3uuud2BpC", kfuncs.I3uuud2BpC[0], raw[:, 17] + raw[:, 26]),
    ("I3uuud3BpC", kfuncs.I3uuud3BpC[0], raw[:, 18] + raw[:, 27]),
    ("I4uuuu2BpC", kfuncs.I4uuuu2BpC[0], raw[:, 20] + raw[:, 29]),
    ("I4uuuu3BpC", kfuncs.I4uuuu3BpC[0], raw[:, 21] + raw[:, 30]),
    ("I4uuuu4BpC", kfuncs.I4uuuu4BpC[0], raw[:, 22] + raw[:, 31]),
    ("Pb1b2", kfuncs.Pb1b2[0], raw[:, 35]), ("Pb1bs2", kfuncs.Pb1bs2[0], raw[:, 36]),
    ("Pb22", kfuncs.Pb22[0], raw[:, 37]), ("Pb2s2", kfuncs.Pb2s2[0], raw[:, 38]),
    ("Ps22", kfuncs.Ps22[0], raw[:, 39]),
    ("Pb2theta", kfuncs.Pb2theta[0], raw[:, 40]), ("Pbs2theta", kfuncs.Pbs2theta[0], raw[:, 41]),
    ("sigma32PSL", kfuncs.sigma32PSL[0], raw[:, 42]),
]:
    if ours is None:
        continue
    ours_np = np.asarray(ours)
    xgrid = kout if ours_np.shape[0] == kout.shape[0] else k_tab
    ours_i = np.interp(k_tab, xgrid, ours_np)
    mask = (k_tab >= 0.01) & (k_tab <= 0.5)
    pct = 100.0 * (ours_i - smgpt_col) / np.where(np.abs(smgpt_col) > 1e-8, smgpt_col, 1.0)
    print(f"{name:6s} max|pct diff| over k in [0.01,0.5] = {np.max(np.abs(pct[mask])):8.2f}%   mean={np.mean(np.abs(pct[mask])):8.2f}%")

print("kernel_constants (A, Ap, CFD3, CFD3p):", kernel_constants)
print("sigma2, sigma2v (row0) sMGPT:", raw[0, 32], raw[0, 33], " f0 sMGPT:", raw[0, 34])
