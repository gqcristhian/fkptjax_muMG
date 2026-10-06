import os, re, sys
import numpy as np
from scipy.signal import savgol_filter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "src")
from fkptjax import mg_jax as bj
from fkptjax.jax_ode import DP_jax
from fkptjax import MG_kernels as mgk
from fkptjax.kfuncs_to_tables import Kfuncs_to_tables_jax

DATA_DIR = "examples/sMGPT_reference_data"
Om = 0.31519
z_target = 0.295
xnow = -3.912023
xstop = float(np.log(1.0 / (1.0 + z_target)))

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)
f0 = 0.6826138224244257

# ---------------------------------------------------------------------------
# Part 1: D2/D3 (Gamma2evR / C3Gamma3-type) direct comparison, fkptjax vs sMGPT
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "D2D3_kcompare_wolfram.txt")) as f:
    lines = [l for l in f if l.startswith("NumberForm")]
rows = np.array([[float(v) for v in re.findall(r"NumberForm\[([^,]+),", line)] for line in lines])
k_slice, D2_smgpt, D2p_smgpt, CFD3_smgpt, CFD3p_smgpt = rows.T
print(f"Loaded {len(k_slice)} (k, D2, D2p, CFD3, CFD3p) rows from sMGPT's own solveD2fusedFull/solveD3fused")

r_fixed, x_fixed = 0.3, 0.3
k_j = jnp.asarray(k_slice)
p_j = r_fixed * k_j

D2_q, D2p_q = mgk.D2_fused_grid(x_fixed, k_j, p_j, P, xnow, xstop, solver='adaptive')
CFD3_q, CFD3p_q = mgk.D3_fused_grid(x_fixed, k_j, p_j, P, xnow, xstop, f0, solver='adaptive')
D2_q, D2p_q, CFD3_q, CFD3p_q = (np.asarray(v) for v in (D2_q, D2p_q, CFD3_q, CFD3p_q))

print("\n=== D2/D3 direct comparison (fkptjax vs sMGPT) ===")
for name, ours, smgpt in [("D2 (Gamma2evR)", D2_q, D2_smgpt), ("D2' ", D2p_q, D2p_smgpt),
                          ("CFD3", CFD3_q, CFD3_smgpt), ("CFD3'", CFD3p_q, CFD3p_smgpt)]:
    pct = 100.0 * (ours - smgpt) / np.where(np.abs(smgpt) > 1e-8, smgpt, 1.0)
    print(f"{name:16s} max|pct diff| = {np.max(np.abs(pct)):.4g}%   mean = {np.mean(np.abs(pct)):.4g}%")

# ---------------------------------------------------------------------------
# Part 2: I2uud2A / I3uuu3A zero-crossing investigation
# ---------------------------------------------------------------------------
k_ab, pk_ab_z0 = np.loadtxt(os.path.join(DATA_DIR, "Abacus_pklin_z0.dat"), unpack=True)
logpk = np.log(pk_ab_z0)
pk_ab_z0_nw = np.exp(savgol_filter(logpk, window_length=41, polyorder=3))
k_pivot = k_ab[1]
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

print("\n=== I2uud2A / I3uuu3A: absolute values around the reported max-pct-diff region ===")
for name, ours_raw, smgpt_col in [
    ("I2uud2A", kfuncs.I2uud2A[0], raw[:, 11]),
    ("I3uuu3A", kfuncs.I3uuu3A[0], raw[:, 13]),
]:
    ours_i = interp_ours(ours_raw)
    pct = 100.0 * (ours_i - smgpt_col) / np.where(np.abs(smgpt_col) > 1e-8, smgpt_col, 1.0)
    mask = (k_tab >= 0.01) & (k_tab <= 0.5)
    i_worst = np.argmax(np.abs(pct * mask))
    print(f"\n{name}: worst point at k={k_tab[i_worst]:.4g}, pct diff={pct[i_worst]:.2f}%")
    lo = max(0, i_worst - 3)
    hi = min(len(k_tab), i_worst + 4)
    for i in range(lo, hi):
        print(f"  k={k_tab[i]:8.4g}  ours={ours_i[i]: .6e}  sMGPT={smgpt_col[i]: .6e}  "
              f"absdiff={abs(ours_i[i]-smgpt_col[i]):.3e}  pct={pct[i]:8.2f}%")
    print(f"  field range over full k-grid: ours [{ours_i.min():.3e}, {ours_i.max():.3e}]  "
          f"sMGPT [{smgpt_col.min():.3e}, {smgpt_col.max():.3e}]")

print("\n=== CFD3 raw value comparison (diagnosing the 320% factor) ===")
for i in [0, 10, 20, 30, 39]:
    print(f"k={k_slice[i]:.4g}  CFD3_ours={CFD3_q[i]: .6f}  CFD3_smgpt={CFD3_smgpt[i]: .6f}  "
          f"ratio={CFD3_q[i]/CFD3_smgpt[i]: .6f}  diff={CFD3_q[i]-CFD3_smgpt[i]: .6f}")
    print(f"   CFD3p_ours={CFD3p_q[i]: .6f}  CFD3p_smgpt={CFD3p_smgpt[i]: .6f}  ratio={CFD3p_q[i]/CFD3p_smgpt[i]: .6f}")

print("\n=== CFD3 with the 5/21 convention factor applied ===")
CFD3_q_conv = CFD3_q * 5.0 / 21.0
CFD3p_q_conv = CFD3p_q * 5.0 / 21.0
for name, ours, smgpt in [("CFD3*5/21", CFD3_q_conv, CFD3_smgpt), ("CFD3'*5/21", CFD3p_q_conv, CFD3p_smgpt)]:
    pct = 100.0 * (ours - smgpt) / np.where(np.abs(smgpt) > 1e-8, smgpt, 1.0)
    print(f"{name:16s} max|pct diff| = {np.max(np.abs(pct)):.4g}%   mean = {np.mean(np.abs(pct)):.4g}%")
