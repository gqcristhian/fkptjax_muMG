"""
Isolate whether the ~2-7% BZ_Mass P_ell residual (see
check_bzmass_vs_sMGPT_full_v3.py) comes from the beyond-EdS KERNEL TABLES
or from the RSD/IR-resummation ASSEMBLY code, by running the SAME
downstream code (fkptjax's own FOLPS-based tables_to_poles) on BOTH sets
of raw kernels:

  (A) fkptjax's own raw kernel table (already computed, "v3")
  (B) sMGPT's own raw kernel table (AllFunctions_BGS_BZMass_NoScreen(.dat/_nw.dat)),
      repacked into fkptjax's internal table_w/table_now layout
      (fkptjax.types.KFunctionsOut's field names are a near-exact 1:1 match
      to sMGPT's AllFunctions columns -- both derive from the same
      FOLPS/EFT bias-expansion naming convention -- so this repacking is a
      column relabelling + B+C summation, not a reinterpretation).

Both (A) and (B) are then run through the IDENTICAL
fkptjax.pipelines.poles_from_tables / fkptjax.rsd.tables_to_poles code
(FOLPS's own Python IR-resummation + multipole projection), with the SAME
bias parameters, mu-quadrature, and AP settings.

If (B) agrees with sMGPT's OWN official (Wolfram-produced) multipoles to
<1%, the FOLPS Python assembly code reproduces sMGPT's Wolfram IR-resummation
just fine when given sMGPT's own kernels -- meaning the residual seen in
(A) vs sMGPT-official is a genuine, still-not-fully-explained KERNEL-TABLE
content difference (contradicting the earlier <0.5% raw P22/P13 check --
would point at one of the OTHER raw fields: I1udd1A family or the bias
Pb2b1-type terms).

If (B) shows a SIMILAR ~2-7% residual against sMGPT-official (even though
its kernel content IS sMGPT's own, by construction exact), that proves the
residual lives in the RSD/IR-resummation ASSEMBLY step itself (FOLPS's
Python implementation vs sMGPT's Wolfram one), not in the kernel tables --
consistent with the same conclusion drawn independently from the
Hu-Sawicki/F5 notebook.

sigma2_NW / delta_sigma2_NW (BAO-scale, spherical-Bessel-weighted
integrals of the no-wiggle *linear* P(k) -- MG-kernel-independent, i.e.
identical between the two kernel-table sources here since the underlying
linear P(k) already matches to <0.1%, see check_growth_ratio_pivot.py)
have no sMGPT-table analogue, so they are taken verbatim from fkptjax's
own (A) run rather than recomputed.
"""
import os
import sys
import time

import numpy as np
from scipy.signal import savgol_filter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fkptjax import mg_jax as bj
from fkptjax.jax_ode import DP_jax
from fkptjax.pipelines import binning_jax_poles, make_table_state, poles_from_tables
from fkptjax.kfuncs_to_tables import build_jax_static_ctx
from fkptjax.rsd import pack_fkpt_bias

DATA_DIR = os.path.join(os.path.dirname(__file__), "sMGPT_reference_data")

Om = 0.31519
z_target = 0.295
xnow = -3.912023
xstop = float(np.log(1.0 / (1.0 + z_target)))

# ----------------------------------------------------------------------------
# sMGPT reference tables
# ----------------------------------------------------------------------------
raw = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen.dat"))
raw_nw = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen_nw.dat"))
k_tab = raw[:, 0]
mult = np.loadtxt(os.path.join(DATA_DIR, "multipoles_BGS_BZMass_NoScreen.dat"))
k_ref, P0_ref, P2_ref, P4_ref = mult[:, 0], mult[:, 1], mult[:, 2], mult[:, 3]

# ----------------------------------------------------------------------------
# fkptjax's OWN kernel run (v3, corrected growth pivot) -- provides (A), and
# lends its sigma2_NW / delta_sigma2_NW to (B).
# ----------------------------------------------------------------------------
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

def make_mu_grid(nmu=12):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)

mu, wmu = make_mu_grid(nmu=12)
jac = jnp.asarray(1.0)
k_tab_j = jnp.asarray(k_tab)
kap = k_tab_j[:, None] * jnp.ones_like(mu)[None, :]
muap = jnp.ones_like(k_tab_j)[:, None] * mu[None, :]

bias = dict(
    b1=1.6215433618068629, b2=0.23677842354547154, bs2=-0.35516763531820733,
    b3nl=0.06314091294545908, alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0,
    alpha0shot=1.0, alpha2shot=0.0, PshotP=5000.0,
)
pars = pack_fkpt_bias(bias)

t0 = time.time()
static_ctx = build_jax_static_ctx(
    k_ab_j, kmin=0.001, kmax=1.0, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
)
print(f"static_ctx built in {time.time() - t0:.2f}s")

t0 = time.time()
poles_A, state_A = binning_jax_poles(
    k=k_ab_j, pk=jnp.asarray(pk_target), pk_now=jnp.asarray(pk_target_nw),
    jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
    ells=(0, 2, 4), bias_scheme="folps", IR_resummation=True,
    damping=None, A_full=False, use_TNS_model=False,
    return_kernel_constants=True,
    static_ctx=static_ctx,
    z=z_target, Om=Om, beyond_eds=True, fkpt_approximation=False,
    kmin=0.001, kmax=1.0, xnow=xnow, f0_kmax=0.001,
    model="HDKI", mg_variant="BZ_Mass",
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,
)
poles_A.block_until_ready()
print(f"(A) fkptjax kernels + FOLPS: {time.time() - t0:.2f}s")

sigma2_NW_reuse = state_A.table_now[29]
delta_sigma2_NW_reuse = state_A.table_now[30]

# ----------------------------------------------------------------------------
# (B) sMGPT's own raw kernel table, repacked into fkptjax's table_w/table_now
# layout, run through the SAME FOLPS tables_to_poles code.
# ----------------------------------------------------------------------------
def col(a, i):
    return jnp.asarray(a[:, i - 1])  # 1-indexed like the file header

zeros = jnp.zeros_like(k_tab_j)

f0_smgpt = float(raw[0, 34])
sigma2w_smgpt = col(raw, 34)[0]      # sMGPT's "sigma2v" == fkptjax's sigma2w (same f^2-weighted integral)
sigma2w_NW_smgpt = col(raw_nw, 34)[0]

fk_norm_smgpt = col(raw, 3) / f0_smgpt

table_w_smgpt = (
    k_tab_j, col(raw, 2), fk_norm_smgpt,
    col(raw, 4) + col(raw, 7), col(raw, 5) + col(raw, 8), col(raw, 6) + col(raw, 9),
    col(raw, 36), col(raw, 37), col(raw, 38), col(raw, 39), col(raw, 40), col(raw, 43),
    col(raw, 41), col(raw, 42),
    col(raw, 10), col(raw, 11), col(raw, 12), col(raw, 13), col(raw, 14),
    col(raw, 15) + col(raw, 24), col(raw, 16) + col(raw, 25),
    col(raw, 18) + col(raw, 27), col(raw, 19) + col(raw, 28),
    col(raw, 21) + col(raw, 30), col(raw, 22) + col(raw, 31), col(raw, 23) + col(raw, 32),
    zeros, zeros,
    sigma2w_smgpt, jnp.asarray(f0_smgpt),
)

table_now_smgpt = (
    k_tab_j, col(raw_nw, 2), fk_norm_smgpt,
    col(raw_nw, 4) + col(raw_nw, 7), col(raw_nw, 5) + col(raw_nw, 8), col(raw_nw, 6) + col(raw_nw, 9),
    col(raw_nw, 36), col(raw_nw, 37), col(raw_nw, 38), col(raw_nw, 39), col(raw_nw, 40), col(raw_nw, 43),
    col(raw_nw, 41), col(raw_nw, 42),
    col(raw_nw, 10), col(raw_nw, 11), col(raw_nw, 12), col(raw_nw, 13), col(raw_nw, 14),
    col(raw_nw, 15) + col(raw_nw, 24), col(raw_nw, 16) + col(raw_nw, 25),
    col(raw_nw, 18) + col(raw_nw, 27), col(raw_nw, 19) + col(raw_nw, 28),
    col(raw_nw, 21) + col(raw_nw, 30), col(raw_nw, 22) + col(raw_nw, 31), col(raw_nw, 23) + col(raw_nw, 32),
    zeros, zeros,
    sigma2w_NW_smgpt, sigma2_NW_reuse, delta_sigma2_NW_reuse, jnp.asarray(f0_smgpt),
)

assert len(table_w_smgpt) == len(state_A.table_w), (len(table_w_smgpt), len(state_A.table_w))
assert len(table_now_smgpt) == len(state_A.table_now), (len(table_now_smgpt), len(state_A.table_now))

state_B = make_table_state(table_w_smgpt, table_now_smgpt, kernel_constants=None)

t0 = time.time()
poles_B = poles_from_tables(
    state_B, jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
    ells=(0, 2, 4), bias_scheme="folps", IR_resummation=True,
    damping=None, A_full=False, use_TNS_model=False,
)
poles_B.block_until_ready()
print(f"(B) sMGPT kernels + FOLPS: {time.time() - t0:.2f}s")

# ----------------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------------
P0_A, P2_A, P4_A = (np.asarray(poles_A[i]) for i in range(3))
P0_B, P2_B, P4_B = (np.asarray(poles_B[i]) for i in range(3))

mask = (k_tab >= 0.01) & (k_tab <= 0.5)

def report(label, ours, refs):
    print(f"\n--- {label} ---")
    for name, o, r in zip(("P0", "P2", "P4"), ours, refs):
        pct = 100.0 * (o - r) / np.where(np.abs(r) > 1.0, r, 1.0)
        print(f"{name}: max={np.max(np.abs(pct[mask])):7.3f}%   mean={np.mean(np.abs(pct[mask])):7.3f}%")

report("(A) fkptjax kernels + FOLPS  vs  sMGPT official", (P0_A, P2_A, P4_A), (P0_ref, P2_ref, P4_ref))
report("(B) sMGPT kernels  + FOLPS  vs  sMGPT official", (P0_B, P2_B, P4_B), (P0_ref, P2_ref, P4_ref))
report("(A) vs (B): fkptjax-kernels vs sMGPT-kernels, BOTH through FOLPS", (P0_A, P2_A, P4_A), (P0_B, P2_B, P4_B))

# ----------------------------------------------------------------------------
# (C) Hybrid: sMGPT's raw kernels EXCEPT the Pb22-family bias-loop terms
# (Pb22, Pb2s2, Ps22, Pb2theta, Pbs2theta, sigma32PSL), swapped for fkptjax's
# own independently-computed values -- to test whether sMGPT's own exported
# "Pb22" column (numerically IDENTICAL to its "Pbs2b1" column at every k --
# an apparent export bug for this MG/no-screening configuration; sMGPT's own
# KPb22 formula depends only on the linear P(k) shape, no MG kernels at all)
# accounts for the entire remaining residual.
# ----------------------------------------------------------------------------
def interp_ours(field, kout):
    return jnp.asarray(np.interp(k_tab, np.asarray(kout), np.asarray(field)))

kout_A = np.asarray(state_A.kt)

# Pull fkptjax's own raw KFunctionsOut fields directly (re-run with
# return_raw_kfuncs=True is avoided; instead recompute via the same call
# binning_jax_poles already made internally is not exposed, so call the
# lower-level builder once more for the raw fields only).
from fkptjax.kfuncs_to_tables import Kfuncs_to_tables_jax
_, _, _, kfuncs_A = Kfuncs_to_tables_jax(
    k=k_ab_j, pk=jnp.asarray(pk_target), pk_now=jnp.asarray(pk_target_nw),
    z=z_target, Om=Om, beyond_eds=True, fkpt_approximation=False,
    kmin=0.001, kmax=1.0, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    xnow=xnow, f0_kmax=0.001,
    model="HDKI", mg_variant="BZ_Mass",
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,
    return_kernel_constants=True, return_raw_kfuncs=True,
    static_ctx=static_ctx,
)

def swap_bias(kf):
    return (
        interp_ours(kf.Pb22[0], kout_A), interp_ours(kf.Pb2s2[0], kout_A),
        interp_ours(kf.Ps22[0], kout_A), interp_ours(kf.Pb2theta[0], kout_A),
        interp_ours(kf.Pbs2theta[0], kout_A), interp_ours(kf.sigma32PSL[0], kout_A),
    )

Pb22_o, Pb2s2_o, Ps22_o, Pb2theta_o, Pbs2theta_o, sigma32PSL_o = swap_bias(kfuncs_A)

table_w_C = (
    table_w_smgpt[0], table_w_smgpt[1], table_w_smgpt[2],
    table_w_smgpt[3], table_w_smgpt[4], table_w_smgpt[5],
    table_w_smgpt[6], table_w_smgpt[7],
    Pb22_o, Pb2s2_o, Ps22_o, sigma32PSL_o, Pb2theta_o, Pbs2theta_o,
) + table_w_smgpt[14:]

table_now_C = (
    table_now_smgpt[0], table_now_smgpt[1], table_now_smgpt[2],
    table_now_smgpt[3], table_now_smgpt[4], table_now_smgpt[5],
    table_now_smgpt[6], table_now_smgpt[7],
    Pb22_o, Pb2s2_o, Ps22_o, sigma32PSL_o, Pb2theta_o, Pbs2theta_o,
) + table_now_smgpt[14:]

state_C = make_table_state(table_w_C, table_now_C, kernel_constants=None)
poles_C = poles_from_tables(
    state_C, jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
    ells=(0, 2, 4), bias_scheme="folps", IR_resummation=True,
    damping=None, A_full=False, use_TNS_model=False,
)
poles_C.block_until_ready()
P0_C, P2_C, P4_C = (np.asarray(poles_C[i]) for i in range(3))

report("(C) sMGPT kernels + fkptjax's OWN Pb22-family, + FOLPS  vs  fkptjax-official (A)",
       (P0_C, P2_C, P4_C), (P0_A, P2_A, P4_A))
report("(C) sMGPT kernels + fkptjax's OWN Pb22-family, + FOLPS  vs  sMGPT official",
       (P0_C, P2_C, P4_C), (P0_ref, P2_ref, P4_ref))

# ----------------------------------------------------------------------------
# (D) sMGPT's raw kernels, but with the CONFIRMED off-by-one column bug in
# their own src/4_Together.wl line 131 corrected:
#
#   Pb2b1T[[i]], Pbs2b1T[[i]], Pbs2b1T[[i]], Pb22T[[i]], Pb2s2T[[i]],
#   Pbs22T[[i]], Pb2tT[[i]], Pbs2tT[[i]], sigma32pkT[[i]]
#
# writes Pbs2b1T TWICE (9 values into 8 header-labeled slots 36-43), so
# every field from column 38 onward is the PREVIOUS field's value, and the
# true sigma32pk never gets exported at all. Corrected mapping (1-based):
#   36 Pb2b1 (ok), 37 Pbs2b1 (ok), 38 [duplicate Pbs2b1, discard],
#   39 Pb22 (true), 40 Pb2s2 (true), 41 Pbs22/Ps22 (true),
#   42 Pb2theta (true), 43 Pbs2theta (true), sigma32pk MISSING entirely
#   -> use fkptjax's own sigma32PSL (no sMGPT analogue survives the bug).
# ----------------------------------------------------------------------------
def col_now(i):
    return jnp.asarray(raw_nw[:, i - 1])

table_w_D = (
    k_tab_j, col(raw, 2), fk_norm_smgpt,
    col(raw, 4) + col(raw, 7), col(raw, 5) + col(raw, 8), col(raw, 6) + col(raw, 9),
    col(raw, 36), col(raw, 37),
    col(raw, 39), col(raw, 40), col(raw, 41), sigma32PSL_o, col(raw, 42), col(raw, 43),
    col(raw, 10), col(raw, 11), col(raw, 12), col(raw, 13), col(raw, 14),
    col(raw, 15) + col(raw, 24), col(raw, 16) + col(raw, 25),
    col(raw, 18) + col(raw, 27), col(raw, 19) + col(raw, 28),
    col(raw, 21) + col(raw, 30), col(raw, 22) + col(raw, 31), col(raw, 23) + col(raw, 32),
    zeros, zeros,
    sigma2w_smgpt, jnp.asarray(f0_smgpt),
)
table_now_D = (
    k_tab_j, col(raw_nw, 2), fk_norm_smgpt,
    col(raw_nw, 4) + col(raw_nw, 7), col(raw_nw, 5) + col(raw_nw, 8), col(raw_nw, 6) + col(raw_nw, 9),
    col(raw_nw, 36), col(raw_nw, 37),
    col(raw_nw, 39), col(raw_nw, 40), col(raw_nw, 41), sigma32PSL_o, col(raw_nw, 42), col(raw_nw, 43),
    col(raw_nw, 10), col(raw_nw, 11), col(raw_nw, 12), col(raw_nw, 13), col(raw_nw, 14),
    col(raw_nw, 15) + col(raw_nw, 24), col(raw_nw, 16) + col(raw_nw, 25),
    col(raw_nw, 18) + col(raw_nw, 27), col(raw_nw, 19) + col(raw_nw, 28),
    col(raw_nw, 21) + col(raw_nw, 30), col(raw_nw, 22) + col(raw_nw, 31), col(raw_nw, 23) + col(raw_nw, 32),
    zeros, zeros,
    sigma2w_NW_smgpt, sigma2_NW_reuse, delta_sigma2_NW_reuse, jnp.asarray(f0_smgpt),
)

state_D = make_table_state(table_w_D, table_now_D, kernel_constants=None)
poles_D = poles_from_tables(
    state_D, jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
    ells=(0, 2, 4), bias_scheme="folps", IR_resummation=True,
    damping=None, A_full=False, use_TNS_model=False,
)
poles_D.block_until_ready()
P0_D, P2_D, P4_D = (np.asarray(poles_D[i]) for i in range(3))

report("(D) sMGPT kernels, column-bug CORRECTED, + FOLPS  vs  fkptjax-official (A)",
       (P0_D, P2_D, P4_D), (P0_A, P2_A, P4_A))
report("(D) sMGPT kernels, column-bug CORRECTED, + FOLPS  vs  sMGPT official (buggy)",
       (P0_D, P2_D, P4_D), (P0_ref, P2_ref, P4_ref))
