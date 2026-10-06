"""
Full production-pipeline validation, v2: same as check_bzmass_vs_sMGPT_full.py,
but instead of using sMGPT's own sparse 121-point OUTPUT PSL(k)/f(k) table as
the fkptjax INPUT spectrum, this uses the full, dense z=0 Abacus_pklin_z0.dat
grid (888 points, k in [1e-5, 500]) as input, rescaled to z=0.295 via
fkptjax's OWN scale-dependent BZ_Mass growth ODE.

Why: v1 fed the 121-point, [0.001,1.0]-only AllFunctions PSL column into
Kfuncs_to_tables_jax's `k`/`pk` arguments, which ALSO set the domain/
resolution of the internal loop-integral extrapolation grid (via
extrapolate_pklin). That starves the P22/P13/sigma2/sigma2v integrals of
both range and resolution relative to sMGPT's own internal integration
(pmin=0.00001, pmax=16, ~183+ points) -- likely the dominant source of
v1's 2-6% residual, NOT a kernel-formula bug (confirmed separately: (a)
fkpt_approximation True/False agree to ~5e-7 for scale-independent models,
(b) fkptjax's own growth ODE for BZ_Mass matches sMGPT's to ~3e-5%, so
growth-ODE convention is not the source either).
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
from fkptjax.pipelines import binning_jax_poles
from fkptjax.kfuncs_to_tables import build_jax_static_ctx
from fkptjax.rsd import pack_fkpt_bias

DATA_DIR = os.path.join(os.path.dirname(__file__), "sMGPT_reference_data")

Om = 0.31519
z_target = 0.295
xnow = -3.912023
xstop = float(np.log(1.0 / (1.0 + z_target)))

# ----------------------------------------------------------------------------
# Full z=0 Abacus input, rescaled to z=0.295 via fkptjax's OWN BZ_Mass growth
# ----------------------------------------------------------------------------
k_ab, pk_ab_z0 = np.loadtxt(os.path.join(DATA_DIR, "Abacus_pklin_z0.dat"), unpack=True)
logpk = np.log(pk_ab_z0)
pk_ab_z0_nw = np.exp(savgol_filter(logpk, window_length=41, polyorder=3))

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

k_ab_j = jnp.asarray(k_ab)
Y0 = DP_jax(k_ab_j, P, xnow, 0.0)          # D(k, z=0)  [xstop=0 <-> a=1]
Yz = DP_jax(k_ab_j, P, xnow, xstop)        # D(k, z=z_target)
growth_ratio2 = np.asarray((Yz[0] / Y0[0]) ** 2)

pk_target = pk_ab_z0 * growth_ratio2
pk_target_nw = pk_ab_z0_nw * growth_ratio2
print(f"growth ratio^2(k): min={growth_ratio2.min():.4f}  max={growth_ratio2.max():.4f}")

# ----------------------------------------------------------------------------
# sMGPT reference
# ----------------------------------------------------------------------------
mult = np.loadtxt(os.path.join(DATA_DIR, "multipoles_BGS_BZMass_NoScreen.dat"))
k_ref, P0_ref, P2_ref, P4_ref = mult[:, 0], mult[:, 1], mult[:, 2], mult[:, 3]

# ----------------------------------------------------------------------------
# fkptjax, direct (non-rescale) JAX route, fkpt_approximation=False,
# full-resolution wide input grid, internal growth (fk=None, self-consistent)
# ----------------------------------------------------------------------------
def make_mu_grid(nmu=12):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)

mu, wmu = make_mu_grid(nmu=12)
jac = jnp.asarray(1.0)

k_out_eval = jnp.asarray(np.geomspace(0.001, 1.0, 121))
kap = k_out_eval[:, None] * jnp.ones_like(mu)[None, :]
muap = jnp.ones_like(k_out_eval)[:, None] * mu[None, :]

bias = dict(
    b1=1.6215433618068629, b2=0.23677842354547154, bs2=-0.35516763531820733,
    b3nl=0.06314091294545908, alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0,
    alpha0shot=1.0, alpha2shot=0.0, PshotP=5000.0,
)
pars = pack_fkpt_bias(bias)

t0 = time.time()
static_ctx = build_jax_static_ctx(
    k_ab_j, kmin=0.001, kmax=1.0,
    Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
)
print(f"static_ctx built in {time.time() - t0:.2f}s")

t0 = time.time()
poles, state = binning_jax_poles(
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
poles.block_until_ready()
print(f"binning_jax_poles (fkpt_approximation=False, wide grid): {time.time() - t0:.2f}s")
print("kernel_constants (A, Ap, CFD3, CFD3p):", state.kernel_constants)

k_out = np.asarray(static_ctx["kout"])
P0_ours = np.asarray(poles[0])
P2_ours = np.asarray(poles[1])
P4_ours = np.asarray(poles[2])

# ----------------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------------
P0_i = np.interp(k_ref, k_out_eval, P0_ours)
P2_i = np.interp(k_ref, k_out_eval, P2_ours)
P4_i = np.interp(k_ref, k_out_eval, P4_ours)

mask = (k_ref >= 0.01) & (k_ref <= 0.5)
print("\n=== fkptjax (v2, wide input grid) vs sMGPT, BZ_Mass, fkpt_approximation=False, k in [0.01, 0.5] ===")
for name, ours, ref_ in [("P0", P0_i, P0_ref), ("P2", P2_i, P2_ref), ("P4", P4_i, P4_ref)]:
    pct = 100.0 * (ours - ref_) / np.where(np.abs(ref_) > 1.0, ref_, 1.0)
    print(f"{name}: max |ours - sMGPT| / sMGPT = {np.max(np.abs(pct[mask])):.2f}%   "
          f"mean = {np.mean(np.abs(pct[mask])):.2f}%")

np.savez(os.path.join(DATA_DIR, "bzmass_full_comparison_v2.npz"),
          k_ref=k_ref, P0_ref=P0_ref, P2_ref=P2_ref, P4_ref=P4_ref,
          k_out=k_out_eval, P0_ours=P0_ours, P2_ours=P2_ours, P4_ours=P4_ours)
print("\nSaved bzmass_full_comparison_v2.npz")
