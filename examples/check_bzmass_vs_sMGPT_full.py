"""
Full production-pipeline validation: fkptjax's fkpt_approximation=False
(genuine, non-squeezed beyond-EdS) kernels vs sMGPT's own independent
Mathematica reference, for the HDKI/BZ_Mass model (mu_kinf=1.2,
lambda_a=lambda_dS=100.0), Om=0.31519, z=0.295, BGS tracer/bias.

Methodology: take sMGPT's OWN linear PSL(k) and f(k) (columns 2/3 of
AllFunctions_BGS_BZMass_NoScreen(.dat/_nw.dat), i.e. their own
scale-dependent-growth-evolved z=0.295 spectrum and growth rate) as the
INPUT to fkptjax, instead of re-deriving the growth-ODE evolution
ourselves. This isolates exactly what the user wants to check -- whether
the two codes' beyond-EdS LOOP KERNEL formulas and RSD/IR-resummation
assembly agree -- from any incidental disagreement in the growth-ODE
integration itself (already spot-checked separately this session; see
diagnose_growth_ode_fixed.wl).

Final P_ell(k) multipoles are compared directly against
outputs/BGS_BZMass/NoScreen/multipoles_BGS_BZMass_NoScreen.dat (their own
IR-resummed, bias-projected output), which used
b1=1.6215433618068629, b2=0.23677842354547154, bs2=-0.35516763531820733,
b3nl=0.06314091294545908, alpha0=alpha2=alpha4=0, ctilde=0,
alphashot0=1, alphashot2=0, PshotP=5000.
"""
import os
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fkptjax.pipelines import binning_jax_poles
from fkptjax.kfuncs_to_tables import build_jax_static_ctx
from fkptjax.rsd import pack_fkpt_bias

DATA_DIR = os.path.join(os.path.dirname(__file__), "sMGPT_reference_data")

# ----------------------------------------------------------------------------
# sMGPT's own reference tables (already-evolved PSL/f(k), and final P_ell)
# ----------------------------------------------------------------------------
tab = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen.dat"))
tab_nw = np.loadtxt(os.path.join(DATA_DIR, "AllFunctions_BGS_BZMass_NoScreen_nw.dat"))
k_tab, PSL, fk_tab, f0_tab = tab[:, 0], tab[:, 1], tab[:, 2], tab[0, 34]
PSL_nw = tab_nw[:, 1]

mult = np.loadtxt(os.path.join(DATA_DIR, "multipoles_BGS_BZMass_NoScreen.dat"))
k_ref, P0_ref, P2_ref, P4_ref = mult[:, 0], mult[:, 1], mult[:, 2], mult[:, 3]

print(f"sMGPT table: {len(k_tab)} k-points, k in [{k_tab.min():.4g}, {k_tab.max():.4g}]")
print(f"sMGPT f0 = {f0_tab!r}")

# ----------------------------------------------------------------------------
# fkptjax, direct (non-rescale) JAX route, fkpt_approximation=False
# ----------------------------------------------------------------------------
Om = 0.31519
z_target = 0.295
xnow = -3.912023

k_j = jnp.asarray(k_tab)
pk_j = jnp.asarray(PSL)
pk_now_j = jnp.asarray(PSL_nw)
fk_j = jnp.asarray(fk_tab)

def make_mu_grid(nmu=12):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)

mu, wmu = make_mu_grid(nmu=12)
jac = jnp.asarray(1.0)
kap = k_j[:, None] * jnp.ones_like(mu)[None, :]
muap = jnp.ones_like(k_j)[:, None] * mu[None, :]

bias = dict(
    b1=1.6215433618068629, b2=0.23677842354547154, bs2=-0.35516763531820733,
    b3nl=0.06314091294545908, alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0,
    alpha0shot=1.0, alpha2shot=0.0, PshotP=5000.0,
)
pars = pack_fkpt_bias(bias)

t0 = time.time()
static_ctx = build_jax_static_ctx(
    k_j, kmin=float(k_j.min()), kmax=float(k_j.max()),
    Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
)
print(f"static_ctx built in {time.time() - t0:.2f}s")

t0 = time.time()
poles, state = binning_jax_poles(
    k=k_j, pk=pk_j, pk_now=pk_now_j,
    jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
    ells=(0, 2, 4), bias_scheme="folps", IR_resummation=True,
    damping=None, A_full=False, use_TNS_model=False,
    return_kernel_constants=True,
    static_ctx=static_ctx,
    z=z_target, Om=Om, beyond_eds=True, fkpt_approximation=False,
    kmin=float(k_j.min()), kmax=float(k_j.max()), xnow=xnow, f0_kmax=float(k_j.min()),
    fk=fk_j, f0=float(f0_tab),
    model="HDKI", mg_variant="BZ_Mass",
    mu_kinf_BZmass=1.2, lambda_a_BZmass=100.0, lambda_dS_BZmass=100.0,
)
poles.block_until_ready()
print(f"binning_jax_poles (fkpt_approximation=False): {time.time() - t0:.2f}s")
print("kernel_constants (A, Ap, CFD3, CFD3p):", state.kernel_constants)

P0_ours = np.asarray(poles[0])
P2_ours = np.asarray(poles[1])
P4_ours = np.asarray(poles[2])

# ----------------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------------
P0_i = np.interp(k_ref, k_tab, P0_ours)
P2_i = np.interp(k_ref, k_tab, P2_ours)
P4_i = np.interp(k_ref, k_tab, P4_ours)

mask = (k_ref >= 0.01) & (k_ref <= 0.5)
print("\n=== fkptjax vs sMGPT, BZ_Mass, fkpt_approximation=False, k in [0.01, 0.5] ===")
for name, ours, ref_ in [("P0", P0_i, P0_ref), ("P2", P2_i, P2_ref), ("P4", P4_i, P4_ref)]:
    pct = 100.0 * (ours - ref_) / np.where(np.abs(ref_) > 1.0, ref_, 1.0)
    print(f"{name}: max |ours - sMGPT| / sMGPT = {np.max(np.abs(pct[mask])):.2f}%   "
          f"mean = {np.mean(np.abs(pct[mask])):.2f}%")

np.savez(os.path.join(DATA_DIR, "bzmass_full_comparison.npz"),
          k_ref=k_ref, P0_ref=P0_ref, P2_ref=P2_ref, P4_ref=P4_ref,
          k_tab=k_tab, P0_ours=P0_ours, P2_ours=P2_ours, P4_ours=P4_ours)
print("\nSaved bzmass_full_comparison.npz")
