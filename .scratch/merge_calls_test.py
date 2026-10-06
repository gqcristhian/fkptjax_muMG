import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk
from fkptjax.kfuncs_to_tables import build_jax_static_ctx

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

Om = 0.315
z = 0.3
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + z)))
P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.MU_OMDE, mu0=-0.5)

k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 256))
RES = dict(Nk_kernel=120, nquadSteps=300, NQ=10, NR=10)
static_ctx = build_jax_static_ctx(
    k_lin, kmin=1.0e-3, kmax=1.0, rbao=104.0, pmax_bao=0.4, Np_bao=100, **RES,
)
calculator = static_ctx["calculator"]

k_ext_q = calculator.logk_grid_jax
r_q = calculator.r_jax
x_q = calculator.x_jax
y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
q_loop_Q = r_q * k_ext_q
kminus_Q = k_ext_q * y_q
kf_full = k_ext_q * jnp.ones_like(y_q)
kf_full, q_loop_Q, kminus_Q = jnp.broadcast_arrays(kf_full, q_loop_Q, kminus_Q)
shape = kf_full.shape
npts = kf_full.size
print(f"\nQ-loop grid shape: {shape}  npts={npts}  (x3 orderings = {3*npts})")

# Q, N, RQ orderings -- the exact three A_B_grid calls kfuncs_to_tables.py
# makes for the I1udd1-family (beyond_eds=True, fkpt_approximation=False).
KF_Q, K1_Q, K2_Q = kf_full, q_loop_Q, kminus_Q
KF_N, K1_N, K2_N = q_loop_Q, kf_full, kminus_Q
KF_R, K1_R, K2_R = kminus_Q, kf_full, q_loop_Q


def run_separate():
    A_Q, B_Q, _, _ = mgk.A_B_grid(KF_Q, K1_Q, K2_Q, P, xnow, xstop)
    A_N, B_N, _, _ = mgk.A_B_grid(KF_N, K1_N, K2_N, P, xnow, xstop)
    A_R, B_R, _, _ = mgk.A_B_grid(KF_R, K1_R, K2_R, P, xnow, xstop)
    out = (A_Q, B_Q, A_N, B_N, A_R, B_R)
    jax.block_until_ready(out)
    return out


def run_merged():
    kf_cat = jnp.concatenate([KF_Q.reshape(-1), KF_N.reshape(-1), KF_R.reshape(-1)])
    k1_cat = jnp.concatenate([K1_Q.reshape(-1), K1_N.reshape(-1), K1_R.reshape(-1)])
    k2_cat = jnp.concatenate([K2_Q.reshape(-1), K2_N.reshape(-1), K2_R.reshape(-1)])
    A_cat, B_cat, _, _ = mgk.A_B_grid(kf_cat, k1_cat, k2_cat, P, xnow, xstop)
    A_Q = A_cat[:npts].reshape(shape); A_N = A_cat[npts:2*npts].reshape(shape); A_R = A_cat[2*npts:].reshape(shape)
    B_Q = B_cat[:npts].reshape(shape); B_N = B_cat[npts:2*npts].reshape(shape); B_R = B_cat[2*npts:].reshape(shape)
    out = (A_Q, B_Q, A_N, B_N, A_R, B_R)
    jax.block_until_ready(out)
    return out


print("\n=== 3 separate A_B_grid calls (current approach) ===")
t0 = time.time()
out_sep = run_separate()
warm_sep = time.time() - t0
t0 = time.time()
out_sep = run_separate()
iter_sep = time.time() - t0
print(f"warmup={warm_sep:.3f}s  post-compile={iter_sep:.3f}s")

print("\n=== 1 merged A_B_grid call (concatenated orderings) ===")
t0 = time.time()
out_merged = run_merged()
warm_merged = time.time() - t0
t0 = time.time()
out_merged = run_merged()
iter_merged = time.time() - t0
print(f"warmup={warm_merged:.3f}s  post-compile={iter_merged:.3f}s"
      f"  (speedup vs separate: {iter_sep/iter_merged:.2f}x)")

print("\nCorrectness check (merged vs separate, should be ~identical):")
for name, a, b in zip(("A_Q", "B_Q", "A_N", "B_N", "A_R", "B_R"), out_sep, out_merged):
    rel = float(jnp.max(jnp.abs(a - b) / jnp.maximum(1.0, jnp.abs(a))))
    print(f"  {name}: max relerr = {rel:.3e}")
