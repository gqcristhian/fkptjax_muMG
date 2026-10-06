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

xnow = -3.912023
z_pk_cmp = 0.3
xstop = float(jnp.log(1.0 / (1.0 + z_pk_cmp)))

k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 256))
static_ctx = build_jax_static_ctx(
    k_lin, kmin=1.0e-3, kmax=1.0, Nk_kernel=20, nquadSteps=48, NQ=6, NR=6,
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
)
calculator = static_ctx["calculator"]

fR0_test = -1.0e-5
P = bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.HS, fR0_HS=fR0_test, beta2=1.0 / 6.0, n_HS=1)

k_ext_q = calculator.logk_grid_jax
r_q = calculator.r_jax
x_q = calculator.x_jax
y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
q_loop_Q = r_q * k_ext_q
kminus_Q = k_ext_q * y_q
kf_full, q_full, kminus_full = jnp.broadcast_arrays(k_ext_q * jnp.ones_like(y_q), q_loop_Q, kminus_Q)
shape = kf_full.shape
print(f"\nQ-loop grid shape: {shape}  npts={kf_full.size}")

print("\n--- Separate calls (Q, N, RQ orderings) ---")
A_Q_sep, B_Q_sep, Ap_Q_sep, Bp_Q_sep = mgk.A_B_grid(kf_full, q_full, kminus_full, P, xnow, xstop)
A_N_sep, B_N_sep, Ap_N_sep, Bp_N_sep = mgk.A_B_grid(q_full, kf_full, kminus_full, P, xnow, xstop)
A_R_sep, B_R_sep, Ap_R_sep, Bp_R_sep = mgk.A_B_grid(kminus_full, kf_full, q_full, P, xnow, xstop)

print("--- Merged call (concatenated) ---")
flat = lambda a: a.reshape(-1)
npts = kf_full.size
kf_cat = jnp.concatenate([flat(kf_full), flat(q_full), flat(kminus_full)])
k1_cat = jnp.concatenate([flat(q_full), flat(kf_full), flat(kf_full)])
k2_cat = jnp.concatenate([flat(kminus_full), flat(kminus_full), flat(q_full)])
A_cat, B_cat, Ap_cat, Bp_cat = mgk.A_B_grid(kf_cat, k1_cat, k2_cat, P, xnow, xstop)
split = lambda a, i: a[i * npts:(i + 1) * npts].reshape(shape)
A_Q_m, B_Q_m, Ap_Q_m, Bp_Q_m = split(A_cat, 0), split(B_cat, 0), split(Ap_cat, 0), split(Bp_cat, 0)
A_N_m, B_N_m, Ap_N_m, Bp_N_m = split(A_cat, 1), split(B_cat, 1), split(Ap_cat, 1), split(Bp_cat, 1)
A_R_m, B_R_m, Ap_R_m, Bp_R_m = split(A_cat, 2), split(B_cat, 2), split(Ap_cat, 2), split(Bp_cat, 2)

print("\nCorrectness check (merged vs separate, Hu-Sawicki f(R), coarse grid):")
for name, sep, m in [
    ("A_Q", A_Q_sep, A_Q_m), ("B_Q", B_Q_sep, B_Q_m),
    ("A_N", A_N_sep, A_N_m), ("B_N", B_N_sep, B_N_m),
    ("A_R", A_R_sep, A_R_m), ("B_R", B_R_sep, B_R_m),
]:
    rel = jnp.abs(m - sep) / jnp.maximum(1.0, jnp.abs(sep))
    print(f"  {name}: max relerr = {float(jnp.max(rel)):.3e}")
