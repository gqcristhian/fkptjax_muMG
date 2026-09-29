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

xnow = -3.912023
xstop = float(jnp.log(1.0 / 1.3))
f0 = jnp.asarray(0.68)

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

r_r = calculator.r_r_jax
x_r = calculator.x_r_jax
q_loop_R = r_r * k_ext_q

print(f"\nQ-grid npts={k_ext_q.size * y_q.size // k_ext_q.size}  (production resolution)")


def make_P(mu0):
    return bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.MU_OMDE, mu0=mu0)


# --- un-jitted (current production behavior) ---
print("\n--- un-jitted (current) ---")
P1 = make_P(-0.5)
t0 = time.time()
out1 = mgk.I1udd1_and_P13_grid(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P1, xnow, xstop, f0)
jax.block_until_ready(out1)
warm1 = time.time() - t0
print(f"call 1 (mu0=-0.5): {warm1:.3f}s")

P2 = make_P(-0.55)
t0 = time.time()
out2 = mgk.I1udd1_and_P13_grid(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P2, xnow, xstop, f0)
jax.block_until_ready(out2)
warm2 = time.time() - t0
print(f"call 2 (mu0=-0.55, DIFFERENT params, same 'kind'): {warm2:.3f}s")

P3 = make_P(-0.6)
t0 = time.time()
out3 = mgk.I1udd1_and_P13_grid(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P3, xnow, xstop, f0)
jax.block_until_ready(out3)
warm3 = time.time() - t0
print(f"call 3 (mu0=-0.6): {warm3:.3f}s")

# --- jit-wrapped ---
print("\n--- jax.jit-wrapped ---")
I1udd1_and_P13_grid_jit = jax.jit(mgk.I1udd1_and_P13_grid)

t0 = time.time()
out1j = I1udd1_and_P13_grid_jit(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P1, xnow, xstop, f0)
jax.block_until_ready(out1j)
warm1j = time.time() - t0
print(f"call 1 (mu0=-0.5, compiles): {warm1j:.3f}s")

t0 = time.time()
out2j = I1udd1_and_P13_grid_jit(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P2, xnow, xstop, f0)
jax.block_until_ready(out2j)
warm2j = time.time() - t0
print(f"call 2 (mu0=-0.55, should reuse compiled program): {warm2j:.3f}s")

t0 = time.time()
out3j = I1udd1_and_P13_grid_jit(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P3, xnow, xstop, f0)
jax.block_until_ready(out3j)
warm3j = time.time() - t0
print(f"call 3 (mu0=-0.6, should reuse compiled program): {warm3j:.3f}s")

print(f"\nspeedup (call 3, un-jitted vs jitted): {warm3/warm3j:.2f}x")

print("\nCorrectness (jitted vs un-jitted, mu0=-0.6):")
for name, a, b in zip(
    ("A_Q", "B_Q", "ApOverf0_Q", "BpOverf0_Q", "A_N", "B_N", "ApOverf0_N", "BpOverf0_N",
     "A_RQ", "B_RQ", "ApOverf0_RQ", "BpOverf0_RQ", "A_fused", "ApOverf0_fused", "CFD3", "CFD3p"),
    out3, out3j,
):
    rel = float(jnp.max(jnp.abs(b - a) / jnp.maximum(1.0, jnp.abs(a))))
    print(f"  {name:16s} max relerr = {rel:.3e}")
