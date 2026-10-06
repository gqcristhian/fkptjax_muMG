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

print("JAX version:", jax.__version__, " devices:", jax.devices(), " backend:", jax.default_backend())

xnow = -3.912023
xstop = float(jnp.log(1.0 / 1.3))
f0 = jnp.asarray(0.68)

k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 256))
RES = dict(Nk_kernel=120, nquadSteps=300, NQ=10, NR=10)
t0 = time.time()
static_ctx = build_jax_static_ctx(
    k_lin, kmin=1.0e-3, kmax=1.0, rbao=104.0, pmax_bao=0.4, Np_bao=100, **RES,
)
print(f"static_ctx build: {time.time()-t0:.3f}s")
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

print("Q-grid shape:", k_ext_q.shape, "quadrature-flattened size:", int(np.prod(kminus_Q.shape)))


def make_P_muomde(mu0):
    return bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.MU_OMDE, mu0=mu0)


def make_P_hs(fR0):
    return bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.HS, fR0_HS=fR0, beta2=1.0 / 6.0, n_HS=1)


for label, Pfunc, params in [
    ("HDKI/mu_OmDE (scale-independent)", make_P_muomde, [-0.5, -0.55, -0.6]),
    ("Hu-Sawicki f(R) (scale-dependent)", make_P_hs, [-1.0e-5, -1.2e-5, -0.8e-5]),
]:
    print(f"\n=== {label}: solver='rk4', n_steps=64 (production call site) ===")
    for i, p in enumerate(params):
        t0 = time.time()
        out = mgk.I1udd1_and_P13_grid(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R,
                                       Pfunc(p), xnow, xstop, f0, solver='rk4', n_steps=64)
        jax.block_until_ready(out)
        dt = time.time() - t0
        tag = "compile+call1" if i == 0 else f"call{i+1}"
        print(f"  {tag}: {dt:.3f}s")
