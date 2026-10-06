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

# Confirm domain safety (same check as the earlier step-count diagnostic).
y_ratio = np.asarray((kminus_Q / k_ext_q).reshape(-1))
print(f"\nQ-grid y=kminus/kf range: min={y_ratio.min():.4f}  max={y_ratio.max():.2f} "
      f"(collinear danger zone is y->0; confirming still far from it)")


def run(model_label, P, mu0_list, solver, n_steps):
    kwargs = dict(solver=solver)
    if solver == "rk4":
        kwargs["n_steps"] = n_steps
    outs = []
    times = []
    for mu0 in mu0_list:
        t0 = time.time()
        out = mgk.I1udd1_and_P13_grid(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R,
                                       P(mu0), xnow, xstop, f0, **kwargs)
        jax.block_until_ready(out)
        dt = time.time() - t0
        outs.append(out)
        times.append(dt)
    return outs, times


def make_P_muomde(mu0):
    return bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.MU_OMDE, mu0=mu0)


def make_P_hs(fR0):
    return bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.HS, fR0_HS=fR0, beta2=1.0 / 6.0, n_HS=1)


MODELS = [
    ("HDKI/mu_OmDE (scale-independent)", make_P_muomde, [-0.5, -0.55, -0.6]),
    ("Hu-Sawicki f(R) (scale-dependent)", make_P_hs, [-1.0e-5, -1.2e-5, -0.8e-5]),
]

for label, Pfunc, param_list in MODELS:
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    print("--- adaptive (ground truth) ---")
    outs_ref, times_ref = run(label, Pfunc, param_list, "adaptive", None)
    print(f"  compile+call1={times_ref[0]:.3f}s  call2={times_ref[1]:.3f}s  call3={times_ref[2]:.3f}s")

    for n_steps in (24, 32, 49, 64, 96, 128):
        outs_rk4, times_rk4 = run(label, Pfunc, param_list, "rk4", n_steps)
        print(f"\n--- rk4, n_steps={n_steps} ---")
        print(f"  compile+call1={times_rk4[0]:.3f}s  call2={times_rk4[1]:.3f}s  call3={times_rk4[2]:.3f}s"
              f"  (speedup vs adaptive call3: {times_ref[2]/times_rk4[2]:.2f}x)")
        max_rel = 0.0
        for a, b in zip(outs_ref[-1], outs_rk4[-1]):
            rel = float(jnp.max(jnp.abs(b - a) / jnp.maximum(1.0, jnp.abs(a))))
            max_rel = max(max_rel, rel)
        print(f"  max relerr (all 16 outputs, last call) vs adaptive: {max_rel:.3e}")
