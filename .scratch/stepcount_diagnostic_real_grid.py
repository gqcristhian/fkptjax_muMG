import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

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
t0 = time.time()
static_ctx = build_jax_static_ctx(
    k_lin, kmin=1.0e-3, kmax=1.0, rbao=104.0, pmax_bao=0.4, Np_bao=100, **RES,
)
print(f"static_ctx build: {time.time()-t0:.2f}s")
calculator = static_ctx["calculator"]

# EXACT same construction as kfuncs_to_tables.py's Kfuncs_to_tables_jax
# (Q-ordering block).
k_ext_q = calculator.logk_grid_jax
r_q = calculator.r_jax
x_q = calculator.x_jax
y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
q_loop_Q = r_q * k_ext_q
kminus_Q = k_ext_q * y_q
kf_grid = k_ext_q * jnp.ones_like(y_q)
kf_grid, q_loop_Q, kminus_Q = jnp.broadcast_arrays(kf_grid, q_loop_Q, kminus_Q)

print(f"\nREAL Q-loop grid shape: {kf_grid.shape}  npts={kf_grid.size}")


def solve_one_with_stats(kf, k1, k2):
    e0 = jnp.exp(xnow)
    D2plusi = 3.0 * jnp.exp(2.0 * xnow) / 7.0
    dD2plusi = 6.0 * jnp.exp(2.0 * xnow) / 7.0
    y0 = jnp.stack([e0, e0, e0, e0, D2plusi, dD2plusi, D2plusi, dD2plusi])
    term = diffrax.ODETerm(lambda t, y, args: mgk.secondOrderAB(t, y, kf, k1, k2, P))
    solver = diffrax.Tsit5()
    ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-11)
    sol = diffrax.diffeqsolve(
        term, solver, t0=xnow, t1=xstop, dt0=0.01, y0=y0,
        stepsize_controller=ctrl, saveat=diffrax.SaveAt(t1=True),
        max_steps=100000,
    )
    return sol.stats["num_steps"]


solve_batch = jax.vmap(solve_one_with_stats)

kf_flat = kf_grid.reshape(-1)
k1_flat = q_loop_Q.reshape(-1)
k2_flat = kminus_Q.reshape(-1)
y_ratio = k2_flat / kf_flat
npts = kf_flat.size

t0 = time.time()
step_counts = solve_batch(kf_flat, k1_flat, k2_flat)
jax.block_until_ready(step_counts)
warm_t = time.time() - t0
t0 = time.time()
step_counts = solve_batch(kf_flat, k1_flat, k2_flat)
jax.block_until_ready(step_counts)
solve_t = time.time() - t0
print(f"warmup={warm_t:.3f}s  post-compile solve time={solve_t:.3f}s")

sc = np.asarray(step_counts)
yr = np.asarray(y_ratio)
print(f"\nstep-count distribution across all {npts} points:")
print(f"  min={sc.min()}  p10={np.percentile(sc,10):.0f}  median={np.median(sc):.0f}  "
      f"p90={np.percentile(sc,90):.0f}  p99={np.percentile(sc,99):.0f}  max={sc.max()}")
print(f"  fraction at/near max_steps (>=99000): {np.mean(sc >= 99000)*100:.3f}%")
print(f"  fraction with step_count > 2x median: {np.mean(sc > 2*np.median(sc))*100:.3f}%")
print(f"  y=kminus/kf range in this real grid: min={yr.min():.6e}  max={yr.max():.4f}")

bins = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf]
print("\nmean step count by y=kminus/kf bin (smaller y = closer to collinear):")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (yr >= lo) & (yr < hi)
    n = mask.sum()
    if n == 0:
        continue
    print(f"  y in [{lo:8.4f},{hi:8.4f}):  n={n:7d}  mean_steps={sc[mask].mean():8.1f}  "
          f"max_steps={sc[mask].max():6d}  frac_of_total_stepwork~={sc[mask].sum()/sc.sum()*100:5.1f}%")

worst_idx = np.argsort(sc)[-10:][::-1]
print("\nworst 10 points (highest step count):")
for i in worst_idx:
    print(f"  kf={kf_flat[i]:.6f}  q={k1_flat[i]:.6f}  kminus={k2_flat[i]:.6f}  "
          f"y=kminus/kf={yr[i]:.6e}  steps={sc[i]}")
