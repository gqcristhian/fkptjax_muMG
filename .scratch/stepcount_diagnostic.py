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

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

Om = 0.315
z = 0.3
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + z)))

# Scale-independent model (mu_OmDE) -- deliberately, to isolate the KINEMATIC
# (SPT-source) near-collinear divergence from any MG-model-specific
# k-dependence in mu(eta,k). The collinear-singularity source terms
# (S2FL/SD2/S3FLplus etc.) divide by momentum combinations that shrink near
# q~k_ext regardless of the MG model, so this should still show the effect
# if it's real.
P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.MU_OMDE, mu0=-0.5)


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

# Same production Q-loop grid shape as the real pipeline (Nk_kernel=120,
# NQ=10, nquadSteps=300 -> (10, 299, 120) points).
Nk, NQ, Nq = 120, 10, 300
k_ext_arr = jnp.geomspace(1e-3, 1.0, Nk)
r_arr = jnp.geomspace(1e-3, 4.0, NQ)
x_arr = jnp.linspace(-0.999, 0.999, Nq - 1)

k_ext_b = k_ext_arr[None, None, :]
r_b = r_arr[:, None, None]
x_b = x_arr[None, :, None]
y_b = jnp.sqrt(1.0 + r_b * r_b - 2.0 * r_b * x_b)
q_loop = r_b * k_ext_b
kminus = k_ext_b * y_b
kf_full = k_ext_b * jnp.ones_like(y_b)
kf_grid, k1_grid, k2_grid = jnp.broadcast_arrays(kf_full, q_loop, kminus)

kf_flat = kf_grid.reshape(-1)
k1_flat = k1_grid.reshape(-1)
k2_flat = k2_grid.reshape(-1)
y_ratio = (k2_flat / kf_flat)  # "closeness to collinear" proxy: kminus/kf -> 0 is dangerous
npts = kf_flat.size
print(f"\ngrid: {kf_grid.shape}  npts={npts}")

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

# Bin by y=kminus/kf (proxy for collinearity) to see if step count correlates.
bins = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, np.inf]
print("\nmean step count by y=kminus/kf bin (smaller y = closer to collinear):")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (yr >= lo) & (yr < hi)
    n = mask.sum()
    if n == 0:
        print(f"  y in [{lo},{hi}):  (no points)")
        continue
    print(f"  y in [{lo:5.2f},{hi:5.2f}):  n={n:7d}  mean_steps={sc[mask].mean():8.1f}  "
          f"max_steps={sc[mask].max():6d}  frac_of_total_time~={sc[mask].sum()/sc.sum()*100:5.1f}%")

# Also identify the actual worst offenders directly.
worst_idx = np.argsort(sc)[-10:][::-1]
print("\nworst 10 points (highest step count):")
for i in worst_idx:
    print(f"  kf={kf_flat[i]:.6f}  k1(q)={k1_flat[i]:.6f}  k2(kminus)={k2_flat[i]:.6f}  "
          f"y=kminus/kf={yr[i]:.5f}  steps={sc[i]}")
