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

# Same coarse resolution + Hu-Sawicki model as compute_mg_multipoles.ipynb's
# beyond-EdS comparison section.
k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 256))
static_ctx = build_jax_static_ctx(
    k_lin, kmin=1.0e-3, kmax=1.0, Nk_kernel=20, nquadSteps=48, NQ=6, NR=6,
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
)
calculator = static_ctx["calculator"]

fR0_test = -1.0e-5
P = bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.HS, fR0_HS=fR0_test, beta2=1.0 / 6.0, n_HS=1)
f0_test = 0.684640  # from the notebook's own printed HS kernel_constants, close enough for this check

k_ext_q = calculator.logk_grid_jax
r_r = calculator.r_r_jax
x_r = calculator.x_r_jax
q_loop_R = r_r * k_ext_q
k_ext_full = k_ext_q * jnp.ones_like(q_loop_R)
k_ext_full, q_loop_R, x_r_full = jnp.broadcast_arrays(k_ext_full, q_loop_R, x_r)
print(f"\nR-loop grid shape: {k_ext_full.shape}  npts={k_ext_full.size}")

t0 = time.time()
A_old, Ap_old = mgk.D2_fused_grid(-x_r_full, k_ext_full, q_loop_R, P, xnow, xstop)
CFD3_old, CFD3p_old = mgk.D3_fused_grid(x_r_full, k_ext_full, q_loop_R, P, xnow, xstop, f0_test)
jax.block_until_ready((A_old, Ap_old, CFD3_old, CFD3p_old))
t_old = time.time() - t0
print(f"separate D2_fused_grid+D3_fused_grid: {t_old:.3f}s")

t0 = time.time()
A_new, Ap_new, CFD3_new, CFD3p_new = mgk.D2_D3_fused_grid(x_r_full, k_ext_full, q_loop_R, P, xnow, xstop, f0_test)
jax.block_until_ready((A_new, Ap_new, CFD3_new, CFD3p_new))
t_new = time.time() - t0
print(f"combined D2_D3_fused_grid: {t_new:.3f}s")

for name, old, new in [("A_fused", A_old, A_new), ("Ap_fused", Ap_old, Ap_new),
                       ("CFD3", CFD3_old, CFD3_new), ("CFD3p", CFD3p_old, CFD3p_new)]:
    rel = jnp.abs(new - old) / jnp.maximum(1.0, jnp.abs(old))
    print(f"  {name}: max relerr = {float(jnp.max(rel)):.3e}  "
          f"(at index {int(jnp.argmax(rel))} of {old.size})")

# Also print the raw values at the worst point for both, for a sanity look.
worst_flat_idx = int(jnp.argmax(jnp.abs(CFD3_new.reshape(-1) - CFD3_old.reshape(-1))
                                / jnp.maximum(1.0, jnp.abs(CFD3_old.reshape(-1)))))
print(f"\nworst CFD3 point: old={float(CFD3_old.reshape(-1)[worst_flat_idx]):.8f}  "
      f"new={float(CFD3_new.reshape(-1)[worst_flat_idx]):.8f}")
