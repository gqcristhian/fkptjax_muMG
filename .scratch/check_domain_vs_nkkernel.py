import sys, os
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax.kfuncs_to_tables import build_jax_static_ctx

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())

# desilike's real fkpt settings (full_shape.py):
#   kmin = min(1e-3, min(self.k)); kmax = max(1.0, max(self.k))
#   Nk_kernel = min(len(self.k), 120); nquadSteps=300, NQ=10, NR=10
# Data k-range 0.02-0.2 (or up to 0.3) -> kmin=1e-3, kmax=1.0 regardless.
k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 256))

for Nk_kernel in (16, 20, 28, 40, 56, 80, 120):
    static_ctx = build_jax_static_ctx(
        k_lin, kmin=1.0e-3, kmax=1.0, Nk_kernel=Nk_kernel,
        nquadSteps=300, NQ=10, NR=10,
        rbao=104.0, pmax_bao=0.4, Np_bao=100,
    )
    calculator = static_ctx["calculator"]
    k_ext_q = calculator.logk_grid_jax
    r_q = calculator.r_jax
    x_q = calculator.x_jax
    y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
    kminus_Q = k_ext_q * y_q
    kf_full = k_ext_q * jnp.ones_like(y_q)
    y_ratio = np.asarray((kminus_Q / kf_full).reshape(-1))

    r_r = calculator.r_r_jax
    x_r = calculator.x_r_jax
    y_r = jnp.sqrt(1.0 + r_r * r_r - 2.0 * r_r * x_r)
    kminus_R = k_ext_q * y_r
    kf_full_R = k_ext_q * jnp.ones_like(y_r)
    y_ratio_R = np.asarray((kminus_R / kf_full_R).reshape(-1))

    print(f"Nk_kernel={Nk_kernel:4d}:  Q-grid y=kminus/kf min={y_ratio.min():.4f}  "
          f"R-grid y min={y_ratio_R.min():.4f}   (danger zone is y->0)")
