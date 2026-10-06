import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax import mg_jax as bj
from fkptjax.jax_ode import DP_jax

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

xnow = -3.912023
xstop = float(jnp.log(1.0 / 1.3))
P1 = bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.MU_OMDE, mu0=-0.5)
P2 = bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.MU_OMDE, mu0=-0.55)

# Representative extrapolated-k-grid size (a few hundred points, matching
# extrapolate_pklin's typical output for a ~256-point input P(k)).
k_ext = jnp.asarray(np.geomspace(1e-5, 500.0, 400))

print("\n--- un-jitted (as currently called in kfuncs_to_tables.py) ---")
t0 = time.time()
out1 = DP_jax(k_ext, P1, xnow, xstop)
jax.block_until_ready(out1)
print(f"call 1: {time.time()-t0:.3f}s")
t0 = time.time()
out2 = DP_jax(k_ext, P2, xnow, xstop)
jax.block_until_ready(out2)
t_unjit = time.time() - t0
print(f"call 2 (different mu0): {t_unjit:.3f}s")

print("\n--- jax.jit-wrapped ---")
DP_jax_jit = jax.jit(DP_jax)
t0 = time.time()
out1j = DP_jax_jit(k_ext, P1, xnow, xstop)
jax.block_until_ready(out1j)
print(f"call 1 (compiles): {time.time()-t0:.3f}s")
t0 = time.time()
out2j = DP_jax_jit(k_ext, P2, xnow, xstop)
jax.block_until_ready(out2j)
t_jit = time.time() - t0
print(f"call 2 (different mu0, should reuse compiled program): {t_jit:.3f}s")

print(f"\nspeedup: {t_unjit/t_jit:.1f}x")
rel = float(jnp.max(jnp.abs(out2j - out2) / jnp.maximum(1.0, jnp.abs(out2))))
print(f"correctness: max relerr = {rel:.3e}")
