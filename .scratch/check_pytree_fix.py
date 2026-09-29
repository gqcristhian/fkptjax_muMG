import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax import mg_jax as bj
from fkptjax.jax_ode import kernel_constants_jax

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

Om = 0.315
xnow = -3.912023
xstop = float(jnp.log(1.0 / 1.3))
P_muomde = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.MU_OMDE, mu0=-0.5)
P_hs = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.HS, fR0_HS=-1.0e-5, beta2=1.0 / 6.0, n_HS=1)
f0 = jnp.asarray(0.68)

# Reference (un-jitted math, computed via the OLD path logic manually is not
# available anymore since we patched the function itself -- instead, verify
# self-consistency: same P/f0/xnow/xstop must give the same answer every
# time, and different mu0 values (same "kind") must NOT trigger a recompile.

print("\n--- mu_OmDE model, first call (compiles) ---")
t0 = time.time()
out1 = kernel_constants_jax(f0, P_muomde, xnow, xstop)
jax.block_until_ready(out1)
print(f"warmup={time.time()-t0:.3f}s  ->  {out1}")

print("\n--- mu_OmDE model, SAME params, second call (should be cached/fast) ---")
t0 = time.time()
out2 = kernel_constants_jax(f0, P_muomde, xnow, xstop)
jax.block_until_ready(out2)
t_same = time.time() - t0
print(f"post-compile={t_same:.3f}s")
rel = [float(jnp.abs(a - b) / jnp.maximum(1.0, jnp.abs(a))) for a, b in zip(out1, out2)]
print(f"relerr vs first call: {max(rel):.3e}")

print("\n--- mu_OmDE model, DIFFERENT mu0 (same kind) -- should reuse compiled program, NOT recompile ---")
P_muomde2 = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.MU_OMDE, mu0=-0.3)
t0 = time.time()
out3 = kernel_constants_jax(f0, P_muomde2, xnow, xstop)
jax.block_until_ready(out3)
t_diff_mu0 = time.time() - t0
print(f"time={t_diff_mu0:.3f}s (should be ~fast, like the cached case, NOT a fresh ~1.4s compile)")
print(f"values differ from mu0=-0.5 case (expected, different physics): {out3}")

print("\n--- Hu-Sawicki f(R) model (DIFFERENT kind) -- SHOULD trigger a fresh compile ---")
t0 = time.time()
out4 = kernel_constants_jax(f0, P_hs, xnow, xstop)
jax.block_until_ready(out4)
t_diff_kind = time.time() - t0
print(f"time={t_diff_kind:.3f}s (expected to be slower -- new model kind, new compiled program)")

print("\n--- Hu-Sawicki f(R), same params again (should now be cached) ---")
t0 = time.time()
out5 = kernel_constants_jax(f0, P_hs, xnow, xstop)
jax.block_until_ready(out5)
t_hs_cached = time.time() - t0
print(f"time={t_hs_cached:.3f}s")
