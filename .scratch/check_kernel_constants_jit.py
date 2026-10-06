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
P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.MU_OMDE, mu0=-0.5)
f0 = jnp.asarray(0.68)

print("\n--- un-jitted (as currently called in kfuncs_to_tables.py) ---")
t0 = time.time()
out = kernel_constants_jax(f0, P, xnow, xstop)
jax.block_until_ready(out)
warm = time.time() - t0
t0 = time.time()
out = kernel_constants_jax(f0, P, xnow, xstop)
jax.block_until_ready(out)
iter_t = time.time() - t0
print(f"warmup={warm:.3f}s  post-compile={iter_t:.3f}s")

print("\n--- explicitly jax.jit-wrapped (P captured via closure, since MGConstants isn't a registered pytree) ---")
kernel_constants_jax_jit = jax.jit(lambda f0_, xnow_, xstop_: kernel_constants_jax(f0_, P, xnow_, xstop_))
t0 = time.time()
out2 = kernel_constants_jax_jit(f0, xnow, xstop)
jax.block_until_ready(out2)
warm2 = time.time() - t0
t0 = time.time()
out2 = kernel_constants_jax_jit(f0, xnow, xstop)
jax.block_until_ready(out2)
iter_t2 = time.time() - t0
print(f"warmup={warm2:.3f}s  post-compile={iter_t2:.3f}s  (speedup vs un-jitted: {iter_t/iter_t2:.1f}x)")

rel = [float(jnp.abs(a - b) / jnp.maximum(1.0, jnp.abs(a))) for a, b in zip(out, out2)]
print(f"\ncorrectness (jit vs un-jitted): max relerr = {max(rel):.3e}")
