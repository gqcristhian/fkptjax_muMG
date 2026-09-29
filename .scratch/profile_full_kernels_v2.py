import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.signal import savgol_filter

from fkptjax.kfuncs_to_tables import build_jax_static_ctx
from fkptjax.pipelines import binning_jax_poles
from fkptjax.rsd import pack_fkpt_bias
from fkptjax import MG_kernels as mgk

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

TIMINGS = {}


def timed(name, fn):
    def wrapped(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        dt = time.time() - t0
        TIMINGS.setdefault(name, []).append(dt)
        return out
    return wrapped


# Patch the CURRENT production entry points: kfuncs_to_tables.py now calls
# A_B_grid ONCE (merged Q/N/R) and D2_D3_fused_grid ONCE (combined D2+D3),
# not the old 3-separate-calls + D2_fused_grid + D3_fused_grid pattern.
mgk.A_B_grid = timed("A_B_grid (merged Q/N/R)", mgk.A_B_grid)
mgk.D2_D3_fused_grid = timed("D2_D3_fused_grid (combined)", mgk.D2_D3_fused_grid)


def toy_pk(k):
    k = np.asarray(k, dtype=float)
    return 2.0e4 * k / (1.0 + (k / 0.05) ** 2) ** 2


def make_mu_grid(nmu=8):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)


k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 256))
pk_lin_np = toy_pk(np.asarray(k_lin))
pk_now_np = np.exp(savgol_filter(np.log(pk_lin_np), window_length=41, polyorder=3))
pk_lin = jnp.asarray(pk_lin_np)
pk_now_lin = jnp.asarray(pk_now_np)

k_eval = jnp.asarray(np.linspace(0.02, 0.20, 40))
mu, wmu = make_mu_grid(nmu=8)
jac = jnp.asarray(1.0)
kap = k_eval[:, None] * jnp.ones_like(mu)[None, :]
muap = jnp.ones_like(k_eval)[:, None] * mu[None, :]

bias = dict(b1=2.0, b2=0.0, bs2=0.0, b3nl=0.0, alpha0=0.0, alpha2=0.0, alpha4=0.0,
            ctilde=0.0, alpha0shot=0.0, alpha2shot=0.0)
pars = pack_fkpt_bias(bias, nd=1.0e-4)
z_pk_cmp = 0.3

RES = dict(Nk_kernel=120, nquadSteps=300, NQ=10, NR=10)
t0 = time.time()
static_ctx = build_jax_static_ctx(
    k_lin, kmin=1.0e-3, kmax=1.0, rbao=104.0, pmax_bao=0.4, Np_bao=100, **RES,
)
print(f"static_ctx build: {time.time()-t0:.2f}s  (grid: {RES})")


def compute_poles(mu0):
    poles, _ = binning_jax_poles(
        k=k_lin, pk=pk_lin, pk_now=pk_now_lin,
        jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
        ells=(0, 2), bias_scheme="folps", IR_resummation=True,
        damping=None, A_full=False, use_TNS_model=False,
        return_kernel_constants=True,
        static_ctx=static_ctx,
        z=z_pk_cmp, Om=0.315,
        beyond_eds=True, fkpt_approximation=False,
        kmin=1.0e-3, kmax=1.0, xnow=-3.912023, f0_kmax=1.0e-3,
        model="HDKI", mg_variant="mu_OmDE", mu0=mu0,
    )
    jax.block_until_ready(poles)
    return poles


print("\nWarmup (compile) call...")
t0 = time.time()
compute_poles(mu0=-0.5)
warm_t = time.time() - t0
print(f"warmup total: {warm_t:.3f}s")
for name in list(TIMINGS.keys()):
    TIMINGS[name].clear()

print("\nTimed (post-compile) call...")
t0 = time.time()
compute_poles(mu0=-0.55)
total_t = time.time() - t0
print(f"total pipeline (post-compile): {total_t:.3f}s")

print("\nPer-function breakdown (each includes its own block_until_ready --")
print("note this adds sync points not present in the unprofiled pipeline,")
print("so treat these as relative proportions, not an exact sum-to-total):")
accounted = 0.0
for name, times in TIMINGS.items():
    s = sum(times)
    accounted += s
    print(f"  {name:30s} called {len(times)}x  total={s:.3f}s")
print(f"  {'(sum of above)':30s}          total={accounted:.3f}s")
print(f"  {'remainder (RSD/IR-resum via folps, combination, etc.)':55s} ~= {total_t - accounted:.3f}s")
