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


mgk.A_B_grid = timed("A_B_grid", mgk.A_B_grid)
mgk.D2_fused_grid = timed("D2_fused_grid", mgk.D2_fused_grid)
mgk.D3_fused_grid = timed("D3_fused_grid", mgk.D3_fused_grid)

# Re-point kfuncs_to_tables's already-imported `_mgk` alias (it does
# `from fkptjax import MG_kernels as _mgk` LAZILY inside the function body,
# so patching the module's own attributes above is picked up automatically
# on every call -- no need to patch kfuncs_to_tables itself.)


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
    print(f"  {name:20s} called {len(times)}x  total={s:.3f}s")
print(f"  {'(sum of above)':20s}          total={accounted:.3f}s")
print(f"  {'remainder (RSD/IR-resum/combination/etc.)':40s} ~= {total_t - accounted:.3f}s"
      f"  (or negative if profiling sync points changed overlap/dispatch)")


# ---------------------------------------------------------------------------
# Tolerance-loosening experiment: temporarily patch the ORIGINAL functions'
# default rtol/atol (via __defaults__), on the real production pipeline path
# -- no changes to MG_kernels.py's source. Compares speed AND the resulting
# P0/P2 accuracy against the tight-tolerance (default) result above.
# ---------------------------------------------------------------------------
print("\n=== Tolerance-loosening experiment (same production pipeline) ===")

poles_tight = compute_poles(mu0=-0.55)
P0_tight, P2_tight = np.asarray(poles_tight[0]), np.asarray(poles_tight[1])

# Grab the ORIGINAL (unwrapped) functions from inside the `timed` closures
# is not possible after wrapping -- so re-import fresh originals here and
# patch THEIR __defaults__ before re-wrapping with the same timer.
import importlib
from fkptjax import MG_kernels as mgk_fresh
importlib.reload(mgk_fresh)

for rtol_new, atol_new in [(1e-5, 1e-8), (1e-4, 1e-7)]:
    orig_A_B_grid = mgk_fresh.A_B_grid
    orig_D2 = mgk_fresh.D2_fused_grid
    orig_D3 = mgk_fresh.D3_fused_grid

    # __defaults__ order matches (solver, n_steps, rtol, atol) tail of each signature.
    orig_A_B_grid.__defaults__ = (orig_A_B_grid.__defaults__[0], orig_A_B_grid.__defaults__[1], rtol_new, atol_new)
    orig_D2.__defaults__ = (orig_D2.__defaults__[0], orig_D2.__defaults__[1], rtol_new, atol_new)
    orig_D3.__defaults__ = (orig_D3.__defaults__[0], orig_D3.__defaults__[1], rtol_new, atol_new)

    TIMINGS.clear()
    mgk.A_B_grid = timed("A_B_grid", orig_A_B_grid)
    mgk.D2_fused_grid = timed("D2_fused_grid", orig_D2)
    mgk.D3_fused_grid = timed("D3_fused_grid", orig_D3)

    # warmup (new tolerance -> new compiled trace)
    compute_poles(mu0=-0.5)
    TIMINGS.clear()

    t0 = time.time()
    poles_loose = compute_poles(mu0=-0.55)
    total_loose = time.time() - t0
    P0_loose, P2_loose = np.asarray(poles_loose[0]), np.asarray(poles_loose[1])

    rel0 = np.max(np.abs(P0_loose - P0_tight) / np.maximum(1.0, np.abs(P0_tight)))
    rel2 = np.max(np.abs(P2_loose - P2_tight) / np.maximum(1.0, np.abs(P2_tight)))

    print(f"\nrtol={rtol_new:.0e}, atol={atol_new:.0e}:")
    print(f"  total pipeline (post-compile): {total_loose:.3f}s  (tight-tolerance was {total_t:.3f}s, "
          f"speedup {total_t/total_loose:.2f}x)")
    for name, times in TIMINGS.items():
        print(f"    {name:20s} called {len(times)}x  total={sum(times):.3f}s")
    print(f"  max relative diff vs tight-tolerance:  P0={rel0:.3e}  P2={rel2:.3e}")
