import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")
# NOTE: JAX_PLATFORMS is intentionally NOT forced here -- let JAX pick GPU
# when one is visible (gpu_test node), CPU otherwise. Compare by running
# this same script once with JAX_PLATFORMS=cpu and once without.

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.signal import savgol_filter
from fkptjax.kfuncs_to_tables import build_jax_static_ctx
from fkptjax.pipelines import binning_jax_poles
from fkptjax.rsd import pack_fkpt_bias

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())


def toy_pk(k):
    # Simple smooth analytic P(k) -- no need for CAMB/ISiTGR here, this is
    # purely a *speed* test, not a physics-accuracy test.
    k = np.asarray(k, dtype=float)
    return 2.0e4 * k / (1.0 + (k / 0.05) ** 2) ** 2


def make_mu_grid(nmu=6):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)


# Small grid: "few k-points", per the request -- just enough to get a
# reliable per-iteration timing, not a production run.
N_K_EVAL = 6
k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 200))
pk_lin_np = toy_pk(np.asarray(k_lin))
logpk = np.log(pk_lin_np)
pk_now_np = np.exp(savgol_filter(logpk, window_length=41, polyorder=3))
pk_lin = jnp.asarray(pk_lin_np)
pk_now_lin = jnp.asarray(pk_now_np)

k_eval = jnp.asarray(np.linspace(0.05, 0.15, N_K_EVAL))
mu, wmu = make_mu_grid(nmu=6)
jac = jnp.asarray(1.0)
kap = k_eval[:, None] * jnp.ones_like(mu)[None, :]
muap = jnp.ones_like(k_eval)[:, None] * mu[None, :]

bias = dict(b1=2.0, b2=0.0, bs2=0.0, b3nl=0.0, alpha0=0.0, alpha2=0.0, alpha4=0.0,
            ctilde=0.0, alpha0shot=0.0, alpha2shot=0.0)
pars = pack_fkpt_bias(bias, nd=1.0e-4)

z_pk_cmp = 0.3

VARIANTS = [
    ("EdS",                       dict(beyond_eds=False, fkpt_approximation=True)),
    ("beyond-EdS (fkPT approx.)", dict(beyond_eds=True,  fkpt_approximation=True)),
    ("beyond-EdS (full kernels)", dict(beyond_eds=True,  fkpt_approximation=False)),
]

# Small resolution: N_K_EVAL external k's, small loop grid.
RES_KWARGS = dict(Nk_kernel=N_K_EVAL, nquadSteps=24, NQ=4, NR=4)
N_TIMED_BY_VARIANT = {
    "EdS": 8,
    "beyond-EdS (fkPT approx.)": 8,
    "beyond-EdS (full kernels)": 5,
}


def compute_poles(static_ctx, mu0, beyond_eds, fkpt_approximation):
    poles, _ = binning_jax_poles(
        k=k_lin, pk=pk_lin, pk_now=pk_now_lin,
        jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
        ells=(0, 2), bias_scheme="folps", IR_resummation=True,
        damping=None, A_full=False, use_TNS_model=False,
        return_kernel_constants=True,
        static_ctx=static_ctx,
        z=z_pk_cmp, Om=0.315,
        beyond_eds=beyond_eds, fkpt_approximation=fkpt_approximation,
        kmin=float(k_lin[0]), kmax=float(k_lin[-1]), xnow=-3.912023, f0_kmax=1.0e-3,
        model="HDKI", mg_variant="mu_OmDE", mu0=mu0,
    )
    poles.block_until_ready()
    return poles


t0 = time.time()
static_ctx = build_jax_static_ctx(
    k_lin, kmin=float(k_lin[0]), kmax=float(k_lin[-1]),
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
    **RES_KWARGS,
)
print(f"static_ctx build: {time.time()-t0:.2f}s  (grid: {RES_KWARGS})")

for label, kwargs in VARIANTS:
    t0 = time.time()
    compute_poles(static_ctx, mu0=-0.5, **kwargs)
    warm_t = time.time() - t0

    n_timed = N_TIMED_BY_VARIANT[label]
    mu0_vals = np.linspace(-0.6, -0.4, n_timed)
    t0 = time.time()
    for mu0 in mu0_vals:
        compute_poles(static_ctx, mu0=float(mu0), **kwargs)
    elapsed = time.time() - t0
    per_iter = elapsed / n_timed
    print(f"  {label:28s} warmup(incl. compile)={warm_t:7.3f}s   "
          f"post-compile ({n_timed} iters): {per_iter*1000:8.2f} ms/iter   {1.0/per_iter:8.3f} it/s")
