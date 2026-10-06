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

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())


def toy_pk(k):
    k = np.asarray(k, dtype=float)
    return 2.0e4 * k / (1.0 + (k / 0.05) ** 2) ** 2


def make_mu_grid(nmu=6):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)


bias = dict(b1=2.0, b2=0.0, bs2=0.0, b3nl=0.0, alpha0=0.0, alpha2=0.0, alpha4=0.0,
            ctilde=0.0, alpha0shot=0.0, alpha2shot=0.0)
pars = pack_fkpt_bias(bias, nd=1.0e-4)
z_pk_cmp = 0.3
mu, wmu = make_mu_grid(nmu=6)

# Sweep of grid sizes -- Q-loop ODE-grid point count ~= Nk_kernel*(nquadSteps-1)*NQ.
# Chosen geometrically to trace out the CPU-vs-GPU crossover, if any, without
# jumping straight to the ~3.6e5-point "production" size in one shot.
RESOLUTIONS = [
    dict(Nk_kernel=30,  nquadSteps=150, NQ=10, NR=10),  # ~44700 pts (already benchmarked)
    dict(Nk_kernel=60,  nquadSteps=225, NQ=10, NR=10),  # ~134400 pts (~3x)
    dict(Nk_kernel=120, nquadSteps=300, NQ=10, NR=10),  # ~359280 pts (production resolution)
]


def compute_full_kernels(k_lin, pk_lin, pk_now_lin, k_eval, jac, kap, muap, static_ctx, mu0):
    poles, _ = binning_jax_poles(
        k=k_lin, pk=pk_lin, pk_now=pk_now_lin,
        jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
        ells=(0, 2), bias_scheme="folps", IR_resummation=True,
        damping=None, A_full=False, use_TNS_model=False,
        return_kernel_constants=True,
        static_ctx=static_ctx,
        z=z_pk_cmp, Om=0.315,
        beyond_eds=True, fkpt_approximation=False,
        kmin=float(k_lin[0]), kmax=float(k_lin[-1]), xnow=-3.912023, f0_kmax=1.0e-3,
        model="HDKI", mg_variant="mu_OmDE", mu0=mu0,
    )
    poles.block_until_ready()
    return poles


for res in RESOLUTIONS:
    npts = res["Nk_kernel"] * (res["nquadSteps"] - 1) * res["NQ"]
    N_K_EVAL = res["Nk_kernel"]

    k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 200))
    pk_lin_np = toy_pk(np.asarray(k_lin))
    logpk = np.log(pk_lin_np)
    pk_now_np = np.exp(savgol_filter(logpk, window_length=41, polyorder=3))
    pk_lin = jnp.asarray(pk_lin_np)
    pk_now_lin = jnp.asarray(pk_now_np)

    k_eval = jnp.asarray(np.linspace(0.05, 0.15, N_K_EVAL))
    jac = jnp.asarray(1.0)
    kap = k_eval[:, None] * jnp.ones_like(mu)[None, :]
    muap = jnp.ones_like(k_eval)[:, None] * mu[None, :]

    t0 = time.time()
    static_ctx = build_jax_static_ctx(
        k_lin, kmin=float(k_lin[0]), kmax=float(k_lin[-1]),
        rbao=104.0, pmax_bao=0.4, Np_bao=100,
        **res,
    )
    ctx_t = time.time() - t0

    t0 = time.time()
    compute_full_kernels(k_lin, pk_lin, pk_now_lin, k_eval, jac, kap, muap, static_ctx, mu0=-0.5)
    warm_t = time.time() - t0

    t0 = time.time()
    compute_full_kernels(k_lin, pk_lin, pk_now_lin, k_eval, jac, kap, muap, static_ctx, mu0=-0.55)
    iter_t = time.time() - t0

    print(f"grid={res}  npts~{npts:7d}  ctx_build={ctx_t:6.2f}s  "
          f"warmup(compile)={warm_t:8.2f}s  post-compile 1 iter={iter_t:8.2f}s")
