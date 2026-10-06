import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import isitgr
from isitgr import model as isitgr_model
from scipy.signal import savgol_filter

from fkptjax.kfuncs_to_tables import build_jax_static_ctx
import fkptjax.pipelines as pipelines
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


mgk.I1udd1_and_P13_grid = timed("fkptjax: MG_kernels ODE solve", mgk.I1udd1_and_P13_grid)
pipelines.tables_to_poles = timed("folps: RSD/IR-resummation", pipelines.tables_to_poles)

import fkptjax.jax_ode as jax_ode_mod
jax_ode_mod.kernel_constants_jax = timed("fkptjax: kernel_constants_jax (squeezed-limit ODE)", jax_ode_mod.kernel_constants_jax)


# ---------------------------------------------------------------------------
# 1) Boltzmann solver (ISiTGR/CAMB) -- same linear P(k) feeds all 3 cases.
# ---------------------------------------------------------------------------
def smooth_now_savgol(k, pk, window=41, polyorder=3):
    k = np.asarray(k, dtype=float); pk = np.asarray(pk, dtype=float)
    logpk = np.log(pk)
    n = len(k)
    window = min(int(window), n - 1 if (n - 1) % 2 == 1 else n - 2)
    if window < polyorder + 2: window = polyorder + 3
    if window % 2 == 0: window += 1
    return np.exp(savgol_filter(logpk, window_length=window, polyorder=polyorder))


def get_isitgr_gr_linear_spectra(z=0.7, nk=256, minkh=1.0e-3, maxkh=1.0):
    pars = isitgr.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.12, mnu=0.06, omk=0.0, tau=0.0544,
                        nnu=3.046, MG_parameterization="muSigma", mu0=0.0)
    pars.InitPower.set_params(As=2.083e-9, ns=0.9649, r=0.0)
    pars.set_accuracy(AccuracyBoost=2)
    pars.NonLinear = isitgr_model.NonLinear_none
    pars.set_matter_power(redshifts=[z], kmax=maxkh)
    results = isitgr.get_results(pars)
    k, _, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=nk)
    pk = pk[0]
    pk_now = smooth_now_savgol(k, pk, window=41, polyorder=3)
    return jnp.asarray(k), jnp.asarray(pk), jnp.asarray(pk_now)


t0 = time.time()
k_lin, pk_lin, pk_now_lin = get_isitgr_gr_linear_spectra(z=0.7, nk=256, minkh=1.0e-3, maxkh=1.0)
jax.block_until_ready((k_lin, pk_lin, pk_now_lin))
t_isitgr_cold = time.time() - t0
t0 = time.time()
k_lin, pk_lin, pk_now_lin = get_isitgr_gr_linear_spectra(z=0.7, nk=256, minkh=1.0e-3, maxkh=1.0)
jax.block_until_ready((k_lin, pk_lin, pk_now_lin))
t_isitgr_warm = time.time() - t0
print(f"\nISiTGR/CAMB linear P(k): cold={t_isitgr_cold:.3f}s  warm(repeat-call)={t_isitgr_warm:.3f}s")


def make_mu_grid(nmu=8):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)


k_eval = jnp.asarray(np.linspace(0.02, 0.20, 36))
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
t_static_ctx = time.time() - t0
print(f"fkptjax: static_ctx build (ONE-TIME setup, not per-likelihood-call): {t_static_ctx:.3f}s")


def compute_poles(beyond_eds, fkpt_approximation, mu0):
    poles, _ = pipelines.binning_jax_poles(
        k=k_lin, pk=pk_lin, pk_now=pk_now_lin,
        jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
        ells=(0, 2), bias_scheme="folps", IR_resummation=True,
        damping=None, A_full=False, use_TNS_model=False,
        return_kernel_constants=True,
        static_ctx=static_ctx,
        z=z_pk_cmp, Om=0.315,
        beyond_eds=beyond_eds, fkpt_approximation=fkpt_approximation,
        kmin=1.0e-3, kmax=1.0, xnow=-3.912023, f0_kmax=1.0e-3,
        model="HDKI", mg_variant="mu_OmDE", mu0=mu0,
    )
    jax.block_until_ready(poles)
    return poles


CASES = [
    ("EdS", False, True),
    ("bEdS (fkPT approx.)", True, True),
    ("bEdS (Full)", True, False),
]

results = {}
for label, beyond_eds, fkpt_approx in CASES:
    for name in list(TIMINGS.keys()):
        TIMINGS[name].clear()
    t0 = time.time()
    compute_poles(beyond_eds, fkpt_approx, mu0=-0.5)
    warm_t = time.time() - t0
    for name in list(TIMINGS.keys()):
        TIMINGS[name].clear()

    t0 = time.time()
    compute_poles(beyond_eds, fkpt_approx, mu0=-0.55)
    total_t = time.time() - t0

    mgk_t = sum(TIMINGS.get("fkptjax: MG_kernels ODE solve", []))
    folps_t = sum(TIMINGS.get("folps: RSD/IR-resummation", []))
    kconst_t = sum(TIMINGS.get("fkptjax: kernel_constants_jax (squeezed-limit ODE)", []))
    fkptjax_overhead_t = total_t - mgk_t - folps_t - kconst_t
    results[label] = dict(total=total_t, warm=warm_t, mgk=mgk_t, folps=folps_t,
                          kconst=kconst_t, overhead=fkptjax_overhead_t)
    print(f"\n{label}: warmup(compile)={warm_t:.3f}s  post-compile total={total_t:.3f}s")
    print(f"  fkptjax: kernel_constants_jax (squeezed-limit ODE, beyond_eds only): {kconst_t:.3f}s")
    print(f"  fkptjax (remaining table build/combination):                       {fkptjax_overhead_t:.3f}s")
    print(f"  fkptjax: MG_kernels ODE solve (only nonzero for Full):              {mgk_t:.3f}s")
    print(f"  folps: RSD/IR-resummation:                                         {folps_t:.3f}s")

print("\n" + "=" * 90)
print("SUMMARY TABLE -- per-likelihood-call time breakdown (seconds), production resolution")
print("(Boltzmann solver time is amortized separately -- see note below)")
print("=" * 90)
header = f"{'Stage':<45} {'EdS':>12} {'bEdS approx':>14} {'bEdS Full':>12}"
print(header)
print("-" * len(header))


def row(label, key):
    vals = [results[c[0]][key] for c in CASES]
    print(f"{label:<45} " + " ".join(f"{v:12.3f}" for v in vals))


row("fkptjax: kernel_constants_jax (squeezed ODE)", "kconst")
row("fkptjax: table build / combination (remaining)", "overhead")
row("fkptjax: MG_kernels ODE solve (beyond-EdS full)", "mgk")
row("folps: RSD/IR-resummation", "folps")
print("-" * len(header))
row("TOTAL (post-compile, per-likelihood-call)", "total")
print(f"\nBoltzmann solver (ISiTGR/CAMB), same P(k) for all 3 cases: "
      f"{t_isitgr_warm:.3f}s (repeat call) -- typically run once per cosmology step, "
      f"not necessarily every bias/MG-param step depending on your sampler setup.")
print(f"One-time fkptjax setup (build_jax_static_ctx, NOT per-call): {t_static_ctx:.3f}s")
