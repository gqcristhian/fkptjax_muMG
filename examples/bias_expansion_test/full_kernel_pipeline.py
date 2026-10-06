"""P_ell,full^standard(k): the real full-kernel BGS pipeline, constant b1.

Standalone ``fkptjax`` route (no desilike) -- the desilike wrapper for full,
non-squeezed MG kernels (``beyond_eds=True, fkpt_approximation=False``) is not
yet wired up for this. Instead this mirrors the already-validated pattern from
``examples/compute_mg_multipoles.ipynb`` (the "beyond-EdS full kernels"
comparison) and ``examples/check_bzmass_vs_sMGPT_full_v3.py`` (the
sMGPT-cross-checked production-resolution settings): ISiTGR supplies the
GR/LCDM linear input P(k); ``fkptjax.kfuncs_to_tables.Kfuncs_to_tables_jax``
applies the full, non-squeezed Hu-Sawicki MG kernels internally with
``beyond_eds=True, fkpt_approximation=False``, and the resulting FOLPS tables
are then projected to RSD multipoles for any bias/EFT/stochastic vector.

The expensive step (building the FKPT/FOLPS kernel tables) depends only on
cosmology/MG parameters, not on bias/EFT/stochastic nuisance parameters, so
it is built once per MG point (``build_state``, cached) and reused very
cheaply (``poles_for_bias``, a single jitted projection) for e.g. scanning
over EFT counterterm values (see ``alpha_fit.py``).
"""

import os

os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax.kfuncs_to_tables import build_jax_static_ctx, Kfuncs_to_tables_jax
from fkptjax.pipelines import make_table_state, poles_from_tables
from fkptjax.rsd import pack_fkpt_bias

import config
import linear_power

# Production-resolution kernel-table settings (matches
# examples/check_bzmass_vs_sMGPT_full_v3.py, sMGPT-cross-checked).
_KERNEL_SETTINGS = dict(Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
                         rbao=104.0, pmax_bao=0.4, Np_bao=100)

_static_ctx_cache = {}
_table_state_cache = {}

# Standard-basis zero defaults for everything but b1/b2 (config.STANDARD_NUISANCE_DEFAULTS,
# renamed to the FKPT_BIAS_ORDER names used by fkptjax.rsd.pack_fkpt_bias).
_BASE_BIAS = dict(
    b1=config.B1_BGS, b2=config.B2_BGS, bs2=0.0, b3nl=0.0,
    alpha0=0.0, alpha2=0.0, alpha4=0.0, ctilde=0.0,
    alpha0shot=0.0, alpha2shot=0.0, X_FoG_p=0.0,
)


def _mu_grid(nmu=12):
    x, w = np.polynomial.legendre.leggauss(nmu)
    return jnp.asarray(0.5 * (x + 1.0)), jnp.asarray(0.5 * w)


def _get_static_ctx(k_lin_j):
    key = (float(k_lin_j[0]), float(k_lin_j[-1]), k_lin_j.shape[0])
    if key not in _static_ctx_cache:
        _static_ctx_cache[key] = build_jax_static_ctx(
            k_lin_j, kmin=config.KERNEL_KMIN, kmax=config.KERNEL_KMAX,
            **_KERNEL_SETTINGS,
        )
    return _static_ctx_cache[key]


def build_state(fR0_HS, n_HS=config.N_HS):
    """Build (and cache) the FKPT/FOLPS kernel-table state for one MG point.

    Independent of bias/EFT/stochastic nuisance parameters -- built once,
    reused for any number of ``poles_for_bias`` calls.
    """
    key = (float(fR0_HS), float(n_HS))
    if key in _table_state_cache:
        return _table_state_cache[key]

    k_lin, pk_lin, pk_now_lin = linear_power.linear_power_gr(z=config.Z_EFF, nk=300)
    k_lin_j = jnp.asarray(k_lin)
    static_ctx = _get_static_ctx(k_lin_j)

    table_w, table_now, kernel_constants = Kfuncs_to_tables_jax(
        k=k_lin_j, pk=jnp.asarray(pk_lin), pk_now=jnp.asarray(pk_now_lin),
        return_kernel_constants=True, static_ctx=static_ctx,
        z=config.Z_EFF, Om=config.OM,
        beyond_eds=True, fkpt_approximation=False,
        kmin=config.KERNEL_KMIN, kmax=config.KERNEL_KMAX,
        xnow=config.XNOW, f0_kmax=config.KERNEL_KMIN,
        model="HS", mg_variant=None,
        fR0_HS=float(fR0_HS), beta2=config.BETA2_HS, n_HS=float(n_HS),
    )
    state = make_table_state(table_w, table_now, kernel_constants=kernel_constants)
    _table_state_cache[key] = state
    return state


def poles_for_bias(state, bias_overrides=None, k=None, ells=(0, 2, 4), nmu=12):
    """Project an already-built table `state` to P_ell(k) for a given bias vector.

    `bias_overrides` updates `_BASE_BIAS` (constant b1=B1_BGS, everything else
    standard-basis zero) -- e.g. ``{"alpha0": 12.0, "alpha2": -30.0}``.
    """
    if k is None:
        k = config.K_FIT
    k_eval = jnp.asarray(np.asarray(k, dtype="f8"))
    mu, wmu = _mu_grid(nmu)
    jac = jnp.asarray(1.0)
    kap = k_eval[:, None] * jnp.ones_like(mu)[None, :]
    muap = jnp.ones_like(k_eval)[:, None] * mu[None, :]

    bias = dict(_BASE_BIAS)
    bias.update(bias_overrides or {})
    pars = pack_fkpt_bias(bias, nd=config.NBAR_BGS)

    poles = poles_from_tables(
        state, jac=jac, kap=kap, muap=muap, pars=pars, mu=mu, wmu=wmu,
        ells=tuple(ells), bias_scheme="folps", IR_resummation=True,
        damping=None, A_full=False, use_TNS_model=False,
    )
    poles.block_until_ready()
    return {ell: np.asarray(poles[i]) for i, ell in enumerate(ells)}


def compute_standard_poles(fR0_HS, n_HS=config.N_HS, k=None, ells=(0, 2, 4), nmu=12):
    """Return P_ell,full^standard(k) (constant b1, no EFT/stochastic tilt)."""
    state = build_state(fR0_HS, n_HS=n_HS)
    return poles_for_bias(state, {}, k=k, ells=ells, nmu=nmu)


if __name__ == "__main__":
    import time
    for name, pt in config.MG_POINTS.items():
        t0 = time.time()
        poles = compute_standard_poles(pt["fR0_HS"])
        t1 = time.time()
        poles2 = poles_for_bias(build_state(pt["fR0_HS"]), {"alpha0": 10.0, "alpha2": -20.0})
        t2 = time.time()
        finite = all(np.all(np.isfinite(poles[ell])) for ell in poles)
        print(f"{name}: build+project {t1-t0:.1f}s, cached re-project {t2-t1:.3f}s, finite={finite}")
