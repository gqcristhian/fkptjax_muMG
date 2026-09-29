"""P_ell(k) for the three theory cases: EdS / bEdS-fkPT-approx / bEdS-full.

Direct ``fkptjax`` route (no desilike): desilike's FKPT theory wrapper has no
``fkpt_approximation`` parameter and no cosmological-parameter layer of its
own, so this mirrors ``bias_expansion_test/full_kernel_pipeline.py``'s
build_state/poles_for_bias split, generalized with a `theory` switch
(``"eds"``, ``"approx"``, ``"full"`` -- see ``config.THEORY_KWARGS``) and
with cosmological parameters (H0, ombh2, omch2, logA, ns) as explicit
arguments, since every cosmological finite-difference step needs a fresh
ISiTGR linear spectrum AND a fresh FKPT/FOLPS kernel-table build.

The expensive step (building the kernel tables, especially for "full") only
depends on cosmology + mu0 + theory, not on bias/EFT/stochastic nuisance
parameters, so it is cached and reused very cheaply (``poles_for_bias``) for
any nuisance-parameter finite-difference step.
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

_static_ctx_cache = {}
_table_state_cache = {}

_MU, _WMU = np.polynomial.legendre.leggauss(12)
MU = jnp.asarray(0.5 * (_MU + 1.0))
WMU = jnp.asarray(0.5 * _WMU)
JAC = jnp.asarray(1.0)


def _round_key(*vals, ndig=10):
    return tuple(round(float(v), ndig) for v in vals)


def _get_static_ctx(k_lin_j):
    key = _round_key(k_lin_j[0], k_lin_j[-1]) + (k_lin_j.shape[0],)
    if key not in _static_ctx_cache:
        _static_ctx_cache[key] = build_jax_static_ctx(
            k_lin_j, kmin=config.KERNEL_KMIN, kmax=config.KERNEL_KMAX,
            **config.KERNEL_SETTINGS,
        )
    return _static_ctx_cache[key]


def build_state(theory, *, H0=config.H0_FID, ombh2=config.OMBH2_FID, omch2=config.OMCH2_FID,
                logA=None, ns=config.NS_FID, mg_param=None):
    """Build (and cache) the FKPT/FOLPS kernel-table state for one
    (theory, cosmology, MG parameter) point. Independent of bias/EFT/stochastic
    nuisance parameters -- built once, reused for any number of
    ``poles_for_bias`` calls.

    ``mg_param`` is the value of whichever parameter ``config.MG_PARAM_NAME``
    points to (``mu0`` for mu_OmDE, ``fR0_HS`` for Hu-Sawicki f(R)); defaults
    to its fiducial value.
    """
    if logA is None:
        As = config.AS_FID
    else:
        As = float(np.exp(logA) * 1.0e-10)
    if mg_param is None:
        mg_param = config.MG_PARAM_FID

    key = (theory,) + _round_key(H0, ombh2, omch2, As, ns, mg_param)
    if key in _table_state_cache:
        return _table_state_cache[key]

    k_lin, pk_lin, pk_now_lin = linear_power.linear_power_gr(
        H0=H0, ombh2=ombh2, omch2=omch2, As=As, ns=ns, z=config.Z_EFF, nk=300,
    )
    k_lin_j = jnp.asarray(k_lin)
    static_ctx = _get_static_ctx(k_lin_j)

    Om = config.Om_of(H0=H0, ombh2=ombh2, omch2=omch2, mnu=config.MNU)

    theory_kwargs = config.THEORY_KWARGS[theory]
    mg_kwargs = dict(config.MG_EXTRA_KWARGS)
    mg_kwargs[config.MG_PARAM_NAME] = float(mg_param)
    table_w, table_now, kernel_constants = Kfuncs_to_tables_jax(
        k=k_lin_j, pk=jnp.asarray(pk_lin), pk_now=jnp.asarray(pk_now_lin),
        return_kernel_constants=True, static_ctx=static_ctx,
        z=config.Z_EFF, Om=Om,
        kmin=config.KERNEL_KMIN, kmax=config.KERNEL_KMAX,
        xnow=config.XNOW, f0_kmax=config.KERNEL_KMIN,
        model=config.MG_MODEL, mg_variant=config.MG_VARIANT,
        **mg_kwargs, **theory_kwargs,
    )
    state = make_table_state(table_w, table_now, kernel_constants=kernel_constants)
    _table_state_cache[key] = state
    return state


def poles_for_bias(state, bias_overrides=None, k=None, ells=config.ELLS, nmu=12):
    """Project an already-built table `state` to P_ell(k) for a given bias vector."""
    if k is None:
        k = config.K_FIT
    k_eval = jnp.asarray(np.asarray(k, dtype="f8"))
    kap = k_eval[:, None] * jnp.ones_like(MU)[None, :]
    muap = jnp.ones_like(k_eval)[:, None] * MU[None, :]

    bias = dict(config.BASE_BIAS)
    bias.update(bias_overrides or {})
    pars = pack_fkpt_bias(bias, nd=config.NBAR_BGS)

    poles = poles_from_tables(
        state, jac=JAC, kap=kap, muap=muap, pars=pars, mu=MU, wmu=WMU,
        ells=tuple(ells), bias_scheme="folps", IR_resummation=True,
        damping=None, A_full=False, use_TNS_model=False,
    )
    poles.block_until_ready()
    return {ell: np.asarray(poles[i]) for i, ell in enumerate(ells)}


def compute_poles(theory, cosmo_overrides=None, mg_param=None, bias_overrides=None,
                   k=None, ells=config.ELLS):
    """Convenience one-shot: build_state + poles_for_bias for one point."""
    cosmo_overrides = cosmo_overrides or {}
    state = build_state(theory, mg_param=mg_param, **cosmo_overrides)
    return poles_for_bias(state, bias_overrides, k=k, ells=ells)


if __name__ == "__main__":
    import time
    for theory in config.THEORIES:
        t0 = time.time()
        poles = compute_poles(theory)
        t1 = time.time()
        finite = all(np.all(np.isfinite(poles[ell])) for ell in config.ELLS)
        print(f"{theory:8s}: build+project {t1 - t0:.1f}s, finite={finite}, "
              f"P0[0]={poles[0][0]:.4g}")
