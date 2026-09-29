"""Numerical dP_ell/dtheta for every parameter, for each of the 3 theories.

Central (2-point) finite differences throughout (``config.PARAMS[name]["step"]``).
Cosmological/mu0 parameters require rebuilding the FKPT/FOLPS kernel table at
theta +/- step (expensive, especially for "full"); bias/EFT/stochastic
parameters only re-project the already-built fiducial table (cheap) -- see
``kernel_pipeline.build_state`` vs ``poles_for_bias``.
"""

import numpy as np

import config
import kernel_pipeline

# Parameter name -> keyword accepted by kernel_pipeline.build_state.
COSMO_KEYS = {"H0": "H0", "ombh2": "ombh2", "omch2": "omch2", "logA": "logA", "ns": "ns"}


def _state_for(theory, param=None, sign=0):
    """Fiducial-point state, or with one cosmological/MG parameter shifted by
    ``sign * step`` (sign in {-1, 0, +1})."""
    cosmo_overrides = {}
    mg_param = config.MG_PARAM_FID
    if param is not None and sign != 0:
        spec = config.PARAMS[param]
        val = spec["fid"] + sign * spec["step"]
        if param == config.MG_PARAM_NAME:
            mg_param = val
        else:
            cosmo_overrides[COSMO_KEYS[param]] = val
    return kernel_pipeline.build_state(theory, mg_param=mg_param, **cosmo_overrides)


def _bias_overrides_for(param, sign):
    if sign == 0 or param not in config.BASE_BIAS:
        return {}
    spec = config.PARAMS[param]
    return {param: spec["fid"] + sign * spec["step"]}


def stacked_poles(poles_dict, ells=config.ELLS):
    """Flatten a {ell: array(nk,)} dict into one (n_ell*nk,) data vector, in
    ``ells`` order -- matches the block order covariance.py's covariance uses.
    """
    return np.concatenate([poles_dict[ell] for ell in ells])


def fiducial_poles(theory, ells=config.ELLS, k=None):
    state = kernel_pipeline.build_state(theory, mg_param=config.MG_PARAM_FID)
    return kernel_pipeline.poles_for_bias(state, {}, k=k, ells=ells)


def derivative(theory, param, ells=config.ELLS, k=None):
    """Central-difference dP/dtheta (stacked P0,P2,P4 data vector) for one
    parameter, for one theory.
    """
    spec = config.PARAMS[param]
    group = spec["group"]
    if group in ("mgparam", "cosmo"):
        state_plus = _state_for(theory, param, +1)
        state_minus = _state_for(theory, param, -1)
        p_plus = kernel_pipeline.poles_for_bias(state_plus, {}, k=k, ells=ells)
        p_minus = kernel_pipeline.poles_for_bias(state_minus, {}, k=k, ells=ells)
    else:
        state = kernel_pipeline.build_state(theory, mg_param=config.MG_PARAM_FID)
        p_plus = kernel_pipeline.poles_for_bias(state, _bias_overrides_for(param, +1), k=k, ells=ells)
        p_minus = kernel_pipeline.poles_for_bias(state, _bias_overrides_for(param, -1), k=k, ells=ells)
    return (stacked_poles(p_plus, ells) - stacked_poles(p_minus, ells)) / (2.0 * spec["step"])


def all_derivatives(theory, params=config.ALL_PARAMS, ells=config.ELLS, k=None):
    """{param_name: dP/dtheta array} for every parameter, one theory."""
    return {p: derivative(theory, p, ells=ells, k=k) for p in params}


if __name__ == "__main__":
    import time

    for theory in config.THEORIES:
        t0 = time.time()
        derivs = all_derivatives(theory, params=(config.MG_PARAM_NAME, "H0", "b1", "alpha0"))
        t1 = time.time()
        finite = all(np.all(np.isfinite(v)) for v in derivs.values())
        print(f"{theory:8s}: 4-param derivative smoke test {t1 - t0:.1f}s, finite={finite}")
        for p, v in derivs.items():
            print(f"    d P/d{p}: min={v.min():.4g} max={v.max():.4g}")
