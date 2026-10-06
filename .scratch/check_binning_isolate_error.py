"""
Isolate WHERE binning's A/B emulation error actually lives: is it (a) the
(kf,k1,k2) INTERPOLATION (fixed MG params, sweep k), (b) the MG-PARAMETER
Chebyshev fit (fixed k triple, sweep params), or (c) neither (a training-vs
-query MISMATCH, e.g. the triangle clamp)?

Cheap: no bispectrum/multipole pipeline, no Boltzmann calls -- just
ab_ingredients.build_ingredients_provider (A/B only) vs a DIRECT live
MG_kernels.A_B_grid call, at hand-picked triples straddling k_TGR/k_c/k_S.
"""
import os
os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk
from fkptjax import ab_ingredients as abi
from fkptjax.kfuncs_to_tables import _resolve_mg_kind

KF_MIN, KF_MAX = 0.001, 0.5
KLEG_MIN, KLEG_MAX = 1e-5, 8.0
Om = 0.315192
xnow = -3.912023
xstop = float(jnp.log(1.0 / 1.5))
kind = _resolve_mg_kind('PHENOM', 'binning')
PARAM_BOUNDS = {'mu1': (0.5, 1.5), 'mu2': (0.5, 1.5), 'mu3': (0.5, 1.5), 'mu4': (0.5, 1.5)}


def P_from_params(params):
    return bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=kind, scale_bins=True, **params)


import os as _os2
_NKF = int(_os2.environ.get("CHECK_NKF", "16"))
_NKLEG = int(_os2.environ.get("CHECK_NKLEG", "32"))
print(f"Training A/B-only provider (budget=2, k_attractors=(0.01,0.1,0.2), "
      f"cluster_width=0.005, n_cluster=16, Nkf={_NKF}, Nkleg={_NKLEG})...")
grid_ab = abi.build_coarse_grid(KF_MIN, KF_MAX, KLEG_MIN, KLEG_MAX,
                                 Nkf=_NKF, Nkleg=_NKLEG,
                                 k_attractors=(0.01, 0.1, 0.2), cluster_width=0.005, n_cluster=16)

from cosmoprimo.emulators.tools import Emulator as _Emulator, Space as _Space


class _ABTarget:
    def __call__(self, params):
        params = {name: float(np.atleast_1d(value)[0]) for name, value in params.items()}
        P = P_from_params(params)
        A, B, Ap, Bp = abi.compute_ab_coarse(grid_ab, P, xnow, xstop, solver='rk4', n_steps=64)
        return {'A': A, 'B': B, 'Ap': Ap, 'Bp': Bp}


import os as _os
_ENGINE = _os.environ.get("CHECK_ENGINE", "chebyshev")
_BUDGET = int(_os.environ.get("CHECK_BUDGET", "2"))
_LEVEL = _os.environ.get("CHECK_LEVEL")
_space_kwargs = {}
if _LEVEL is not None:
    _space_kwargs["levels"] = {name: int(_LEVEL) for name in PARAM_BOUNDS}

space = _Space(bounds=PARAM_BOUNDS, **_space_kwargs)
emu_ab = _Emulator(_ABTarget(), space, engine=_ENGINE, budget=_BUDGET)
print(f"engine={_ENGINE} budget={_BUDGET} space_kwargs={_space_kwargs}")
emu_ab.train()
print("done training\n")

# Triples straddling the thresholds: (kf, k1, k2) chosen so at least one leg
# sits in a DIFFERENT window than another (genuine cross-window triangle).
TRIPLES = {
    'straddle k_c (0.15,0.08,0.09)': (0.15, 0.08, 0.09),
    'straddle k_S (0.25,0.15,0.12)': (0.25, 0.15, 0.12),
}

PARAM_POINTS = {
    'ref (0.8,1.3,0.9,1.1)': dict(mu1=0.8, mu2=1.3, mu3=0.9, mu4=1.1),
    'mixed corner': dict(mu1=0.55, mu2=1.45, mu3=0.55, mu4=1.45),
}

print(f"{'triple':40s} {'params':22s} {'A_live':>12s} {'A_emu':>12s} {'relerr':>10s} "
      f"{'B_live':>12s} {'B_emu':>12s} {'relerr':>10s}")
for tname, (kf, k1, k2) in TRIPLES.items():
    for pname, params in PARAM_POINTS.items():
        P = P_from_params(params)
        A_live, B_live, _, _ = mgk.A_B_grid(
            jnp.asarray(kf), jnp.asarray(k1), jnp.asarray(k2), P, xnow, xstop,
            solver='rk4', n_steps=64)
        ab_pred = emu_ab.predict(**params)
        A_c, B_c = np.asarray(ab_pred['A']), np.asarray(ab_pred['B'])
        _INTERP_ORDER = int(_os2.environ.get("CHECK_INTERP_ORDER", "1"))
        A_emu, B_emu, _, _ = abi.interpolate_ab(
            jnp.asarray(kf), jnp.asarray(k1), jnp.asarray(k2), grid_ab, A_c, B_c,
            np.asarray(ab_pred['Ap']), np.asarray(ab_pred['Bp']), order=_INTERP_ORDER)
        A_live_f, B_live_f = float(A_live), float(B_live)
        A_emu_f, B_emu_f = float(A_emu), float(B_emu)
        relA = abs(A_emu_f - A_live_f) / (abs(A_live_f) + 1e-30)
        relB = abs(B_emu_f - B_live_f) / (abs(B_live_f) + 1e-30)
        print(f"{tname:40s} {pname:22s} {A_live_f:12.6f} {A_emu_f:12.6f} {relA:10.3e} "
              f"{B_live_f:12.6f} {B_emu_f:12.6f} {relB:10.3e}")
    print()

print("DONE")
