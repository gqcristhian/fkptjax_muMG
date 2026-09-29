"""
Does the trained BZ_Mass ingredients provider actually respond to changes in
(mu_kinf_BZmass, lambda_a_BZmass, lambda_dS_BZmass)? Direct, cheap check:
predict_coarse() at two clearly different parameter points and compare the
raw coarse A/B/A_fused/CFD3 arrays -- if they're nearly identical despite
very different params, the emulator isn't tracking parameter variation at
all (as opposed to a resolution/edge-case problem, which WOULD show
different, if wrong, predictions at different points).
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
from fkptjax import ab_ingredients as abi
from fkptjax.kfuncs_to_tables import _resolve_mg_kind

KF_MIN, KF_MAX = 0.001, 0.5
KLEG_MIN, KLEG_MAX = 1e-5, 8.0
Om = 0.315192
xnow = -3.912023
xstop = float(jnp.log(1.0 / 1.5))
kind = _resolve_mg_kind('HDKI', 'BZ_Mass')
PARAM_BOUNDS = {'mu_kinf_BZmass': (0.8, 1.6), 'lambda_a_BZmass': (0.5, 1.5),
                'lambda_dS_BZmass': (0.5, 1.5)}


def P_from_params(params):
    return bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=kind,
                                  mu_kinf=params['mu_kinf_BZmass'],
                                  lambda_a=params['lambda_a_BZmass'],
                                  lambda_dS=params['lambda_dS_BZmass'])


print("Training provider (budget=4, chebyshev)...")
provider = abi.build_ingredients_provider(
    kf_min=KF_MIN, kf_max=KF_MAX, kleg_min=KLEG_MIN, kleg_max=KLEG_MAX,
    param_bounds=PARAM_BOUNDS, P_from_params=P_from_params,
    xnow=xnow, xstop=xstop, budget=4,
)
print("done training\n")

POINTS = {
    'reference (1.2, 1.0, 1.0)': dict(mu_kinf_BZmass=1.2, lambda_a_BZmass=1.0, lambda_dS_BZmass=1.0),
    'low corner (0.8, 0.5, 0.5)': dict(mu_kinf_BZmass=0.8, lambda_a_BZmass=0.5, lambda_dS_BZmass=0.5),
    'high corner (1.6, 1.5, 1.5)': dict(mu_kinf_BZmass=1.6, lambda_a_BZmass=1.5, lambda_dS_BZmass=1.5),
}

results = {}
for name, params in POINTS.items():
    ab_pred, fused_pred = provider.predict_coarse(**params)
    results[name] = (np.asarray(ab_pred['A']), np.asarray(ab_pred['B']),
                      np.asarray(fused_pred['Af']), np.asarray(fused_pred['CFD3']))
    A, B, Af, CFD3 = results[name]
    print(f"{name}: A[mean,std]=({A.mean():.6f},{A.std():.3e})  "
          f"B[mean,std]=({B.mean():.6f},{B.std():.3e})  "
          f"Af[mean,std]=({Af.mean():.6f},{Af.std():.3e})  "
          f"CFD3[mean,std]=({CFD3.mean():.6f},{CFD3.std():.3e})")

names = list(POINTS)
print("\nPairwise differences (should be LARGE if the emulator tracks these very different params):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        A_i, B_i, Af_i, CFD3_i = results[names[i]]
        A_j, B_j, Af_j, CFD3_j = results[names[j]]
        dA = np.abs(A_i - A_j).mean() / (np.abs(A_i).mean() + 1e-30)
        dB = np.abs(B_i - B_j).mean() / (np.abs(B_i).mean() + 1e-30)
        dAf = np.abs(Af_i - Af_j).mean() / (np.abs(Af_i).mean() + 1e-30)
        dCFD3 = np.abs(CFD3_i - CFD3_j).mean() / (np.abs(CFD3_i).mean() + 1e-30)
        print(f"  {names[i]!r} vs {names[j]!r}: "
              f"mean|dA/A|={dA:.3e}  mean|dB/B|={dB:.3e}  mean|dAf/Af|={dAf:.3e}  mean|dCFD3/CFD3|={dCFD3:.3e}")

print("\nDONE")
