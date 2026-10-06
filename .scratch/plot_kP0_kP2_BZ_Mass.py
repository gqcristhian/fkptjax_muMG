"""
Same kP0/kP2 exact-vs-emulated comparison as the PHENOM/binning one, but for
HDKI/BZ_Mass -- a SMOOTH mu(k,eta) model (a rational function of k, no tanh
step transitions at all), used as a contrast case: no k_attractors clustering
should be needed here, and accuracy should be much better than binning's.
"""
import os
os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/synthetic/scripts")

import generate_noiseless_synthetic_data_psbs_fkptbs as gen  # noqa: E402
from fkptjax import ab_ingredients as abi  # noqa: E402
from fkptjax import mg_jax as bj  # noqa: E402
from fkptjax.kfuncs_to_tables import _resolve_mg_kind  # noqa: E402

OUTDIR = Path("/n/home12/cgarciaquintero/DESI/synthetic/notebooks/binning_ingredients_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

TRACER = next(iter(gen.TRACERS))
TRACER_INFO = gen.TRACERS[TRACER]
Z = float(TRACER_INFO["z_pk"])

# The notebook's own illustrative BZ_Mass reference point.
PARAMS = dict(mu_kinf_BZmass=1.2, lambda_a_BZmass=1.0, lambda_dS_BZmass=1.0)

KF_MIN, KF_MAX = 0.001, 0.5
KLEG_MIN, KLEG_MAX = 1e-5, 8.0  # realistic, matches setup_kfunctions' pmax=16*kmax
PARAM_BOUNDS = {
    'mu_kinf_BZmass': (0.8, 1.6),
    'lambda_a_BZmass': (0.5, 1.5),
    'lambda_dS_BZmass': (0.5, 1.5),
}


def build_pt(fkpt_approximation, ingredients_provider=None):
    cosmo = gen.build_cosmology({})
    template = gen.DirectSpectrum2Template(
        z=Z, fiducial=gen.FIDUCIAL, engine="isitgr", cosmo=cosmo, with_now="peakaverage",
    )
    pt = gen.FKPTJAXPTSpectrum2Poles(
        k=gen.K_FIT, template=template, ells=gen.ELLS_PS,
        model="HDKI", mg_variant="BZ_Mass",
        beyond_eds=True, fkpt_approximation=fkpt_approximation,
        use_numba=gen.USE_NUMBA,
        include_neutrino_corrections=False,
        mg_params_override=dict(PARAMS),
        growth_source="ode",
        ingredients_provider=ingredients_provider,
    )
    theory = gen.FKPTJAXTracerSpectrum2Poles(
        k=gen.K_FIT, pt=pt, ells=gen.ELLS_PS, prior_basis=gen.PRIOR_BASIS,
        nbar=float(TRACER_INFO["nbar"]), damping=gen.DAMPING, damping_method=gen.DAMPING_METHOD,
        tracers=TRACER,
    )
    nuisance_values = {
        "b1": float(TRACER_INFO["b1"]), "b2": float(TRACER_INFO["b2"]),
        **gen.STANDARD_NUISANCE_DEFAULTS,
    }
    gen._fix_tracer_parameters(theory, nuisance_values)
    return gen.compile_calculator(theory)


def P_from_params(params):
    kind = _resolve_mg_kind('HDKI', 'BZ_Mass')
    Om = 0.315192
    return bj.pack_constants_jnp(
        om=Om, ol=1.0 - Om, kind=kind,
        mu_kinf=params['mu_kinf_BZmass'],
        lambda_a=params['lambda_a_BZmass'],
        lambda_dS=params['lambda_dS_BZmass'],
    )


print("=" * 78)
print("LIVE (fkpt_approximation=False, no ingredients_provider) -- ground truth")
print("=" * 78)
t0 = time.perf_counter()
pipeline_live = build_pt(False)
poles_live = np.asarray(pipeline_live())
jax.block_until_ready(poles_live)
print(f"build+compile+call: {time.perf_counter()-t0:.2f}s")

print()
print("=" * 78)
print("Training ingredients provider (BZ_Mass -- smooth model, no k_attractors)")
print("=" * 78)
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + Z)))
f0 = jnp.asarray(0.8)
t0 = time.perf_counter()
provider = abi.build_ingredients_provider(
    kf_min=KF_MIN, kf_max=KF_MAX, kleg_min=KLEG_MIN, kleg_max=KLEG_MAX,
    param_bounds=PARAM_BOUNDS, P_from_params=P_from_params,
    xnow=xnow, xstop=xstop, f0=f0, budget=4,  # no k_attractors -- smooth mu(k)
)
print(f"train time: {time.perf_counter()-t0:.2f}s")

print()
print("=" * 78)
print("EMULATED (fkpt_emulation route)")
print("=" * 78)
t0 = time.perf_counter()
pipeline_emu = build_pt(False, ingredients_provider=provider)
poles_emu = np.asarray(pipeline_emu())
jax.block_until_ready(poles_emu)
print(f"build+compile+call: {time.perf_counter()-t0:.2f}s")

print()
print("=" * 78)
print("fkpt_approximation=True (for context)")
print("=" * 78)
pipeline_approx = build_pt(True)
poles_approx = np.asarray(pipeline_approx())

poles_live_d = gen.split_multipoles_ps(poles_live)
poles_emu_d = gen.split_multipoles_ps(poles_emu)
poles_approx_d = gen.split_multipoles_ps(poles_approx)
P0_live, P2_live = poles_live_d[0], poles_live_d[2]
P0_emu, P2_emu = poles_emu_d[0], poles_emu_d[2]
P0_approx, P2_approx = poles_approx_d[0], poles_approx_d[2]

k = gen.K_FIT
rel_P0 = np.abs(P0_emu - P0_live) / (np.abs(P0_live) + 1e-30)
rel_P2 = np.abs(P2_emu - P2_live) / (np.abs(P2_live) + 1e-30)
print(f"\nkP0: max rel err={rel_P0.max():.3e}  median={np.median(rel_P0):.3e}")
print(f"kP2: max rel err={rel_P2.max():.3e}  median={np.median(rel_P2):.3e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (label, P_live, P_emu, P_approx, rel) in zip(
    axes[0], [('kP0(k)', P0_live, P0_emu, P0_approx, rel_P0),
              ('kP2(k)', P2_live, P2_emu, P2_approx, rel_P2)]
):
    ax.plot(k, k * P_live, color='C0', lw=1.8, label='exact (fkpt_approximation=False, live)')
    ax.plot(k, k * P_emu, color='C1', lw=1.4, ls='--', label='emulated (fkpt_emulation)')
    ax.plot(k, k * P_approx, color='C2', lw=1.0, ls=':', label='fkpt_approximation=True')
    ax.set_title(label)
    ax.set_xlabel('k [h/Mpc]')
    ax.legend(fontsize=8)

for ax, (label, rel) in zip(axes[1], [('P0 rel err', rel_P0), ('P2 rel err', rel_P2)]):
    ax.plot(k, rel, color='C3')
    ax.set_yscale('log')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_title(f'{label}: max={rel.max():.2e}, median={np.median(rel):.2e}')

fig.suptitle(f"kP0/kP2: exact vs emulated -- HDKI/BZ_Mass, {PARAMS}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
outpath = OUTDIR / "kP0_kP2_exact_vs_emulated_BZ_Mass_budget4.png"
fig.savefig(outpath, dpi=130)
print(f"\nsaved {outpath}")
print("DONE")
