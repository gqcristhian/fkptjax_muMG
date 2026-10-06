"""
One comprehensive diagnostic figure per model: the raw emulated ingredients
(A, B, A_fused, CFD3 -- scatter of every point on the REAL production Q-loop/
R-loop grid, live vs emulated) PLUS the reference (not emulated, shown for
context) linear P(k) and growth f(k), PLUS the final products that actually
get fit to data: kP0(k), kP2(k), exact vs emulated vs fkpt_approximation=True.

Usage: set MODEL_NAME below to 'BZ_Mass' or 'binning' and run.
"""
import os
os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/synthetic/scripts")

import generate_noiseless_synthetic_data_psbs_fkptbs as gen  # noqa: E402
from fkptjax import ab_ingredients as abi  # noqa: E402
from fkptjax import mg_jax as bj  # noqa: E402
from fkptjax import MG_kernels as mgk  # noqa: E402
from fkptjax.util import setup_kfunctions  # noqa: E402
from fkptjax.calculate_jax import JaxCalculator  # noqa: E402
from fkptjax.kfuncs_to_tables import _resolve_mg_kind  # noqa: E402

OUTDIR = Path("/n/home12/cgarciaquintero/DESI/synthetic/notebooks/binning_ingredients_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model / engine configuration -- edit via env vars.
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("DIAG_MODEL", "BZ_Mass")
ENGINE_NAME = os.environ.get("DIAG_ENGINE", "chebyshev")

ENGINE_CONFIGS = {
    'chebyshev': {},
    # order=3: cubic per parameter: matches chebyshev budget=4's degree-3 polynomial
    # (4 Chebyshev nodes per axis -> exact up to degree 3). oversampling=2.0 (the
    # engine's own default): tolerant of a dropped/NaN node, at the cost of not being
    # exact anywhere (a regression, not an interpolant) -- see ab_ingredients.py's
    # build_ingredients_provider docstring.
    'polynomial': dict(order=3, oversampling=2.0),
}

MODEL_CONFIGS = {
    'BZ_Mass': dict(
        model='HDKI', mg_variant='BZ_Mass', scale_bins=False,
        params=dict(mu_kinf_BZmass=1.2, lambda_a_BZmass=1.0, lambda_dS_BZmass=1.0),
        # near the box corner (bounds: mu_kinf in [0.8,1.6], lambda_a/lambda_dS in [0.5,1.5])
        extreme_params=dict(mu_kinf_BZmass=1.58, lambda_a_BZmass=1.45, lambda_dS_BZmass=1.45),
        param_bounds={'mu_kinf_BZmass': (0.8, 1.6), 'lambda_a_BZmass': (0.5, 1.5),
                      'lambda_dS_BZmass': (0.5, 1.5)},
        k_attractors=(), budget=4,
        pack_kwargs=lambda p: dict(mu_kinf=p['mu_kinf_BZmass'], lambda_a=p['lambda_a_BZmass'],
                                    lambda_dS=p['lambda_dS_BZmass']),
    ),
    'binning': dict(
        model='PHENOM', mg_variant='binning', scale_bins=True,
        params=dict(mu1=0.8, mu2=1.3, mu3=0.9, mu4=1.1),
        # near the box corners (bounds: all mu_i in [0.5,1.5])
        extreme_params=dict(mu1=0.55, mu2=1.45, mu3=0.55, mu4=1.45),
        param_bounds={'mu1': (0.5, 1.5), 'mu2': (0.5, 1.5), 'mu3': (0.5, 1.5), 'mu4': (0.5, 1.5)},
        k_attractors=(0.01, 0.1, 0.2), budget=2,
        # k_tw (the model's own tanh transition half-width) is a FIXED 0.001 at
        # all three thresholds (mg_jax.py), but _clustered_log_nodes's default
        # cluster window scales as 0.3*k0 -- at k_c=0.1/k_S=0.2 that's a window
        # 10-30x wider than the step itself with only 8 nodes in it, so the
        # step is badly under-resolved there (unlike k_TGR=0.01, where 0.3*k0
        # happens to match). Use a fixed absolute width instead, tied to k_tw.
        # This is the fix that actually moved the needle (kP0 max 4.4%->1.2%,
        # kP2 max 8.9%->3.2%). Geometric (vs flat) clustering, budget 2->3->4,
        # fused-grid Nx 64->128, and Chebyshev `levels` 2->3 (parameter-space
        # resolution, 5x the training cost) were ALL tried on top of this and
        # made ZERO further difference -- the ~1.2%/3.2% floor is not a
        # resolution problem of any kind tried so far.
        # Nkleg default (32) log-uniform over kleg_min..kleg_max=1e-5..8.0 is
        # very coarse (~55% relative node spacing) OUTSIDE the narrow +-0.005
        # cluster windows -- isolated diagnostic (check_binning_nodeexact.py)
        # proved the Chebyshev parameter fit is essentially EXACT at a coarse
        # node (~0.05-0.1% even at extreme corners), so the whole ~1.2-3.2%
        # residual is from k-INTERPOLATION between generic-region nodes, not
        # parameter resolution at all -- confirmed: Nkleg=48 alone cut the
        # full-pipeline max error ~13-14% (1.22%->1.06%, 3.16%->2.73%), the
        # first real (if modest) improvement since the cluster_width fix.
        grid_kwargs=dict(cluster_width=0.005, n_cluster=24, Nkleg=160, Nkf=48),
        fused_grid_kwargs=dict(cluster_width=0.005, n_cluster=24, Np=160, Nk=64),
        # JAX's map_coordinates supports only order 0/1 (no cubic) -- real
        # ceiling for a curved function independent of node density. Local
        # tricubic interpolation (_interp_grid_cubic, same node set, no extra
        # training cost) roughly HALVED the error on isolated straddling-
        # threshold triangles (check_binning_isolate_error.py, e.g. straddle
        # k_c/mixed-corner: 4.99%->2.72%) -- the single biggest lever found.
        interp_order=3,
        # build_fused_coarse_grid's signature has no `Nkleg` (it's `Nk`/`Np`/`Nx`) --
        # kept separate above so the ab-only `Nkleg` override doesn't leak into it
        # as an unexpected keyword.
        pack_kwargs=lambda p: dict(p),
    ),
    'mu_OmDE': dict(
        model='HDKI', mg_variant='mu_OmDE', scale_bins=False,
        params=dict(mu0=0.5),
        # near the box edge (bounds: mu0 in [-0.5,1.5])
        extreme_params=dict(mu0=1.45),
        param_bounds={'mu0': (-0.5, 1.5)},
        k_attractors=(), budget=4,
        pack_kwargs=lambda p: dict(mu0=p['mu0']),
    ),
}
CFG = MODEL_CONFIGS[MODEL_NAME]
if os.environ.get("DIAG_EXTREME", "0") == "1":
    CFG = dict(CFG, params=CFG['extreme_params'])

TRACER = next(iter(gen.TRACERS))
TRACER_INFO = gen.TRACERS[TRACER]
Z = float(TRACER_INFO["z_pk"])
Om = 0.315192

KF_MIN, KF_MAX = 0.001, 0.5
KLEG_MIN, KLEG_MAX = 1e-5, 8.0


def build_pt(fkpt_approximation, ingredients_provider=None, beyond_eds=True):
    cosmo = gen.build_cosmology({})
    template = gen.DirectSpectrum2Template(
        z=Z, fiducial=gen.FIDUCIAL, engine="isitgr", cosmo=cosmo, with_now="peakaverage",
    )
    pt = gen.FKPTJAXPTSpectrum2Poles(
        k=gen.K_FIT, template=template, ells=gen.ELLS_PS,
        model=CFG['model'], mg_variant=CFG['mg_variant'], scale_bins=CFG['scale_bins'],
        beyond_eds=beyond_eds, fkpt_approximation=fkpt_approximation,
        use_numba=gen.USE_NUMBA,
        include_neutrino_corrections=False,
        mg_params_override=dict(CFG['params']),
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
    return gen.compile_calculator(theory), pt


def P_from_params(params):
    kind = _resolve_mg_kind(CFG['model'], CFG['mg_variant'])
    return bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=kind, scale_bins=CFG['scale_bins'],
                                  **CFG['pack_kwargs'](params))


print(f"MODEL: {MODEL_NAME}  ENGINE: {ENGINE_NAME}  params={CFG['params']}")
print("=" * 78)
print("LIVE pt")
print("=" * 78)
pipeline_live, pt_live = build_pt(False)
poles_live = np.asarray(pipeline_live())
jax.block_until_ready(poles_live)

k_lin = np.asarray(pt_live.template.k)
pk_lin = np.asarray(pt_live.template.pk_dd)
# pt.fk lives on the internal `kout` grid (table_w[0], Nk_kernel points), NOT template.k
# (a much finer Boltzmann/FFTLog-only grid) -- use its own matching k-axis.
k_fk = np.asarray(pt_live._table_w[0])
fk_live = np.asarray(pt_live.fk)
f0_live = float(pt_live.f0)

print()
print("=" * 78)
_engine_kwargs = CFG.get('engine_kwargs', ENGINE_CONFIGS[ENGINE_NAME])
print(f"Training ingredients provider (engine={ENGINE_NAME}, budget={CFG['budget']}, "
      f"k_attractors={CFG['k_attractors']}, engine_kwargs={_engine_kwargs})")
print("=" * 78)
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + Z)))
t0 = time.perf_counter()
provider = abi.build_ingredients_provider(
    kf_min=KF_MIN, kf_max=KF_MAX, kleg_min=KLEG_MIN, kleg_max=KLEG_MAX,
    param_bounds=CFG['param_bounds'], P_from_params=P_from_params,
    xnow=xnow, xstop=xstop, budget=CFG['budget'], k_attractors=CFG['k_attractors'],
    engine=ENGINE_NAME, engine_kwargs=_engine_kwargs, space_kwargs=CFG.get('space_kwargs'),
    ab_grid_kwargs=CFG.get('grid_kwargs'),
    fused_grid_kwargs=CFG.get('fused_grid_kwargs', CFG.get('grid_kwargs')),
    interp_order=CFG.get('interp_order', 1),
)
print(f"train time: {time.perf_counter()-t0:.2f}s")

print()
print("=" * 78)
print("EMULATED pt (fkpt_emulation)")
print("=" * 78)
pipeline_emu, pt_emu = build_pt(False, ingredients_provider=provider)
poles_emu = np.asarray(pipeline_emu())
jax.block_until_ready(poles_emu)

print()
print("=" * 78)
print("fkpt_approximation=True (context)")
print("=" * 78)
pipeline_approx, _ = build_pt(True)
poles_approx = np.asarray(pipeline_approx())

print()
print("=" * 78)
print("EdS (beyond_eds=False -- no MG beyond-EdS kernel correction at all)")
print("=" * 78)
pipeline_eds, _ = build_pt(True, beyond_eds=False)
poles_eds = np.asarray(pipeline_eds())

# --- Raw ingredients on the REAL production Q-loop/R-loop grid ---
print()
print("=" * 78)
print("Raw ingredients on the real production grid")
print("=" * 78)
Nk_kernel = int(min(len(gen.K_FIT), 120))
FKPT_KMIN = float(min(1e-3, float(np.min(gen.K_FIT))))
FKPT_KMAX = float(max(1.0, float(np.max(gen.K_FIT))))
k_ext = jnp.geomspace(1e-5, 10.0, 500)
init_data = setup_kfunctions(k_in=np.asarray(k_ext), kmin=FKPT_KMIN, kmax=FKPT_KMAX,
                              Nk=Nk_kernel, nquadSteps=300, NQ=10, NR=10)
calculator = JaxCalculator()
calculator.initialize(init_data)
k_ext_q = calculator.logk_grid_jax
r_q = calculator.r_jax
x_q = calculator.x_jax
y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
q_loop_Q = r_q * k_ext_q
kminus_Q = k_ext_q * y_q
r_r = calculator.r_r_jax
x_r = calculator.x_r_jax
q_loop_R = r_r * k_ext_q

P = P_from_params(CFG['params'])
f0_jax = jnp.asarray(f0_live, dtype=jnp.float64)
(A_Q_live, B_Q_live, *_, Af_live, Apf_live, CFD3_live, CFD3p_live) = mgk.I1udd1_and_P13_grid(
    k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P, float(xnow), xstop,
    f0_jax, solver='rk4', n_steps=64)
ingredients_fn = provider.bind(**CFG['params'])
(A_Q_emu, B_Q_emu, *_, Af_emu, Apf_emu, CFD3_emu, CFD3p_emu) = ingredients_fn(
    k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, f0_jax)
jax.block_until_ready((A_Q_live, Af_live, CFD3_live, A_Q_emu, Af_emu, CFD3_emu))

# --- Multipoles ---
poles_live_d = gen.split_multipoles_ps(poles_live)
poles_emu_d = gen.split_multipoles_ps(poles_emu)
poles_approx_d = gen.split_multipoles_ps(poles_approx)
poles_eds_d = gen.split_multipoles_ps(poles_eds)
P0_live, P2_live = poles_live_d[0], poles_live_d[2]
P0_emu, P2_emu = poles_emu_d[0], poles_emu_d[2]
P0_approx, P2_approx = poles_approx_d[0], poles_approx_d[2]
P0_eds, P2_eds = poles_eds_d[0], poles_eds_d[2]
k = gen.K_FIT
rel_P0 = np.abs(P0_emu - P0_live) / (np.abs(P0_live) + 1e-30)
rel_P2 = np.abs(P2_emu - P2_live) / (np.abs(P2_live) + 1e-30)
print(f"kP0: max={rel_P0.max():.3e} median={np.median(rel_P0):.3e}")
print(f"kP2: max={rel_P2.max():.3e} median={np.median(rel_P2):.3e}")

# ---------------------------------------------------------------------------
# One 3x3 figure: ingredients (row 0-1) + final products (row 2)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(16, 13))

ax = axes[0, 0]
ax.plot(k_lin, pk_lin, color='C0', lw=1.5)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_title('Linear P(k) [reference: live in both routes]')
ax.set_xlabel('k [h/Mpc]')

ax = axes[0, 1]
ax.plot(k_fk, fk_live / f0_live, color='C0', lw=1.5)
ax.set_xscale('log')
ax.set_title(f'Growth f(k)/f0 [reference: live in both routes] (f0={f0_live:.4f})')
ax.set_xlabel('k [h/Mpc]')

scatter_data = [
    ('A (Q-ordering)', np.asarray(A_Q_live).ravel(), np.asarray(A_Q_emu).ravel(), axes[0, 2]),
    ('B (Q-ordering)', np.asarray(B_Q_live).ravel(), np.asarray(B_Q_emu).ravel(), axes[1, 0]),
    ('A_fused', np.asarray(Af_live).ravel(), np.asarray(Af_emu).ravel(), axes[1, 1]),
    ('CFD3', np.asarray(CFD3_live).ravel(), np.asarray(CFD3_emu).ravel(), axes[1, 2]),
]
for name, live_vals, emu_vals, ax in scatter_data:
    rel = np.abs(emu_vals - live_vals) / (np.abs(live_vals) + 1e-30)
    lo, hi = np.percentile(live_vals, [0.5, 99.5])
    ax.scatter(live_vals, emu_vals, s=3, alpha=0.3, color='C1')
    ax.plot([lo, hi], [lo, hi], color='k', lw=1, ls='--', label='y=x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('exact'); ax.set_ylabel('emulated')
    ax.set_title(f'{name}: max|rel err|={rel.max():.2e}, median={np.median(rel):.2e}')
    ax.legend(fontsize=8)

for ax, (label, P_live_, P_emu_, P_approx_, P_eds_) in zip(
    [axes[2, 0], axes[2, 1]],
    [('kP0(k)', P0_live, P0_emu, P0_approx, P0_eds), ('kP2(k)', P2_live, P2_emu, P2_approx, P2_eds)]
):
    ax.plot(k, k * P_eds_, color='C3', lw=1.0, ls='-.', label='EdS (beyond_eds=False)')
    ax.plot(k, k * P_approx_, color='C2', lw=1.0, ls=':', label='fkpt_approx (fkpt_approximation=True)')
    ax.plot(k, k * P_live_, color='C0', lw=1.8, label='fkpt_full (live, fkpt_approximation=False)')
    ax.plot(k, k * P_emu_, color='C1', lw=1.4, ls='--', label='emulated fkpt_full (fkpt_emulation)')
    ax.set_title(f'FINAL PRODUCT: {label}')
    ax.set_xlabel('k [h/Mpc]')
    ax.legend(fontsize=7)

ax = axes[2, 2]
ax.plot(k, rel_P0, color='C0', label=f'P0 (max={rel_P0.max():.2e})')
ax.plot(k, rel_P2, color='C3', label=f'P2 (max={rel_P2.max():.2e})')
ax.set_yscale('log')
ax.set_xlabel('k [h/Mpc]')
ax.set_title('FINAL PRODUCT: relative error')
ax.legend(fontsize=8)

EXTREME_TAG = "_extreme" if os.environ.get("DIAG_EXTREME", "0") == "1" else ""
fig.suptitle(f"Full diagnostic -- {CFG['model']}/{CFG['mg_variant']}, engine={ENGINE_NAME}"
             f"{' (EXTREME point)' if EXTREME_TAG else ''}, {CFG['params']}", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outpath = OUTDIR / f"full_diagnostic_{MODEL_NAME}_{ENGINE_NAME}{EXTREME_TAG}.png"
fig.savefig(outpath, dpi=130)
print(f"\nsaved {outpath}")
print("DONE")
