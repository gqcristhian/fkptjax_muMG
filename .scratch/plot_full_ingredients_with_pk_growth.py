"""
Complete picture for PHENOM/binning at the "reference" point: linear P(k) and
growth f(k) as REFERENCE panels (live in both routes -- NOT emulated by
ab_ingredients.py, shown here just for context/completeness), plus scatter
plots of every A/B/A_fused/CFD3 value on the ACTUAL production Q-loop/R-loop
grid (not a synthetic (r,x) sweep) -- live vs emulated, all points at once,
which gives the true overall accuracy distribution rather than a
cherry-picked slice.
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
from fkptjax import MG_kernels as mgk  # noqa: E402
from fkptjax.util import setup_kfunctions  # noqa: E402
from fkptjax.calculate_jax import JaxCalculator  # noqa: E402
from fkptjax.kfuncs_to_tables import _resolve_mg_kind  # noqa: E402

OUTDIR = Path("/n/home12/cgarciaquintero/DESI/synthetic/notebooks/binning_ingredients_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

TRACER = next(iter(gen.TRACERS))
TRACER_INFO = gen.TRACERS[TRACER]
Z = float(TRACER_INFO["z_pk"])
PARAMS = dict(mu1=0.8, mu2=1.3, mu3=0.9, mu4=1.1)

KF_MIN, KF_MAX = 0.001, 0.5
KLEG_MIN, KLEG_MAX = 1e-5, 8.0
K_ATTRACTORS = (0.01, 0.1, 0.2)
PARAM_BOUNDS = {'mu1': (0.5, 1.5), 'mu2': (0.5, 1.5), 'mu3': (0.5, 1.5), 'mu4': (0.5, 1.5)}


def build_pt(fkpt_approximation, ingredients_provider=None):
    cosmo = gen.build_cosmology({})
    template = gen.DirectSpectrum2Template(
        z=Z, fiducial=gen.FIDUCIAL, engine="isitgr", cosmo=cosmo, with_now="peakaverage",
    )
    pt = gen.FKPTJAXPTSpectrum2Poles(
        k=gen.K_FIT, template=template, ells=gen.ELLS_PS,
        model="PHENOM", mg_variant="binning", scale_bins=True,
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
    return gen.compile_calculator(theory), pt


def P_from_params(params):
    kind = _resolve_mg_kind('PHENOM', 'binning')
    Om = 0.315192
    return bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=kind, scale_bins=True, **params)


print("=" * 78)
print("LIVE pt (also gives us template.pk_dd / pt.fk / pt.f0 -- unaffected by")
print("the ingredients route either way)")
print("=" * 78)
pipeline_live, pt_live = build_pt(False)
_ = pipeline_live()
jax.block_until_ready(_)

k_lin = np.asarray(pt_live.template.k)
pk_lin = np.asarray(pt_live.template.pk_dd)
fk_live = np.asarray(pt_live.fk)
f0_live = float(pt_live.f0)
print(f"f0 = {f0_live:.6f}")

print()
print("=" * 78)
print("Training ingredients provider (v4: clustered + realistic kleg range/resolution)")
print("=" * 78)
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + Z)))
f0 = jnp.asarray(0.8)
t0 = time.perf_counter()
provider = abi.build_ingredients_provider(
    kf_min=KF_MIN, kf_max=KF_MAX, kleg_min=KLEG_MIN, kleg_max=KLEG_MAX,
    param_bounds=PARAM_BOUNDS, P_from_params=P_from_params,
    xnow=xnow, xstop=xstop, f0=f0, budget=2, k_attractors=K_ATTRACTORS,
)
print(f"train time: {time.perf_counter()-t0:.2f}s")

print()
print("=" * 78)
print("Building the REAL production Q-loop/R-loop grid (matches what pt itself uses)")
print("=" * 78)
Nk_kernel = int(min(len(gen.K_FIT), 120))
FKPT_KMIN = float(min(1e-3, float(np.min(gen.K_FIT))))
FKPT_KMAX = float(max(1.0, float(np.max(gen.K_FIT))))
print(f"Nk_kernel={Nk_kernel}, kmin={FKPT_KMIN}, kmax={FKPT_KMAX}")

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

P = P_from_params(PARAMS)

print("Live (genuine ODE) ingredients on this grid...")
(A_Q_live, B_Q_live, ApQ_live, BpQ_live, A_N_live, B_N_live, ApN_live, BpN_live,
 A_RQ_live, B_RQ_live, ApRQ_live, BpRQ_live, Af_live, Apf_live, CFD3_live, CFD3p_live) = (
    mgk.I1udd1_and_P13_grid(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R,
                             P, float(xnow), xstop, jnp.asarray(f0, dtype=jnp.float64),
                             solver='rk4', n_steps=64)
)

print("Emulated (fkpt_emulation) ingredients on this grid...")
ingredients_fn = provider.bind(**PARAMS)
(A_Q_emu, B_Q_emu, ApQ_emu, BpQ_emu, A_N_emu, B_N_emu, ApN_emu, BpN_emu,
 A_RQ_emu, B_RQ_emu, ApRQ_emu, BpRQ_emu, Af_emu, Apf_emu, CFD3_emu, CFD3p_emu) = (
    ingredients_fn(k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R)
)

jax.block_until_ready((A_Q_live, Af_live, CFD3_live, A_Q_emu, Af_emu, CFD3_emu))

# ---------------------------------------------------------------------------
# Plot: 2 reference panels (P_lin, growth) + 4 scatter panels (A, B, A_fused, CFD3)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

ax = axes[0, 0]
ax.plot(k_lin, pk_lin, color='C0', lw=1.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title('Linear P(k) -- template output\n(NOT emulated: live in both routes)')
ax.set_xlabel('k [h/Mpc]')

ax = axes[0, 1]
ax.plot(k_lin, fk_live / f0_live, color='C0', lw=1.5)
ax.set_xscale('log')
ax.set_title(f'Growth f(k)/f0 -- fkptjax ODE (f0={f0_live:.4f})\n(NOT emulated: live in both routes)')
ax.set_xlabel('k [h/Mpc]')

scatter_data = [
    ('A (Q-ordering)', np.asarray(A_Q_live).ravel(), np.asarray(A_Q_emu).ravel()),
    ('B (Q-ordering)', np.asarray(B_Q_live).ravel(), np.asarray(B_Q_emu).ravel()),
    ('A_fused', np.asarray(Af_live).ravel(), np.asarray(Af_emu).ravel()),
    ('CFD3', np.asarray(CFD3_live).ravel(), np.asarray(CFD3_emu).ravel()),
]
scatter_axes = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
for ax, (name, live_vals, emu_vals) in zip(scatter_axes, scatter_data):
    rel = np.abs(emu_vals - live_vals) / (np.abs(live_vals) + 1e-30)
    lo, hi = np.percentile(live_vals, [0.5, 99.5])
    ax.scatter(live_vals, emu_vals, s=3, alpha=0.3, color='C1')
    ax.plot([lo, hi], [lo, hi], color='k', lw=1, ls='--', label='y=x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('exact')
    ax.set_ylabel('emulated')
    ax.set_title(f'{name}: max|rel err|={rel.max():.2e}\nmedian={np.median(rel):.2e} '
                 f'(n={live_vals.size} pts, real Q/R-loop grid)')
    ax.legend(fontsize=8)

fig.suptitle(f"Full ingredient picture -- PHENOM/binning, {PARAMS}", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outpath = OUTDIR / "full_ingredients_with_pk_growth_reference.png"
fig.savefig(outpath, dpi=130)
print(f"\nsaved {outpath}")
print("DONE")
