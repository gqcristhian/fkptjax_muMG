import os
os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/synthetic/scripts")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generate_noiseless_synthetic_data_psbs_fkptbs as gen  # noqa: E402

print("tracers available:", list(gen.TRACERS))



# ---------------------------------------------------------------------------
# Shared machinery: build pt + PS/BS theories directly (mirrors the generator
# script's own build_theory_ps / bispectrum_data_vector_fkpt, generalized
# over model/variant/route flags). No emulators anywhere in this notebook.
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "mu_OmDE": dict(model="HDKI", mg_variant="mu_OmDE", scale_bins=False),
    "BZ_Mass": dict(model="HDKI", mg_variant="BZ_Mass", scale_bins=False),
    "binning_z": dict(model="PHENOM", mg_variant="binning", scale_bins=False),
    "binning_zk": dict(model="PHENOM", mg_variant="binning", scale_bins=True),
}

# (label, kwargs) -- order matters for legend/plot layering.
ROUTES = [
    ("EdS nonu", dict(beyond_eds=False, fkpt_approximation=True, include_neutrino_corrections=False)),
    ("fkpt approx nonu", dict(beyond_eds=True, fkpt_approximation=True, include_neutrino_corrections=False)),
    ("fkpt full nonu", dict(beyond_eds=True, fkpt_approximation=False, include_neutrino_corrections=False)),
    ("fkpt full nu", dict(beyond_eds=True, fkpt_approximation=False, include_neutrino_corrections=True)),
]
ROUTE_STYLE = {
    "EdS nonu": ("C3", "-."),
    "fkpt approx nonu": ("C2", ":"),
    "fkpt full nonu": ("C0", "-"),
    "fkpt full nu": ("C1", "--"),
}


def build_pt_and_ps(cfg, tracer, params, *, fkpt_approximation, beyond_eds, include_neutrino_corrections):
    tracer_info = gen.TRACERS[tracer]
    z = float(tracer_info["z_pk"])
    cosmo = gen.build_cosmology({})
    template = gen.DirectSpectrum2Template(
        z=z, fiducial=gen.FIDUCIAL, engine="isitgr", cosmo=cosmo, with_now="peakaverage",
    )
    pt = gen.FKPTJAXPTSpectrum2Poles(
        k=gen.K_FIT, template=template, ells=gen.ELLS_PS,
        model=cfg["model"], mg_variant=cfg.get("mg_variant"), scale_bins=cfg.get("scale_bins", False),
        beyond_eds=beyond_eds, fkpt_approximation=fkpt_approximation,
        use_numba=gen.USE_NUMBA,
        include_neutrino_corrections=include_neutrino_corrections,
        mg_params_override=dict(params),
        growth_source="ode",  # the ONE route where include_neutrino_corrections has any effect at all
    )
    theory = gen.FKPTJAXTracerSpectrum2Poles(
        k=gen.K_FIT, pt=pt, ells=gen.ELLS_PS, prior_basis=gen.PRIOR_BASIS,
        nbar=float(tracer_info["nbar"]), damping=gen.DAMPING, damping_method=gen.DAMPING_METHOD,
        tracers=tracer,
    )
    nuisance_values = {
        "b1": float(tracer_info["b1"]), "b2": float(tracer_info["b2"]),
        **gen.STANDARD_NUISANCE_DEFAULTS,
    }
    gen._fix_tracer_parameters(theory, nuisance_values)
    return gen.compile_calculator(theory), pt


def build_bs(tracer, pt, ell_triplet):
    tracer_info = gen.TRACERS[tracer]
    k_pairs = np.column_stack([gen.K_FIT, gen.K_FIT])
    theory = gen.FKPTJAXTracerSpectrum3Poles(
        k=k_pairs, pt=pt, ells=[ell_triplet], prior_basis=gen.PRIOR_BASIS,
        nbar=float(tracer_info["nbar"]), damping=gen.DAMPING, tracers=tracer,
    )
    nuisance_values = {
        "b1": float(tracer_info["b1"]), "b2": float(tracer_info["b2"]),
        **gen.BS_NUISANCE_DEFAULTS,
    }
    gen._fix_tracer_parameters(theory, nuisance_values)
    return gen.compile_calculator(theory)


def compute_route(cfg, tracer, params, route_kwargs):
    ps_pipeline, pt = build_pt_and_ps(cfg, tracer, params, **route_kwargs)
    raw_ps = np.asarray(ps_pipeline(), dtype="f8")
    if not np.all(np.isfinite(raw_ps)):
        raise RuntimeError(f"Non-finite P0/P2 for route {route_kwargs}")
    poles = gen.split_multipoles_ps(raw_ps)
    b000 = np.asarray(build_bs(tracer, pt, gen.BS_ELL_TRIPLETS["B000"])(), dtype="f8").reshape(-1)
    b202 = np.asarray(build_bs(tracer, pt, gen.BS_ELL_TRIPLETS["B202"])(), dtype="f8").reshape(-1)
    if not (np.all(np.isfinite(b000)) and np.all(np.isfinite(b202))):
        raise RuntimeError(f"Non-finite B000/B202 for route {route_kwargs}")
    return dict(k=gen.K_FIT, P0=poles[0], P2=poles[2], B000=b000, B202=b202)


def run_all_routes(model_key, params, tracer):
    cfg = MODEL_CONFIGS[model_key]
    results = {}
    for label, route_kwargs in ROUTES:
        try:
            results[label] = compute_route(cfg, tracer, params, route_kwargs)
        except Exception as exc:
            print(f"[{model_key}] route {label!r} FAILED: {exc}")
            results[label] = None
    return results


def plot_comparison(model_key, params, tracer):
    results = run_all_routes(model_key, params, tracer)
    have = {name: r for name, r in results.items() if r is not None}
    if "EdS nonu" not in have or "fkpt full nonu" not in have:
        print(f"[{model_key}] missing baseline route(s), cannot plot.")
        return results

    fig, axes = plt.subplots(2, 4, figsize=(20, 7), gridspec_kw={"height_ratios": [3, 1]})
    quantities = [("P0", 1, "k P0(k)"), ("P2", 1, "k P2(k)"),
                  ("B000", 2, "k^2 B000"), ("B202", 2, "k^2 B202")]

    for col, (qty, power, title) in enumerate(quantities):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for label, _ in ROUTES:
            r = have.get(label)
            if r is None:
                continue
            color, ls = ROUTE_STYLE[label]
            ax_top.plot(r["k"], r["k"] ** power * r[qty], color=color, ls=ls, label=label, lw=1.6)
        ax_top.set_title(title)
        ax_top.set_xlabel("k [h/Mpc]")
        if col == 0:
            ax_top.legend(fontsize=7)

        eds = have["EdS nonu"][qty]
        full_nonu = have["fkpt full nonu"][qty]
        mg_effect = np.abs(full_nonu - eds) / (np.abs(eds) + 1e-30)
        ax_bot.plot(have["EdS nonu"]["k"], mg_effect, color="C0", label="MG: |full nonu - EdS|/EdS")
        if "fkpt full nu" in have:
            full_nu = have["fkpt full nu"][qty]
            nu_effect = np.abs(full_nu - full_nonu) / (np.abs(full_nonu) + 1e-30)
            ax_bot.plot(have["fkpt full nonu"]["k"], nu_effect, color="C1", label="nu: |full nu - full nonu|/full nonu")
        ax_bot.set_yscale("log")
        ax_bot.set_xlabel("k [h/Mpc]")
        if col == 0:
            ax_bot.legend(fontsize=7)

    fig.suptitle(f"{model_key} -- tracer={tracer}, params={params}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/n/home12/cgarciaquintero/DESI/synthetic/notebooks/verify_neutrino_bzmass_smoketest.png', dpi=100)
    return results



TRACER = "LRG2"
params = dict(mu0=0.5)
_ = plot_comparison("mu_OmDE", params, TRACER)

TRACER = "LRG2"
params = dict(mu_kinf_BZmass=1.2, lambda_a_BZmass=1.0, lambda_dS_BZmass=1.0)
_ = plot_comparison("BZ_Mass", params, TRACER)
