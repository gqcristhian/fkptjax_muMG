"""Shared fiducial constants and parameter list for the EdS / bEdS-fkPT / bEdS-full
Fisher-forecast comparison ("Fisher_bEdS_FoM").

Reuses the exact BGS tracer/cosmology/covariance recipe already validated in
``examples/bias_expansion_test`` (same fiducial CAMB-ISiTGR cosmology, same
``thecov`` Gaussian covariance), with a nonzero fiducial ``mu0=0.5`` (HDKI/
mu_OmDE: mu(a) = 1 + mu0 * Omega_DE(a)/Omega_Lambda) as the MG signal whose
recovery this test is about.

This is a standalone diagnostic/stress test -- nothing under ``src/fkptjax``
is modified.

PLACEHOLDER PRIORS: no repo precedent exists for standard prior widths on the
bias/EFT/stochastic nuisance parameters (only ``mu0`` has a documented prior
elsewhere in this codebase: ``desilike``'s FKPT theory class uses
``uniform[-3,1]``). The priors below are broad, weakly-informative Gaussians
chosen only to keep every Fisher submatrix invertible -- every value is
listed explicitly here so it can be retuned to a real analysis's priors.
"""

import numpy as np

# ----------------------------------------------------------------------
# Fiducial cosmology (matches bias_expansion_test/config.py exactly)
# ----------------------------------------------------------------------
H0_FID = 67.36
OMBH2_FID = 0.02237
OMCH2_FID = 0.12
AS_FID = 2.083e-9
NS_FID = 0.9649
MNU = 0.06          # fixed, not varied
TAU_REIO = 0.0544


def Om_of(H0=H0_FID, ombh2=OMBH2_FID, omch2=OMCH2_FID, mnu=MNU):
    """Omega_m,0 consistent with a given (H0, ombh2, omch2, mnu) -- fkptjax's
    growth ODE (mg_jax.py) takes Om/Ol directly, so this must be recomputed
    for every cosmological finite-difference step, not held fixed at the
    fiducial value.
    """
    h = H0 / 100.0
    omnuh2 = mnu / 93.14  # single massive-neutrino approx, standard relation
    return (omch2 + ombh2 + omnuh2) / h**2


OM_FID = Om_of()
OL_FID = 1.0 - OM_FID

# FKPT growth-ODE start: eta = ln(a) at z_init ~ 49.
XNOW = -3.912023

# ----------------------------------------------------------------------
# BGS tracer (matches bias_expansion_test/config.py / generate_noiseless_synthetic_data.py)
# ----------------------------------------------------------------------
Z_EFF = 0.295
NBAR_BGS = 9.59e-4       # (Mpc/h)^-3
VEFF_BGS = 1.207077e9    # (Mpc/h)^3

# ----------------------------------------------------------------------
# MG model + target parameter: flip this and rerun to get an independent
# result set (outputs/figures are suffixed by MODEL_CHOICE, so nothing gets
# overwritten between runs).
#   "mu_omde" -- HDKI/mu_OmDE, mu(a) = 1 + mu0*Omega_DE(a)/Omega_Lambda.
#                Scale-INDEPENDENT: fkPT-approx and full kernels are
#                mathematically identical here (see compute_mg_multipoles.ipynb's
#                beyond-EdS section), so this variant cannot by itself test
#                whether fkPT-approx mis-estimates sigma(mu0) -- only EdS vs
#                {approx,full} is a meaningful comparison for this model.
#   "hs"      -- Hu-Sawicki f(R), fR0_HS. Genuinely scale-dependent (chameleon),
#                so fkPT-approx and full kernels really do differ (see
#                check_G22_vs_Pell_correction.ipynb) -- this is the variant
#                that can actually test the central question.
# ----------------------------------------------------------------------
MODEL_CHOICE = "hs"

if MODEL_CHOICE == "mu_omde":
    MG_MODEL = "HDKI"
    MG_VARIANT = "mu_OmDE"
    MG_PARAM_NAME = "mu0"
    MG_PARAM_FID = 0.5
    MG_PARAM_STEP = 0.01
    MG_PARAM_PRIOR_SIGMA = 10.0
    MG_EXTRA_KWARGS = {}
elif MODEL_CHOICE == "hs":
    MG_MODEL = "HS"
    MG_VARIANT = None
    MG_PARAM_NAME = "fR0_HS"
    MG_PARAM_FID = -1.0e-5   # F5, matching check_G22_vs_Pell_correction.ipynb
    # 10% of |fid|, not 1%: a plain 1% step (1e-7) risks being dominated by the
    # full-kernel ODE solver's own tolerance rather than the real derivative.
    MG_PARAM_STEP = 1.0e-6
    MG_PARAM_PRIOR_SIGMA = 1.0e-4  # placeholder, order of the F4-F6 range
    MG_EXTRA_KWARGS = dict(beta2=1.0 / 6.0, n_HS=1.0)
else:
    raise ValueError(f"Unknown MODEL_CHOICE={MODEL_CHOICE!r}")

# ----------------------------------------------------------------------
# Nuisance fiducial values (standard FOLPS basis, fkptjax.rsd.FKPT_BIAS_ORDER)
# ----------------------------------------------------------------------
B1_FID = 1.5
B2_FID = -0.5247206065
BS2_FID = 0.0
B3NL_FID = 0.0
ALPHA0_FID = 0.0
ALPHA2_FID = 0.0
ALPHA4_FID = 0.0
CTILDE_FID = 0.0
ALPHA0SHOT_FID = 0.0
ALPHA2SHOT_FID = 0.0

BASE_BIAS = dict(
    b1=B1_FID, b2=B2_FID, bs2=BS2_FID, b3nl=B3NL_FID,
    alpha0=ALPHA0_FID, alpha2=ALPHA2_FID, alpha4=ALPHA4_FID, ctilde=CTILDE_FID,
    alpha0shot=ALPHA0SHOT_FID, alpha2shot=ALPHA2SHOT_FID, X_FoG_p=0.0,
)

# ----------------------------------------------------------------------
# Parameter list: name -> (fiducial, central-difference step, prior sigma, group)
#
# Groups (drive the staged-marginalization stages (a)-(e)):
#   "mgparam"    -- stage (a); whichever parameter MG_PARAM_NAME points to
#   "cosmo"      -- stage (b) adds this to (a)
#   "bias"       -- stage (c) adds this to (b)
#   "counterterm"-- stage (d) adds this to (c)
#   "stochastic" -- stage (e) adds this to (d) -> "full nuisance model"
#
# Steps: central (2-point) differences, ~1% relative step for parameters with
# a nonzero, O(1)-or-larger fiducial value; a fixed absolute step for
# parameters fiducially at/near zero (bs2, b3nl, every counterterm/stochastic
# term) since a relative step is degenerate there. All absolute-step choices
# are placeholders -- see the module docstring.
# ----------------------------------------------------------------------
PARAMS = {
    # name              fiducial        step      prior_sigma  group
    MG_PARAM_NAME: dict(fid=MG_PARAM_FID, step=MG_PARAM_STEP,
                         prior_sigma=MG_PARAM_PRIOR_SIGMA, group="mgparam"),
    "H0":         dict(fid=H0_FID,      step=0.6736,   prior_sigma=5.0,   group="cosmo"),
    "ombh2":      dict(fid=OMBH2_FID,   step=2.237e-4, prior_sigma=5.0e-4, group="cosmo"),
    "omch2":      dict(fid=OMCH2_FID,   step=1.2e-3,   prior_sigma=2.0e-2, group="cosmo"),
    "logA":       dict(fid=float(np.log(1.0e10 * AS_FID)), step=2.0e-2, prior_sigma=0.5, group="cosmo"),
    "ns":         dict(fid=NS_FID,      step=9.649e-3, prior_sigma=2.0e-2, group="cosmo"),
    "b1":         dict(fid=B1_FID,      step=0.015,    prior_sigma=1.0,   group="bias"),
    "b2":         dict(fid=B2_FID,      step=0.05,     prior_sigma=5.0,   group="bias"),
    "bs2":        dict(fid=BS2_FID,     step=0.5,      prior_sigma=5.0,   group="bias"),
    "b3nl":       dict(fid=B3NL_FID,    step=0.5,      prior_sigma=5.0,   group="bias"),
    "alpha0":     dict(fid=ALPHA0_FID,  step=1.0,      prior_sigma=50.0,  group="counterterm"),
    "alpha2":     dict(fid=ALPHA2_FID,  step=1.0,      prior_sigma=50.0,  group="counterterm"),
    "alpha4":     dict(fid=ALPHA4_FID,  step=1.0,      prior_sigma=50.0,  group="counterterm"),
    "ctilde":     dict(fid=CTILDE_FID,  step=1.0,      prior_sigma=50.0,  group="counterterm"),
    "alpha0shot": dict(fid=ALPHA0SHOT_FID, step=1.0,   prior_sigma=50.0,  group="stochastic"),
    "alpha2shot": dict(fid=ALPHA2SHOT_FID, step=1.0,   prior_sigma=50.0,  group="stochastic"),
}

GROUP_ORDER = ("mgparam", "cosmo", "bias", "counterterm", "stochastic")

STAGES = {
    f"(a) {MG_PARAM_NAME} only": ("mgparam",),
    "(b) + cosmological": ("mgparam", "cosmo"),
    "(c) + galaxy biases": ("mgparam", "cosmo", "bias"),
    "(d) + counterterms": ("mgparam", "cosmo", "bias", "counterterm"),
    "(e) full nuisance model": ("mgparam", "cosmo", "bias", "counterterm", "stochastic"),
}


def params_in_groups(groups):
    """Ordered parameter-name list for a tuple of group names."""
    names = []
    for name, spec in PARAMS.items():
        if spec["group"] in groups:
            names.append(name)
    # keep GROUP_ORDER, then insertion order within a group
    order_key = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(names, key=lambda n: order_key[PARAMS[n]["group"]])


ALL_PARAMS = params_in_groups(GROUP_ORDER)

# ----------------------------------------------------------------------
# Theory cases
# ----------------------------------------------------------------------
THEORIES = ("eds", "approx", "full")
THEORY_KWARGS = {
    "eds":    dict(beyond_eds=False, fkpt_approximation=True),   # fkpt_approximation irrelevant when beyond_eds=False
    "approx": dict(beyond_eds=True, fkpt_approximation=True),
    "full":   dict(beyond_eds=True, fkpt_approximation=False),
}

# ----------------------------------------------------------------------
# k grid: same fixed bin-center grid as bias_expansion_test (uniform, needed
# by thecov's covariance recipe).
# ----------------------------------------------------------------------
KMIN, KMAX = 0.02, 0.20
DK_FIT = 0.005
_n_bins = int(round((KMAX - KMIN) / DK_FIT))
K_FIT = KMIN + DK_FIT / 2.0 + DK_FIT * np.arange(_n_bins)

ELLS = (0, 2, 4)

# Internal FKPT kernel-table k-range and production-resolution grid (matches
# bias_expansion_test/full_kernel_pipeline.py, itself matching
# examples/check_bzmass_vs_sMGPT_full_v3.py's sMGPT-cross-checked settings).
KERNEL_KMIN, KERNEL_KMAX = 1.0e-3, 1.0
KERNEL_SETTINGS = dict(Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
                        rbao=104.0, pmax_bao=0.4, Np_bao=100)

OUTDIR = f"outputs_{MODEL_CHOICE}"
FIGDIR = f"figures_{MODEL_CHOICE}"
