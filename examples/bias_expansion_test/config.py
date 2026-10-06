"""Shared fiducial constants for the BGS scale-dependent bias-expansion test.

This is a standalone diagnostic/stress test (Desjacques, Jeong & Schmidt
arXiv:1611.09787, Sec. 8.3): it does not modify anything under ``src/fkptjax``.
All cosmology/tracer constants here mirror the production BGS setup used in
``/n/home12/cgarciaquintero/DESI/synthetic/scripts/generate_noiseless_synthetic_data.py``
(copied, not imported, to keep this test self-contained).
"""

import numpy as np

# ----------------------------------------------------------------------
# Fiducial cosmology (matches generate_noiseless_synthetic_data.py)
# ----------------------------------------------------------------------
H0 = 0.6736 * 100.0
H = 0.6736
OMBH2 = 0.02237
OMCH2 = 0.12
AS = 2.083e-9
NS = 0.9649
MNU = 0.06
TAU_REIO = 0.0544

# Omega_m,0 / Omega_Lambda,0 used directly by fkptjax's growth ODE (mg_jax.py).
OM = 0.31519
OL = 1.0 - OM

# FKPT growth-ODE start: eta = ln(a) at z_init ~ 49 (matches jax_ode/production).
XNOW = -3.912023

# ----------------------------------------------------------------------
# BGS tracer (matches TRACERS["BGS"] in generate_noiseless_synthetic_data.py)
# ----------------------------------------------------------------------
Z_EFF = 0.295
NBAR_BGS = 9.59e-4       # (Mpc/h)^-3
VEFF_BGS = 1.207077e9    # (Mpc/h)^3
B1_BGS = 1.5
B2_BGS = -0.5247206065

# Standard-basis nuisance defaults (prior_basis="standard" FOLPS/FKPT order),
# all zero: a purely analytic bias-expansion baseline, nothing physical behind
# a nonzero EFT counterterm or shot-noise excess here.
STANDARD_NUISANCE_DEFAULTS = {
    "bs": 0.0,
    "b3": 0.0,
    "alpha0": 0.0,
    "alpha2": 0.0,
    "alpha4": 0.0,
    "ct": 0.0,
    "sn0": 0.0,
    "sn2": 0.0,
    "X_FoG": 0.0,
}

# ----------------------------------------------------------------------
# Hu-Sawicki f(R) MG points: F4/F5/F6
# ----------------------------------------------------------------------
N_HS = 1.0
BETA2_HS = 1.0 / 6.0

MG_POINTS = {
    "F4": dict(fR0_HS=1.0e-4, n_HS=N_HS, beta2=BETA2_HS),
    "F5": dict(fR0_HS=1.0e-5, n_HS=N_HS, beta2=BETA2_HS),
    "F6": dict(fR0_HS=1.0e-6, n_HS=N_HS, beta2=BETA2_HS),
}

# Tracer "formation" redshifts used to build the Schmidt bias envelope.
Z_STAR = (1.0, 2.0, 3.0, 5.0)

# ----------------------------------------------------------------------
# k grids
# ----------------------------------------------------------------------
KMIN, KMAX = 0.01, 0.20

# Dense, cheap (analytic-growth-only) grid for plots 1-3.
K_PLOT = np.geomspace(KMIN, KMAX, 200)

# Coarser, uniform grid for anything that needs a full-kernel PT evaluation
# (expensive) and/or the analytic covariance (needs uniform bins): bin edges
# from KMIN to KMAX with dk=0.005, bin centers below.
DK_FIT = 0.005
_n_bins = int(round((KMAX - KMIN) / DK_FIT))
K_FIT = KMIN + DK_FIT / 2.0 + DK_FIT * np.arange(_n_bins)

# k used as a numerical proxy for k -> 0 (deep in the model's GR-like limit).
K_ZERO_PROXY = 1.0e-4

# Internal FKPT kernel-table k-range (must safely cover K_PLOT/K_FIT and the
# PT loop integral's own internal range); matches examples/compute_mg_multipoles.ipynb.
KERNEL_KMIN, KERNEL_KMAX = 1.0e-3, 1.0

OUTDIR = "outputs"
FIGDIR = "figures"
