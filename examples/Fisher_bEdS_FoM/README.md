# Fisher_bEdS_FoM

FKPTJAX Fisher-forecast test: **EdS vs bEdS-fkPT vs bEdS-full**.

**Goal**: test whether the EdS approximation (or the historical fkPT
large-scale-limit approximation) gives an unbiased recovery of the MG
parameter while underestimating its marginalized uncertainty, because
nuisance parameters/counterterms silently absorb the missing beyond-EdS
kernel physics -- i.e. whether the parameter-degeneracy geometry the
approximate theories imply is *wrong*, even if their central value looks
fine.

This is a standalone diagnostic/stress test -- nothing under `src/fkptjax` is
modified. It hand-rolls the Fisher matrix directly against `fkptjax`
(desilike's own Fisher/differentiation classes were removed in a refactor,
and its FKPT theory wrapper has no `fkpt_approximation` parameter and no
cosmological-parameter layer of its own), following the same
`build_state`/`poles_for_bias` caching pattern as
`examples/bias_expansion_test/full_kernel_pipeline.py`.

## Two model variants (`config.MODEL_CHOICE`)

- **`"mu_omde"`** -- HDKI/mu_OmDE, mu(a) = 1 + mu0\*Omega_DE(a)/Omega_Lambda,
  fiducial mu0=0.5. **Scale-independent**: fkPT-approx and full kernels are
  mathematically identical for this model (see
  `compute_mg_multipoles.ipynb`'s beyond-EdS section), so `bEdS-fkPT` and
  `bEdS-full` come out numerically identical here -- this variant can only
  test EdS vs {approx≡full}, not whether fkPT-approx itself matters.
- **`"hs"`** -- Hu-Sawicki f(R), fiducial fR0_HS=-1e-5 (F5). Genuinely
  scale-dependent (chameleon), so fkPT-approx and full kernels really do
  differ (see `check_G22_vs_Pell_correction.ipynb`) -- **this is the variant
  that actually stress-tests the central question**.

To (re)run a variant: set `MODEL_CHOICE` at the top of `config.py`, then

```
source ~/.bashrc && cosmodesi_new
cd examples/Fisher_bEdS_FoM
python run_all.py
```

Outputs land in `outputs_<MODEL_CHOICE>/summary.npz` and
`figures_<MODEL_CHOICE>/plot{1..5}_*.png` (suffixed so the two variants never
overwrite each other). `python replot.py` regenerates just the figures from
an already-saved `summary.npz` (cheap -- no fkptjax/CAMB calls), useful after
a `make_plots.py` styling tweak.

Fiducial cosmology/tracer/covariance: BGS, same as `bias_expansion_test`
(z_eff=0.295, Om=0.31519, thecov Gaussian covariance, nbar=9.59e-4,
veff=1.207e9 (Mpc/h)^3, b1=1.5, b2=-0.525).

**Placeholder priors**: no repo precedent exists for standard prior widths on
the bias/EFT/stochastic nuisance parameters -- see `config.py`'s module
docstring and the `PARAMS` dict for the explicit (broad, weakly-informative)
values used, easy to retune to a real analysis's priors.

## Modules

- `config.py` -- fiducial cosmology/tracer/MG-model constants, the full
  parameter list (5 cosmological + 1 MG + 4 bias + 4 counterterm + 2
  stochastic) with fiducials/finite-difference steps/prior sigmas, and the
  staged-marginalization groups (a)-(e).
- `linear_power.py` -- GR/LCDM linear P(k) via CAMB-ISiTGR, parametrized on
  cosmology (fresh spectrum per cosmological finite-difference step).
- `kernel_pipeline.py` -- `build_state(theory, ...)` builds (and caches) the
  FKPT/FOLPS kernel table for one (theory, cosmology, MG parameter) point,
  calling `fkptjax.kfuncs_to_tables.Kfuncs_to_tables_jax` directly with
  `beyond_eds`/`fkpt_approximation` set by `config.THEORY_KWARGS[theory]`;
  `poles_for_bias(state, bias_overrides, ...)` cheaply re-projects an
  already-built table for any nuisance-parameter vector.
- `covariance.py` -- shared analytic Gaussian P_ell covariance (`thecov`),
  built once from the fiducial bEdS-full P_ell and reused identically for
  all three theories.
- `derivatives.py` -- central (2-point) finite differences for every
  parameter, for each theory. Cosmological/MG-parameter derivatives rebuild
  the kernel table at theta +/- step (expensive, especially "full"); nuisance
  derivatives just re-project the cached fiducial table (cheap).
- `fisher.py` -- `F_ij = dP/dtheta_i^T C^-1 dP/dtheta_j + F_prior`; staged
  marginalization (a)-(e); `sigma_and_corr` for sigma(MG param) and its
  correlations with every other parameter; `ellipse_sub` for the 2-parameter
  Fisher-ellipse geometry used in the plots.
- `param_bias.py` -- the standard linear Fisher-bias formula: takes
  bEdS-full at the fiducial point as synthetic truth, computes
  `Delta_theta = F_approx^-1 [dP_approx/dtheta]^T C^-1 DeltaP` for EdS and
  bEdS-fkPT separately.
- `make_plots.py` / `run_all.py` -- diagnostics 1-4, the final compact
  summary figure/table, and the orchestration driver.

## Headline results so far

**mu_omde** (`outputs_mu_omde`/`figures_mu_omde`): bEdS-fkPT ≡ bEdS-full
exactly (as expected, scale-independent model). EdS slightly
*over*-estimates sigma(mu0): sigma_EdS/sigma_full = 1.014. EdS's
parameter-bias is small: Delta(mu0)/sigma = 0.014.

**hs** (`outputs_hs`/`figures_hs`): EdS and bEdS-fkPT now differ meaningfully
from bEdS-full, but *both* still **over**-estimate sigma(fR0_HS), not
under-estimate it: sigma_EdS/sigma_full = 1.070, sigma_approx/sigma_full =
1.072. Both recover fR0_HS with a small bias once nuisance parameters are
free (Delta/sigma = -0.014 for EdS, -0.005 for fkPT-approx). The
parameter-degeneracy *geometry* is where the real difference shows up: e.g.
rho(fR0_HS, alpha0) flips sign between full (-0.058) and EdS/approx (+0.022),
and rho(fR0_HS, alpha4) is noticeably larger for full (0.574) than for
EdS/approx (0.366) -- i.e. even though the approximate theories happen to
give a *larger* (more conservative) sigma here rather than an artificially
tight one, their implied degeneracy structure with the counterterms is
measurably wrong relative to the full-kernel truth.
