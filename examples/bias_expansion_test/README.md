# BGS scale-dependent bias-expansion test

Diagnostic/stress test for Sec. 8.3 of Desjacques, Jeong & Schmidt
(arXiv:1611.09787): quantifies the possible impact of a Schmidt
conserved-tracer scale-dependent linear bias in a Hu-Sawicki f(R) model whose
chameleon transition sits inside the BGS fitting range, `0.01 < k < 0.20 h/Mpc`.
This is a standalone test — nothing under `src/fkptjax` is modified.

## Setup

BGS tracer at `z_eff=0.295`, `Om=0.31519`, fiducial `b1_BGS=1.5` (same
constants as the production `generate_noiseless_synthetic_data.py` synthetic
pipeline). MG model: Hu-Sawicki f(R), tested for `F4` (`fR0=1e-4`), `F5`
(`fR0=1e-5`), `F6` (`fR0=1e-6`). Their linear chameleon transition scales at
`z_eff` are approximately `k_MG = 0.038, 0.119, 0.378 h/Mpc` respectively
(see `growth.k_MG_estimate`) — F5 sits almost exactly on the requested
`k_MG~0.1`, F4 shows an already-substantial in-range effect, F6's transition
is mostly beyond `k_max=0.2` (a weak-effect control case). Tracer "formation"
redshifts: `z* = 1, 2, 3, 5`.

## Modules

- `config.py` — fiducial cosmology/BGS/MG constants and k grids.
- `growth.py` — exact scale-dependent `D_+(k,z)`, `f(k,z)=D'/D` via
  `fkptjax.jax_ode.DP_jax` (fkptjax's own linear-growth ODE solver).
- `schmidt_bias.py` — the Schmidt `b1(k;z*)` formula, `Delta_b1`, and the
  exact (non-linearized) `Delta_P_bias^s(k,mu;z*)` correction projected to
  `ell=0,2,4` via `fkptjax.rsd.project_to_poles`. Kept fully separate from
  production bias/kernel code.
- `linear_power.py` — GR and Hu-Sawicki linear `P_L(k,z_eff)` via CAMB-ISiTGR.
- `full_kernel_pipeline.py` — `P_ell,full^standard(k)`: the real full-kernel
  BGS pipeline (desilike + fkptjax + ISiTGR, `beyond_eds=True`, no fkPT
  approximation, ordinary constant-`b1` bias expansion, EFT counterterms,
  stochastic terms, standard RSD), adapted from
  `/n/home12/cgarciaquintero/DESI/synthetic/scripts/generate_noiseless_synthetic_data.py`.
- `covariance.py` — analytic Gaussian `P_ell` covariance for BGS
  (`thecov`), same recipe as that same production script.
- `toy_spectrum.py` — combines everything into
  `P_ell,toy(k;z*) = P_ell,full^standard(k) + Delta_P_ell,bias(k;z*)`.
- `make_plots.py` / `run_all.py` — Plots 1-5 and all numeric outputs.

## Running

```
source ~/.bashrc && cosmodesi_new
cd examples/bias_expansion_test
python run_all.py
```

Outputs land in `outputs/*.npz` (+ `outputs/summary.json`) and `figures/*.png`,
one set of 5 plots per MG point (15 figures total).

## What is deferred

Sections 9-10 of the original spec — refitting the toy spectra with the
ordinary nuisance parameters (and then also the MG parameter) to see whether
the missing bias is absorbed, reporting `Delta(theta_MG)/sigma(theta_MG)` —
are **not** implemented yet. This repo has no real DESI BGS covariance or
fitting/likelihood pipeline of its own to fit against; `run_all.py` instead
generates the analytic Gaussian covariance and reports a simple raw (no-fit)
`|Delta_P_ell| / sigma_diag` significance on Plot 5, as an interim diagnostic.
A follow-up can wire up an actual least-squares/MCMC nuisance refit using
`desilike`'s own fitting machinery once that is wanted.
