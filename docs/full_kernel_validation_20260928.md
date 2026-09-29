# Full-kernel audit: scale-dependent BZ_Mass

The full **unscreened** BZ_Mass kernels reproduce sMGPT. This does not establish
the most general screened-gravity implementation in arXiv:2012.05077. The
desilike route initially did not select these kernels; the missing wiring has
been restored and reconciled with the current refactor-jax branch.

All numerical validation uses the `cosmodesi_new` environment and the workspace
repositories. Results, logs, plots and provenance are in
`../validation_results/bzmass_20260928/`. Changes are uncommitted; the recovery
stashes and archived reference outputs remain intact.

## Definition and scope

The paper's equations (20)--(24) require independent A, B, dA/dln(a), dB/dln(a).
Its equations (28)--(33) and Appendix C require their corresponding second- and
third-order kernels in the loop integrals. Merely retaining f(k) is insufficient.
Reference: https://arxiv.org/html/2012.05077.

The benchmark is HDKI/BZ_Mass with Omega_m=0.31519, h=0.6736, z=0.295,
mu_kinf=1.2, lambda_a=lambda_dS=100 Mpc/h. Its Poisson factor is

```
mu(a,k) = (1 + mu_kinf X)/(1 + X)
X = [k lambda_a lambda_dS / (a (lambda_dS a^-3 + lambda_a))]^2
```

This is scale dependent. Both nonlinear screening sources vanish in this
model/input deck. sMGPT's appropriate mode is PTkernels=2 (NoScreen).

The fkptjax full path supplies separate Q, N and R triangle orderings to the
TNS A terms, separate A/B to P22 and bias terms, and genuine fused D2/D3 to P13.
The restored bispectrum callback evaluates all three AP-transformed cyclic
triangles. A and B differ by as much as 0.1251 on the tested bispectrum mesh.

## Repairs

- Reconciled the desilike stash with the newer bispectrum basis argument and
  emulator layout. Forwarded fkpt_approximation and configurable quadrature to
  the loop builder. Restored per-triangle bispectrum A/B and BZ_Mass parameter
  name translation. Carried model constants and flags through the JAX pytree.
- Extended the recovered FolpsD callback to Scoccimarro multipoles as well as
  Sugiyama multipoles. Corrected the RSD Z2 mapping: each linear velocity uses
  its own f(k), and the velocity kernel G2 normalized by f0 is multiplied by f0,
  rather than by the mean of the two leg growth rates.
- Preserved the FKPT tracer's compatible damping default through both
  constructor and post-initialization. Removed an accidental overwrite of
  the routed f0 during pytree reconstruction.
- Fixed cosmoprimo's zero-total-neutrino-mass fraction normalization: fractions
  must sum to one for the actual number of eigenstates, not assume three.
- Set sMGPT's IR reader default to detect its 43/44-column layout. Existing
  output files are preserved; new corrected multipoles are stored separately.

The sMGPT AllFunctions header labels 43 columns but the data contain 44: Pbs2b1
is duplicated after column 37. The archived run used LegacyColumnOrder=True,
so it assigned the wrong columns to later bias terms. That caused roughly
2.1% monopole and 5.7% quadrupole differences over the tested interval even
when using the same sMGPT loops in both projections. The new Wolfram run uses
the actual layout. This corrects reference parsing, not the kernel equations.

## Kernel and power-spectrum evidence

The existing independent Wolfram A/B evaluation was rerun: its 60 rows are
identical to the archive. Against that reference, adaptive fkptjax A/B agree
within 5.7e-8 absolute; production RK4/64 within 3.1e-7. Derivative differences
are below 8e-7 absolute. The D2/D3 comparison also passes after accounting for
the reference's actual D3 normalization (its header says CFD3 but its exported
expression lacks fkptjax's factor 21/5).

The solver comparison covers 36 leg lengths x 16 Gauss-Legendre angles x
3 cyclic terms. RK4/64 versus adaptive differences are below 3e-7 for A/B and
9e-7 for their derivatives. Relative derivative errors can be large near
zero crossings; the previously quoted universal relative-error bound is not
valid for these stronger BZ_Mass parameters.

The production loop comparison uses the exact normal/no-wiggle input pair
recorded in the sMGPT log (`temp_pklin_ext_64878*.dat`), its fixed low-k growth
normalization pivot, linear-growth start ln(a)=-6, nonlinear start -4, identical
183 radial nodes and 10 angular nodes, and 121 output k values. No bias fitting
or counterterm absorption is used to improve agreement.

Maximum relative differences over 0.01 <= k <= 0.3 h/Mpc:

| Quantity | Maximum difference |
|---|---:|
| Linear P(k) | 0.0021% |
| P22 dd/dtheta/thetatheta | 0.00037% |
| P13 dd/dtheta/thetatheta | 0.0141% |
| TNS A-family terms | 0.0271% |
| sigma3 squared times P_L | 0.0393% |
| P0 versus corrected Wolfram multipoles | 0.0043% |
| P2 versus corrected Wolfram multipoles | 0.0396% |
| P4 versus corrected Wolfram multipoles | 0.0486% |

The same multipole assembly applied to the two loop sets differs by at most
0.00094%, 0.00532%, 0.00298% for ell=0,2,4. The remaining difference against
the Wolfram projection includes interpolation/IR-integral conventions.

This is implementation agreement at matched quadrature, not an assertion
that the archived sMGPT quadrature is converged. The live default is 300 radial
nodes, not 183. Increasing 300 to 600 changes P0/P2/P4 by up to
0.047%/0.230%/0.300%. Additional convergence results are recorded in
`convergence.json`; do not choose 183 simply because it reproduces the archive.

The completed 600-to-1200 radial-node comparison changes P0/P2/P4 by at most
0.0142%/0.0704%/0.0892%. At 600 radial nodes, increasing the angular quadrature
from 10 to 20 changes them by 0.00107%/0.00295%/0.00464%. These are successive
resolution differences, not rigorous error bounds. The default 300-node setting
therefore should not be advertised as sub-0.1% converged for this benchmark.
For precision runs, use higher radial resolution and verify the tolerance needed
for the actual cosmology, bias parameters, and k range.

## End-to-end execution

Both `pipeline` (job 49119110) and `sensitivity` (49119785) completed successfully
in cosmodesi_new. All four kernel modes produced finite P0/P2/P4, Sugiyama
B000/B202 and Scoccimarro B0/B2, with distinct outputs. ISiTGR's scale-dependent
growth agrees with the ODE growth within 0.0141%; its f(k) spans 0.68265--0.76185.
Changing the shared mu_kinf parameter from 1.2 to 1.3 updates both the linear
spectrum and the nonlinear kernel state. Pytree reconstruction passes, the
emulator layout has 113 leaves, and eager/JIT bispectra agree to 2.2e-16 relative.
This checks emulator state compatibility, not trained emulator accuracy; old
saved emulators need compatibility review or rebuilding after the layout change.

## Selecting kernels

`FKPTJAXPTSpectrum2Poles` now accepts `kernel_mode`:

| Mode | Nonlinear growth coefficients | Linear velocity |
|---|---|---|
| `full` | Genuine momentum-dependent A/B/D3 | f(k) |
| `fkpt` | Large-scale A/Ap/D3 constants | f(k) |
| `eds` | EdS coefficients | Constant template f0 |
| `eds_fk` | EdS coefficients | f(k) |

Without kernel_mode, the existing booleans keep their meanings: True/False
for beyond_eds/fkpt_approximation selects full; True/True selects fkpt;
beyond_eds=False selects EdSfk. The synthetic CLI's existing `--no-beyond-eds`
therefore retains f(k). Pure EdS is explicitly available via the calculator's
`kernel_mode='eds'`; this audit does not add a new synthetic CLI argument.

For live BZ_Mass, ISiTGR also needs `MG_parameterization='muSigma'` and
`use_BZ_Mass_form=True`. Lengths enter cosmoprimo in Mpc/h; its adapter divides
by h for ISiTGR. The nonlinear and linear sectors must receive the same MG
parameter values. The validation script shows both fixed overrides and shared
Parameter objects. `growth_source='template'` consumes ISiTGR's linear f(k).

## Limits of the certification

- Nonlinear screening is absent. Full HS with screening requested now fails
  explicitly; EFT_DE full kernels also fail explicitly pending interpolator
  translation. BZ_Mass NoScreen is the validated case.
- This run does not certify massive-neutrino corrections, arbitrary background
  histories, or trained ingredient/emulator accuracy. The archived ingredient
  emulator script references a now-missing provider; its mesh/solver comparison
  was reproduced without that dependency.
- A static audit finds further work needed before claiming general w0/wa or
  curved-background support: the wrapper assumes ol=1-Om, does not generally
  forward cosmological w0/wa, and the underlying ODE friction still uses
  2-3 Omega_m(a)/2. That is the flat LambdaCDM friction, not general 2+H'/H.
- sMGPT in this workspace has no bispectrum implementation to provide an
  independent full B000/B202 reference. Bispectrum validation checks its
  underlying independent kernels, the Z2 equation, callback routing, finite
  multipoles and mode sensitivity; it must not be described as a direct
  sMGPT bispectrum-multipole comparison.

## Reproduction

From `/n/home12/cgarciaquintero/DESI/src`:

```bash
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch reference
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch kernels
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch tables --nquad 183
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch tables --nquad 300
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch tables --nquad 600
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch tables --nquad 1200
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch tables --nquad 600 --nangle 20
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch pipeline
sbatch fkptjax_muMG/examples/validate_bzmass_pipeline.sbatch sensitivity
```

Run the reference stage before the table stages. After jobs finish:

```bash
/n/home12/cgarciaquintero/.conda/envs/cosmodesi_new/bin/python fkptjax_muMG/examples/summarize_bzmass_validation.py
```
