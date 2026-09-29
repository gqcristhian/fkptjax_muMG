"""
Plot every meaningful table_w column, exact (live) vs emulated, for several
PHENOM/binning (mu1, mu2, mu3, mu4) points spanning distinct
redshift/scale-binning combinations.

Output: one PNG per parameter point in OUTDIR, each a grid of subplots (one
per loop-table column), k*value(k) for live (solid) vs emulated (dashed).
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
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")

from fkptjax import mg_jax as bj
from fkptjax import ab_ingredients as abi
from fkptjax.kfuncs_to_tables import Kfuncs_to_tables, _resolve_mg_kind

OUTDIR = Path("/n/home12/cgarciaquintero/DESI/synthetic/notebooks/binning_ingredients_validation")
OUTDIR.mkdir(parents=True, exist_ok=True)

Nk_kernel, nquadSteps, NQ, NR = 120, 300, 10, 10
KMIN, KMAX = 0.001, 0.5
Om = 0.315192
Z = 0.5
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + Z)))
f0 = jnp.asarray(0.8)
kind = _resolve_mg_kind('PHENOM', 'binning')

k_lin = jnp.geomspace(1e-4, 2.0, 400)
pk_lin = 2e4 * (k_lin / 0.05) ** (-1.5) / (1.0 + (k_lin / 0.05) ** 2.5)
pk_now_lin = pk_lin

PARAM_BOUNDS = {'mu1': (0.5, 1.5), 'mu2': (0.5, 1.5), 'mu3': (0.5, 1.5), 'mu4': (0.5, 1.5)}

# mu1/mu2 gate the LOW-z bin's two k-sub-ranges; mu3/mu4 gate the HIGH-z bin's
# (see mg_jax.py's `_mu_binning`, scale_bins=True branch) -- chosen to
# distinctly probe redshift- and scale-binning separately, plus the
# notebook's own illustrative reference point.
TEST_POINTS = {
    'GR (mu=1,1,1,1)':        dict(mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0),
    'low-z boost (1.4,1.4,1,1)':  dict(mu1=1.4, mu2=1.4, mu3=1.0, mu4=1.0),
    'high-z boost (1,1,1.4,1.4)': dict(mu1=1.0, mu2=1.0, mu3=1.4, mu4=1.4),
    'reference (0.8,1.3,0.9,1.1)': dict(mu1=0.8, mu2=1.3, mu3=0.9, mu4=1.1),
}

COLUMN_LABELS = {
    3: 'P22dd+P13dd', 4: 'P22du+P13du', 5: 'P22uu+P13uu',
    6: 'Pb1b2', 7: 'Pb1bs2', 8: 'Pb22', 9: 'Pb2s2', 10: 'Ps22',
    11: 'sigma32PSL', 12: 'Pb2theta', 13: 'Pbs2theta',
    14: 'I1udd1A', 15: 'I2uud1A', 16: 'I2uud2A', 17: 'I3uuu2A', 18: 'I3uuu3A',
    19: 'I2uudd1BpC', 20: 'I2uudd2BpC', 21: 'I3uuud2BpC', 22: 'I3uuud3BpC',
    23: 'I4uuuu2BpC', 24: 'I4uuuu3BpC', 25: 'I4uuuu4BpC',
}
COL_INDICES = sorted(COLUMN_LABELS)


def P_from_params(params):
    return bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=kind, scale_bins=True, **params)


def live_kfuncs(params):
    return Kfuncs_to_tables(
        k_lin, pk_lin, pk_now_lin, z=Z, Om=Om, beyond_eds=True, fkpt_approximation=False,
        model='PHENOM', mg_variant='binning', scale_bins=True, **params,
        kmin=KMIN, kmax=KMAX, Nk_kernel=Nk_kernel, nquadSteps=nquadSteps, NQ=NQ, NR=NR,
        return_kernel_constants=False,
    )


print("=" * 78)
print("Training the 4D (mu1..mu4) ingredients provider (budget=2, 16 nodes)")
print("=" * 78)
t0 = time.perf_counter()
provider = abi.build_ingredients_provider(
    kmin=KMIN, kmax=KMAX, rmin=1e-4, rmax=1e4,
    param_bounds=PARAM_BOUNDS, P_from_params=P_from_params,
    xnow=xnow, xstop=xstop, f0=f0, budget=2,
)
print(f"train time: {time.perf_counter()-t0:.2f}s")


def emulated_kfuncs(params):
    ingredients_fn = provider.bind(**params)
    return Kfuncs_to_tables(
        k_lin, pk_lin, pk_now_lin, z=Z, Om=Om, beyond_eds=True, fkpt_approximation=False,
        model='PHENOM', mg_variant='binning', scale_bins=True, **params,
        kmin=KMIN, kmax=KMAX, Nk_kernel=Nk_kernel, nquadSteps=nquadSteps, NQ=NQ, NR=NR,
        return_kernel_constants=False, ingredients_fn=ingredients_fn,
    )


print("\nwarm-up calls (JIT compile)")
_ = live_kfuncs(TEST_POINTS['GR (mu=1,1,1,1)']); jax.block_until_ready(_)
_ = emulated_kfuncs(TEST_POINTS['GR (mu=1,1,1,1)']); jax.block_until_ready(_)

N_PANELS = len(COL_INDICES)
NCOLS = 5
NROWS = int(np.ceil(N_PANELS / NCOLS))

for name, params in TEST_POINTS.items():
    print(f"\n--- {name}: {params} ---")
    table_w_live, _ = live_kfuncs(params)
    jax.block_until_ready(table_w_live)
    table_w_emu, _ = emulated_kfuncs(params)
    jax.block_until_ready(table_w_emu)

    kout = np.asarray(table_w_live[0])

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(4.2 * NCOLS, 3.0 * NROWS))
    axes = np.asarray(axes).ravel()

    for ax, idx in zip(axes, COL_INDICES):
        live_col = np.asarray(table_w_live[idx])
        emu_col = np.asarray(table_w_emu[idx])
        rel = np.abs(emu_col - live_col) / (np.abs(live_col) + 1e-30)

        ax.plot(kout, kout * live_col, color='C0', lw=1.6, label='exact')
        ax.plot(kout, kout * emu_col, color='C1', lw=1.2, ls='--', label='emulated')
        ax.set_xscale('log')
        ax.set_title(f"[{idx}] {COLUMN_LABELS[idx]}\nmax|rel err|={rel.max():.2e}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color='grey', lw=0.5)

    for ax in axes[N_PANELS:]:
        ax.axis('off')
    axes[0].legend(fontsize=8)

    fig.suptitle(f"PHENOM/binning table_w columns, exact vs emulated -- {name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    safe_name = name.split(' (')[0].replace(' ', '_')
    outpath = OUTDIR / f"table_w_exact_vs_emulated_{safe_name}.png"
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")

print("\nDONE")
