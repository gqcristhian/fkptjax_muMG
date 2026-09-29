"""
Isolate the PARAMETER-SPACE Chebyshev fit ALONE, with NO k-interpolation at
all: evaluate A/B at a coarse-grid node's EXACT (kf,k1,k2) value (a direct
array index, not interpolate_ab), and compare the trained emulator's
predict_coarse(**params) at that index against a DIRECT live A_B_grid call
at that same exact (kf,k1,k2) triple. If this already disagrees, the
parameter fit itself is the problem; if it agrees closely, the problem is
purely in interpolate_ab's k-interpolation between nodes.
"""
import os
os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp

import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk
from fkptjax import ab_ingredients as abi
from fkptjax.kfuncs_to_tables import _resolve_mg_kind

Om = 0.315192
xnow = -3.912023
xstop = float(jnp.log(1.0 / 1.5))
kind = _resolve_mg_kind('PHENOM', 'binning')
PARAM_BOUNDS = {'mu1': (0.5, 1.5), 'mu2': (0.5, 1.5), 'mu3': (0.5, 1.5), 'mu4': (0.5, 1.5)}


def P_from_params(params):
    return bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=kind, scale_bins=True, **params)


print("Building A/B-only coarse grid (budget=2, k_attractors=(0.01,0.1,0.2))...")
grid_ab = abi.build_coarse_grid(0.001, 0.5, 1e-5, 8.0,
                                 k_attractors=(0.01, 0.1, 0.2), cluster_width=0.005, n_cluster=16)
kf_nodes = np.exp(np.asarray(grid_ab.log_kf_nodes))
k1_nodes = np.exp(np.asarray(grid_ab.log_k1_nodes))
k2_nodes = np.exp(np.asarray(grid_ab.log_k2_nodes))
print(f"kf_nodes: {len(kf_nodes)}, k1/k2_nodes: {len(k1_nodes)}")

# Find node indices bracketing the problem triangle (kf~0.15, k1~0.08, k2~0.09)
i_kf = int(np.argmin(np.abs(kf_nodes - 0.15)))
i_k1 = int(np.argmin(np.abs(k1_nodes - 0.08)))
i_k2 = int(np.argmin(np.abs(k2_nodes - 0.09)))
kf0, k1_0, k2_0 = kf_nodes[i_kf], k1_nodes[i_k1], k2_nodes[i_k2]
print(f"Nearest coarse node to (0.15,0.08,0.09): "
      f"(kf={kf0:.6f}[i={i_kf}], k1={k1_0:.6f}[i={i_k1}], k2={k2_0:.6f}[i={i_k2}])")

from cosmoprimo.emulators.tools import Emulator as _Emulator, Space as _Space


class _ABTarget:
    def __call__(self, params):
        params = {name: float(np.atleast_1d(value)[0]) for name, value in params.items()}
        P = P_from_params(params)
        A, B, Ap, Bp = abi.compute_ab_coarse(grid_ab, P, xnow, xstop, solver='rk4', n_steps=64)
        return {'A': A, 'B': B, 'Ap': Ap, 'Bp': Bp}


space = _Space(bounds=PARAM_BOUNDS)
emu_ab = _Emulator(_ABTarget(), space, engine='chebyshev', budget=2)
emu_ab.train()
print("done training\n")

PARAM_POINTS = {
    'ref (0.8,1.3,0.9,1.1)': dict(mu1=0.8, mu2=1.3, mu3=0.9, mu4=1.1),
    'mixed corner': dict(mu1=0.55, mu2=1.45, mu3=0.55, mu4=1.45),
    'low corner': dict(mu1=0.55, mu2=0.55, mu3=0.55, mu4=0.55),
    'high corner': dict(mu1=1.45, mu2=1.45, mu3=1.45, mu4=1.45),
}

print(f"{'params':22s} {'A_live(node)':>14s} {'A_emu(node)':>14s} {'relerr':>10s} "
      f"{'B_live':>12s} {'B_emu':>12s} {'relerr':>10s}")
for pname, params in PARAM_POINTS.items():
    P = P_from_params(params)
    A_live, B_live, _, _ = mgk.A_B_grid(
        jnp.asarray(kf0), jnp.asarray(k1_0), jnp.asarray(k2_0), P, xnow, xstop,
        solver='rk4', n_steps=64)
    ab_pred = emu_ab.predict(**params)
    A_c = np.asarray(ab_pred['A'])
    B_c = np.asarray(ab_pred['B'])
    A_emu_node = A_c[i_kf, i_k1, i_k2]
    B_emu_node = B_c[i_kf, i_k1, i_k2]
    A_live_f, B_live_f = float(A_live), float(B_live)
    relA = abs(A_emu_node - A_live_f) / (abs(A_live_f) + 1e-30)
    relB = abs(B_emu_node - B_live_f) / (abs(B_live_f) + 1e-30)
    print(f"{pname:22s} {A_live_f:14.6f} {A_emu_node:14.6f} {relA:10.3e} "
          f"{B_live_f:12.6f} {B_emu_node:12.6f} {relB:10.3e}")

print("\nDONE")
