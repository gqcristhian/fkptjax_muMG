import os
os.environ.setdefault("JAX_ENABLE_X64", "True")
import sys
import time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")

import numpy as np

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk
from fkptjax import ab_ingredients as abi

Om = 0.315192
xnow = -3.912023
xstop = float(np.log(1.0 / 1.5))
f0 = 0.8
P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.HS, fR0_HS=1e-4, beta2=1.0 / 6.0, n_HS=1.0)

KMIN, KMAX = 0.005, 0.5
RMIN, RMAX = 1e-3, 20.0

rng = np.random.default_rng(1)
n_test = 40
k_q = np.exp(rng.uniform(np.log(KMIN * 1.5), np.log(KMAX * 0.9), n_test))
r_q = np.exp(rng.uniform(np.log(RMIN * 2), np.log(RMAX * 0.9), n_test))
x_q = rng.uniform(-0.95, 0.95, n_test)
p_q = r_q * k_q

Af_true, Apf_true, CFD3_true, CFD3p_true = mgk.D2_D3_fused_grid(x_q, k_q, p_q, P, xnow, xstop, f0, solver='adaptive')
Af_true, CFD3_true = np.asarray(Af_true), np.asarray(CFD3_true)

for Nk, Nr, Nx in [(24, 16, 12), (32, 24, 16)]:
    grid = abi.build_fused_coarse_grid(KMIN, KMAX, RMIN, RMAX, Nk=Nk, Nr=Nr, Nx=Nx)
    t0 = time.perf_counter()
    Af_c, Apf_c, CFD3_c, CFD3p_c = abi.compute_fused_coarse(grid, P, xnow, xstop, f0, solver='adaptive')
    Af_c.block_until_ready()
    t_solve = time.perf_counter() - t0

    Af_i, Apf_i, CFD3_i, CFD3p_i = abi.interpolate_fused(k_q, p_q, x_q, grid, Af_c, Apf_c, CFD3_c, CFD3p_c)
    Af_i, CFD3_i = np.asarray(Af_i), np.asarray(CFD3_i)

    rel_Af = np.abs(Af_i - Af_true) / (np.abs(Af_true) + 1e-30)
    rel_CFD3 = np.abs(CFD3_i - CFD3_true) / (np.abs(CFD3_true) + 1e-30)
    print(f"fused grid {Nk}x{Nr}x{Nx}={Nk*Nr*Nx:5d} pts, solve={t_solve:.2f}s: "
          f"max rel err A_fused={rel_Af.max():.3e} (median {np.median(rel_Af):.3e})  "
          f"CFD3={rel_CFD3.max():.3e} (median {np.median(rel_CFD3):.3e})")
    order = np.argsort(-rel_CFD3)
    print("  worst 8 CFD3 cases (k, p, x, rel_err):")
    for i in order[:8]:
        print(f"    k={k_q[i]:.4f} p={p_q[i]:.4f} x={x_q[i]:+.4f} rel_err={rel_CFD3[i]:.3e}")
    only_away_from_collinear = np.abs(x_q) < 0.7
    print(f"  restricted to |x|<0.7 ({only_away_from_collinear.sum()} pts): "
          f"max={rel_CFD3[only_away_from_collinear].max():.3e} "
          f"median={np.median(rel_CFD3[only_away_from_collinear]):.3e}")

print("DONE")
