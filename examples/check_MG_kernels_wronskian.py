"""
Validate MG_kernels_Wronskian.{A_B_grid,D2_fused_grid,D3_fused_grid} against
MG_kernels's existing, adaptive-per-point-solve implementation.

MG_kernels.py is NOT modified or imported-around here -- it is the trusted
ground truth. This script only checks the new module; production code
(calculate_jax.py / kfuncs_to_tables.py) is untouched and unused here.

Sections:
  1. Single-point regression (well-separated + near-collinear), sweeping
     n_eta, for A_B_grid / D2_fused_grid / D3_fused_grid.
  2. All three I1udd1-family leg orderings (Q/R/N) at both test triples.
  3. A realistic batched-grid test (actual Q-loop-shaped arrays), spot
     checking a handful of points and reporting real measured wall-clock
     for the whole batch vs. MG_kernels.A_B_grid on the same batch.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk
from fkptjax import MG_kernels_Wronskian as mgw

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

# ---------------------------------------------------------------------------
# Model: HDKI/BZ_Mass -- validated single-point against an independent
# Wolfram evaluation of sMGPT's own equations earlier this session, so
# MG_kernels.A_B_grid/D2_fused_grid/D3_fused_grid are trustworthy references.
# ---------------------------------------------------------------------------
Om = 0.315192
z = 0.295
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + z)))
f0_test = 0.7  # only enters CFD3p's normalization; used identically on both sides

P = bj.pack_constants_jnp(
    om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
    mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0,
)

TEST_TRIPLES = {
    "well-separated (kf=0.2,q=0.05,x=0.3)": (0.2, 0.05, 0.3),
    "near-collinear  (kf=0.2,q=0.196,x=0.995)": (0.2, 0.196, 0.995),
}


def leg_triple(kf, q, x):
    r = q / kf
    kminus = kf * float(jnp.sqrt(1.0 + r * r - 2.0 * r * x))
    return kf, q, kminus


print("\n=== 1a. A_B_grid: MG_kernels_Wronskian vs MG_kernels ===")
for label, (kf, q, x) in TEST_TRIPLES.items():
    kf_, k1_, k2_ = leg_triple(kf, q, x)
    A_ref, B_ref, _, _ = mgk.A_B_grid(kf_, k1_, k2_, P, xnow, xstop)
    print(f"\n{label}: kf={kf_}, k1=q={k1_}, k2=kminus={k2_:.6f}")
    print(f"  ground truth (MG_kernels, adaptive):  A={float(A_ref):.8f}  B={float(B_ref):.8f}")
    for n_eta in (32, 64, 128, 256, 512, 1024):
        A_w, B_w, _, _ = mgw.A_B_grid(kf_, k1_, k2_, P, xnow, xstop, n_eta=n_eta)
        relA = abs(float(A_w) - float(A_ref)) / abs(float(A_ref))
        relB = abs(float(B_w) - float(B_ref)) / abs(float(B_ref))
        print(f"    n_eta={n_eta:5d}:  A_w={float(A_w): .8f} (relerr={relA:.2e})   "
              f"B_w={float(B_w): .8f} (relerr={relB:.2e})")


print("\n=== 1b. D2_fused_grid: MG_kernels_Wronskian vs MG_kernels ===")
for label, (kf, q, x) in TEST_TRIPLES.items():
    A_ref, _ = mgk.D2_fused_grid(x, kf, q, P, xnow, xstop)
    print(f"\n{label}: x={x}, k={kf}, p={q}")
    print(f"  ground truth (MG_kernels, adaptive):  D2={float(A_ref):.8f}")
    for n_eta in (32, 64, 128, 256, 512, 1024):
        A_w, _ = mgw.D2_fused_grid(x, kf, q, P, xnow, xstop, n_eta=n_eta)
        rel = abs(float(A_w) - float(A_ref)) / abs(float(A_ref))
        print(f"    n_eta={n_eta:5d}:  D2_w={float(A_w): .8f}  (relerr={rel:.2e})")


print("\n=== 1c. D3_fused_grid (nested): MG_kernels_Wronskian vs MG_kernels ===")
for label, (kf, q, x) in TEST_TRIPLES.items():
    CFD3_ref, _ = mgk.D3_fused_grid(x, kf, q, P, xnow, xstop, f0=f0_test)
    print(f"\n{label}: x={x}, k={kf}, p={q}")
    print(f"  ground truth (MG_kernels, adaptive):  CFD3={float(CFD3_ref):.8f}")
    for n_eta in (32, 64, 128, 256, 512, 1024):
        CFD3_w, _ = mgw.D3_fused_grid(x, kf, q, P, xnow, xstop, f0=f0_test, n_eta=n_eta)
        rel = abs(float(CFD3_w) - float(CFD3_ref)) / abs(float(CFD3_ref))
        print(f"    n_eta={n_eta:5d}:  CFD3_w={float(CFD3_w): .8f}  (relerr={rel:.2e})")


print("\n=== 2. I1udd1-family orderings (Q/R/N leg permutations) ===")
ORDERINGS = {
    "Q (kf=k_ext, k1=q, k2=kminus)": lambda k_ext, q, kminus: (k_ext, q, kminus),
    "R (kf=kminus, k1=k_ext, k2=q)": lambda k_ext, q, kminus: (kminus, k_ext, q),
    "N (kf=q, k1=k_ext, k2=kminus)": lambda k_ext, q, kminus: (q, k_ext, kminus),
}
for label, (kf, q, x) in TEST_TRIPLES.items():
    k_ext, q_, kminus = leg_triple(kf, q, x)
    print(f"\n{label}: k_ext={k_ext}, q={q_}, kminus={kminus:.6f}")
    for ord_label, perm in ORDERINGS.items():
        kf_o, k1_o, k2_o = perm(k_ext, q_, kminus)
        A_ref, B_ref, _, _ = mgk.A_B_grid(kf_o, k1_o, k2_o, P, xnow, xstop)
        A_w, B_w, _, _ = mgw.A_B_grid(kf_o, k1_o, k2_o, P, xnow, xstop, n_eta=256)
        relA = abs(float(A_w) - float(A_ref)) / abs(float(A_ref))
        relB = abs(float(B_w) - float(B_ref)) / abs(float(B_ref))
        print(f"  {ord_label:32s} (kf={kf_o:.6f}):  n_eta=256  A relerr={relA:.2e}  B relerr={relB:.2e}")


print("\n=== 3. Realistic batched Q-loop grid ===")


def build_q_loop_grid(k_ext_arr, r_arr, x_arr):
    """Mirrors kfuncs_to_tables.py's Q-loop grid construction:
    q_loop = r*k_ext, kminus = k_ext*sqrt(1+r^2-2*r*x), all shape
    (NR_or_NQ, nquad-1, Nk_kernel)-broadcastable."""
    k_ext_b = k_ext_arr[None, None, :]
    r_b = r_arr[:, None, None]
    x_b = x_arr[None, :, None]
    y_b = jnp.sqrt(1.0 + r_b * r_b - 2.0 * r_b * x_b)
    q_loop = r_b * k_ext_b
    kminus = k_ext_b * y_b
    kf_full = k_ext_b * jnp.ones_like(y_b)
    return jnp.broadcast_arrays(kf_full, q_loop, kminus)


for res_label, (Nk, Nq, Nx) in [("coarse", (30, 149, 10)), ("production", (120, 299, 10))]:
    k_ext_arr = jnp.geomspace(1e-3, 1.0, Nk)
    r_arr = jnp.geomspace(1e-3, 4.0, Nx)  # NQ-ish
    x_arr = jnp.linspace(-0.999, 0.999, Nq)
    kf_grid, k1_grid, k2_grid = build_q_loop_grid(k_ext_arr, r_arr, x_arr)
    npts = kf_grid.size
    print(f"\n{res_label} grid: shape={kf_grid.shape}  npts={npts}")

    t0 = time.time()
    A_w, B_w, _, _ = mgw.A_B_grid(kf_grid, k1_grid, k2_grid, P, xnow, xstop)
    jax.block_until_ready((A_w, B_w))
    warm_w = time.time() - t0
    t0 = time.time()
    A_w, B_w, _, _ = mgw.A_B_grid(kf_grid, k1_grid, k2_grid, P, xnow, xstop)
    jax.block_until_ready((A_w, B_w))
    iter_w = time.time() - t0
    print(f"  MG_kernels_Wronskian.A_B_grid:  warmup={warm_w:.3f}s  post-compile={iter_w:.3f}s")

    # Spot-check a handful of points against MG_kernels' adaptive ground truth.
    flat_kf = np.asarray(kf_grid).reshape(-1)
    flat_k1 = np.asarray(k1_grid).reshape(-1)
    flat_k2 = np.asarray(k2_grid).reshape(-1)
    flat_Aw = np.asarray(A_w).reshape(-1)
    flat_Bw = np.asarray(B_w).reshape(-1)
    rng = np.random.default_rng(0)
    idx = rng.choice(npts, size=min(5, npts), replace=False)
    max_rel = 0.0
    for i in idx:
        A_ref, B_ref, _, _ = mgk.A_B_grid(float(flat_kf[i]), float(flat_k1[i]), float(flat_k2[i]), P, xnow, xstop)
        relA = abs(flat_Aw[i] - float(A_ref)) / abs(float(A_ref))
        relB = abs(flat_Bw[i] - float(B_ref)) / abs(float(B_ref))
        max_rel = max(max_rel, relA, relB)
    print(f"  spot-check ({len(idx)} random points vs MG_kernels): max relerr = {max_rel:.2e}")

    t0 = time.time()
    A_ref_grid, B_ref_grid, _, _ = mgk.A_B_grid(kf_grid, k1_grid, k2_grid, P, xnow, xstop)
    jax.block_until_ready((A_ref_grid, B_ref_grid))
    warm_ref = time.time() - t0
    t0 = time.time()
    A_ref_grid, B_ref_grid, _, _ = mgk.A_B_grid(kf_grid, k1_grid, k2_grid, P, xnow, xstop)
    jax.block_until_ready((A_ref_grid, B_ref_grid))
    iter_ref = time.time() - t0
    print(f"  MG_kernels.A_B_grid (adaptive): warmup={warm_ref:.3f}s  post-compile={iter_ref:.3f}s"
          f"  (speedup: {iter_ref / iter_w:.1f}x)")
