"""
Feasibility probe for the Green's-function/Wronskian reformulation of
MG_kernels.A_B_grid's second-order (DA, DB) kernel.

Part A -- CORRECTNESS: build the homogeneous solution basis + Wronskian for a
fixed output leg kf via the SAME adaptive diffrax solver already validated in
this codebase, then get DA/DB at several (kf,k1,k2) triples -- including one
picked to sit right in the near-collinear regime the module's own docstring
flags as numerically dangerous -- via discretized variation-of-parameters on
a FIXED, shared eta-grid. Compare against MG_kernels.A_B_grid's existing,
already-validated adaptive per-point solve, sweeping the shared grid's
density (Nt) to check for convergence (the tell for "just needs a finer
shared grid" vs "structurally broken near collinear", per this session's own
established convergence-vs-bug methodology).

Part B -- SPEED PROXY: time the two pieces that would replace the current
~3.6e5 independent adaptive ODE solves: (1) a modest number of *dense-output*
homogeneous-basis solves over a k-interpolation grid, and (2) a large batched
linear-algebra contraction standing in for "evaluate every quadrature point
via interpolation + a source/dot-product against the precomputed basis".
"""
import sys, os, time
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

# ---------------------------------------------------------------------------
# Model: HDKI/BZ_Mass -- the same scale-dependent model already validated
# single-point against an independent Wolfram evaluation of sMGPT's own
# equations this session, so MG_kernels.A_B_grid is a trustworthy reference.
# ---------------------------------------------------------------------------
Om = 0.315192
z = 0.295
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + z)))

P = bj.pack_constants_jnp(
    om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
    mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0,
)

# ---------------------------------------------------------------------------
# Part A: correctness of the fixed-shared-eta-grid Green's function approach
# ---------------------------------------------------------------------------

def homogeneous_rhs(kf):
    def rhs(eta, y, args):
        f2 = bj.f1(eta, P)
        fr = 2.0 - f2
        return jnp.stack([y[1], f2 * bj.mu(eta, kf, P) * y[0] - fr * y[1]])
    return rhs


def solve_homogeneous_dense(kf, y0, eta_grid):
    """One homogeneous D''+fr D' - f2 mu(kf) D = 0 solve, dense output at eta_grid."""
    term = diffrax.ODETerm(homogeneous_rhs(kf))
    solver = diffrax.Tsit5()
    ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-11)
    sol = diffrax.diffeqsolve(
        term, solver, t0=xnow, t1=xstop, dt0=0.01, y0=y0,
        stepsize_controller=ctrl, saveat=diffrax.SaveAt(ts=eta_grid),
        max_steps=200000,
    )
    return sol.ys[:, 0], sol.ys[:, 1]  # D(eta_grid), D'(eta_grid)


def green_DA_DB(kf, k1, k2, eta_grid):
    """DA(xstop), DB(xstop) via discretized variation of parameters on a
    FIXED shared eta_grid, reusing the SAME source terms as
    MG_kernels.secondOrderAB (S2a+S2FL for A, S2b for B)."""
    e0 = jnp.exp(xnow)
    D2plusi = 3.0 * jnp.exp(2.0 * xnow) / 7.0
    dD2plusi = 6.0 * jnp.exp(2.0 * xnow) / 7.0

    # Homogeneous basis for the OUTPUT leg kf: two independent solutions.
    y1_0 = jnp.array([e0, e0])
    y2_0 = jnp.array([e0, -2.0 * e0])
    y1, dy1 = solve_homogeneous_dense(kf, y1_0, eta_grid)
    y2, dy2 = solve_homogeneous_dense(kf, y2_0, eta_grid)
    W = y1 * dy2 - y2 * dy1  # Wronskian, sampled on eta_grid

    # Leg growth factors at the two INPUT legs (their own physical
    # growing-mode solution -- same equation, k1/k2 instead of kf).
    Dk1, _ = solve_homogeneous_dense(k1, jnp.array([e0, e0]), eta_grid)
    Dk2, _ = solve_homogeneous_dense(k2, jnp.array([e0, e0]), eta_grid)

    x = mgk._x_from_triple(kf, k1, k2)
    src_A = bj.S2a(eta_grid, x, k1, k2, P) + bj.S2FL(eta_grid, x, k1, k2, P)
    src_B = bj.S2b(eta_grid, x, k1, k2, P)
    gA = src_A * Dk1 * Dk2
    gB = src_B * Dk1 * Dk2

    def particular(g):
        integrand1 = y1 * g / W
        integrand2 = y2 * g / W
        I1 = jnp.trapezoid(integrand1, eta_grid)
        I2 = jnp.trapezoid(integrand2, eta_grid)
        # Full-range particular solution + derivative at xstop (eta_grid[-1]).
        u = y2[-1] * I1 - y1[-1] * I2
        # d/deta [y2*I1 - y1*I2] = y2'*I1 - y1'*I2  (boundary terms cancel,
        # standard variation-of-parameters identity).
        du = dy2[-1] * I1 - dy1[-1] * I2
        return u, du

    uA, duA = particular(gA)
    uB, duB = particular(gB)

    # Match physical initial conditions DA(xnow)=D2plusi, DA'(xnow)=dD2plusi
    # (same as MG_kernels._solve_AB's y0) via a homogeneous correction.
    Mmat = jnp.array([[y1_0[0], y2_0[0]], [y1_0[1], y2_0[1]]])
    cA = jnp.linalg.solve(Mmat, jnp.array([D2plusi, dD2plusi]))
    cB = jnp.linalg.solve(Mmat, jnp.array([D2plusi, dD2plusi]))

    DA = cA[0] * y1[-1] + cA[1] * y2[-1] + uA
    dDA = cA[0] * dy1[-1] + cA[1] * dy2[-1] + duA
    DB = cB[0] * y1[-1] + cB[1] * y2[-1] + uB
    dDB = cB[0] * dy1[-1] + cB[1] * dy2[-1] + duB

    C = (3.0 / 7.0) * Dk1[-1] * Dk2[-1]
    A = DA / C
    B = DB / C
    return float(A), float(B)


def ground_truth_AB(kf, k1, k2):
    A, B, _, _ = mgk.A_B_grid(kf, k1, k2, P, xnow, xstop)
    return float(A), float(B)


TEST_TRIPLES = {
    "well-separated (kf=0.2,q=0.05,x=0.3)": (0.2, 0.05, None, 0.3),
    "near-collinear  (kf=0.2,q=0.196,x=0.995)": (0.2, 0.196, None, 0.995),
}

print("\n=== Part A: fixed-shared-eta-grid Green's function vs adaptive ground truth ===")
for label, (kf, q, _unused, x) in TEST_TRIPLES.items():
    r = q / kf
    kminus = kf * float(jnp.sqrt(1.0 + r * r - 2.0 * r * x))
    A_ref, B_ref = ground_truth_AB(kf, q, kminus)
    print(f"\n{label}: kf={kf}, k1=q={q}, k2=kminus={kminus:.6f}  (y=kminus/kf={kminus/kf:.4f})")
    print(f"  ground truth (adaptive, per-point):  A={A_ref:.8f}  B={B_ref:.8f}")
    for Nt in (32, 64, 128, 256, 512, 1024):
        eta_grid = jnp.linspace(xnow, xstop, Nt)
        A_g, B_g = green_DA_DB(kf, q, kminus, eta_grid)
        relA = abs(A_g - A_ref) / abs(A_ref)
        relB = abs(B_g - B_ref) / abs(B_ref)
        print(f"    Nt={Nt:5d}:  A_green={A_g: .8f} (relerr={relA:.2e})   "
              f"B_green={B_g: .8f} (relerr={relB:.2e})")


# ---------------------------------------------------------------------------
# Part B: speed proxy for the two pieces that would REPLACE the current
# ~3.6e5 independent adaptive ODE solves at production resolution:
#   (1) a modest, one-time set of dense-output homogeneous-basis solves over
#       a k-interpolation grid (this replaces essentially ALL of the current
#       adaptive-ODE cost);
#   (2) a large batched array evaluation (source terms + interpolation +
#       the variation-of-parameters contraction) over every quadrature point
#       -- pure vectorized array ops, no per-point sequential ODE solving.
# This is a real, measured proxy (actual jnp/diffrax calls), not a made-up
# number -- but the DATA involved (e.g. the "source" array in step 2) is
# synthetic, since building the full physical pipeline is future work; the
# point here is only to measure whether the *shape* of computation each
# piece implies is cheap or not.
# ---------------------------------------------------------------------------
print("\n=== Part B: speed proxy for the Green's-function replacement ===")

N_KTABLE = 300          # k-interpolation-grid size (replaces ~3.6e5 solves)
NT_PROXY = 128           # shared eta-grid density used for the proxy
N_POINTS_PROD = 358800   # production quadrature grid size (measured earlier)

k_table = jnp.geomspace(1e-3, 2.0, N_KTABLE)
eta_grid_proxy = jnp.linspace(xnow, xstop, NT_PROXY)

t0 = time.time()
solve_y1 = jax.vmap(lambda kf: solve_homogeneous_dense(kf, jnp.array([jnp.exp(xnow), jnp.exp(xnow)]), eta_grid_proxy))
solve_y2 = jax.vmap(lambda kf: solve_homogeneous_dense(kf, jnp.array([jnp.exp(xnow), -2.0 * jnp.exp(xnow)]), eta_grid_proxy))
y1_tab, dy1_tab = solve_y1(k_table)
y2_tab, dy2_tab = solve_y2(k_table)
jax.block_until_ready((y1_tab, dy1_tab, y2_tab, dy2_tab))
warm_t = time.time() - t0

t0 = time.time()
y1_tab, dy1_tab = solve_y1(k_table)
y2_tab, dy2_tab = solve_y2(k_table)
jax.block_until_ready((y1_tab, dy1_tab, y2_tab, dy2_tab))
basis_t = time.time() - t0
print(f"  (1) basis-table build ({N_KTABLE} k-nodes x 2 solutions, dense NT={NT_PROXY}):"
      f"  warmup={warm_t:.3f}s   post-compile={basis_t:.3f}s")


def batched_eval_proxy(key):
    # Representative of: interpolate Dk1/Dk2/y1/y2/W from the k-table for
    # every quadrature point, evaluate the source there, and contract
    # against the propagator weights -- all as dense array ops.
    k1 = jax.random.uniform(key, (N_POINTS_PROD,), minval=1e-3, maxval=1.0)
    k2 = jax.random.uniform(jax.random.fold_in(key, 1), (N_POINTS_PROD,), minval=1e-3, maxval=1.0)
    x = jax.random.uniform(jax.random.fold_in(key, 2), (N_POINTS_PROD,), minval=-1.0, maxval=1.0)

    def interp_table(kq, table):
        return jnp.interp(kq, k_table, table)

    # "Interpolate" a couple of representative table columns at each point
    # (in the real pipeline this would be per-eta-node, i.e. NT_PROXY-many
    # interpolations; approximate the aggregate cost with one column each
    # broadcast over the batch, since jnp.interp itself is O(N_POINTS)).
    Dk1 = interp_table(k1, y1_tab[:, -1])
    Dk2 = interp_table(k2, y1_tab[:, -1])

    src_A = bj.S2a(eta_grid_proxy[-1], x, k1, k2, P) + bj.S2FL(eta_grid_proxy[-1], x, k1, k2, P)
    src_B = bj.S2b(eta_grid_proxy[-1], x, k1, k2, P)

    # Stand in for the full NT_PROXY-length source vector per point via an
    # outer product (this is the honest FLOP-shape of "evaluate the source
    # at every shared eta-node for every point"), then do the
    # variation-of-parameters contraction (a (N_POINTS,NT)x(NT,) matmul).
    gA = (src_A * Dk1 * Dk2)[:, None] * jnp.ones((1, NT_PROXY))
    gB = (src_B * Dk1 * Dk2)[:, None] * jnp.ones((1, NT_PROXY))
    w1 = y1_tab[0] / (y1_tab[0] * dy2_tab[0] - y2_tab[0] * dy1_tab[0])  # placeholder weight vector, shape (NT,)
    w2 = y2_tab[0] / (y1_tab[0] * dy2_tab[0] - y2_tab[0] * dy1_tab[0])
    IA1 = jnp.einsum('pt,t->p', gA, w1)
    IA2 = jnp.einsum('pt,t->p', gA, w2)
    IB1 = jnp.einsum('pt,t->p', gB, w1)
    IB2 = jnp.einsum('pt,t->p', gB, w2)
    return IA1, IA2, IB1, IB2


key = jax.random.PRNGKey(0)
t0 = time.time()
out = batched_eval_proxy(key)
jax.block_until_ready(out)
warm_t2 = time.time() - t0

t0 = time.time()
out = batched_eval_proxy(key)
jax.block_until_ready(out)
eval_t = time.time() - t0
print(f"  (2) batched per-point evaluation ({N_POINTS_PROD} points, NT={NT_PROXY}):"
      f"  warmup={warm_t2:.3f}s   post-compile={eval_t:.3f}s")

total = basis_t + eval_t
print(f"\n  Projected total (post-compile): {total:.3f}s"
      f"  vs. measured current full-kernel cost: 19.3s (GPU) / 70.9s (CPU) at this same resolution")


# ---------------------------------------------------------------------------
# Part C: third-order D3 (nested Green's function) -- the genuinely new
# structural test. thirdOrder's D3 kernel is sourced by S3I+S3II+S3FL, which
# themselves need the FULL eta-trajectory of two second-order-type
# quantities (D2f at output leg kplusp=kpp(x,k,p), D2mf at kpluspm=kpp(-x,k,p))
# -- not just their value at xstop. So this needs a *cumulative* (trajectory)
# variation-of-parameters, one layer for D2f/D2mf, then a third homogeneous
# layer (output leg = k, matching thirdOrder's own D3 equation) on top.
# ---------------------------------------------------------------------------
print("\n=== Part C: third-order D3 (nested Green's function) vs adaptive ground truth ===")


def cumtrapz(y, x):
    dx = jnp.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    inc = avg * dx
    return jnp.concatenate([jnp.array([0.0]), jnp.cumsum(inc)])


def green_D2_traj(kf_leg, x_local, k, p, eta_grid, ic_val, ic_deriv):
    """Full eta-trajectory of the single fused D2-type equation (mirrors
    mg_jax.secondOrder's y[4]/y[5] pair) for output leg kf_leg."""
    e1 = jnp.exp(xnow)
    y1_0 = jnp.array([e1, e1]); y2_0 = jnp.array([e1, -2.0 * e1])
    y1, dy1 = solve_homogeneous_dense(kf_leg, y1_0, eta_grid)
    y2, dy2 = solve_homogeneous_dense(kf_leg, y2_0, eta_grid)
    W = y1 * dy2 - y2 * dy1
    Dpk, _ = solve_homogeneous_dense(k, jnp.array([e1, e1]), eta_grid)
    Dpp, _ = solve_homogeneous_dense(p, jnp.array([e1, e1]), eta_grid)
    src = bj.SD2(eta_grid, x_local, k, p, P) * Dpk * Dpp
    I1 = cumtrapz(y1 * src / W, eta_grid)
    I2 = cumtrapz(y2 * src / W, eta_grid)
    u = y2 * I1 - y1 * I2
    du = dy2 * I1 - dy1 * I2
    Mmat = jnp.array([[y1_0[0], y2_0[0]], [y1_0[1], y2_0[1]]])
    c = jnp.linalg.solve(Mmat, jnp.array([ic_val, ic_deriv]))
    D2 = c[0] * y1 + c[1] * y2 + u
    dD2 = c[0] * dy1 + c[1] * dy2 + du
    return D2, dD2, Dpk, Dpp


def green_D3(k, p, x, eta_grid):
    e1 = jnp.exp(xnow); e2 = jnp.exp(2.0 * xnow); e3 = jnp.exp(3.0 * xnow)
    one_m_x2 = 1.0 - x * x
    kplusp = float(bj.kpp(x, k, p)); kpluspm = float(bj.kpp(-x, k, p))
    D2i = 3.0 * e2 / 7.0 * one_m_x2; dD2i = 6.0 * e2 / 7.0 * one_m_x2

    D2f, _dD2f, Dpk, Dpp = green_D2_traj(kplusp, x, k, p, eta_grid, D2i, dD2i)
    D2mf, _dD2mf, _, _ = green_D2_traj(kpluspm, -x, k, p, eta_grid, D2i, dD2i)

    src3 = (bj.S3I(eta_grid, x, k, p, Dpk, Dpp, D2f, D2mf, P)
            + bj.S3II(eta_grid, x, k, p, Dpk, Dpp, D2f, D2mf, P)
            + bj.S3FL(eta_grid, x, k, p, Dpk, Dpp, D2f, D2mf, P))

    y1_0 = jnp.array([e1, e1]); y2_0 = jnp.array([e1, -2.0 * e1])
    y1, dy1 = solve_homogeneous_dense(k, y1_0, eta_grid)
    y2, dy2 = solve_homogeneous_dense(k, y2_0, eta_grid)
    W = y1 * dy2 - y2 * dy1

    pk = p / k
    ang = 1.0 / (1.0 + pk * pk + 2.0 * pk * x) + 1.0 / (1.0 + pk * pk - 2.0 * pk * x)
    D3i = (5.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang
    dD3i = (15.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang

    I1 = jnp.trapezoid(y1 * src3 / W, eta_grid)
    I2 = jnp.trapezoid(y2 * src3 / W, eta_grid)
    u = y2[-1] * I1 - y1[-1] * I2

    Mmat = jnp.array([[y1_0[0], y2_0[0]], [y1_0[1], y2_0[1]]])
    c = jnp.linalg.solve(Mmat, jnp.array([D3i, dD3i]))
    D3 = c[0] * y1[-1] + c[1] * y2[-1] + u

    denom = Dpk[-1] * Dpp[-1] * Dpp[-1]
    CFD3 = (21.0 / 5.0) * D3 / denom
    return float(CFD3)


def ground_truth_CFD3(k, p, x):
    CFD3, _CFD3p, = mgk.D3_fused_grid(x, k, p, P, xnow, xstop, f0=0.7)
    return float(CFD3)


for label, (kf, q, _unused, x) in TEST_TRIPLES.items():
    k_ext, p_loop = kf, q  # for the P13-loop's D3_fused_grid(x, k, p, ...) convention
    ref = ground_truth_CFD3(k_ext, p_loop, x)
    print(f"\n{label}: k={k_ext}, p={p_loop}, x={x}")
    print(f"  ground truth (adaptive, per-point):  CFD3={ref:.8f}")
    for Nt in (32, 64, 128, 256, 512, 1024):
        eta_grid = jnp.linspace(xnow, xstop, Nt)
        val = green_D3(k_ext, p_loop, x, eta_grid)
        rel = abs(val - ref) / abs(ref)
        print(f"    Nt={Nt:5d}:  CFD3_green={val: .8f}  (relerr={rel:.2e})")


# ---------------------------------------------------------------------------
# Part D: I1udd1-family's three orderings (Q/R/N) -- same A_B_grid formula
# as Part A, just with different leg permutations of the SAME (k_ext,q,kminus)
# triple. Confirms the Green's-function approach is agnostic to which
# physical leg plays the "output" role, which is all the three-ordering sum
# (calculate_jax.py's unified Q+R+N formula) actually needs.
# ---------------------------------------------------------------------------
print("\n=== Part D: I1udd1-family orderings (Q/R/N leg permutations) ===")

ORDERINGS = {
    "Q (kf=k_ext, k1=q, k2=kminus)": lambda k_ext, q, kminus: (k_ext, q, kminus),
    "R (kf=kminus, k1=k_ext, k2=q)": lambda k_ext, q, kminus: (kminus, k_ext, q),
    "N (kf=q, k1=k_ext, k2=kminus)": lambda k_ext, q, kminus: (q, k_ext, kminus),
}

for label, (kf, q, _unused, x) in TEST_TRIPLES.items():
    r = q / kf
    kminus = kf * float(jnp.sqrt(1.0 + r * r - 2.0 * r * x))
    print(f"\n{label}: k_ext={kf}, q={q}, kminus={kminus:.6f}")
    for ord_label, perm in ORDERINGS.items():
        kf_o, k1_o, k2_o = perm(kf, q, kminus)
        A_ref, B_ref = ground_truth_AB(kf_o, k1_o, k2_o)
        Nt = 256
        eta_grid = jnp.linspace(xnow, xstop, Nt)
        A_g, B_g = green_DA_DB(kf_o, k1_o, k2_o, eta_grid)
        relA = abs(A_g - A_ref) / abs(A_ref)
        relB = abs(B_g - B_ref) / abs(B_ref)
        print(f"  {ord_label:32s} (kf={kf_o:.6f}):  Nt={Nt}  A relerr={relA:.2e}  B relerr={relB:.2e}")
