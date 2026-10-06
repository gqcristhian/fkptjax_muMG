"""
Test whether A_B_grid (8-state secondOrderAB, Q-loop grid) and
D2_D3_fused_grid (10-state thirdOrder, R-loop grid) can be merged into ONE
vmap call, via state-padding (pad the 8-state system to 10) + a per-point
mode flag selecting which RHS/IC to use (both branches computed for every
point via jnp.where -- no actual control-flow branching under vmap/jit).

Validates against BOTH a scale-independent model (mu_OmDE) and a
scale-dependent one (Hu-Sawicki f(R)) -- the earlier A_B_grid merge bug was
invisible in the scale-independent case and only showed up for f(R).
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
from fkptjax.kfuncs_to_tables import build_jax_static_ctx

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

xnow = -3.912023
z_pk_cmp = 0.3
xstop = float(jnp.log(1.0 / (1.0 + z_pk_cmp)))
f0_test = 0.7


def combined_rhs(eta, y, kf, k1, k2, x, k, p, mode, P):
    y8 = y[:8]
    rhs8 = mgk.secondOrderAB(eta, y8, kf, k1, k2, P)
    rhs_A_padded = jnp.concatenate([rhs8, jnp.zeros(2, dtype=y.dtype)])
    rhs_D3 = bj.thirdOrder(eta, y, x, k, p, P)
    return jnp.where(mode < 0.5, rhs_A_padded, rhs_D3)


def combined_y0(x, k, p, mode):
    e0 = jnp.exp(xnow); e2 = jnp.exp(2.0 * xnow); e3 = jnp.exp(3.0 * xnow)
    D2plusi = 3.0 * e2 / 7.0
    dD2plusi = 6.0 * e2 / 7.0
    y0_A = jnp.stack([e0, e0, e0, e0, D2plusi, dD2plusi, D2plusi, dD2plusi, 0.0 * e0, 0.0 * e0])

    one_m_x2 = 1.0 - x * x
    pk = p / k
    ang = 1.0 / (1.0 + pk * pk + 2.0 * pk * x) + 1.0 / (1.0 + pk * pk - 2.0 * pk * x)
    D2i = 3.0 * e2 / 7.0 * one_m_x2
    dD2i = 6.0 * e2 / 7.0 * one_m_x2
    D3i = (5.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang
    dD3i = (15.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang
    y0_D3 = jnp.stack([e0, e0, e0, e0, D2i, dD2i, D2i, dD2i, D3i, dD3i])

    return jnp.where(mode < 0.5, y0_A, y0_D3)


def solve_one(kf, k1, k2, x, k, p, mode, P):
    y0 = combined_y0(x, k, p, mode)
    term = diffrax.ODETerm(lambda t, y, args: combined_rhs(t, y, kf, k1, k2, x, k, p, mode, P))
    solver = diffrax.Tsit5()
    ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-11)
    sol = diffrax.diffeqsolve(term, solver, t0=xnow, t1=xstop, dt0=0.01, y0=y0,
                              stepsize_controller=ctrl, saveat=diffrax.SaveAt(t1=True),
                              max_steps=100000)
    return sol.ys[0]


solve_batch = jax.vmap(solve_one, in_axes=(0, 0, 0, 0, 0, 0, 0, None))


def run_combined(kf_Q, q_Q, kminus_Q, x_r, k_ext_r, q_loop_R, P, f0):
    npts_Q = kf_Q.size
    npts_R = x_r.size
    zeros_Q3 = jnp.zeros(3 * npts_Q)
    zeros_R = jnp.zeros(npts_R)
    ones_R = jnp.ones(npts_R)
    flat = lambda a: a.reshape(-1)  # noqa: E731

    # Q, N, R orderings (mode=0, secondOrderAB) concatenated, THEN the R-grid
    # D2_D3 points (mode=1, thirdOrder) -- i.e. the full scope of both
    # existing merged calls, combined into one.
    kf_qnr = jnp.concatenate([flat(kf_Q), flat(q_Q), flat(kminus_Q)])
    k1_qnr = jnp.concatenate([flat(q_Q), flat(kf_Q), flat(kf_Q)])
    k2_qnr = jnp.concatenate([flat(kminus_Q), flat(kminus_Q), flat(q_Q)])

    kf_all = jnp.concatenate([kf_qnr, flat(k_ext_r)])
    k1_all = jnp.concatenate([k1_qnr, zeros_R])
    k2_all = jnp.concatenate([k2_qnr, zeros_R])
    x_all = jnp.concatenate([zeros_Q3, flat(x_r)])
    k_all = jnp.concatenate([zeros_Q3, flat(k_ext_r)])
    p_all = jnp.concatenate([zeros_Q3, flat(q_loop_R)])
    mode_all = jnp.concatenate([zeros_Q3, ones_R])

    Y = solve_batch(kf_all, k1_all, k2_all, x_all, k_all, p_all, mode_all, P)
    jax.block_until_ready(Y)

    split = lambda i: Y[i * npts_Q:(i + 1) * npts_Q]  # noqa: E731

    def AB_from(Yb):
        Dk1, dDk1, Dk2, dDk2, DA, dDA, DB, dDB = (Yb[:, i] for i in range(8))
        C = (3.0 / 7.0) * Dk1 * Dk2
        Cp = (3.0 / 7.0) * (dDk1 * Dk2 + Dk1 * dDk2)
        A = DA / C; B = DB / C
        Ap = dDA / C - DA * Cp / (C * C)
        Bp = dDB / C - DB * Cp / (C * C)
        return A, B, Ap, Bp

    A_Q, B_Q, Ap_Q, Bp_Q = AB_from(split(0))
    A_N, B_N, Ap_N, Bp_N = AB_from(split(1))
    A_R, B_R, Ap_R, Bp_R = AB_from(split(2))

    Y_R = Y[3 * npts_Q:]
    Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3 = (Y_R[:, i] for i in range(10))
    Cr = (3.0 / 7.0) * Dk * Dp
    Cpr = (3.0 / 7.0) * (dDk * Dp + Dk * dDp)
    A_fused = D2m / Cr
    Ap_fused = dD2m / Cr - D2m * Cpr / (Cr * Cr)
    denom = Dk * Dp * Dp
    CFD3 = (21.0 / 5.0) * D3 / denom
    CFD3p = (21.0 / 5.0) * dD3 / denom / (3.0 * f0)

    return (A_Q, B_Q, A_N, B_N, A_R, B_R, A_fused, Ap_fused, CFD3, CFD3p)


def build_grids(static_ctx):
    calculator = static_ctx["calculator"]
    k_ext_q = calculator.logk_grid_jax
    r_q = calculator.r_jax
    x_q = calculator.x_jax
    y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
    q_loop_Q = r_q * k_ext_q
    kminus_Q = k_ext_q * y_q
    kf_Q, q_Q, kminus_Q = jnp.broadcast_arrays(k_ext_q * jnp.ones_like(y_q), q_loop_Q, kminus_Q)

    r_r = calculator.r_r_jax
    x_r = calculator.x_r_jax
    q_loop_R = r_r * k_ext_q
    k_ext_r, x_r, q_loop_R = jnp.broadcast_arrays(k_ext_q * jnp.ones_like(x_r), x_r, q_loop_R)
    return kf_Q, q_Q, kminus_Q, x_r, k_ext_r, q_loop_R


def check_model(label, P, f0):
    print(f"\n=== {label} ===")
    k_lin = jnp.asarray(np.geomspace(1e-3, 1.0, 256))
    static_ctx = build_jax_static_ctx(
        k_lin, kmin=1.0e-3, kmax=1.0, Nk_kernel=20, nquadSteps=48, NQ=6, NR=6,
        rbao=104.0, pmax_bao=0.4, Np_bao=100,
    )
    kf_Q, q_Q, kminus_Q, x_r, k_ext_r, q_loop_R = build_grids(static_ctx)
    print(f"Q-grid npts={kf_Q.size}  R-grid npts={x_r.size}")

    # Reference: current production functions (post A_B_grid-merge and
    # D2_D3_fused_grid fixes), called the same way kfuncs_to_tables.py does.
    def run_ref():
        A_r, B_r, _, _ = mgk.A_B_grid(kf_Q, q_Q, kminus_Q, P, xnow, xstop)
        A_Nr, B_Nr, _, _ = mgk.A_B_grid(q_Q, kf_Q, kminus_Q, P, xnow, xstop)
        A_Rr, B_Rr, _, _ = mgk.A_B_grid(kminus_Q, kf_Q, q_Q, P, xnow, xstop)
        Af_r, Apf_r, CFD3_r, CFD3p_r = mgk.D2_D3_fused_grid(x_r, k_ext_r, q_loop_R, P, xnow, xstop, f0)
        out = (A_r, B_r, A_Nr, B_Nr, A_Rr, B_Rr, Af_r, Apf_r, CFD3_r, CFD3p_r)
        jax.block_until_ready(out)
        return out

    t0 = time.time()
    run_ref()
    warm_ref = time.time() - t0
    t0 = time.time()
    (A_ref, B_ref, A_N_ref, B_N_ref, A_R_ref, B_R_ref,
     Afus_ref, Apfus_ref, CFD3_ref, CFD3p_ref) = run_ref()
    iter_ref = time.time() - t0
    print(f"reference (current production, separate calls): warmup={warm_ref:.3f}s  post-compile={iter_ref:.3f}s")

    t0 = time.time()
    out_new = run_combined(kf_Q, q_Q, kminus_Q, x_r, k_ext_r, q_loop_R, P, f0)
    warm_new = time.time() - t0
    t0 = time.time()
    out_new = run_combined(kf_Q, q_Q, kminus_Q, x_r, k_ext_r, q_loop_R, P, f0)
    iter_new = time.time() - t0
    A_Q_new, B_Q_new, A_N_new, B_N_new, A_R_new, B_R_new, Afus_new, Apfus_new, CFD3_new, CFD3p_new = out_new
    print(f"combined-merge call: warmup={warm_new:.3f}s  post-compile={iter_new:.3f}s"
          f"  (speedup vs current production: {iter_ref/iter_new:.2f}x)")

    for name, ref, new in [("A_Q", A_ref, A_Q_new), ("B_Q", B_ref, B_Q_new),
                            ("A_N", A_N_ref, A_N_new), ("B_N", B_N_ref, B_N_new),
                            ("A_R", A_R_ref, A_R_new), ("B_R", B_R_ref, B_R_new),
                            ("A_fused", Afus_ref, Afus_new), ("Ap_fused", Apfus_ref, Apfus_new),
                            ("CFD3", CFD3_ref, CFD3_new), ("CFD3p", CFD3p_ref, CFD3p_new)]:
        rel = jnp.abs(new.reshape(-1) - ref.reshape(-1)) / jnp.maximum(1.0, jnp.abs(ref.reshape(-1)))
        print(f"  {name:10s} max relerr = {float(jnp.max(rel)):.3e}")


P_mu = bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.MU_OMDE, mu0=-0.5)
check_model("HDKI/mu_OmDE (scale-independent)", P_mu, 0.7)

P_hs = bj.pack_constants_jnp(om=0.315, ol=0.685, kind=bj.HS, fR0_HS=-1.0e-5, beta2=1.0 / 6.0, n_HS=1)
check_model("Hu-Sawicki f(R) (scale-dependent)", P_hs, 0.684640)
