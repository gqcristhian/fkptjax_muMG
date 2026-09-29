"""
Standalone cross-check: fkptjax's MG_kernels.py loop-integral machinery vs the
sMGPT Wolfram Cloud reference in ../wolfram_bzmass_loopintegral_snippet.txt.

Reuses fkptjax's actual, already-validated building blocks
(MG_kernels.A_B_grid / D2_fused_grid / D3_fused_grid, jax_ode.DP_jax) and the
exact same kernel-assembly formulas as calculate_jax.py (F2evQ/G2evQ,
F3K/G3K), but glues them together with the SAME simple Nq=10 log-spaced
trapezoidal / Nx=8 Gauss-Legendre quadrature the Wolfram snippet uses --
NOT fkptjax's own production quadrature grid -- so the only thing being
tested is the kernel building blocks, apples-to-apples against the Wolfram
numbers.

Model: HDKI/BZ_Mass, mu_kinf=1.2, lambda_a=lambda_dS=100.0,
Om=0.315192, z=0.295. Toy P(k) = 2000*(k/0.05)^-1.5*exp(-(k/0.3)^2).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fkptjax import mg_jax as bj
from fkptjax import jax_ode
from fkptjax import MG_kernels as mgk

# ----------------------------------------------------------------------------
# Model / grid setup -- matches wolfram_bzmass_loopintegral_snippet.txt exactly
# ----------------------------------------------------------------------------
om = 0.315192
z = 0.295
etaini = -6.0
etaev = float(np.log(1.0 / (1.0 + z)))
kmin, kmax = 0.001, 1.0
k_ext = 0.0545559527
Nq, Nx = 10, 8
C2 = 3.0 / 7.0

P = bj.pack_constants_jnp(om=om, ol=1.0 - om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

def PSL(kk):
    return 2000.0 * (kk / 0.05) ** (-1.5) * np.exp(-(kk / 0.3) ** 2)

def fgrowth(kk):
    """f(k) = D'(k)/D(k) at etaev, for a scalar or array kk."""
    kk = jnp.atleast_1d(jnp.asarray(kk, dtype=jnp.float64))
    D, dD = jax_ode.DP_jax(kk, P, etaini, etaev, solver='adaptive')
    return np.asarray(dD / D)

f0 = fgrowth(1.0e-6)[0]
fk = fgrowth(k_ext)[0]
print(f"f0 = {f0!r}   fk = {fk!r}")

# log-spaced q grid, Nq points from kmin to kmax inclusive (matches Wolfram's
# Table[10.^lq, {lq, Log10[kmin], Log10[kmax], (Log10[kmax]-Log10[kmin])/(Nq-1)}])
qGrid = np.logspace(np.log10(kmin), np.log10(kmax), Nq)

# Gauss-Legendre nodes/weights on an arbitrary [a,b] (mirrors Wolfram's
# GaussianQuadratureWeights[n, a, b])
def gauss_legendre_ab(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    xs = 0.5 * (b - a) * x + 0.5 * (a + b)
    ws = 0.5 * (b - a) * w
    return xs, ws

def trapz_q(vals):
    vals = np.asarray(vals)
    dq = (qGrid[1:] - qGrid[:-1]).reshape((-1,) + (1,) * (vals.ndim - 1))
    return np.sum(0.5 * (vals[1:] + vals[:-1]) * dq, axis=0)

# ----------------------------------------------------------------------------
# P22 (Q-loop)
# ----------------------------------------------------------------------------
def angular_sum_P22(q):
    r = q / k_ext
    rmax, rmin = kmax / k_ext, kmin / k_ext
    muMin = max(-1.0, (1.0 + r * r - rmax * rmax) / (2.0 * r))
    muMax = 0.5 / r if r >= 0.5 else min(1.0, (1.0 + r * r - rmin * rmin) / (2.0 * r))
    xNodes, wNodes = gauss_legendre_ab(Nx, muMin, muMax)

    y = np.sqrt(1.0 + r * r - 2.0 * r * xNodes)
    kminus = k_ext * y

    A_Q, B_Q, Ap_Q, Bp_Q = mgk.A_B_grid(k_ext, q, kminus, P, etaini, etaev, solver='adaptive')
    A_Q, B_Q, Ap_Q, Bp_Q = map(np.asarray, (A_Q, B_Q, Ap_Q, Bp_Q))
    fp = fgrowth(q)[0]
    fkm = fgrowth(kminus)

    AngleEvQ = (xNodes - r) / y
    F2v = 0.5 + 3.0 / 14.0 * A_Q + (0.5 - 3.0 / 14.0 * B_Q) * AngleEvQ ** 2 \
        + AngleEvQ / 2.0 * (y / r + r / y)
    G2v = 3.0 / 14.0 * A_Q * (fp + fkm) / f0 + 3.0 / 14.0 * Ap_Q / f0 \
        + (0.5 * (fp + fkm) - 3.0 / 14.0 * B_Q * (fp + fkm) - 3.0 / 14.0 * Bp_Q) * AngleEvQ ** 2 / f0 \
        + AngleEvQ / (2.0 * f0) * (fkm * y / r + fp * r / y)

    total = np.sum(wNodes * PSL(q) * PSL(kminus) * np.stack(
        [2.0 * r * r * F2v ** 2, 2.0 * r * r * F2v * G2v, 2.0 * r * r * G2v ** 2]), axis=1)
    return q * q * total

print("Computing P22 loop...")
valsP22 = np.stack([angular_sum_P22(q) for q in qGrid])  # (Nq, 3)
P22dd, P22du, P22uu = (1.0 / (2.0 * np.pi ** 2)) * trapz_q(valsP22)
print(f"=== P22 (Q-loop) at k={k_ext} ===")
print(f"P22dd={P22dd!r}  P22du={P22du!r}  P22uu={P22uu!r}")

# ----------------------------------------------------------------------------
# P13 (R-loop)
# ----------------------------------------------------------------------------
xNodesR, wNodesR = gauss_legendre_ab(Nx, -1.0, 1.0)

def angular_sum_P13(q):
    r = q / k_ext

    Gamma2evR, Gamma2evR_p = mgk.D2_fused_grid(-xNodesR, k_ext, q, P, etaini, etaev, solver='adaptive')
    Gamma2evR, Gamma2evR_p = np.asarray(Gamma2evR), np.asarray(Gamma2evR_p)
    fp = fgrowth(q)[0]

    kminusPlus = k_ext * np.sqrt(1.0 + r * r - 2.0 * r * xNodesR)
    fkmP = fgrowth(kminusPlus)

    Gamma2fevR = Gamma2evR * (fkmP + fp) / (2.0 * f0) + Gamma2evR_p / (2.0 * f0)

    CFD3, CFD3p = mgk.D3_fused_grid(xNodesR, k_ext, q, P, etaini, etaev, f0, solver='adaptive')
    c3gamma3 = np.asarray(CFD3) * 5.0 / 21.0
    c3gamma3f = np.asarray(CFD3p) * 5.0 / 21.0

    k2, q2 = k_ext ** 2, q ** 2
    denomR = k2 + q2 - 2.0 * q * k_ext * xNodesR
    F3Kv = c3gamma3 / 6.0 + 1.0 / 3.0 * C2 * ((k2 - k_ext * q * xNodesR) * k_ext * q * xNodesR) / (q2 * denomR) * Gamma2evR \
        - 1.0 / 6.0 * k2 / q2 * xNodesR ** 2
    G3Kv = c3gamma3f / 2.0 + 2.0 / 3.0 * C2 * (k_ext * xNodesR) / q * Gamma2fevR \
        + 1.0 / 3.0 * C2 * (fk / f0) * (k2 - k_ext * q * xNodesR) / denomR * Gamma2evR \
        - 1.0 / 6.0 * k2 / q2 * xNodesR ** 2 * fk / f0 \
        - 1.0 / 3.0 * C2 * (2.0 * Gamma2fevR + Gamma2evR * fp / f0) * (1.0 - (k_ext * xNodesR - q) ** 2 / denomR)

    r2 = r * r
    total = np.sum(wNodesR * PSL(q) * np.stack(
        [6.0 * r2 * F3Kv, 3.0 * r2 * G3Kv + 3.0 * r2 * F3Kv * fk / f0, 6.0 * r2 * G3Kv * fk / f0]), axis=1)
    return total

print(f"Computing P13 loop ({Nq * Nx} points, 3 ODE solves each)...")
valsP13 = np.stack([angular_sum_P13(q) for q in qGrid])  # (Nq, 3)
P13dd, P13du, P13uu = (k_ext ** 2 * PSL(k_ext) / (4.0 * np.pi ** 2)) * trapz_q(valsP13)
print(f"=== P13 (R-loop) at k={k_ext} ===")
print(f"P13dd={P13dd!r}  P13du={P13du!r}  P13uu={P13uu!r}")
