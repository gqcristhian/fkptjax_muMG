"""
Diagnostic (not wired into production): does replacing fkptjax's current
"I1udd1tA(Q, clipped domain) + 2*I1udd1a(R, UNCLIPPED domain)" shortcut with
sMGPT's true formula -- tA11(Q)+a11(R)+A11(N), all evaluated at the SAME
(r,x) point on the SAME clipped Q-loop domain, one combined integral
(2_P22type.wl's computeOneKBothExact/AKernelsT) -- change the answer, and by
how much, for a genuine (non-squeezed) HDKI/BZ_Mass model?

Uses fkptjax's own MG_kernels.A_B_grid for all three momentum orderings
(Q: kf=k,k1=q,k2=kminus; R: kf=kminus,k1=k,k2=q; N: kf=q,k1=k,k2=kminus) --
no new ODE code, just three calls to the existing, already-validated
function. Uses the REAL Abacus linear P(k) (GR-evolved to z=0.295, same
convention as check_vs_sMGPT_reference.ipynb), not a toy analytic spectrum.
"""
import numpy as np
from scipy.interpolate import CubicSpline
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fkptjax import mg_jax as bj
from fkptjax import jax_ode
from fkptjax import MG_kernels as mgk
from fkptjax.ode import ModelDerivatives, DP, ODESolver

# ----------------------------------------------------------------------------
# Real P(k): Abacus linear spectrum, GR-evolved z=0 -> z=0.295 (same recipe as
# examples/check_vs_sMGPT_reference.ipynb)
# ----------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "sMGPT_reference_data")
Om = 0.315192
z_target = 0.295
xnow_gr = -3.912023

k_abacus, pk_abacus_z0 = np.loadtxt(os.path.join(DATA_DIR, "Abacus_pklin_z0.dat"), unpack=True)

gr_derivs = ModelDerivatives(om=Om, ol=1.0 - Om, model="HDKI", mg_variant="mu_OmDE", mu0=0.0)
D_gr_0 = DP(1e-3, gr_derivs, ODESolver(zout=0.0, xnow=xnow_gr, method="RKQS"))[0]
D_gr_target = DP(1e-3, gr_derivs, ODESolver(zout=z_target, xnow=xnow_gr, method="RKQS"))[0]
gr_growth_ratio2 = (D_gr_target / D_gr_0) ** 2
print(f"pure-GR (D(z={z_target})/D(z=0))^2 = {gr_growth_ratio2:.6f}")

pk_target = pk_abacus_z0 * gr_growth_ratio2
logk_abacus = np.log(k_abacus)
logpk_spline = CubicSpline(logk_abacus, np.log(pk_target))
kmin_tab, kmax_tab = k_abacus.min(), k_abacus.max()

def PSL(kk):
    kk = np.clip(np.asarray(kk, dtype=float), kmin_tab, kmax_tab)
    return np.exp(logpk_spline(np.log(kk)))

# ----------------------------------------------------------------------------
# Model / grid -- same BZ_Mass point used in the earlier loop-integral check
# ----------------------------------------------------------------------------
etaini = -6.0
etaev = float(np.log(1.0 / (1.0 + z_target)))
k_ext = 0.18205642
Nq, Nx = 40, 16
# Modest, "reasonable" loop-integral range (matches the kmin/kmax used in the
# real Kfuncs_to_tables_jax smoke test) -- NOT the raw table's full extent
# (1e-4 to 20), which drives the SPT kernels into an unphysically wide r=q/k
# range and was the actual cause of an earlier ~1e5-magnitude blowup here,
# not a real formula bug (see chat/memory).
kmin, kmax = 0.005, 2.0

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

def fgrowth(kk):
    kk = jnp.atleast_1d(jnp.asarray(kk, dtype=jnp.float64))
    D, dD = jax_ode.DP_jax(kk, P, etaini, etaev, solver='adaptive')
    return np.asarray(dD / D)

f0 = fgrowth(1.0e-6)[0]
fk = fgrowth(k_ext)[0]
print(f"f0 = {f0!r}   fk = {fk!r}")

qGrid = np.logspace(np.log10(kmin), np.log10(kmax), Nq)

def gauss_legendre_ab(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w

def trapz_q(vals):
    vals = np.asarray(vals)
    dq = (qGrid[1:] - qGrid[:-1]).reshape((-1,) + (1,) * (vals.ndim - 1))
    return np.sum(0.5 * (vals[1:] + vals[:-1]) * dq, axis=0)

def clip_domain(r):
    rmax, rmin = kmax / k_ext, kmin / k_ext
    muMin = max(-1.0, (1.0 + r * r - rmax * rmax) / (2.0 * r))
    muMax = 0.5 / r if r >= 0.5 else min(1.0, (1.0 + r * r - rmin * rmin) / (2.0 * r))
    return muMin, muMax

C2 = 3.0 / 7.0

# ----------------------------------------------------------------------------
# (a) CURRENT fkptjax formula: I1udd1tA(Q, clipped) + 2*I1udd1a(R, UNCLIPPED)
# ----------------------------------------------------------------------------
def angular_I1udd1tA(q):
    r = q / k_ext
    muMin, muMax = clip_domain(r)
    xv, wv = gauss_legendre_ab(Nx, muMin, muMax)
    y = np.sqrt(1.0 + r * r - 2.0 * r * xv)
    kminus = k_ext * y

    A_Q, B_Q, Ap_Q, Bp_Q = mgk.A_B_grid(k_ext, q, kminus, P, etaini, etaev, solver='adaptive')
    A_Q, B_Q, Ap_Q, Bp_Q = map(np.asarray, (A_Q, B_Q, Ap_Q, Bp_Q))
    fp = fgrowth(q)[0]
    fkm = fgrowth(kminus)

    AngleEvQ = (xv - r) / y
    F2Q = 0.5 + 3./14.*A_Q + (0.5 - 3./14.*B_Q)*AngleEvQ**2 + AngleEvQ/2.*(y/r + r/y)
    tA11 = 2.0 * F2Q * (xv*r*fp/f0 + (r*r*(1.0 - r*xv))/(y*y) * fkm/f0)
    return q*q * np.sum(wv * PSL(q) * PSL(kminus) * tA11)

def angular_I1udd1a_v2(q):
    r = q / k_ext
    xv, wv = gauss_legendre_ab(Nx, -1.0, 1.0)
    y2 = 1.0 + r*r - 2.0*r*xv

    A_R, B_R, Ap_R, Bp_R = mgk.A_B_grid(k_ext, k_ext, q, P, etaini, etaev, solver='adaptive')
    A_R, B_R, Ap_R, Bp_R = map(np.asarray, (A_R, B_R, Ap_R, Bp_R))
    fp = fgrowth(q)[0]

    AngleEvR = -xv
    F2R = 0.5 + 3./14.*A_R + (0.5 - 3./14.*B_R)*AngleEvR**2 + AngleEvR/2.*(1./r + r)
    G2R = 3./14.*A_R*(fp+fk)/f0 + 3./14.*Ap_R/f0 + (0.5*(fp+fk) - 3./14.*B_R*(fp+fk) - 3./14.*Bp_R)*AngleEvR**2/f0 \
        + AngleEvR/(2*f0)*(fk/r + fp*r)
    a11 = 2.0*(F2R*xv*r*fp/f0 + G2R*(r*r*(1.0 - r*xv))/y2)
    return q*q * np.sum(wv * PSL(q) * a11)

print("Computing OLD formula (current fkptjax: Q-clipped + 2xR-unclipped)...")
valsQ = np.array([angular_I1udd1tA(q) for q in qGrid])
valsR = np.array([angular_I1udd1a_v2(q) for q in qGrid])
I1udd1tA_total = (1.0/(2.0*np.pi**2)) * trapz_q(valsQ)
I1udd1a_total  = (k_ext**2 * PSL(k_ext) / (2.0*np.pi**2)) * trapz_q(valsR)
old_total = I1udd1tA_total + 2.0 * I1udd1a_total
print(f"OLD I1udd1A = {old_total!r}  [Q-part={I1udd1tA_total!r}  R-part={I1udd1a_total!r}]")

# ----------------------------------------------------------------------------
# (b) TRUE sMGPT formula: tA11(Q)+a11(R)+A11(N), all on the SAME clipped
#     Q-domain, ONE combined integral (matches computeOneKBothExact exactly)
# ----------------------------------------------------------------------------
def angular_I1udd1_true(q):
    r = q / k_ext
    muMin, muMax = clip_domain(r)
    xv, wv = gauss_legendre_ab(Nx, muMin, muMax)
    y = np.sqrt(1.0 + r * r - 2.0 * r * xv)
    y2 = y * y
    kminus = k_ext * y

    A_Q, B_Q, Ap_Q, Bp_Q = mgk.A_B_grid(k_ext, q, kminus, P, etaini, etaev, solver='adaptive')
    A_R, B_R, Ap_R, Bp_R = mgk.A_B_grid(kminus, k_ext, q, P, etaini, etaev, solver='adaptive')
    A_N, B_N, Ap_N, Bp_N = mgk.A_B_grid(q, k_ext, kminus, P, etaini, etaev, solver='adaptive')
    A_Q, B_Q, Ap_Q, Bp_Q = map(np.asarray, (A_Q, B_Q, Ap_Q, Bp_Q))
    A_R, B_R, Ap_R, Bp_R = map(np.asarray, (A_R, B_R, Ap_R, Bp_R))
    A_N, B_N, Ap_N, Bp_N = map(np.asarray, (A_N, B_N, Ap_N, Bp_N))

    fp = fgrowth(q)[0]
    fkm = fgrowth(kminus)

    AngleEvQ = (xv - r) / y
    F2Q = 0.5 + 3./14.*A_Q + (0.5 - 3./14.*B_Q)*AngleEvQ**2 + AngleEvQ/2.*(y/r + r/y)
    tA11 = 2.0 * F2Q * (xv*r*fp/f0 + (r*r*(1.0 - r*xv))/y2 * fkm/f0)

    AngleEvR = -xv
    F2R = 0.5 + 3./14.*A_R + (0.5 - 3./14.*B_R)*AngleEvR**2 + AngleEvR/2.*(1./r + r)
    G2R = 3./14.*A_R*(fp+fk)/f0 + 3./14.*Ap_R/f0 + (0.5*(fp+fk) - 3./14.*B_R*(fp+fk) - 3./14.*Bp_R)*AngleEvR**2/f0 \
        + AngleEvR/(2*f0)*(fk/r + fp*r)
    a11 = 2.0*(F2R*xv*r*fp/f0 + G2R*(r*r*(1.0 - r*xv))/y2)

    AngleEvN = -((1.0 - r*xv) / y)
    F2N = 0.5 + 3./14.*A_N + (0.5 - 3./14.*B_N)*AngleEvN**2 + AngleEvN/2.*(y + 1./y)
    G2N = 3./14.*A_N*(fk+fkm)/f0 + 3./14.*Ap_N/f0 + (0.5*(fk+fkm) - 3./14.*B_N*(fk+fkm) - 3./14.*Bp_N)*AngleEvN**2/f0 \
        + AngleEvN/(2*f0)*(fkm*y + fk/y)
    A11 = 2.0*(G2N*xv*r + F2N*(r*r*(1.0 - r*xv))/y2 * fkm/f0)

    PSLk, PSLp, PSLkm = PSL(k_ext), PSL(q), PSL(kminus)
    kern = A11*PSLk*PSLkm + tA11*PSLp*PSLkm + a11*PSLk*PSLp
    return q*q * np.sum(wv * kern)

print("Computing TRUE formula (single clipped-domain, 3-ordering combined integral)...")
valsTrue = np.array([angular_I1udd1_true(q) for q in qGrid])
true_total = (1.0/(2.0*np.pi**2)) * trapz_q(valsTrue)
print(f"TRUE I1udd1 = {true_total!r}")

print()
print(f"ratio true/old = {true_total/old_total!r}")
