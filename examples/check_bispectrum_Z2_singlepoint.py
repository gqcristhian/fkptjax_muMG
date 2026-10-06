"""
Single-point (no quadrature, no ODE re-derivation) cross-check of
folps.BispectrumCalculator_fk.Z2's F2/G2 formula against eq. (2.20)-(2.21)
of arXiv:2012.05077, for a genuinely scale-dependent model (HDKI/BZ_Mass),
at one hand-picked, non-squeezed triangle.

This does NOT re-validate the underlying A(k_f,k1,k2)/B(k_f,k1,k2) ODE --
that machinery (fkptjax.MG_kernels.A_B_grid) was already cross-checked
against a from-scratch Wolfram evaluation of sMGPT's own equations in
examples/check_I1udd1_singlepoint.py. What's new here, and what this
script isolates, is purely: given a set of already-computed numeric
A/B/Ap/Bp values, does folps.py's Z2 correctly implement eq. (2.20)-(2.21)'s
F2/G2 algebra? Prints every intermediate quantity so it can be diffed
against a matching Wolfram Cloud/wolframscript snippet
(wolfram_bispectrum_Z2_singlepoint_snippet.wl).
"""
import os
os.environ.setdefault("FOLPS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk
import folps.folps as _folps_module

# ------------------------------------------------------------------
# Model and hand-picked, non-squeezed triangle (same style/model as
# check_I1udd1_singlepoint.py: HDKI/BZ_Mass, mu_kinf=1.2).
# ------------------------------------------------------------------
Om = 0.315192
z = 0.295
xnow = -6.0
xstop = float(np.log(1.0 / (1.0 + z)))

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=1.0, lambda_dS=1.0)

k1 = 0.10
k2 = 0.15
x12 = 0.30
k3 = float(np.sqrt(k1**2 + k2**2 + 2.0 * k1 * k2 * x12))
x23 = -(k2 + k1 * x12) / k3
x31 = -(k1 + k2 * x12) / k3

print(f"model: HDKI/BZ_Mass, mu_kinf=1.2, lambda_a=1.0, lambda_dS=1.0, Om={Om}, z={z}")
print(f"triangle: k1={k1}, k2={k2}, x12={x12} -> k3={k3!r}, x23={x23!r}, x31={x31!r}")


def fgrowth(kk):
    kk = jnp.atleast_1d(jnp.asarray(kk, dtype=jnp.float64))
    from fkptjax import jax_ode
    D, dD = jax_ode.DP_jax(kk, P, xnow, xstop, solver='adaptive')
    return float(np.asarray(dD / D)[0])


f1 = fgrowth(k1)
f2 = fgrowth(k2)
f3 = fgrowth(k3)
print(f"f(k1)={f1!r}  f(k2)={f2!r}  f(k3)={f3!r}")

# ------------------------------------------------------------------
# A(kf,k1,k2)/B(kf,k1,k2)/Ap/Bp at the 3 cyclic leg assignments
# (matches folps.py's bispectrum(): B12 kf=k3, B23 kf=k1, B31 kf=k2).
# ------------------------------------------------------------------
def ab_at(kf, ka, kb):
    A, B, Ap, Bp = mgk.A_B_grid(kf, ka, kb, P, xnow, xstop, solver='adaptive')
    return float(A), float(B), float(Ap), float(Bp)


A12, B12c, Ap12, Bp12 = ab_at(k3, k1, k2)
A23, B23c, Ap23, Bp23 = ab_at(k1, k2, k3)
A31, B31c, Ap31, Bp31 = ab_at(k2, k3, k1)

print(f"leg B12 (kf=k3): A={A12!r} B={B12c!r} Ap={Ap12!r} Bp={Bp12!r}")
print(f"leg B23 (kf=k1): A={A23!r} B={B23c!r} Ap={Ap23!r} Bp={Bp23!r}")
print(f"leg B31 (kf=k2): A={A31!r} B={B31c!r} Ap={Ap31!r} Bp={Bp31!r}")

# ------------------------------------------------------------------
# folps.py's Z2 (isolated: b1=1, b2=bs=0 so term1=term2=0 and Z2 reduces
# to exactly F2 + (mu-weighted G2 piece) -- print F2/G2 directly by
# calling Z2 at mui=muj=0 too, which also kills term2/term4's mu-dependent
# pieces, leaving term3 = b1*F2 = F2 alone; then separately with mui=muj
# chosen nonzero to also expose G2 through term4.
# ------------------------------------------------------------------
bc = _folps_module.BispectrumCalculator_fk(model='FOLPSD')


def z2_F2_only(ki, kj, xij, fi, fj, A, Ap, B, Bp):
    # mui=muj=0 -> term1=term2=term4=0 (km=0), term3=b1*F2 with b1=1.
    return float(bc.Z2(ki, kj, xij, 0.0, 0.0, fi, fi, fj, 1.0, 0.0, 0.0, A, Ap, B, Bp))


F2_12 = z2_F2_only(k1, k2, x12, f1, f2, A12, Ap12, B12c, Bp12)
F2_23 = z2_F2_only(k2, k3, x23, f2, f3, A23, Ap23, B23c, Bp23)
F2_31 = z2_F2_only(k3, k1, x31, f3, f1, A31, Ap31, B31c, Bp31)
print(f"folps Z2-isolated F2: F2(k1,k2)={F2_12!r}  F2(k2,k3)={F2_23!r}  F2(k3,k1)={F2_31!r}")

# ------------------------------------------------------------------
# Independent hand-evaluation of eq. (2.20) (F2 only -- Ap/Bp don't
# enter F2, only G2), for a direct print-to-print diff against the
# Wolfram companion snippet.
# ------------------------------------------------------------------
def F2_eq220(ki, kj, xij, A, B):
    return 0.5 + 3.0 / 14.0 * A + xij / 2.0 * (ki / kj + kj / ki) + (0.5 - 3.0 / 14.0 * B) * xij**2


F2_12_eq = F2_eq220(k1, k2, x12, A12, B12c)
F2_23_eq = F2_eq220(k2, k3, x23, A23, B23c)
F2_31_eq = F2_eq220(k3, k1, x31, A31, B31c)
print(f"eq.(2.20) F2 (this script, independent of Z2): "
      f"F2(k1,k2)={F2_12_eq!r}  F2(k2,k3)={F2_23_eq!r}  F2(k3,k1)={F2_31_eq!r}")

print()
print("Diff Z2-isolated vs this-script eq.(2.20) (should be ~0, both use folps' own A/B):")
print(f"  {abs(F2_12 - F2_12_eq):.3e}  {abs(F2_23 - F2_23_eq):.3e}  {abs(F2_31 - F2_31_eq):.3e}")
print()
print("For the Wolfram cross-check, feed the printed A/B/k/x values above into")
print("wolfram_bispectrum_Z2_singlepoint_snippet.wl and diff its F2 output against")
print("the 'folps Z2-isolated F2' line above.")

# ------------------------------------------------------------------
# G2 (eq. 2.21) -- exercises Ap/Bp, which F2 never touches. Z2 has no
# bias-parameter combination that isolates term4 (G2's home) alone: with
# b1=b2=bs=0, term1=term3=0, but term2 (a purely kinematic RSD-mapping
# term, no calA/calB dependence at all) survives whenever mui/muj are
# nonzero -- and mui=muj=0 would also kill term4 itself (mu2=km^2/... = 0).
# So back G2 out algebraically: with b1=b2=bs=0, Z2 = term2 + term4, and
# term2/mu2/fij are all known kinematic quantities independent of A/B.
# ------------------------------------------------------------------
def z2_full(ki, kj, xij, mui, muj, f0, fi, fj, b1, b2, bs, A, Ap, B, Bp):
    return float(bc.Z2(ki, kj, xij, mui, muj, f0, fi, fj, b1, b2, bs, A, Ap, B, Bp))


def term2_kinematic(ki, kj, mui, muj, fi, fj, b1):
    km = ki * mui + kj * muj
    fij = (fi + fj) / 2.0
    return km / 2.0 * fij * (mui / ki * (b1 + fj * muj**2) + muj / kj * (b1 + fi * mui**2))


def extract_G2(ki, kj, xij, mui, muj, fi, fj, A, Ap, B, Bp):
    b1 = 0.0
    f0 = fi  # only matters through the *(fi/f0*.. + fj/f0*..) terms inside G2 itself
    z2_val = z2_full(ki, kj, xij, mui, muj, f0, fi, fj, b1, 0.0, 0.0, A, Ap, B, Bp)
    t2 = term2_kinematic(ki, kj, mui, muj, fi, fj, b1)
    km = ki * mui + kj * muj
    fij = (fi + fj) / 2.0
    mu2 = km**2 / (ki**2 + kj**2 + 2.0 * ki * kj * xij)
    return (z2_val - t2) / (fij * mu2), z2_val, t2, mu2, fij, f0


mui_pick, muj_pick = 0.5, 0.6
G2_12, z2v_12, t2_12, mu2_12, fij_12, f0_12 = extract_G2(k1, k2, x12, mui_pick, muj_pick, f1, f2, A12, Ap12, B12c, Bp12)
G2_23, z2v_23, t2_23, mu2_23, fij_23, f0_23 = extract_G2(k2, k3, x23, mui_pick, muj_pick, f2, f3, A23, Ap23, B23c, Bp23)
G2_31, z2v_31, t2_31, mu2_31, fij_31, f0_31 = extract_G2(k3, k1, x31, mui_pick, muj_pick, f3, f1, A31, Ap31, B31c, Bp31)

print()
print(f"folps Z2-extracted G2 (mui={mui_pick}, muj={muj_pick}, b1=b2=bs=0):")
print(f"  G2(k1,k2)={G2_12!r}  [z2={z2v_12!r} term2={t2_12!r} mu2={mu2_12!r} fij={fij_12!r} f0={f0_12!r}]")
print(f"  G2(k2,k3)={G2_23!r}  [z2={z2v_23!r} term2={t2_23!r} mu2={mu2_23!r} fij={fij_23!r} f0={f0_23!r}]")
print(f"  G2(k3,k1)={G2_31!r}  [z2={z2v_31!r} term2={t2_31!r} mu2={mu2_31!r} fij={fij_31!r} f0={f0_31!r}]")


def G2_eq221(ki, kj, xij, fi, fj, f0, A, Ap, B, Bp):
    return ((3.0 * A * (fi + fj) + 3.0 * Ap) / (14.0 * f0)
            + xij / 2.0 * (fi / f0 * ki / kj + fj / f0 * kj / ki)
            + ((fi + fj) / (2.0 * f0) - (3.0 * B * (fi + fj) + 3.0 * Bp) / (14.0 * f0)) * xij**2)


G2_12_eq = G2_eq221(k1, k2, x12, f1, f2, f0_12, A12, Ap12, B12c, Bp12)
G2_23_eq = G2_eq221(k2, k3, x23, f2, f3, f0_23, A23, Ap23, B23c, Bp23)
G2_31_eq = G2_eq221(k3, k1, x31, f3, f1, f0_31, A31, Ap31, B31c, Bp31)
print(f"eq.(2.21) G2 (this script, independent of Z2): "
      f"G2(k1,k2)={G2_12_eq!r}  G2(k2,k3)={G2_23_eq!r}  G2(k3,k1)={G2_31_eq!r}")
print("Diff Z2-extracted vs this-script eq.(2.21) (should be ~0):")
print(f"  {abs(G2_12 - G2_12_eq):.3e}  {abs(G2_23 - G2_23_eq):.3e}  {abs(G2_31 - G2_31_eq):.3e}")
print()
print("For the Wolfram G2 cross-check, feed f0_12/f0_23/f0_31 (== f1/f2/f3 here) and")
print("the same A/Ap/B/Bp/k/x values into the G2 section of the .wl companion and")
print("diff its output against the 'folps Z2-extracted G2' line above.")
