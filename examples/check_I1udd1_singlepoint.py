"""
Single-point (no quadrature at all) cross-check of the I1udd1-family
combined-kernel formulas for a genuinely scale-dependent model
(HDKI/BZ_Mass, mu_kinf=1.2), at one hand-picked (k_ext, q, x) triple.

Prints every intermediate quantity (A_Q/A_R/A_N, F2/G2 for each ordering,
and all 15 combined tA_ij/a_ij/A_ij kernels) so they can be diffed
term-by-term against a matching Wolfram Cloud snippet -- no integration,
no quadrature sensitivity, just raw formula evaluation. Uses the exact
same formulas as the validated fix in calculate_jax.py.
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

Om = 0.315192
z = 0.295
etaini = -6.0
etaev = float(np.log(1.0 / (1.0 + z)))

P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                          mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

# Hand-picked, non-special triple (no collinear/squeezed edge cases)
k_ext = 0.18205642
q = 0.05
x = 0.3
y = np.sqrt(1.0 + (q/k_ext)**2 - 2.0*(q/k_ext)*x)
kminus = k_ext * y
r = q / k_ext

print(f"k_ext={k_ext!r}  q={q!r}  x={x!r}")
print(f"r={r!r}  y={y!r}  kminus={kminus!r}")


def fgrowth(kk):
    kk = jnp.atleast_1d(jnp.asarray(kk, dtype=jnp.float64))
    D, dD = jax_ode.DP_jax(kk, P, etaini, etaev, solver='adaptive')
    return float(np.asarray(dD / D)[0])


f0 = fgrowth(1.0e-6)
fk = fgrowth(k_ext)
fp = fgrowth(q)
fkm = fgrowth(kminus)
print(f"f0={f0!r}  fk={fk!r}  fp={fp!r}  fkm={fkm!r}")

# Q ordering: kf=k_ext, k1=q, k2=kminus
A_Q, B_Q, Ap_Q, Bp_Q = mgk.A_B_grid(k_ext, q, kminus, P, etaini, etaev, solver='adaptive')
A_Q, B_Q, Ap_Q, Bp_Q = float(A_Q), float(B_Q), float(Ap_Q), float(Bp_Q)
print(f"Q ordering: A_Q={A_Q!r} B_Q={B_Q!r} Ap_Q={Ap_Q!r} Bp_Q={Bp_Q!r}")

# R ordering: kf=kminus, k1=k_ext, k2=q
A_R, B_R, Ap_R, Bp_R = mgk.A_B_grid(kminus, k_ext, q, P, etaini, etaev, solver='adaptive')
A_R, B_R, Ap_R, Bp_R = float(A_R), float(B_R), float(Ap_R), float(Bp_R)
print(f"R ordering: A_R={A_R!r} B_R={B_R!r} Ap_R={Ap_R!r} Bp_R={Bp_R!r}")

# N ordering: kf=q, k1=k_ext, k2=kminus
A_N, B_N, Ap_N, Bp_N = mgk.A_B_grid(q, k_ext, kminus, P, etaini, etaev, solver='adaptive')
A_N, B_N, Ap_N, Bp_N = float(A_N), float(B_N), float(Ap_N), float(Bp_N)
print(f"N ordering: A_N={A_N!r} B_N={B_N!r} Ap_N={Ap_N!r} Bp_N={Bp_N!r}")

# --- kernel formulas, matching calculate_jax.py exactly ---
r2 = r*r
x2 = x*x
y2 = y*y
rx = r*x

AngleEvQ = (x - r)/y
F2evQ = 0.5 + 3./14.*A_Q + (0.5 - 3./14.*B_Q)*AngleEvQ**2 + AngleEvQ/2.*(y/r + r/y)
G2evQ = (3./14.*A_Q*(fp+fkm) + 3./14.*Ap_Q +
         (0.5*(fp+fkm) - 3./14.*B_Q*(fp+fkm) - 3./14.*Bp_Q)*AngleEvQ**2 +
         AngleEvQ/2.*(fkm*y/r + fp*r/y))
print(f"AngleEvQ={AngleEvQ!r}  F2evQ={F2evQ!r}  G2evQ={G2evQ!r}")

AngleEvR = -x
F2evR = 0.5 + 3./14.*A_R + (0.5 - 3./14.*B_R)*AngleEvR**2 + AngleEvR/2.*(1./r + r)
G2evR = (3./14.*A_R*(fp+fk) + 3./14.*Ap_R +
         (0.5*(fp+fk) - 3./14.*B_R*(fp+fk) - 3./14.*Bp_R)*AngleEvR**2 +
         AngleEvR/2.*(fk/r + fp*r))
print(f"AngleEvR={AngleEvR!r}  F2evR={F2evR!r}  G2evR={G2evR!r}")

AngleEvN = -((1.0 - rx)/y)
F2evN = 0.5 + 3./14.*A_N + (0.5 - 3./14.*B_N)*AngleEvN**2 + AngleEvN/2.*(y + 1./y)
G2evN = (3./14.*A_N*(fk+fkm) + 3./14.*Ap_N +
         (0.5*(fk+fkm) - 3./14.*B_N*(fk+fkm) - 3./14.*Bp_N)*AngleEvN**2 +
         AngleEvN/2.*(fkm*y + fk/y))
print(f"AngleEvN={AngleEvN!r}  F2evN={F2evN!r}  G2evN={G2evN!r}")

tA11 = 2.0*(fp*rx + fkm*r2*(1.0 - rx)/y2)*F2evQ
tA12 = -fp*fkm*r2*(1.0 - x2)/y2*F2evQ
tA22 = (2.0*(fp*rx + fkm*r2*(1.0 - rx)/y2)*G2evQ
        + fp*fkm*(r2*(1.0 - 3.0*x2) + 2.0*rx)/y2*F2evQ)
tA23 = fp*fkm*r2*(x2 - 1.0)/y2*G2evQ
tA33 = fp*fkm*(r2*(1.0 - 3.0*x2) + 2.0*rx)/y2*G2evQ

a11 = 2.0*(F2evR*rx*fp + G2evR*r2*(1.0 - rx)/y2)
a12 = -(r2*(1.0 - x2)/y2)*fp*G2evR
a22 = (((r2*(1.0 - 3.0*x2) + 2.0*rx)/y2*fp + (2.0*r2*(1.0 - rx))/y2*fk)*G2evR
       + 2.0*rx*fk*fp*F2evR)
a23 = r2*(x2 - 1.0)/y2*fp*fk*G2evR
a33 = (r2*(1.0 - 3.0*x2) + 2.0*rx)/y2*fp*fk*G2evR

A11 = 2.0*(G2evN*rx + F2evN*r2*(1.0 - rx)/y2*fkm)
A12 = -(r2*(1.0 - x2)/y2)*fkm*G2evN
A22 = (((r2*(1.0 - 3.0*x2) + 2.0*rx)/y2*fkm + 2.0*rx*fk)*G2evN
       + (2.0*r2*(1.0 - rx))/y2*fkm*fk*F2evN)
A23 = r2*(x2 - 1.0)/y2*fkm*fk*G2evN
A33 = (r2*(1.0 - 3.0*x2) + 2.0*rx)/y2*fkm*fk*G2evN

print()
print(f"tA11={tA11!r}  a11={a11!r}  A11={A11!r}")
print(f"tA12={tA12!r}  a12={a12!r}  A12={A12!r}")
print(f"tA22={tA22!r}  a22={a22!r}  A22={A22!r}")
print(f"tA23={tA23!r}  a23={a23!r}  A23={A23!r}")
print(f"tA33={tA33!r}  a33={a33!r}  A33={A33!r}")
