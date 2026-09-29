import sys, os
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
os.environ.setdefault("FOLPS_BACKEND", "jax")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from fkptjax import mg_jax as bj
from fkptjax import MG_kernels as mgk

Om = 0.315192
z = 0.295
xnow = -3.912023
xstop = float(jnp.log(1.0 / (1.0 + z)))
P = bj.pack_constants_jnp(om=Om, ol=1.0 - Om, kind=bj.BZ_MASS,
                           mu_kinf=1.2, lambda_a=100.0, lambda_dS=100.0)

x, k, p = 0.3, 0.2, 0.05  # same test point used throughout this session

# --- D2_fused_grid's own, separate 6-state solve (output leg kpp(-x,k,p)) ---
A_ref, Ap_ref = mgk.D2_fused_grid(-x, k, p, P, xnow, xstop)
print(f"D2_fused_grid(-x,k,p) [separate solve]:  A={float(A_ref):.10f}  Ap={float(Ap_ref):.10f}")

# --- Replicate D3_fused_grid's FULL 10-state solve, but keep D2m/dD2m too ---
e1 = jnp.exp(xnow); e2 = jnp.exp(2.0 * xnow); e3 = jnp.exp(3.0 * xnow)
one_m_x2 = 1.0 - x * x
pk = p / k
ang = 1.0 / (1.0 + pk * pk + 2.0 * pk * x) + 1.0 / (1.0 + pk * pk - 2.0 * pk * x)
y0 = jnp.stack([
    e1, e1, e1, e1,
    3.0 * e2 / 7.0 * one_m_x2, 6.0 * e2 / 7.0 * one_m_x2,
    3.0 * e2 / 7.0 * one_m_x2, 6.0 * e2 / 7.0 * one_m_x2,
    (5.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang,
    (15.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang,
])
term = diffrax.ODETerm(lambda t, y, args: bj.thirdOrder(t, y, x, k, p, P))
solver = diffrax.Tsit5()
ctrl = diffrax.PIDController(rtol=1e-8, atol=1e-11)
sol = diffrax.diffeqsolve(term, solver, t0=xnow, t1=xstop, dt0=0.01, y0=y0,
                          stepsize_controller=ctrl, saveat=diffrax.SaveAt(t1=True),
                          max_steps=100000)
Y = sol.ys[0]
Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3 = Y

C = (3.0 / 7.0) * Dk * Dp
Cp = (3.0 / 7.0) * (dDk * Dp + Dk * dDp)
A_from_D3 = D2m / C
Ap_from_D3 = dD2m / C - D2m * Cp / (C * C)
print(f"D2mf extracted from D3_fused_grid's OWN state (no separate solve): "
      f"A={float(A_from_D3):.10f}  Ap={float(Ap_from_D3):.10f}")

relA = abs(float(A_from_D3) - float(A_ref)) / abs(float(A_ref))
relAp = abs(float(Ap_from_D3) - float(Ap_ref)) / abs(float(Ap_ref))
print(f"\nrelative diff:  A: {relA:.3e}   Ap: {relAp:.3e}")
