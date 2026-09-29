import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
import jax.numpy as jnp
from fkptjax import mg_jax as _bj
from fkptjax import MG_kernels as _mgk

P = _bj.pack_constants_jnp(om=0.31, ol=0.69, kind=_bj.HS, fR0_HS=1e-4, beta2=1.0/6.0, n_HS=1.0)
xnow = -3.912023
xstop = jnp.log(1.0 / (1.0 + 0.5))

# A representative, non-squeezed, non-collinear triangle around k~0.15 h/Mpc
kf = jnp.array([0.15])
k1 = jnp.array([0.10])
k2 = jnp.array([0.12])

A, B, Ap, Bp = _mgk.A_B_grid(kf, k1, k2, P, xnow, xstop)
print(f"HS fR0=1e-4 at (kf,k1,k2)=(0.15,0.10,0.12): A={float(A[0]):.6f}, B={float(B[0]):.6f}, "
      f"rel_diff={(float(A[0])-float(B[0]))/float(A[0]):.4%}")

# GR limit sanity: fR0_HS -> 0 should give A=B=1 (EdS/GR).
P_gr = _bj.pack_constants_jnp(om=0.31, ol=0.69, kind=_bj.HS, fR0_HS=1e-15, beta2=1.0/6.0, n_HS=1.0)
A_gr, B_gr, _, _ = _mgk.A_B_grid(kf, k1, k2, P_gr, xnow, xstop)
print(f"GR limit (fR0~0): A={float(A_gr[0]):.6f}, B={float(B_gr[0]):.6f}")
