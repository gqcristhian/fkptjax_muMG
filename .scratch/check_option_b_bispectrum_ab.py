import numpy as np
import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/desilike")

from desilike import compile as desilike_compile
from desilike.theories.galaxy_clustering.full_shape import (
    FKPTJAXPTSpectrum2Poles,
    FKPTJAXTracerSpectrum3Poles,
)

k_ps = np.linspace(0.02, 0.2, 21)
k_pairs = np.column_stack([k_ps, k_ps])  # diagonal k1=k2 triangles

def build_bs(fkpt_approx):
    pt = FKPTJAXPTSpectrum2Poles(
        k=k_ps, model='HS', beyond_eds=True,
        fkpt_approximation=fkpt_approx,
        fR0_HS=1e-4,
    )
    bs = FKPTJAXTracerSpectrum3Poles(
        k=k_pairs, pt=pt, ells=[(0, 0, 0)], tracers='LRG',
    )
    return bs

for fkpt_approx in (True, False):
    bs = build_bs(fkpt_approx)
    run = desilike_compile(bs)
    out = np.asarray(run())
    finite = np.all(np.isfinite(out))
    print(f"fkpt_approximation={fkpt_approx}: shape={out.shape}, finite={finite}, "
          f"B000[:5]={out.reshape(-1)[:5]}")

print("SMOKE TEST OK")
