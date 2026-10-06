import numpy as np
import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/desilike")

from desilike import compile as desilike_compile
from desilike.theories.galaxy_clustering.full_shape import FKPTJAXPTSpectrum2Poles

k = np.linspace(0.01, 0.2, 21)

for fkpt_approx in (True, False):
    pt = FKPTJAXPTSpectrum2Poles(
        k=k, model='HDKI', mg_variant='mu_OmDE', beyond_eds=True,
        fkpt_approximation=fkpt_approx,
    )
    run = desilike_compile(pt)
    run(mu0=0.3)
    print(f"fkpt_approximation={fkpt_approx}: f0={pt.f0}, table_w[3][:3]={np.asarray(pt._table_w[3])[:3]}")
print("SMOKE TEST OK")
