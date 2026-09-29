import numpy as np
import sys
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/fkptjax_muMG/src")
sys.path.insert(0, "/n/home12/cgarciaquintero/DESI/src/desilike")

from desilike import compile as desilike_compile
from desilike.theories.galaxy_clustering.full_shape import FKPTJAXPTSpectrum2Poles

k = np.linspace(0.02, 0.2, 11)

pt = FKPTJAXPTSpectrum2Poles(
    k=k, model='PHENOM', mg_variant='binning', scale_bins=True,
    beyond_eds=True, fkpt_approximation=False,
    include_neutrino_corrections=True,
    Nk_kernel=40,
)
run = desilike_compile(pt)
try:
    run(mu1=0.8, mu2=1.2, mu3=0.9, mu4=1.1)
    print("PHENOM/binning scale_bins=True + neutrino_corrections=True + fkpt_approximation=False: OK")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
