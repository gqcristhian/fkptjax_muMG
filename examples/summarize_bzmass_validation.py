"""Make standalone comparison figures and a provenance manifest from saved runs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
out=root/'fkptjax_muMG/validation_results/bzmass_20260928'
ref=np.loadtxt(out/'multipoles_sMGPT_correct_columns.dat')
matched=np.load(out/'tables_183.npz')
k=matched['k']; mask=(k>=.01)&(k<=.3)
fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
for i,ell in enumerate((0,2,4)):
    ax=axes[0,i]
    ax.plot(k[mask],k[mask]*ref[mask,i+1],label='sMGPT (correct column mapping)',color='black')
    ax.plot(k[mask],k[mask]*matched['ours'][i,mask],'--',label='fkptjax full, matched quadrature')
    ax.set_title(rf'$\ell={ell}$'); ax.set_ylabel(r'$k P_\ell(k)$')
    ax=axes[1,i]
    for count in (183,300,600,1200):
        path=out/f'tables_{count}.npz'
        if path.exists():
            data=np.load(path)
            ax.plot(k[mask],100*(data['ours'][i,mask]/ref[mask,i+1]-1),label=f'{count} radial nodes')
    ax.axhline(0,color='black',lw=.6)
    ax.set_xlabel(r'$k\ [h/\mathrm{Mpc}]$'); ax.set_ylabel('difference from sMGPT [%]')
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8); axes[1,0].legend(fontsize=8)
fig.suptitle(r'BZ_Mass: $\mu_\infty=1.2$, $\lambda_a=\lambda_{dS}=100\,\mathrm{Mpc}/h$, $z=0.295$')
fig.tight_layout()
fig.savefig(out/'multipoles_comparison.png',dpi=180)
fig.savefig(out/'multipoles_comparison.pdf')
plt.close(fig)

summary={}
for left,right in [('300','600'),('600','1200'),('600','600_angular20')]:
    a,b=out/f'tables_{left}.npz',out/f'tables_{right}.npz'
    if not (a.exists() and b.exists()): continue
    av,bv=np.load(a)['ours'][:,mask],np.load(b)['ours'][:,mask]
    summary[left+'_vs_'+right]={str(ell):float(100*np.max(abs(x/y-1))) for ell,x,y in zip((0,2,4),av,bv)}
(out/'convergence.json').write_text(json.dumps(summary,indent=2)+'\n')

manifest={'python':sys.executable,'versions':{},'repositories':{},'sha256':{}}
for name in ('jax','jaxlib','numpy','scipy','diffrax','cosmoprimo','desilike','isitgr'):
    try: manifest['versions'][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: pass
for repo in ('fkptjax_muMG','desilike','FolpsD','cosmoprimo','ISiTGR','sMGPT'):
    manifest['repositories'][repo]={'head':subprocess.check_output(['git','-C',str(root/repo),'rev-parse','HEAD'],text=True).strip(),
        'tracked_changes':subprocess.check_output(['git','-C',str(root/repo),'diff','--stat'],text=True)}
paths=['desilike/desilike/theories/galaxy_clustering/full_shape.py','FolpsD/folps/folps.py',
       'cosmoprimo/cosmoprimo/isitgr.py','cosmoprimo/cosmoprimo/camb.py',
       'ISiTGR/fortran/equations.f90','ISiTGR/isitgr/camblib.so',
       'fkptjax_muMG/src/fkptjax/MG_kernels.py','fkptjax_muMG/src/fkptjax/mg_jax.py',
       'fkptjax_muMG/src/fkptjax/kfuncs_to_tables.py','fkptjax_muMG/src/fkptjax/calculate_jax.py',
       'sMGPT/src/IREFTMultipoles.m','sMGPT/src/2_P22type.wl','sMGPT/src/3_P13type.wl',
       'sMGPT/input/temp_pklin_ext_64878.dat','sMGPT/input/temp_pklin_ext_64878_nw.dat']
paths += [str(p.relative_to(root)) for p in (root/'fkptjax_muMG/examples/sMGPT_reference_data').glob('*BZMass*.dat')]
for path in paths:
    file=root/path
    if file.exists(): manifest['sha256'][path]=hashlib.sha256(file.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
