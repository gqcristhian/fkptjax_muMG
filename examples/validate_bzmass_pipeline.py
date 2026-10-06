"""Reproducible BZ_Mass audit of workspace code, without overwriting old references.

Run with the cosmodesi_new Python: validate_bzmass_pipeline.py kernels|tables|pipeline.
The table check uses the exact saved sMGPT input/NW pair and its 44-column layout.
Results go to validation_results/bzmass_20260928 (override with --output).
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for repo in ('cosmoprimo', 'desilike', 'FolpsD', 'fkptjax_muMG/src', 'ISiTGR'):
    sys.path.insert(0, str(ROOT / repo))
os.environ.setdefault('FOLPS_BACKEND', 'jax')
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import jax
import jax.numpy as jnp
from fkptjax import mg_jax as mg, MG_kernels as kernels
from fkptjax.jax_ode import DP_jax

REF = ROOT / 'fkptjax_muMG/examples/sMGPT_reference_data'
OM, Z = 0.31519, 0.295
XNOW, XSTOP = -3.912023, -np.log1p(Z)
MG = dict(mu_kinf_BZmass=1.2, lambda_a_BZmass=100., lambda_dS_BZmass=100.)
P = mg.pack_constants_jnp(om=OM, ol=1-OM, kind=mg.BZ_MASS,
                         mu_kinf=1.2, lambda_a=100., lambda_dS=100.)


def metrics(value, reference):
    value, reference = np.asarray(value), np.asarray(reference)
    assert np.isfinite(value).all()
    delta = np.abs(value - reference)
    return dict(max_abs=float(delta.max()),
                max_rel=float(np.max(delta / np.maximum(np.abs(reference), 1e-12))),
                rms_over_ref_rms=float(np.linalg.norm(delta) / np.linalg.norm(reference)))


def read_wolfram(name):
    rows = []
    for line in (REF / name).read_text().splitlines():
        numbers = re.findall(r'NumberForm\[([^,]+),', line)
        if numbers:
            rows.append([float(n.replace('*^', 'e')) for n in numbers])
    return np.asarray(rows)


def check_kernels(out):
    report = {}
    ref = read_wolfram('AB_kcompare_wolfram.txt')
    if (out/'AB_fresh.dat').exists():
        fresh = np.loadtxt(out/'AB_fresh.dat')
        report['fresh_wolfram_vs_archived'] = metrics(fresh, ref)
        np.testing.assert_allclose(fresh, ref, rtol=1e-7, atol=1e-9)
        ref = fresh
    k = jnp.asarray(ref[:, 0]); q = .3*k; km = k*np.sqrt(1+.3**2-2*.3*.3)
    for solver in ('adaptive', 'rk4'):
        val = kernels.A_B_grid(k, q, km, P, XNOW, XSTOP, solver=solver, n_steps=64)
        report['AB_' + solver] = {n: metrics(v, ref[:, i+1]) for i, (n, v) in enumerate(zip(('A', 'B', 'Ap', 'Bp'), val))}
        np.testing.assert_allclose(np.asarray(val).T, ref[:, 1:], rtol=5e-4, atol=2e-6)
    ref = read_wolfram('D2D3_kcompare_wolfram.txt')
    # The Wolfram header says CFD3, but its exported expression is D3/(Dk Dp^2).
    # fkptjax defines CFD3 = (21/5) D3/(Dk Dp^2); compare the actual conventions.
    ref[:, 3:] *= 21./5.
    k = jnp.asarray(ref[:, 0])
    # The reference evaluates x=+0.3 for both fused solves.
    val = (*kernels.D2_fused_grid(.3, k, .3*k, P, XNOW, XSTOP, solver='rk4', n_steps=64),
           *kernels.D3_fused_grid(.3, k, .3*k, P, XNOW, XSTOP, .6826138224244257, solver='rk4', n_steps=64))
    report['D2D3'] = {n: metrics(v, ref[:, i+1]) for i, (n, v) in enumerate(zip(('D2','D2p','CFD3','CFD3p'), val))}
    np.testing.assert_allclose(np.asarray(val).T, ref[:, 1:], rtol=1e-3, atol=3e-6)
    # Reuse the archived solver test's mesh construction, without its obsolete emulator dependency.
    k = np.linspace(.0227, .198, 36)[:, None]
    x = np.polynomial.legendre.leggauss(16)[0][None, :]
    k3 = np.sqrt(2*k*k*(1+x)); k1 = np.broadcast_to(k, k3.shape)
    triples = np.concatenate([np.stack(t, axis=-1).reshape(-1,3) for t in
                              ((k3,k1,k1),(k1,k1,k3),(k1,k3,k1))])
    vals = {}
    for solver in ('adaptive','rk4'):
        vals[solver] = np.asarray(kernels.A_B_grid(*jnp.asarray(triples).T, P, XNOW, XSTOP, solver=solver, n_steps=64))
    report['bispectrum_mesh_rk4'] = {n: metrics(v, r) for n,v,r in zip(('A','B','Ap','Bp'),vals['rk4'],vals['adaptive'])}
    np.testing.assert_allclose(vals['rk4'], vals['adaptive'], rtol=5e-4, atol=2e-6)
    report['max_A_minus_B'] = float(np.max(np.abs(vals['adaptive'][0]-vals['adaptive'][1])))
    import folps
    b = folps.BispectrumCalculator_fk()
    ki,kj,x,mui,muj,f0,fi,fj = .04,.13,.37,.25,-.6,.68,.72,.81
    A,B,Ap,Bp = 1.07,.96,.02,-.01
    F2 = .5+3*A/14+x/2*(ki/kj+kj/ki)+(.5-3*B/14)*x*x
    G2 = (3*A*(fi+fj)+3*Ap)/(14*f0)+x/2*(fi*ki/kj+fj*kj/ki)/f0+((fi+fj)/2-(3*B*(fi+fj)+3*Bp)/14)*x*x/f0
    km = ki*mui+kj*muj
    target = 1.6*F2+f0*km**2/(ki*ki+kj*kj+2*ki*kj*x)*G2+.2/2-.1/2*(x*x-1/3)
    target += km/2*(fi*mui/ki*(1.6+fj*muj*muj)+fj*muj/kj*(1.6+fi*mui*mui))
    value = b.Z2(ki,kj,x,mui,muj,f0,fi,fj,1.6,.2,-.1,A,Ap,B,Bp)
    np.testing.assert_allclose(value,target,rtol=1e-13)
    report['Z2_equation'] = metrics(value,target)
    return report


def check_tables(out, nq=300, nangle=10):
    from fkptjax.kfuncs_to_tables import Kfuncs_to_tables
    from fkptjax.pipelines import make_table_state, poles_from_tables
    from fkptjax.rsd import pack_fkpt_bias
    raw = np.loadtxt(REF/'AllFunctions_BGS_BZMass_NoScreen.dat')
    rawn = np.loadtxt(REF/'AllFunctions_BGS_BZMass_NoScreen_nw.dat')
    # Header misses the repeated Pbs2b1 column; normalize to the documented 43 columns.
    if raw.shape[1] == 44:
        np.testing.assert_array_equal(raw[:,36],raw[:,37])
        raw, rawn = np.delete(raw,37,axis=1), np.delete(rawn,37,axis=1)
    assert raw.shape[1] == 43
    inp = ROOT/'sMGPT/input/temp_pklin_ext_64878.dat'
    inn = inp.with_name(inp.stem+'_nw.dat')
    k,pk = np.loadtxt(inp,unpack=True); kn,pnw = np.loadtxt(inn,unpack=True)
    np.testing.assert_array_equal(k,kn)
    D,dD = DP_jax(jnp.asarray(k),P,-6.,XSTOP,solver='adaptive')
    D0 = DP_jax(jnp.asarray([k[1]]),P,-6.,0.,solver='adaptive')[0][0]
    growth = np.asarray((D/D0)**2); pk*=growth; pnw*=growth
    fk = np.asarray(dD/D); f0=float(fk[1])
    # Use the same early-time start for the nonlinear kernels as sMGPT.
    start=time.time()
    tw,tn,kc,kfuncs = Kfuncs_to_tables(k,pk,pnw,z=Z,Om=OM,fk=fk,f0=f0,
        model='HDKI',mg_variant='BZ_Mass',**MG,beyond_eds=True,fkpt_approximation=False,
        kmin=.001,kmax=1.,Nk_kernel=121,nquadSteps=nq,NQ=nangle,NR=nangle,xnow=-4.,
        return_raw_kfuncs=True)
    report={'seconds':time.time()-start,'nquad':nq,'nangle':nangle,'raw_columns':44,'f0':f0}
    kt=np.asarray(tw[0]); mask=(kt>=.01)&(kt<=.3)
    report['linear']=metrics(tw[1][mask],raw[mask,1])
    assert report['linear']['max_rel'] < 3e-5
    mapping={'P22dd':3,'P22du':4,'P22uu':5,'P13dd':6,'P13du':7,'P13uu':8,
             'I1udd1A':9,'I2uud1A':10,'I2uud2A':11,'I3uuu2A':12,'I3uuu3A':13,
             'Pb1b2':35,'Pb1bs2':36,'Pb22':37,'Pb2s2':38,'Ps22':39,
             'Pb2theta':40,'Pbs2theta':41,'sigma32PSL':42}
    report['loops']={name:metrics(np.asarray(getattr(kfuncs,name))[0][mask],raw[mask,i]) for name,i in mapping.items()}
    # Repack the independent sMGPT loops, with the SAME IR-displacement integrals.
    def repack(r,ours):
        c=lambda i:jnp.asarray(r[:,i-1])
        fields=[c(1),c(2),c(3)/f0,c(4)+c(7),c(5)+c(8),c(6)+c(9),
                c(36),c(37),c(38),c(39),c(40),c(43),c(41),c(42)]
        fields += [c(i) for i in range(10,15)]
        fields += [c(i)+c(i+9) for i in (15,16,18,19,21,22,23)]
        return tuple(fields)+tuple(ours[26:])
    x,w=np.polynomial.legendre.leggauss(12); mu=jnp.asarray((x+1)/2); w=jnp.asarray(w/2)
    b1=1.12/.6907
    pars=pack_fkpt_bias(dict(b1=b1,b2=8/21*(b1-1),bs2=-4/7*(b1-1),b3nl=32/315*(b1-1),
                            alpha0=0.,alpha2=0.,alpha4=0.,ctilde=0.,alpha0shot=1.,alpha2shot=0.,PshotP=5000.))
    def poles(a,b):
        return np.asarray(poles_from_tables(make_table_state(a,b),jac=1.,kap=kt[:,None]*jnp.ones_like(mu),
            muap=jnp.ones((kt.size,1))*mu,pars=pars,mu=mu,wmu=w,damping=None))
    ours=poles(tw,tn); refloops=poles(repack(raw,tw),repack(rawn,tn))
    official=np.loadtxt(REF/'multipoles_BGS_BZMass_NoScreen.dat')[:,1:4].T
    for label,v,r in [('multipoles_vs_official',ours,official),('same_assembly',ours,refloops),('reference_assembly',refloops,official)]:
        report[label]={str(ell):metrics(a[mask],b[mask]) for ell,a,b in zip((0,2,4),v,r)}
    corrected_path=out/'multipoles_sMGPT_correct_columns.dat'
    if corrected_path.exists():
        corrected=np.loadtxt(corrected_path)[:,1:4].T
        report['multipoles_vs_corrected_sMGPT']={str(ell):metrics(a[mask],b[mask]) for ell,a,b in zip((0,2,4),ours,corrected)}
        report['assembly_vs_corrected_sMGPT']={str(ell):metrics(a[mask],b[mask]) for ell,a,b in zip((0,2,4),refloops,corrected)}
        if nq == 183 and nangle == 10:
            assert max(v['max_rel'] for v in report['multipoles_vs_corrected_sMGPT'].values()) < 1e-3
    suffix='' if nangle == 10 else f'_angular{nangle}'
    np.savez(out/f'tables_{nq}{suffix}.npz',k=kt,ours=ours,reference_loops=refloops,official=official,
             **{name:np.asarray(getattr(kfuncs,name)) for name in mapping})
    return report


def check_pipeline(out):
    import cosmoprimo, desilike, folps, isitgr
    from desilike.base import build, get_params
    from desilike.theories import CosmoprimoCosmology
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template
    from desilike.theories.galaxy_clustering.full_shape import FKPTJAXPTSpectrum2Poles, FKPTJAXTracerSpectrum2Poles
    fid=cosmoprimo.Cosmology(h=.6736,Omega_m=OM,omega_b=.02237,m_ncdm=0.,A_s=2.1e-9,
        engine='isitgr',use_BZ_Mass_form=True,MG_parameterization='muSigma',**MG)
    report={'imports':{m.__name__:m.__file__ for m in (cosmoprimo,desilike,folps,isitgr)}}
    outputs={}
    for name,beyond,approx in [('full',True,False),('fkpt',True,True),('eds',False,True),('eds_fk',False,True)]:
        cosmo=CosmoprimoCosmology(engine='isitgr',fiducial=fid)
        template=DirectSpectrum2Template(cosmo=cosmo,fiducial=fid,z=Z,k=np.geomspace(1e-4,10,200))
        pt=FKPTJAXPTSpectrum2Poles(template=template,k=np.geomspace(.01,.3,20),
            model='HDKI',mg_variant='BZ_Mass',beyond_eds=beyond,fkpt_approximation=approx,kernel_mode=name,
            growth_source='template',mg_params_override=MG,fkpt_nquad_steps=100,fkpt_nq=8,fkpt_nr=8)
        theory=FKPTJAXTracerSpectrum2Poles(pt=pt,k=np.geomspace(.01,.3,20))
        pipe=build(theory)
        if name == 'full':
            d,dp=DP_jax(jnp.asarray(template.k),P,-6.,XSTOP,solver='adaptive')
            mask=(template.k>=.01)&(template.k<=.3)
            report['isitgr_vs_ode_growth']=metrics(np.asarray(template.fk)[mask],np.asarray(dp/d)[mask])
            assert report['isitgr_vs_ode_growth']['max_rel'] < .01
            report['growth_range']=[float(np.min(template.fk)),float(np.max(template.fk))]
            engine_params=fid._engine._camb_params
            report['isitgr_model']={'enabled':bool(engine_params.ISiTGR_BZ_Mass),
                'lambda_a_Mpc':float(engine_params.lambda_a_BZmass),
                'lambda_dS_Mpc':float(engine_params.lambda_dS_BZmass)}
            assert report['isitgr_model']['enabled']
            np.testing.assert_allclose(engine_params.lambda_a_BZmass,100./.6736)
        params={p.name:p._value for p in get_params(theory)}
        outputs[name]=np.asarray(pipe(params))
        assert np.isfinite(outputs[name]).all()
        # Pytree roundtrip is the interface used by desilike's callbacks and emulators.
        leaves,tree=jax.tree_util.tree_flatten(pt)
        restored=jax.tree_util.tree_unflatten(tree,leaves)
        np.testing.assert_allclose(restored._P.mu_kinf,1.2)
        bspars=jnp.asarray([1.6,.2,-.1,0.,0.,0.,0.,0.])
        pairs=jnp.asarray([[.04,.07],[.08,.13]])
        bs=restored.combine_bias_terms_spectrum3_poles(bspars,pairs,('B000','B202'),precision=(4,16,4))
        outputs[name+'_bs']=np.asarray(bs)
        assert np.isfinite(outputs[name+'_bs']).all()
        sc=restored.combine_bias_terms_spectrum3_poles(bspars,jnp.asarray([[.04,.07,.08]]),
                ('B0','B2'),basis='scoccimarro',precision=(4,4))
        assert np.isfinite(np.asarray(sc)).all()
        report[name]={'power_shape':list(outputs[name].shape),'bs':outputs[name+'_bs'].tolist(),
                      'scoccimarro':np.asarray(sc).tolist(),'mu_kinf':float(restored._P.mu_kinf)}
    for mode in ('fkpt','eds','eds_fk'):
        for suffix in ('','_bs'):
            report['full_vs_'+mode+suffix]=metrics(outputs['full'+suffix],outputs[mode+suffix])
            assert not np.allclose(outputs['full'+suffix],outputs[mode+suffix],rtol=1e-7,atol=1e-10)
    np.savez(out/'pipeline.npz',**outputs)
    return report


def check_sensitivity(out):
    """Change one shared MG parameter in both the Boltzmann and nonlinear sectors."""
    from cosmoprimo import Cosmology
    from desilike import Parameter
    from desilike.base import build, get_params
    from desilike.theories import CosmoprimoCosmology
    from desilike.theories.galaxy_clustering import DirectSpectrum2Template
    from desilike.theories.galaxy_clustering.full_shape import (
        FKPTJAXPTSpectrum2Poles, FKPTJAXTracerSpectrum2Poles, FKPTEmulator)
    fid=Cosmology(h=.6736,Omega_m=OM,omega_b=.02237,m_ncdm=0.,A_s=2.1e-9,
        engine='isitgr',use_BZ_Mass_form=True,MG_parameterization='muSigma',**MG)
    cpars=CosmoprimoCosmology.propose_params(fiducial=fid)
    for name,value in MG.items():
        cpars.set(Parameter(name,value=value,fixed=name!='mu_kinf_BZmass'))
    cosmo=CosmoprimoCosmology(engine='isitgr',fiducial=fid,params=cpars)
    template=DirectSpectrum2Template(cosmo=cosmo,fiducial=fid,z=Z,k=np.geomspace(1e-4,10,200))
    pt=FKPTJAXPTSpectrum2Poles(template=template,model='HDKI',mg_variant='BZ_Mass',
        kernel_mode='full',growth_source='template',fkpt_nquad_steps=60,fkpt_nq=6,fkpt_nr=6)
    theory=FKPTJAXTracerSpectrum2Poles(pt=pt,k=np.geomspace(.01,.3,12))
    # Bind after the tracer's update(k=...) rebuilds the PT constructor.
    for name in MG:
        setattr(pt,name,cosmo.params[name])
    pipe=build(theory,output=lambda:(theory.poles,pt._P.mu_kinf,template.pk_dd))
    params={p.name:p._value for p in get_params(theory)}
    base=pipe(params)
    shifted=pipe({**params,'mu_kinf_BZmass':1.3})
    np.testing.assert_allclose(base[1],1.2)
    np.testing.assert_allclose(shifted[1],1.3)
    report={'power_response':metrics(shifted[0],base[0]),'linear_response':metrics(shifted[2],base[2]),
            'kernel_mu_before':float(base[1]),'kernel_mu_after':float(shifted[1])}
    assert report['power_response']['max_rel'] > 1e-4
    assert report['linear_response']['max_rel'] > 1e-4
    leaves,tree=jax.tree_util.tree_flatten(pt)
    restored=jax.tree_util.tree_unflatten(tree,leaves)
    emu=object.__new__(FKPTEmulator); emu.calculator=pt
    emu.set_children_leafnames()
    assert len(emu.children_leafnames)==len(leaves)
    report['emulator_leaf_count']=len(leaves)
    pars=jnp.asarray([1.6,.2,-.1,0.,0.,0.,0.,0.])
    fn=lambda state:state.combine_bias_terms_spectrum3_poles(pars,jnp.asarray([[.04,.07]]),
        ('B000','B202'),precision=(4,16,4))
    eager=fn(restored); compiled=jax.jit(fn)(restored)
    np.testing.assert_allclose(compiled,eager,rtol=1e-10)
    report['bispectrum_jit']=metrics(compiled,eager)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['kernels','tables','pipeline','sensitivity'])
    parser.add_argument('--nquad',type=int,default=300)
    parser.add_argument('--nangle',type=int,default=10)
    parser.add_argument('--output',type=Path,default=ROOT/'fkptjax_muMG/validation_results/bzmass_20260928')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    report=check_tables(args.output,args.nquad,args.nangle) if args.stage=='tables' else globals()['check_'+args.stage](args.output)
    suffix='' if args.nangle == 10 else f'_angular{args.nangle}'
    name=args.stage+(f'_{args.nquad}{suffix}' if args.stage=='tables' else '')
    (args.output/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
