import jax
import numpy as np
import jax.numpy as jnp
import folps as folpsv2    # interp, simpson, extrapolate_pklin

from typing import Any, Dict, Optional, Tuple

from fkptjax.ode import ModelDerivatives, ODESolver, DP
from fkptjax.calculate_jax import JaxCalculator
from fkptjax.util import setup_kfunctions

def Rescaling_MG(
    k_ext,
    pk_ext,
    pk_now_ext,
    *,
    derivs,
    solver,
    Om,
    model,
    mg_variant,
    fR0_HS,
    beta2,
    n_HS,
    screening,
    omegaBD,
    r_c,
    mu0,
    beta_1,
    lambda_1,
    exp_s,
    mu1,
    mu2,
    mu3,
    mu4,
    z_div,
    z_TGR,
    z_tw,
    scale_bins,
    k_TGR,
    k_S,
    k_c,
    k_tw,
    gamma_0,
    gamma_a,
    t_k,
    d_s,
    f0_kmax=1e-3,
):
    """
    Linear-spectrum-only MG rescaling.

    Returns
    -------
    pk_ext_rescaled, pk_now_ext_rescaled
    """

    def build_k_growth(k_ext_np, k_TGR, k_c, k_S, k_tw,
                       kmin=1e-4, kmax=None,
                       nbase=500, nwin=160):
        if kmax is None:
            kmax = max(0.5, float(np.max(k_ext_np)))

        base = np.geomspace(float(kmin), float(kmax), int(nbase))

        local = []
        for kc in [k_TGR, k_c, k_S]:
            kc = float(kc)
            if kc <= 0:
                continue
            w = max(float(k_tw), 1e-5)
            lo = max(float(kmin), kc - 20.0 * w)
            hi = min(float(kmax), kc + 20.0 * w)
            if hi > lo:
                local.append(np.linspace(lo, hi, int(nwin)))

        k_growth = np.unique(np.concatenate([base] + local))
        return k_growth

    def make_derivs(**updates):
        pars = dict(
            om=float(Om), ol=float(1.0 - Om),
            fR0_HS=float(fR0_HS), beta2=float(beta2), n_HS=float(n_HS),
            screening=int(screening), omegaBD=float(omegaBD),
            r_c=float(r_c),
            model=str(model), mg_variant=str(mg_variant),
            mu0=float(mu0),
            beta_1=float(beta_1), lambda_1=float(lambda_1), exp_s=float(exp_s),
            mu1=float(mu1), mu2=float(mu2), mu3=float(mu3), mu4=float(mu4),
            z_div=float(z_div), z_TGR=float(z_TGR), z_tw=float(z_tw),
            scale_bins=bool(scale_bins),
            k_TGR=float(k_TGR), k_S=float(k_S), k_c=float(k_c), k_tw=float(k_tw),
            gamma_0=float(gamma_0), gamma_a=float(gamma_a), t_k=float(t_k), d_s=float(d_s),
        )
        pars.update(updates)
        return ModelDerivatives(**pars)

    def make_gr_derivs():
        return ModelDerivatives(
            om=float(Om), ol=float(1.0 - Om),
            fR0_HS=0.0, beta2=float(beta2), n_HS=float(n_HS),
            screening=int(screening), omegaBD=float(omegaBD),
            r_c=float(r_c),
            model='HDKI', mg_variant='mu_OmDE',
            mu0=0.0,
            beta_1=1.0, lambda_1=0.0, exp_s=0.0,
            mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
            z_div=float(z_div), z_TGR=float(z_TGR), z_tw=float(z_tw),
            scale_bins=bool(scale_bins),
            k_TGR=float(k_TGR), k_S=float(k_S), k_c=float(k_c), k_tw=float(k_tw),
            gamma_0=0.545454, gamma_a=0.0, t_k=float(t_k), d_s=float(d_s),
        )

    k_ext_np = np.asarray(k_ext, dtype=float)

    k_growth = build_k_growth(
        k_ext_np=k_ext_np,
        k_TGR=float(k_TGR),
        k_c=float(k_c),
        k_S=float(k_S),
        k_tw=float(k_tw),
        kmin=min(1e-4, float(np.min(k_ext_np))),
        kmax=max(0.5, float(np.max(k_ext_np))),
        nbase=700,
        nwin=220,
    )
    k_growth_jax = jnp.asarray(k_growth)

    derivs_gr = make_gr_derivs()
    Y_gr = DP(k_growth, derivs_gr, solver)
    D_gr = jnp.asarray(Y_gr[0])

    Y_mg = DP(k_growth, derivs, solver)
    D_mg = jnp.asarray(Y_mg[0])
    scale_growth = (D_mg / D_gr) ** 2

    log_scale_growth = jnp.log(scale_growth)
    log_scale_ext = folpsv2.tools_jax.interp(k_ext, k_growth_jax, log_scale_growth)
    log_scale_ext = jnp.clip(log_scale_ext,
                             jnp.min(log_scale_growth),
                             jnp.max(log_scale_growth))
    scale = jnp.exp(log_scale_ext)

    return pk_ext * scale, pk_now_ext * scale

def Kfuncs_to_tables(
    k,
    pk,
    pk_now,
    *,
    z: float,
    Om: float,
    beyond_eds: bool = False,
    rescale_PS: bool = False,
    kmin: Optional[float] = None,
    kmax: Optional[float] = None,
    Nk_kernel: int = 120,
    nquadSteps: int = 300,
    NQ: int = 10,
    NR: int = 10,
    xnow: float = -3.912023,
    ode_method: str = "RKQS",
    f0_kmax: Optional[float] = None,
    model: str = "HDKI",
    mg_variant: str = "mu_OmDE",
    fR0_HS: float = 1e-15,
    n_HS: float = 1.0,
    beta2: float = 1.0 / 6.0,
    screening: int = 1,
    omegaBD: float = 0.0,
    r_c: float = 1.0e30,
    mu0: float = 0.0,
    beta_1: float = 1.0,
    lambda_1: float = 1.0,
    exp_s: float = 1.0,
    mu1: float = 1.0,
    mu2: float = 1.0,
    mu3: float = 1.0,
    mu4: float = 1.0,
    z_div: float = 1.0,
    z_TGR: float = 10.0,
    z_tw: float = 0.5,
    scale_bins: bool = False,
    k_TGR: float = 0.001,
    k_S: float = 0.5,
    k_c: float = 0.1,
    k_tw: float = 0.01,
    gamma_0: float = 0.54545,
    gamma_a: float = 0.0,
    t_k: float = 100.0,
    d_s: float = 0.0001,
    eftcamb_h1_interp=None,
    eftcamb_h3_interp=None,
    eftcamb_h5_interp=None,
    rbao: float = 104.0,
    pmax_bao: float = 0.4,
    Np_bao: int = 100,
    return_kernel_constants=True,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """
    Return (table_wiggle, table_now) in the A_full=False layout expected by FOLPS.
    Uses fkptjax internal output grid by default (init_data.logk_grid).
    """

    model_u = str(model).upper()
    if model_u in ("HS", "NDGP", "LCDM", "GR"):
        mg_variant = None

    if model_u == "HS":
        if bool(rescale_PS) is not True:
            raise ValueError("For model='HS', rescale_PS must be True (passed False).")

    k = jnp.asarray(k)
    pk = jnp.asarray(pk)
    pk_now = jnp.asarray(pk_now)

    k_ext, pk_ext = folpsv2.tools_jax.extrapolate_pklin(k, pk)
    _, pk_now_ext = folpsv2.tools_jax.extrapolate_pklin(k, pk_now)

    solver = ODESolver(zout=float(z), xnow=float(xnow), method=str(ode_method))

    derivs = ModelDerivatives(
        om=float(Om), ol=float(1.0 - Om),
        fR0_HS=float(fR0_HS), beta2=float(beta2), n_HS=float(n_HS),
        screening=int(screening), omegaBD=float(omegaBD),
        r_c=float(r_c),
        model=str(model), mg_variant=str(mg_variant) if mg_variant is not None else "mu_OmDE",
        mu0=float(mu0),
        beta_1=float(beta_1), lambda_1=float(lambda_1), exp_s=float(exp_s),
        mu1=float(mu1), mu2=float(mu2), mu3=float(mu3), mu4=float(mu4),
        z_div=float(z_div), z_TGR=float(z_TGR), z_tw=float(z_tw),
        scale_bins=bool(scale_bins), k_TGR=float(k_TGR), k_S=float(k_S), k_c=float(k_c), k_tw=float(k_tw),
        gamma_0=float(gamma_0), gamma_a=float(gamma_a), t_k=float(t_k), d_s=float(d_s),
        eftcamb_h1_interp=eftcamb_h1_interp,
        eftcamb_h3_interp=eftcamb_h3_interp,
        eftcamb_h5_interp=eftcamb_h5_interp,
    )

    k_ext_np = np.asarray(k_ext, dtype=float)

    Y = DP(k_ext_np, derivs, solver)
    D_ext, Dp_ext = Y[0], Y[1]

    fk_ext = jnp.asarray(Dp_ext / D_ext)

    # Define the kernel grid range before estimating f0.
    # If f0_kmax is not explicitly provided, use the routine's kmin.
    if kmin is None:
        kmin = float(jnp.minimum(1e-3, jnp.min(k)))
    if kmax is None:
        kmax = float(jnp.maximum(0.5, jnp.max(k)))

    if f0_kmax is None:
        f0_kmax = float(kmin)

    mask0 = (k_ext <= float(f0_kmax))
    nhead = int(min(5, int(k_ext.shape[0])))
    f0_jax = jnp.where(
        jnp.any(mask0),
        jnp.sum(jnp.where(mask0, fk_ext, 0.0)) / jnp.maximum(jnp.sum(mask0), 1),
        jnp.mean(fk_ext[:nhead]),
    )
    f0 = float(f0_jax)

    if bool(rescale_PS):
        pk_ext, pk_now_ext = Rescaling_MG(
            k_ext,
            pk_ext,
            pk_now_ext,
            derivs=derivs,
            solver=solver,
            Om=Om,
            model=model,
            mg_variant=mg_variant,
            fR0_HS=fR0_HS,
            beta2=beta2,
            n_HS=n_HS,
            screening=screening,
            omegaBD=omegaBD,
            r_c=r_c,
            mu0=mu0,
            beta_1=beta_1,
            lambda_1=lambda_1,
            exp_s=exp_s,
            mu1=mu1,
            mu2=mu2,
            mu3=mu3,
            mu4=mu4,
            z_div=z_div,
            z_TGR=z_TGR,
            z_tw=z_tw,
            scale_bins=scale_bins,
            k_TGR=k_TGR,
            k_S=k_S,
            k_c=k_c,
            k_tw=k_tw,
            gamma_0=gamma_0,
            gamma_a=gamma_a,
            t_k=t_k,
            d_s=d_s,
            f0_kmax=f0_kmax,
        )

    init_data = setup_kfunctions(
        k_in=k_ext,
        kmin=float(kmin),
        kmax=float(kmax),
        Nk=int(Nk_kernel),
        nquadSteps=int(nquadSteps),
        NQ=int(NQ),
        NR=int(NR),
    )
    kout = init_data.logk_grid

    fk_out = folpsv2.tools_jax.interp(kout, k_ext, fk_ext)
    fk_norm_out = fk_out / f0

    ff = fk_ext / f0
    sigma2w = float(1.0 / (6.0 * jnp.pi**2) * folpsv2.tools_jax.simpson(pk_ext * ff**2, x=k_ext))
    sigma2w_NW = float(1.0 / (6.0 * jnp.pi**2) * folpsv2.tools_jax.simpson(pk_now_ext * ff**2, x=k_ext))

    p = jnp.exp(jnp.linspace(jnp.log(1e-6), jnp.log(float(pmax_bao)), int(Np_bao)))
    PSL_NW = folpsv2.tools_jax.interp(p, k_ext, pk_now_ext)

    sigma2_NW = float(
        1.0 / (6.0 * jnp.pi**2)
        * folpsv2.tools_jax.simpson(
            PSL_NW * (
                1.0
                - folpsv2.spherical_jn_backend(0, p * float(rbao))
                + 2.0 * folpsv2.spherical_jn_backend(2, p * float(rbao))
            ),
            x=p,
        )
    )
    delta_sigma2_NW = float(
        1.0 / (2.0 * jnp.pi**2)
        * folpsv2.tools_jax.simpson(PSL_NW * folpsv2.spherical_jn_backend(2, p * float(rbao)), x=p)
    )

    if bool(beyond_eds):
        from fkptjax.ode import kernel_constants
        KA, KAp, KR1, KR1p = kernel_constants(f0=f0, derivs=derivs, solver=solver)
        A = float(KA)
        ApOverf0 = float(KAp) / float(f0)
        CFD3 = float(KR1)
        CFD3p = float(KR1p)
    else:
        A = 1.0
        ApOverf0 = 0.0
        CFD3 = 1.0
        CFD3p = 1.0

    calculator = JaxCalculator()
    calculator.initialize(init_data)

    kfuncs = calculator.evaluate(
        Pk_in=pk_ext,
        Pk_nw_in=pk_now_ext,
        fk_in=fk_ext,
        A=A,
        ApOverf0=ApOverf0,
        CFD3=CFD3,
        CFD3p=CFD3p,
        sigma2v=0.0,
        f0=f0,
    )

    def _arr(x):
        return jnp.asarray(x)

    zeros = jnp.zeros_like(_arr(kout))

    pkl_out_w = kfuncs.pkl[0]
    pkl_out_nw = kfuncs.pkl[1]

    table_w = (
        kout,
        _arr(pkl_out_w),
        _arr(fk_norm_out),
        _arr(kfuncs.P22dd[0] + kfuncs.P13dd[0]),
        _arr(kfuncs.P22du[0] + kfuncs.P13du[0]),
        _arr(kfuncs.P22uu[0] + kfuncs.P13uu[0]),
        _arr(kfuncs.Pb1b2[0]),
        _arr(kfuncs.Pb1bs2[0]),
        _arr(kfuncs.Pb22[0]),
        _arr(kfuncs.Pb2s2[0]),
        _arr(kfuncs.Ps22[0]),
        _arr(kfuncs.sigma32PSL[0]),
        _arr(kfuncs.Pb2theta[0]),
        _arr(kfuncs.Pbs2theta[0]),
        _arr(kfuncs.I1udd1A[0]),
        _arr(kfuncs.I2uud1A[0]),
        _arr(kfuncs.I2uud2A[0]),
        _arr(kfuncs.I3uuu2A[0]),
        _arr(kfuncs.I3uuu3A[0]),
        _arr(kfuncs.I2uudd1BpC[0]),
        _arr(kfuncs.I2uudd2BpC[0]),
        _arr(kfuncs.I3uuud2BpC[0]),
        _arr(kfuncs.I3uuud3BpC[0]),
        _arr(kfuncs.I4uuuu2BpC[0]),
        _arr(kfuncs.I4uuuu3BpC[0]),
        _arr(kfuncs.I4uuuu4BpC[0]),
        zeros,
        zeros,
        sigma2w,
        f0,
    )

    table_nw = (
        kout,
        _arr(pkl_out_nw),
        _arr(fk_norm_out),
        _arr(kfuncs.P22dd[1] + kfuncs.P13dd[1]),
        _arr(kfuncs.P22du[1] + kfuncs.P13du[1]),
        _arr(kfuncs.P22uu[1] + kfuncs.P13uu[1]),
        _arr(kfuncs.Pb1b2[1]),
        _arr(kfuncs.Pb1bs2[1]),
        _arr(kfuncs.Pb22[1]),
        _arr(kfuncs.Pb2s2[1]),
        _arr(kfuncs.Ps22[1]),
        _arr(kfuncs.sigma32PSL[1]),
        _arr(kfuncs.Pb2theta[1]),
        _arr(kfuncs.Pbs2theta[1]),
        _arr(kfuncs.I1udd1A[1]),
        _arr(kfuncs.I2uud1A[1]),
        _arr(kfuncs.I2uud2A[1]),
        _arr(kfuncs.I3uuu2A[1]),
        _arr(kfuncs.I3uuu3A[1]),
        _arr(kfuncs.I2uudd1BpC[1]),
        _arr(kfuncs.I2uudd2BpC[1]),
        _arr(kfuncs.I3uuud2BpC[1]),
        _arr(kfuncs.I3uuud3BpC[1]),
        _arr(kfuncs.I4uuuu2BpC[1]),
        _arr(kfuncs.I4uuuu3BpC[1]),
        _arr(kfuncs.I4uuuu4BpC[1]),
        zeros,
        zeros,
        sigma2w_NW,
        sigma2_NW,
        delta_sigma2_NW,
        f0,
    )
    if return_kernel_constants:
        return table_w, table_nw, (A, ApOverf0 * f0, CFD3, CFD3p)
    return table_w, table_nw

@jax.jit(static_argnums=(0, 5, 6))
def scaled_pkmu(ncols, jac, kap, muap, pars, bias_scheme, damping, *table_all: jax.Array):
    folpsv2.MatrixCalculator(A_full=False, use_TNS_model=False)
    calc = folpsv2.RSDMultipolesPowerSpectrumCalculator(model="EFT")
    pars2 = calc.set_bias_scheme(pars=pars, bias_scheme=bias_scheme)
    pkmu = calc.get_rsd_pkmu(
        kap, muap, pars2,
        table_all[:ncols], table_all[ncols:],
        IR_resummation=True,
        damping=damping,
    )

    return jac * pkmu