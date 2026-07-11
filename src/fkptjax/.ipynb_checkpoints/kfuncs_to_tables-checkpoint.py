"""Build FOLPS-compatible FKPT tables from linear power spectra.

This module is the low-level FKPT table layer.  It intentionally does not
project to redshift-space multipoles; that is handled in :mod:`fkptjax.rsd` and
:mod:`fkptjax.pipelines`.

The public entry points are:

``Kfuncs_to_tables``
    Legacy/eager route using the existing FKPT growth and kernel ODE machinery.

``Kfuncs_to_tables_jax``
    Fully JAX-traceable route for the PHENOM/binning implementation.

``Kfuncs_to_tables_rescale_jax``
    JAX-traceable rescaling-branch route: start from a GR/LCDM linear spectrum,
    solve the first-order MG growth with ``diffrax``, rescale the linear spectra,
    and then build live FKPT tables.

``build_jax_static_ctx``
    Precompute the cosmology-independent JAX FKPT context once and reuse it in
    repeated/jitted calls.

"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp


__all__ = [
    "Rescaling_MG",
    "Kfuncs_to_tables",
    "build_jax_static_ctx",
    "Kfuncs_to_tables_jax",
    "Kfuncs_to_tables_rescale_jax",
]

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
    neutrino_correction=None,
    f0_kmax=1e-3,
):
    """
    Linear-spectrum-only MG rescaling.

    Returns
    -------
    pk_ext_rescaled, pk_now_ext_rescaled
    """

    import numpy as np
    import jax.numpy as jnp
    from folps.tools_jax import interp
    from fkptjax.ode import ModelDerivatives, DP

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
            neutrino_correction=neutrino_correction,
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
            neutrino_correction=neutrino_correction,
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
    log_scale_ext = interp(k_ext, k_growth_jax, log_scale_growth)
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
    neutrino_correction=None,
    use_numba: bool = False,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """
    Return (table_wiggle, table_now) in the A_full=False layout expected by FOLPS.
    Uses fkptjax internal output grid by default (init_data.logk_grid).
    """

    import folps as folpsv2
    from folps.tools_jax import extrapolate_pklin, simpson, interp
    from fkptjax.calculate_jax import JaxCalculator
    from fkptjax.util import setup_kfunctions
    from fkptjax.ode import ModelDerivatives, ODESolver, DP

    model_u = str(model).upper()
    if model_u in ("HS", "NDGP", "LCDM", "GR"):
        mg_variant = None

    k = jnp.asarray(k)
    pk = jnp.asarray(pk)
    pk_now = jnp.asarray(pk_now)

    k_ext, pk_ext = extrapolate_pklin(k, pk)
    _, pk_now_ext = extrapolate_pklin(k, pk_now)

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
        neutrino_correction=neutrino_correction,
        # use_numba=bool(use_numba),
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

    # For growth-index models f(k, z) is scale-independent by construction
    # (up to optional scale-dependent corrections).  Use the conventional
    # pivot k = 0.1 h/Mpc instead of the low-k average used for scale-dependent
    # MG models.  Keep the old f0_kmax averaging for all other models.
    if model_u in ("GROWTH_INDEX", "GROWTHINDEX"):
        f0_jax = interp(jnp.asarray([0.1]), k_ext, fk_ext)[0]
    else:
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
            neutrino_correction=neutrino_correction,
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

    fk_out = interp(kout, k_ext, fk_ext)
    fk_norm_out = fk_out / f0

    # sigma^2 (wiggle / no-wiggle) and the BAO sigma^2 integrals are small 1D
    # Simpson quadratures.  Evaluating them eagerly in jax forces a host<->device
    # round-trip per op (device_put / apply_primitive churn ~20 ms/call); do them
    # in numpy instead.  scipy's Simpson rule reproduces folps' jax `simpson`
    # bit-for-bit (same composite rule), so results are unchanged.  The grid
    # (extrapolate_pklin), the interp (interpax cubic) and folps' spherical
    # Bessel backend stay in jax -- only the quadratures move host-side.
    from scipy.integrate import simpson as _np_simpson

    ff = fk_ext / f0
    k_ext_h = np.asarray(k_ext)
    ff_h = np.asarray(ff)
    sigma2w = float(1.0 / (6.0 * np.pi**2) * _np_simpson(np.asarray(pk_ext) * ff_h**2, x=k_ext_h))
    sigma2w_NW = float(1.0 / (6.0 * np.pi**2) * _np_simpson(np.asarray(pk_now_ext) * ff_h**2, x=k_ext_h))

    p = jnp.exp(jnp.linspace(jnp.log(1e-6), jnp.log(float(pmax_bao)), int(Np_bao)))
    PSL_NW = interp(p, k_ext, pk_now_ext)
    p_h = np.asarray(p)
    PSL_NW_h = np.asarray(PSL_NW)
    j0_h = np.asarray(folpsv2.spherical_jn_backend(0, p * float(rbao)))
    j2_h = np.asarray(folpsv2.spherical_jn_backend(2, p * float(rbao)))

    sigma2_NW = float(
        1.0 / (6.0 * np.pi**2)
        * _np_simpson(PSL_NW_h * (1.0 - j0_h + 2.0 * j2_h), x=p_h)
    )
    delta_sigma2_NW = float(
        1.0 / (2.0 * np.pi**2)
        * _np_simpson(PSL_NW_h * j2_h, x=p_h)
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


def build_jax_static_ctx(k, *, kmin, kmax, Nk_kernel, nquadSteps, NQ, NR,
                         rbao=104.0, pmax_bao=0.4, Np_bao=100):
    """Precompute the *static* (cosmology-independent) pieces of the jax fkpt
    loop: the kernel grid (``init_data``), the ``JaxCalculator``, ``kout``, and
    the BAO ``p``-grid + spherical Bessel ``j0``/``j2``.

    These depend only on the k-grid and fixed loop parameters (NOT on pk or the
    cosmology), so they are built once (concretely) and reused inside the jitted
    :func:`Kfuncs_to_tables_jax`, keeping that function fully traceable.
    """
    import numpy as _np
    import folps as folpsv2
    from folps.tools_jax import extrapolate_pklin
    from fkptjax.calculate_jax import JaxCalculator
    from fkptjax.util import setup_kfunctions
    k = jnp.asarray(k)
    # k_ext depends only on k (the extrapolation grid), not pk -> use any pk.
    k_ext, _ = extrapolate_pklin(k, jnp.ones_like(k))
    init_data = setup_kfunctions(
        k_in=_np.asarray(k_ext), kmin=float(kmin), kmax=float(kmax),
        Nk=int(Nk_kernel), nquadSteps=int(nquadSteps), NQ=int(NQ), NR=int(NR))
    calculator = JaxCalculator()
    calculator.initialize(init_data)
    p = jnp.exp(jnp.linspace(jnp.log(1e-6), jnp.log(float(pmax_bao)), int(Np_bao)))
    j0 = jnp.asarray(folpsv2.spherical_jn_backend(0, p * float(rbao)))
    j2 = jnp.asarray(folpsv2.spherical_jn_backend(2, p * float(rbao)))
    return dict(init_data=init_data, calculator=calculator,
                kout=jnp.asarray(init_data.logk_grid), p=p, j0=j0, j2=j2)


def Kfuncs_to_tables_jax(
    k, pk, pk_now, *, z, Om, beyond_eds=True,
    kmin=None, kmax=None, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    xnow=-3.912023, f0_kmax=None,
    mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
    z_div=1.0, z_TGR=10.0, z_tw=0.5, scale_bins=False,
    k_TGR=0.001, k_S=0.5, k_c=0.1, k_tw=0.01,
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
    return_kernel_constants=True, static_ctx=None,
):
    """Fully jax-traceable (jit/vmap-able) ``Kfuncs_to_tables`` for the
    PHENOM/binning model.

    Same physics/outputs as :func:`Kfuncs_to_tables`, but the growth and
    beyond-EdS kernel ODEs are integrated with diffrax (``fkptjax.jax_ode``) on
    the jax RHS (``fkptjax.binning_jax``), and every scalar stays ``jnp`` (no
    ``float()``/``np`` concretisation), so the whole fkpt loop can be ``jax.jit``
    / ``jax.vmap``'d.  The legacy numpy/numba :func:`Kfuncs_to_tables` is
    unchanged.  (The jax ODE is fully converged, so results match the legacy path
    to the legacy RKQS truncation, ~1e-3 in the multipoles.)
    """
    import folps as folpsv2
    from folps.tools_jax import extrapolate_pklin, simpson, interp
    from fkptjax.calculate_jax import JaxCalculator
    from fkptjax.util import setup_kfunctions
    from fkptjax import binning_jax as _bj
    from fkptjax.jax_ode import DP_jax, kernel_constants_jax

    k = jnp.asarray(k); pk = jnp.asarray(pk); pk_now = jnp.asarray(pk_now)
    k_ext, pk_ext = extrapolate_pklin(k, pk)
    _, pk_now_ext = extrapolate_pklin(k, pk_now)

    # binning constants (Om and mu* may be traced); xstop concrete (z is a float).
    P = _bj.pack_constants_jnp(
        om=Om, ol=1.0 - Om,
        mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4,
        z_div=z_div, z_TGR=z_TGR, z_tw=z_tw, scale_bins=scale_bins,
        k_TGR=k_TGR, k_c=k_c, k_S=k_S, k_tw=k_tw)
    xstop = float(np.log(1.0 / (1.0 + float(z))))

    # growth D(k), D'(k) via diffrax
    Y = DP_jax(k_ext, P, float(xnow), xstop)
    D_ext, Dp_ext = Y[0], Y[1]
    fk_ext = Dp_ext / D_ext

    if kmin is None:
        kmin = float(jnp.minimum(1e-3, jnp.min(k)))
    if kmax is None:
        kmax = float(jnp.maximum(0.5, jnp.max(k)))
    if f0_kmax is None:
        f0_kmax = float(kmin)

    mask0 = (k_ext <= float(f0_kmax))
    nhead = int(min(5, int(k_ext.shape[0])))
    f0 = jnp.where(
        jnp.any(mask0),
        jnp.sum(jnp.where(mask0, fk_ext, 0.0)) / jnp.maximum(jnp.sum(mask0), 1),
        jnp.mean(fk_ext[:nhead]),
    )

    # static (cosmology-independent) pieces: precomputed (jit path) or built here.
    if static_ctx is None:
        static_ctx = build_jax_static_ctx(
            k, kmin=kmin, kmax=kmax, Nk_kernel=Nk_kernel, nquadSteps=nquadSteps,
            NQ=NQ, NR=NR, rbao=rbao, pmax_bao=pmax_bao, Np_bao=Np_bao)
    calculator = static_ctx['calculator']
    kout = static_ctx['kout']
    p = static_ctx['p']
    j0 = static_ctx['j0']
    j2 = static_ctx['j2']

    fk_out = interp(kout, k_ext, fk_ext)
    fk_norm_out = fk_out / f0

    ff = fk_ext / f0
    sigma2w = 1.0 / (6.0 * jnp.pi**2) * simpson(pk_ext * ff**2, x=k_ext)
    sigma2w_NW = 1.0 / (6.0 * jnp.pi**2) * simpson(pk_now_ext * ff**2, x=k_ext)

    PSL_NW = interp(p, k_ext, pk_now_ext)
    sigma2_NW = 1.0 / (6.0 * jnp.pi**2) * simpson(PSL_NW * (1.0 - j0 + 2.0 * j2), x=p)
    delta_sigma2_NW = 1.0 / (2.0 * jnp.pi**2) * simpson(PSL_NW * j2, x=p)

    if bool(beyond_eds):
        KA, KAp, KR1, KR1p = kernel_constants_jax(f0, P, float(xnow), xstop)
        A = KA
        ApOverf0 = KAp / f0
        CFD3 = KR1
        CFD3p = KR1p
    else:
        A = 1.0; ApOverf0 = 0.0; CFD3 = 1.0; CFD3p = 1.0

    kfuncs = calculator.evaluate_jax(
        Pk_in=pk_ext, Pk_nw_in=pk_now_ext, fk_in=fk_ext,
        A=A, ApOverf0=ApOverf0, CFD3=CFD3, CFD3p=CFD3p, sigma2v=0.0, f0=f0)

    zeros = jnp.zeros_like(jnp.asarray(kout))

    def _tab(i, tail):
        return (
            kout, kfuncs.pkl[i], fk_norm_out,
            kfuncs.P22dd[i] + kfuncs.P13dd[i],
            kfuncs.P22du[i] + kfuncs.P13du[i],
            kfuncs.P22uu[i] + kfuncs.P13uu[i],
            kfuncs.Pb1b2[i], kfuncs.Pb1bs2[i], kfuncs.Pb22[i], kfuncs.Pb2s2[i],
            kfuncs.Ps22[i], kfuncs.sigma32PSL[i], kfuncs.Pb2theta[i], kfuncs.Pbs2theta[i],
            kfuncs.I1udd1A[i], kfuncs.I2uud1A[i], kfuncs.I2uud2A[i], kfuncs.I3uuu2A[i],
            kfuncs.I3uuu3A[i], kfuncs.I2uudd1BpC[i], kfuncs.I2uudd2BpC[i],
            kfuncs.I3uuud2BpC[i], kfuncs.I3uuud3BpC[i], kfuncs.I4uuuu2BpC[i],
            kfuncs.I4uuuu3BpC[i], kfuncs.I4uuuu4BpC[i], zeros, zeros,
        ) + tail

    table_w = _tab(0, (sigma2w, f0))
    table_nw = _tab(1, (sigma2w_NW, sigma2_NW, delta_sigma2_NW, f0))

    if return_kernel_constants:
        return table_w, table_nw, (A, ApOverf0 * f0, CFD3, CFD3p)
    return table_w, table_nw


def _prepare_jax_neutrino_correction(neutrino_correction):
    """Return JAX arrays needed for mu_nu(k, eta), or ``None``.

    This accepts the table-like object returned by ``fkptjax.neutrinos``
    (``NeutrinoTransferCorrection``): it must expose ``k``, ``eta`` and
    ``mu_nu`` attributes, where ``mu_nu`` has shape ``(neta, nk)``.

    Generic Python callables are intentionally not accepted in this JAX path,
    because they cannot be safely traced inside diffrax.  Use the legacy/eager
    ``Kfuncs_to_tables`` route for arbitrary Python callables.
    """
    if neutrino_correction is None:
        return None

    required = ("k", "eta", "mu_nu")
    if not all(hasattr(neutrino_correction, name) for name in required):
        raise TypeError(
            "JAX neutrino corrections require an object with attributes "
            "'k', 'eta' and 'mu_nu' (e.g. NeutrinoTransferCorrection). "
            "Generic Python callables are supported only by the legacy/eager "
            "Kfuncs_to_tables route."
        )

    k_nu = jnp.asarray(neutrino_correction.k, dtype=jnp.float64)
    eta_nu = jnp.asarray(neutrino_correction.eta, dtype=jnp.float64)
    mu_nu = jnp.asarray(neutrino_correction.mu_nu, dtype=jnp.float64)
    log_interp = bool(getattr(neutrino_correction, "log_interp", True))

    if mu_nu.ndim != 2:
        raise ValueError("neutrino_correction.mu_nu must have shape (neta, nk).")

    xk_nu = jnp.log(k_nu) if log_interp else k_nu
    return k_nu, eta_nu, mu_nu, xk_nu, log_interp


def _jax_mu_nu_from_table(eta, k, nu_data):
    """JAX interpolation of mu_nu(k, eta) from a tabulated correction."""
    if nu_data is None:
        return jnp.ones_like(k)

    k_nu, eta_nu, mu_nu, xk_nu, log_interp = nu_data

    # Clamp outside the transfer table.  This avoids uncontrolled high-z or
    # high-k extrapolation during the ODE integration.
    kk = jnp.clip(k, k_nu[0], k_nu[-1])
    xk = jnp.log(kk) if log_interp else kk
    ee = jnp.clip(eta, eta_nu[0], eta_nu[-1])

    j = jnp.searchsorted(eta_nu, ee, side="right") - 1
    j = jnp.clip(j, 0, eta_nu.size - 2)

    e0 = eta_nu[j]
    e1 = eta_nu[j + 1]
    t = jnp.where(e1 == e0, 0.0, (ee - e0) / (e1 - e0))

    y0 = jnp.interp(xk, xk_nu, mu_nu[j])
    y1 = jnp.interp(xk, xk_nu, mu_nu[j + 1])
    return (1.0 - t) * y0 + t * y1

def _dp_first_order_rescale_jax(
    k_arr,
    *,
    Om,
    xnow,
    xstop,
    model="HDKI",
    mg_variant="mu_OmDE",
    mu0=0.0,
    beta_1=1.0,
    lambda_1=0.0,
    exp_s=0.0,
    r_c=1.0e30,
    neutrino_correction=None,
    rtol=1e-6,
    atol=1e-8,
    max_steps=4096,
):
    """
    First-order JAX growth solver for rescaling branch.

    This is deliberately smaller than the full FKPT/beyond-EdS JAX machinery:
    it solves only

        D'' + (2 - f1) D' - f1 * mu(k, eta) * D = 0

    and returns [D(k), D'(k)] on k_arr.

    Supported here:
      - LCDM / GR
      - HDKI + mu_OmDE
      - HDKI + BZ

    PHENOM/binning should use the existing Kfuncs_to_tables_jax path.
    """
    import jax
    import jax.numpy as jnp
    import diffrax

    k_arr = jnp.asarray(k_arr, dtype=jnp.float64)
    Om = jnp.asarray(Om, dtype=jnp.float64)
    Ol = 1.0 - Om

    model_u = str(model).strip().upper()
    variant_l = str(mg_variant).strip().lower() if mg_variant is not None else ""

    if model_u in ("LCDM", "GR"):
        kind = "gr"
        is_MG_scale_dependent = False
    elif model_u == "NDGP":
        kind = "ndgp"
        is_MG_scale_dependent = False
    elif model_u == "HDKI" and variant_l in ("mu_omde", "muomde"):
        kind = "hdk_muomde"
        is_MG_scale_dependent = False
    elif model_u == "HDKI" and variant_l == "bz":
        kind = "hdk_bz"
        is_MG_scale_dependent = True
    else:
        raise NotImplementedError(
            "Kfuncs_to_tables_rescale_jax currently supports only "
            "LCDM/GR, nDGP, HDKI+mu_OmDE, and HDKI+BZ. "
            "Use the existing Kfuncs_to_tables_jax for PHENOM/binning."
        )

    mu0 = jnp.asarray(mu0, dtype=jnp.float64)
    beta_1 = jnp.asarray(beta_1, dtype=jnp.float64)
    lambda_1 = jnp.asarray(lambda_1, dtype=jnp.float64)
    exp_s = jnp.asarray(exp_s, dtype=jnp.float64)
    r_c = jnp.asarray(r_c, dtype=jnp.float64)

    nu_data = None
    if neutrino_correction is not None:
        nu_data = _prepare_jax_neutrino_correction(neutrino_correction)

    def f1_eta(eta):
        return 3.0 / (2.0 * (1.0 + Ol / Om * jnp.exp(3.0 * eta)))

    def mu_eta_k(eta, k):
        a = jnp.exp(eta)
        k2 = k * k

        if kind == "gr":
            mu_mg = jnp.ones_like(k)
        elif kind == "ndgp":
            Ea2 = Om * a**(-3.0) + Ol
            E = jnp.sqrt(Ea2)
            Oma = Om * a**(-3.0) / Ea2
            beta = 1.0 + 2.0 * E * r_c * (1.0 - 0.5 * Oma)
            mu_mg = (1.0 + 1.0 / (3.0 * beta)) * jnp.ones_like(k)
        elif kind == "hdk_muomde":
            OmDE_over_OmL = 1.0 / (Ol + Om * a**(-3.0))
            mu_mg = 1.0 + mu0 * OmDE_over_OmL * jnp.ones_like(k)
        elif kind == "hdk_bz":
            x = lambda_1**2 * k2 * a**exp_s
            mu_mg = (1.0 + beta_1 * x) / (1.0 + x)
        else:
            # Should never be reached because kind is checked above.
            mu_mg = jnp.ones_like(k)

        if nu_data is not None:
            return mu_mg * _jax_mu_nu_from_table(eta, k, nu_data)
        return mu_mg

    def rhs(eta, y, k):
        D, Dp = y[0], y[1]
        f1 = f1_eta(eta)
        mu = mu_eta_k(eta, k)
        return jnp.stack([
            Dp,
            f1 * mu * D - (2.0 - f1) * Dp,
        ])

    term = diffrax.ODETerm(rhs)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    xnow = float(xnow)
    xstop = float(xstop)
    dt0 = (xstop - xnow) / 128.0

    y0 = jnp.exp(jnp.asarray(xnow, dtype=jnp.float64)) * jnp.ones(2, dtype=jnp.float64)
    saveat = diffrax.SaveAt(ts=jnp.asarray([xstop], dtype=jnp.float64))

    def solve_one(k):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=xnow,
            t1=xstop,
            dt0=dt0,
            y0=y0,
            args=k,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return sol.ys[0]

    # For scale-independent MG models (GR/LCDM, nDGP, HDKI+mu_OmDE) and no
    # tabulated neutrino source, the first-order growth ODE is identical for
    # every k.  Solve it once and broadcast.  Scale-dependent models such as
    # BZ, or any run with a neutrino transfer table, keep the full vmap.
    if (nu_data is None) and (not bool(is_MG_scale_dependent)):
        y_one = solve_one(k_arr[0])
        Y = jnp.broadcast_to(y_one[None, :], (k_arr.shape[0], 2))
    else:
        Y = jax.vmap(solve_one)(k_arr)  # shape: (nk, 2)
    return jnp.moveaxis(Y, 0, 1)     # shape: (2, nk)



def Kfuncs_to_tables_rescale_jax(
    k,
    pk,
    pk_now,
    *,
    z,
    Om,
    model="HDKI",
    mg_variant="mu_OmDE",
    beyond_eds=False,
    rescale_PS=True,
    kmin=None,
    kmax=None,
    Nk_kernel=120,
    nquadSteps=300,
    NQ=10,
    NR=10,
    xnow=-3.912023,
    ode_method=None,
    f0_kmax=None,
    mu0=0.0,
    beta_1=1.0,
    lambda_1=0.0,
    exp_s=0.0,
    r_c=1.0e30,
    rbao=104.0,
    pmax_bao=0.4,
    Np_bao=100,
    return_kernel_constants=True,
    static_ctx=None,
    kernel_constants=None,
    neutrino_correction=None,
    **kwargs,
):
    """
    Rescaling branch JAX FKPT table builder.

    This is the rescale-PS sister of Kfuncs_to_tables_jax:

      emulated GR/LCDM pk, pk_now
          -> extrapolate
          -> solve first-order GR and MG growth with JAX/diffrax
          -> rescale pk and pk_now by (D_MG / D_GR)^2
          -> feed rescaled pk, pk_now, f_MG(k) to JaxCalculator.evaluate_jax
          -> return FOLPS-format tables

    Important:
      - This is NOT the PHENOM/binning full-JAX route.
      - The existing Kfuncs_to_tables_jax remains the PHENOM/binning route.
      - This route supports LCDM/GR, nDGP, HDKI+mu_OmDE, and HDKI+BZ.
      - beyond_eds=True is supported when model-matched kernel constants
        are supplied through ``kernel_constants`` (provided by
        the caller or a caller-side emulator).
      - Massive-neutrino corrections in this JAX route require a tabulated
        NeutrinoTransferCorrection-like object, not a generic Python callable.
    """
    import numpy as np
    import jax.numpy as jnp
    from folps.tools_jax import extrapolate_pklin, simpson, interp

    k = jnp.asarray(k, dtype=jnp.float64)
    pk = jnp.asarray(pk, dtype=jnp.float64)
    pk_now = jnp.asarray(pk_now, dtype=jnp.float64)

    # beyond_eds=True is supported in the rescaling branch live/JAX path only when
    # model-matched kernel constants are provided by a small calculator or
    # emulator.  This keeps this function traceable: no legacy Python ODE /
    # kernel_constants(...) calls are made here.

    # ------------------------------------------------------------------
    # 1. Extrapolate input linear spectra.
    # ------------------------------------------------------------------
    k_ext, pk_ext = extrapolate_pklin(k, pk)
    _, pk_now_ext = extrapolate_pklin(k, pk_now)

    xstop = float(np.log(1.0 / (1.0 + float(z))))
    xnow_f = float(xnow)

    # ------------------------------------------------------------------
    # 2. First-order GR and MG growth.
    # ------------------------------------------------------------------
    Y_gr = _dp_first_order_rescale_jax(
        k_ext,
        Om=Om,
        xnow=xnow_f,
        xstop=xstop,
        model="GR",
        mg_variant=None,
        mu0=0.0,
        beta_1=1.0,
        lambda_1=0.0,
        exp_s=0.0,
        r_c=1.0e30,
        neutrino_correction=neutrino_correction,
    )
    D_gr = Y_gr[0]

    Y_mg = _dp_first_order_rescale_jax(
        k_ext,
        Om=Om,
        xnow=xnow_f,
        xstop=xstop,
        model=model,
        mg_variant=mg_variant,
        mu0=mu0,
        beta_1=beta_1,
        lambda_1=lambda_1,
        exp_s=exp_s,
        r_c=r_c,
        neutrino_correction=neutrino_correction,
    )
    D_mg, Dp_mg = Y_mg[0], Y_mg[1]

    growth_scale = (D_mg / D_gr) ** 2

    pk_ext = pk_ext * growth_scale
    pk_now_ext = pk_now_ext * growth_scale

    fk_ext = Dp_mg / D_mg

    # ------------------------------------------------------------------
    # 3. f0 and static FKPT context.
    # ------------------------------------------------------------------
    if kmin is None:
        kmin = float(jnp.minimum(1e-3, jnp.min(k)))
    if kmax is None:
        kmax = float(jnp.maximum(0.5, jnp.max(k)))
    if f0_kmax is None:
        f0_kmax = float(kmin)

    mask0 = k_ext <= float(f0_kmax)
    nhead = int(min(5, int(k_ext.shape[0])))

    f0 = jnp.where(
        jnp.any(mask0),
        jnp.sum(jnp.where(mask0, fk_ext, 0.0)) / jnp.maximum(jnp.sum(mask0), 1),
        jnp.mean(fk_ext[:nhead]),
    )

    if static_ctx is None:
        static_ctx = build_jax_static_ctx(
            k,
            kmin=kmin,
            kmax=kmax,
            Nk_kernel=Nk_kernel,
            nquadSteps=nquadSteps,
            NQ=NQ,
            NR=NR,
            rbao=rbao,
            pmax_bao=pmax_bao,
            Np_bao=Np_bao,
        )

    calculator = static_ctx["calculator"]
    kout = static_ctx["kout"]
    p = static_ctx["p"]
    j0 = static_ctx["j0"]
    j2 = static_ctx["j2"]

    fk_out = interp(kout, k_ext, fk_ext)
    fk_norm_out = fk_out / f0

    # ------------------------------------------------------------------
    # 4. IR/BAO sigma integrals.
    # ------------------------------------------------------------------
    ff = fk_ext / f0

    sigma2w = 1.0 / (6.0 * jnp.pi**2) * simpson(pk_ext * ff**2, x=k_ext)
    sigma2w_NW = 1.0 / (6.0 * jnp.pi**2) * simpson(pk_now_ext * ff**2, x=k_ext)

    PSL_NW = interp(p, k_ext, pk_now_ext)

    sigma2_NW = (
        1.0 / (6.0 * jnp.pi**2)
        * simpson(PSL_NW * (1.0 - j0 + 2.0 * j2), x=p)
    )
    delta_sigma2_NW = (
        1.0 / (2.0 * jnp.pi**2)
        * simpson(PSL_NW * j2, x=p)
    )

    # ------------------------------------------------------------------
    # 5. beyond-EdS constants.
    #    For rescaling branch, these must be provided by an ingredient calculator or
    #    its emulator; do not call legacy kernel_constants(...) here.
    # ------------------------------------------------------------------
    if bool(beyond_eds):
        if kernel_constants is None:
            raise ValueError(
                "rescale_PS=True with beyond_eds=True requires provided/emulated "
                "kernel constants. Attach kernel-constants provider or its "
                "EmulatedCalculator as kernel_constants."
            )

        def _get_kc(obj, names, default=None):
            if isinstance(names, str):
                names = (names,)
            if isinstance(obj, dict):
                for name in names:
                    if name in obj:
                        return obj[name]
            if isinstance(obj, (tuple, list)):
                mapping = {"A": 0, "Ap": 1, "KAp": 1, "CFD3": 2, "KR1": 2, "CFD3p": 3, "KR1p": 3}
                for name in names:
                    if name in mapping and len(obj) > mapping[name]:
                        return obj[mapping[name]]
            for name in names:
                if hasattr(obj, name):
                    return getattr(obj, name)
            return default

        A = jnp.asarray(_get_kc(kernel_constants, ("A", "KA")), dtype=jnp.float64)
        CFD3 = jnp.asarray(_get_kc(kernel_constants, ("CFD3", "KR1")), dtype=jnp.float64)
        CFD3p = jnp.asarray(_get_kc(kernel_constants, ("CFD3p", "KR1p")), dtype=jnp.float64)

        ap_over = _get_kc(kernel_constants, "ApOverf0", default=None)
        if ap_over is None:
            Ap = jnp.asarray(_get_kc(kernel_constants, ("Ap", "KAp")), dtype=jnp.float64)
            ApOverf0 = Ap / f0
        else:
            ApOverf0 = jnp.asarray(ap_over, dtype=jnp.float64)
    else:
        A = jnp.asarray(1.0, dtype=jnp.float64)
        ApOverf0 = jnp.asarray(0.0, dtype=jnp.float64)
        CFD3 = jnp.asarray(1.0, dtype=jnp.float64)
        CFD3p = jnp.asarray(1.0, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # 6. FKPT JAX calculator.
    # ------------------------------------------------------------------
    kfuncs = calculator.evaluate_jax(
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

    zeros = jnp.zeros_like(jnp.asarray(kout))

    def _tab(i, tail):
        return (
            kout,
            kfuncs.pkl[i],
            fk_norm_out,
            kfuncs.P22dd[i] + kfuncs.P13dd[i],
            kfuncs.P22du[i] + kfuncs.P13du[i],
            kfuncs.P22uu[i] + kfuncs.P13uu[i],
            kfuncs.Pb1b2[i],
            kfuncs.Pb1bs2[i],
            kfuncs.Pb22[i],
            kfuncs.Pb2s2[i],
            kfuncs.Ps22[i],
            kfuncs.sigma32PSL[i],
            kfuncs.Pb2theta[i],
            kfuncs.Pbs2theta[i],
            kfuncs.I1udd1A[i],
            kfuncs.I2uud1A[i],
            kfuncs.I2uud2A[i],
            kfuncs.I3uuu2A[i],
            kfuncs.I3uuu3A[i],
            kfuncs.I2uudd1BpC[i],
            kfuncs.I2uudd2BpC[i],
            kfuncs.I3uuud2BpC[i],
            kfuncs.I3uuud3BpC[i],
            kfuncs.I4uuuu2BpC[i],
            kfuncs.I4uuuu3BpC[i],
            kfuncs.I4uuuu4BpC[i],
            zeros,
            zeros,
        ) + tail

    table_w = _tab(0, (sigma2w, f0))
    table_nw = _tab(1, (sigma2w_NW, sigma2_NW, delta_sigma2_NW, f0))

    if return_kernel_constants:
        return table_w, table_nw, (A, ApOverf0 * f0, CFD3, CFD3p)

    return table_w, table_nw
