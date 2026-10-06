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
    mu_kinf_BZmass=1.0,
    lambda_a_BZmass=0.0,
    lambda_dS_BZmass=0.0,
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
            mu_kinf_BZmass=float(mu_kinf_BZmass),
            lambda_a_BZmass=float(lambda_a_BZmass),
            lambda_dS_BZmass=float(lambda_dS_BZmass),
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
    fkpt_approximation: bool = True,
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
    mu_kinf_BZmass: float = 1.0,
    lambda_a_BZmass: float = 0.0,
    lambda_dS_BZmass: float = 0.0,
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
    t_k: float = 1000.0,
    d_s: float = 0.0001,
    w0: float = -1.0,
    wa: float = 0.0,
    eftcamb_h1_interp=None,
    eftcamb_h3_interp=None,
    eftcamb_h5_interp=None,
    rbao: float = 104.0,
    pmax_bao: float = 0.4,
    Np_bao: int = 100,
    return_kernel_constants=True,
    return_raw_kfuncs: bool = False,
    neutrino_correction=None,
    use_numba: bool = False,
    fk=None, f0=None,
    ingredients_fn=None,
    static_ctx=None,
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """
    Return (table_wiggle, table_now) in the A_full=False layout expected by FOLPS.
    Uses fkptjax internal output grid by default (init_data.logk_grid).

    Parameters
    ----------
    fk : array or None, default=None
        Linear growth rate f(k) on the input grid ``k``. ``None`` (default) integrates the
        growth ODE internally (``fkptjax.ode.DP``), as before. Supply it when ``pk``/``pk_now``
        come from an emulator or a full Boltzmann solve (e.g. desilike's
        ``DirectSpectrum2Template``, ``sqrt(P_theta_cb/P_delta_cb)``) that also determines the
        growth: the ODE and that source are otherwise two independent statements about the same
        cosmology's growth. Extended to the internal extrapolated grid by edge-clamped
        interpolation. Mirrors ``Kfuncs_to_tables_jax``'s ``fk`` -- see there for the full SIGN
        CAVEAT (the internal ODE returns a SIGNED f = D'/D and can go negative in a decaying-mode
        regime; a caller deriving fk from a spectrum ratio such as sqrt(P_theta/P_delta) can only
        supply |f|).
    f0 : float or None, default=None
        Scale-independent growth rate. ``None`` estimates it from ``fk`` (averaged over
        ``k <= f0_kmax``, or at the ``k=0.1`` pivot for growth-index models). Pass it explicitly
        alongside ``fk`` so the two cannot drift apart -- see ``Kfuncs_to_tables_jax``.

        NEUTRINO CAVEAT: ``neutrino_correction`` (above) only corrects the internal ODE
        (``ModelDerivatives``/``DP``); it has no effect once ``fk`` is supplied externally. This
        is fine when ``fk`` already comes from a full Boltzmann solve on the same
        (massive-neutrino) cosmology, since that already captures the cb growth suppression the
        internal correction approximates -- but the two are redundant, not additive, if combined.
    fkpt_approximation : bool, default=True
        ``True`` (default, unchanged historical behavior): beyond-EdS kernels use the fkPT
        large-scale-limit approximation (``ode.kernel_constants``), reused as global scalars for
        every loop mode. ``False``: the genuine, non-squeezed (external k, loop q, angle) full
        kernels are computed via ``fkptjax.MG_kernels.I1udd1_and_P13_grid`` instead, mirroring
        ``Kfuncs_to_tables_jax``'s ``fkpt_approximation=False`` path exactly (same formulas, same
        grids) -- see that function's docstring for the full description. Unlike
        ``Kfuncs_to_tables_jax``, this route also supports ``neutrino_correction`` together with
        ``fkpt_approximation=False``, since the internal growth ODE here is the numpy
        ``ModelDerivatives``/``DP`` one, which already accepts it. Has no effect when
        ``beyond_eds=False``.
    ingredients_fn : callable or None, default=None
        Only consulted when ``beyond_eds=True`` and ``fkpt_approximation=False``. If given,
        called as ``ingredients_fn(k_ext_q, q_loop_Q, kminus_Q, x_r, k_r, p_r, f0)`` -- the SAME
        live ``f0`` this function passes to ``I1udd1_and_P13_grid`` in the ``else`` branch below,
        since the ``ApOverf0``/``CFD3p``-type outputs need to be normalized by it, and it depends
        on the MG parameters (so a callee cannot just assume a fixed value) -- in place of the
        live ``fkptjax.MG_kernels.I1udd1_and_P13_grid(..., P_mg, ...)`` ODE solve, and expected to
        return the SAME 16-tuple ``(A_Q, B_Q, ApOverf0_Q, BpOverf0_Q, A_N, B_N, ApOverf0_N,
        BpOverf0_N, A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ, A_fused_R, ApOverf0_fused_R, CFD3_R,
        CFD3p_R)``. Everything downstream (the P22/P13 combination in ``calculator.evaluate``) is
        unchanged either way -- this hook only replaces HOW the beyond-EdS kernel ingredients
        themselves are obtained, e.g. from a coarse-grid Chebyshev emulator's prediction
        interpolated onto this call's actual Q-loop/R-loop grid (``fkptjax.ab_ingredients``)
        instead of a fresh live ODE solve at every quadrature point. The caller is responsible
        for building the closure so it reflects the CURRENT MG parameter values (this function
        itself does not know they changed if ``ingredients_fn`` was built once and reused --
        mirroring how ``mg_kernel_fn`` is passed fresh into ``full_shape.py``'s
        ``combine_bias_terms_spectrum3_poles`` on every call).
    w0, wa : float, default=-1.0, 0.0
        CPL dark-energy equation of state, ``w(a) = w0 + wa*(1-a)``. Only consumed by the
        ``HS`` and ``PHENOM``/``growth_index``/``growth_index_yukawa`` ``mu(k,eta)`` formulas
        (see ``mg_jax.py``'s module docstring); every other model ignores them. The growth ODE's
        own background (``ModelDerivatives.f1``) is always fixed flat LCDM regardless of these --
        ``growth_index`` with ``gamma_0`` pinned to its GR value (``6/11``) is the way to isolate
        a genuinely evolving dark-energy background's effect on growth, with no MG signal mixed
        in, since its ``mu(a)`` formula is built from the true ``w0``/``wa`` Friedmann equation
        evaluated relative to that fixed fiducial background.
    """

    import jax
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

    # derivs/solver concretize every MG/cosmology parameter via float() to drive fkptjax's
    # own scipy-based growth ODE (DP) and squeezed-limit kernel-constants ODE below -- this is
    # exactly what makes this builder non-jittable ("Wall 2", see this function's docstring).
    # Both are skippable: DP when fk is supplied (the caller's own growth, e.g. from a live/
    # emulated template); the kernel-constants ODE whenever beyond_eds and not
    # fkpt_approximation, since calculate_jax.JaxCalculator.evaluate()'s A_Q/CFD3_R (etc.)
    # kwargs -- always given in that regime, live or via ingredients_fn -- make its own A/
    # ApOverf0/CFD3/CFD3p fallback dead (`A_Q = A if A_Q is None else A_Q`, and the same
    # pattern for B_Q/CFD3_R/...): the squeezed scalars are never read once the per-triangle
    # arrays exist. Building derivs/solver LAZILY (only when one of those two ODEs actually
    # runs, or rescale_PS needs them) lets a caller who supplies both fk and ingredients_fn
    # avoid ALL concretization here -- the remaining computation is genuinely jit/vmap-able.
    def _get_derivs_solver():
        solver = ODESolver(zout=float(z), xnow=float(xnow), method=str(ode_method))
        derivs = ModelDerivatives(
            om=float(Om), ol=float(1.0 - Om),
            fR0_HS=float(fR0_HS), beta2=float(beta2), n_HS=float(n_HS),
            screening=int(screening), omegaBD=float(omegaBD),
            r_c=float(r_c),
            model=str(model), mg_variant=str(mg_variant) if mg_variant is not None else "mu_OmDE",
            mu0=float(mu0),
            beta_1=float(beta_1), lambda_1=float(lambda_1), exp_s=float(exp_s),
            mu_kinf_BZmass=float(mu_kinf_BZmass),
            lambda_a_BZmass=float(lambda_a_BZmass),
            lambda_dS_BZmass=float(lambda_dS_BZmass),
            mu1=float(mu1), mu2=float(mu2), mu3=float(mu3), mu4=float(mu4),
            z_div=float(z_div), z_TGR=float(z_TGR), z_tw=float(z_tw),
            scale_bins=bool(scale_bins), k_TGR=float(k_TGR), k_S=float(k_S), k_c=float(k_c), k_tw=float(k_tw),
            gamma_0=float(gamma_0), gamma_a=float(gamma_a), t_k=float(t_k), d_s=float(d_s),
            w0=float(w0), wa=float(wa),
            eftcamb_h1_interp=eftcamb_h1_interp,
            eftcamb_h3_interp=eftcamb_h3_interp,
            eftcamb_h5_interp=eftcamb_h5_interp,
            neutrino_correction=neutrino_correction,
            # use_numba=bool(use_numba),
        )
        return derivs, solver

    if fk is None:
        derivs, solver = _get_derivs_solver()
        k_ext_np = np.asarray(k_ext, dtype=float)
        Y = DP(k_ext_np, derivs, solver)
        D_ext, Dp_ext = Y[0], Y[1]
        fk_ext = jnp.asarray(Dp_ext / D_ext)
    else:
        # fk arrives on the INPUT grid `k`; lift it to the extrapolated grid the same way
        # Kfuncs_to_tables_jax does: jnp.interp (edge-clamped), not extrapolate_pklin (which
        # fits a power law -- right for P(k), wrong for f(k), which tends to f0 as k -> 0 and
        # stays bounded and smooth at high k).
        fk_in_arr = jnp.asarray(fk, dtype=jnp.float64)
        if fk_in_arr.shape != jnp.asarray(k).shape:
            raise ValueError(
                f"fk must have the same shape as k; got {fk_in_arr.shape} vs "
                f"{jnp.asarray(k).shape}")
        fk_ext = jnp.interp(k_ext, jnp.asarray(k), fk_in_arr)

    # Define the kernel grid range before estimating f0.
    # If f0_kmax is not explicitly provided, use the routine's kmin.
    if kmin is None:
        kmin = float(jnp.minimum(1e-3, jnp.min(k)))
    if kmax is None:
        kmax = float(jnp.maximum(0.5, jnp.max(k)))

    if f0 is not None:
        # Take the caller's f0 verbatim: the estimator below is a DIFFERENT definition (an
        # average of fk_ext, or a fixed-k0 pivot) that would silently drift from the caller's.
        # NOT concretized (no float()) -- a caller feeding a traced f0 (e.g. from an emulated
        # template) needs this to stay jit/vmap-able; jnp.asarray is a no-op either way.
        f0 = jnp.asarray(f0, dtype=jnp.float64)
    else:
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
        derivs, solver = _get_derivs_solver()
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
            mu_kinf_BZmass=mu_kinf_BZmass,
            lambda_a_BZmass=lambda_a_BZmass,
            lambda_dS_BZmass=lambda_dS_BZmass,
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

    if static_ctx is not None:
        # setup_kfunctions (fkptjax.util) does data-DEPENDENT binning (Python-level max()/min()
        # on k_in's own values, not just its shape), so it needs a genuinely concrete k_ext --
        # incompatible with a traced k/pk (e.g. from an emulated template). static_ctx (see
        # build_jax_static_ctx) precomputes init_data/calculator ONCE, concretely, outside any
        # trace, exactly like Kfuncs_to_tables_jax already does; reuse it here instead of
        # rebuilding from this call's (possibly traced) k_ext.
        init_data = static_ctx['init_data']
        calculator_static = static_ctx['calculator']
        kout = static_ctx['kout']
    else:
        init_data = setup_kfunctions(
            k_in=k_ext,
            kmin=float(kmin),
            kmax=float(kmax),
            Nk=int(Nk_kernel),
            nquadSteps=int(nquadSteps),
            NQ=int(NQ),
            NR=int(NR),
        )
        calculator_static = None
        kout = init_data.logk_grid

    fk_out = interp(kout, k_ext, fk_ext)
    fk_norm_out = fk_out / f0

    # sigma^2 (wiggle / no-wiggle) and the BAO sigma^2 integrals are small 1D
    # Simpson quadratures. Evaluating them eagerly in jax forces a host<->device
    # round-trip per op (device_put / apply_primitive churn ~20 ms/call); when nothing here
    # is actually traced (the ordinary eager call this builder was written for), do them in
    # numpy instead -- scipy's Simpson rule reproduces folps' jax `simpson` bit-for-bit (same
    # composite rule), so results are unchanged. But a caller under jit/vmap/grad (e.g.
    # beyond_eds=True, fkpt_approximation=False with both fk/f0 and ingredients_fn supplied,
    # so nothing else below needs concretizing either) hands pk_ext/fk_ext/f0 as tracers,
    # which np.asarray()/float() cannot concretize -- use the already-imported jax-native
    # `simpson` (matching Kfuncs_to_tables_jax's own identical block) in that case instead.
    _traced = any(isinstance(x, jax.core.Tracer) for x in (pk_ext, pk_now_ext, fk_ext, f0))

    ff = fk_ext / f0
    if _traced:
        sigma2w = 1.0 / (6.0 * jnp.pi**2) * simpson(pk_ext * ff**2, x=k_ext)
        sigma2w_NW = 1.0 / (6.0 * jnp.pi**2) * simpson(pk_now_ext * ff**2, x=k_ext)

        p = jnp.exp(jnp.linspace(jnp.log(1e-6), jnp.log(float(pmax_bao)), int(Np_bao)))
        PSL_NW = interp(p, k_ext, pk_now_ext)
        j0 = jnp.asarray(folpsv2.spherical_jn_backend(0, p * float(rbao)))
        j2 = jnp.asarray(folpsv2.spherical_jn_backend(2, p * float(rbao)))

        sigma2_NW = 1.0 / (6.0 * jnp.pi**2) * simpson(PSL_NW * (1.0 - j0 + 2.0 * j2), x=p)
        delta_sigma2_NW = 1.0 / (2.0 * jnp.pi**2) * simpson(PSL_NW * j2, x=p)
    else:
        from scipy.integrate import simpson as _np_simpson

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

    if bool(beyond_eds) and bool(fkpt_approximation):
        from fkptjax.ode import kernel_constants
        derivs, solver = _get_derivs_solver()
        KA, KAp, KR1, KR1p = kernel_constants(f0=f0, derivs=derivs, solver=solver)
        A = float(KA)
        ApOverf0 = float(KAp) / float(f0)
        CFD3 = float(KR1)
        CFD3p = float(KR1p)
    else:
        # beyond_eds=False, or beyond_eds=True with fkpt_approximation=False: either way these
        # squeezed-limit scalars are unused placeholders. In the fkpt_approximation=False case,
        # calculate_jax.JaxCalculator.evaluate() always receives the genuine per-triangle A_Q/
        # B_Q/.../CFD3_R/CFD3p_R below (live or via ingredients_fn) and only falls back to A/
        # ApOverf0/CFD3/CFD3p when those are None (`A_Q = A if A_Q is None else A_Q`, same
        # pattern for the others) -- so skipping kernel_constants's ODE here changes nothing
        # about the output, and (with fk also supplied) avoids concretizing Om/mu0/etc at all.
        A = 1.0
        ApOverf0 = 0.0
        CFD3 = 1.0
        CFD3p = 1.0

    if calculator_static is not None:
        calculator = calculator_static
    else:
        calculator = JaxCalculator()
        calculator.initialize(init_data)

    # Full (non-fkPT-approximated) beyond-EdS kernels: solved at the genuine
    # (external k, loop q, angle) triples of the Q/R quadrature grids instead of
    # reusing the large-scale-limit scalars A/ApOverf0/CFD3/CFD3p above. This
    # mirrors Kfuncs_to_tables_jax's fkpt_approximation=False block exactly (same
    # formulas, same grids, same MG_kernels entry point) -- see that function's
    # docstring for the full description of the Q/R/N orderings. Unlike that jax
    # route, ``derivs``/``solver`` above already carry ``neutrino_correction``, so
    # the growth ODE feeding ``f0``/``fk_ext`` is neutrino-corrected too when
    # requested; this block only adds the k/q/angle-dependent kernel pieces on
    # top of that.
    A_Q = B_Q = ApOverf0_Q = BpOverf0_Q = None
    A_N = B_N = ApOverf0_N = BpOverf0_N = None
    A_RQ = B_RQ = ApOverf0_RQ = BpOverf0_RQ = None
    A_fused_R = ApOverf0_fused_R = None
    CFD3_R = CFD3p_R = None
    if bool(beyond_eds) and not bool(fkpt_approximation):
        from fkptjax import mg_jax as _bj
        from fkptjax import MG_kernels as _mgk

        kind = _resolve_mg_kind(model, mg_variant)
        P_mg = _bj.pack_constants_jnp(
            om=Om, ol=1.0 - Om, kind=kind,
            mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4,
            z_div=z_div, z_TGR=z_TGR, z_tw=z_tw, scale_bins=scale_bins,
            k_TGR=k_TGR, k_c=k_c, k_S=k_S, k_tw=k_tw,
            mu_kinf=mu_kinf_BZmass, lambda_a=lambda_a_BZmass, lambda_dS=lambda_dS_BZmass,
            fR0_HS=fR0_HS, beta2=beta2, n_HS=n_HS,
            r_c=r_c, mu0=mu0,
            beta_1=beta_1, lambda_1=lambda_1, exp_s=exp_s,
            w0=w0, wa=wa,
            gamma_0=gamma_0, gamma_a=gamma_a, t_k=t_k, d_s=d_s,
            neutrino_correction=neutrino_correction)
        xstop = jnp.log(1.0 / (1.0 + jnp.asarray(z, dtype=jnp.float64)))

        k_ext_q = calculator.logk_grid_jax
        r_q = calculator.r_jax
        x_q = calculator.x_jax
        y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
        q_loop_Q = r_q * k_ext_q
        kminus_Q = k_ext_q * y_q

        r_r = calculator.r_r_jax
        x_r = calculator.x_r_jax
        q_loop_R = r_r * k_ext_q

        if ingredients_fn is not None:
            (A_Q, B_Q, ApOverf0_Q, BpOverf0_Q,
             A_N, B_N, ApOverf0_N, BpOverf0_N,
             A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ,
             A_fused_R, ApOverf0_fused_R, CFD3_R, CFD3p_R) = ingredients_fn(
                k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R,
                jnp.asarray(f0, dtype=jnp.float64))
        else:
            (A_Q, B_Q, ApOverf0_Q, BpOverf0_Q,
             A_N, B_N, ApOverf0_N, BpOverf0_N,
             A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ,
             A_fused_R, ApOverf0_fused_R, CFD3_R, CFD3p_R) = _mgk.I1udd1_and_P13_grid(
                k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P_mg, float(xnow), xstop,
                jnp.asarray(f0, dtype=jnp.float64),
                solver='rk4', n_steps=64)

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
        A_Q=A_Q, B_Q=B_Q, ApOverf0_Q=ApOverf0_Q, BpOverf0_Q=BpOverf0_Q,
        A_fused_R=A_fused_R, ApOverf0_fused_R=ApOverf0_fused_R,
        CFD3_R=CFD3_R, CFD3p_R=CFD3p_R,
        A_N=A_N, B_N=B_N, ApOverf0_N=ApOverf0_N, BpOverf0_N=BpOverf0_N,
        A_RQ=A_RQ, B_RQ=B_RQ, ApOverf0_RQ=ApOverf0_RQ, BpOverf0_RQ=BpOverf0_RQ,
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
    if return_kernel_constants and return_raw_kfuncs:
        return table_w, table_nw, (A, ApOverf0 * f0, CFD3, CFD3p), kfuncs
    if return_kernel_constants:
        return table_w, table_nw, (A, ApOverf0 * f0, CFD3, CFD3p)
    if return_raw_kfuncs:
        return table_w, table_nw, kfuncs
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


def _resolve_mg_kind(model, mg_variant):
    """Map desilike's (model, mg_variant) onto an :data:`fkptjax.mg_jax.KINDS` value."""
    from fkptjax import mg_jax as _mg

    model_u = str(model or '').strip().upper()
    variant_u = str(mg_variant or '').strip().upper()
    if model_u in ('LCDM', 'GR'):
        return _mg.LCDM
    if model_u == 'HS':
        return _mg.HS
    if model_u == 'NDGP':
        return _mg.NDGP
    if model_u == 'HDKI' and variant_u in ('MU_OMDE', 'MUOMDE'):
        return _mg.MU_OMDE
    if model_u == 'HDKI' and variant_u == 'BZ':
        return _mg.BZ
    if model_u == 'HDKI' and variant_u in ('BZ_MASS', 'BZMASS'):
        return _mg.BZ_MASS
    if model_u == 'HDKI' and variant_u in ('EFT_DE', 'EFTDE'):
        return _mg.EFT_DE
    if model_u == 'PHENOM' and variant_u == 'BINNING':
        return _mg.BINNING
    if model_u == 'PHENOM' and variant_u in ('GROWTH_INDEX', 'GROWTHINDEX'):
        return _mg.GROWTH_INDEX
    if model_u == 'PHENOM' and variant_u in ('GROWTH_INDEX_YUKAWA', 'GROWTHINDEXYUKAWA'):
        return _mg.GROWTH_INDEX_YUKAWA
    raise NotImplementedError(
        f"Kfuncs_to_tables_jax has no JAX right-hand side for model={model!r}, "
        f"mg_variant={mg_variant!r}; supported: LCDM/GR, HS, NDGP, "
        "HDKI/mu_OmDE, HDKI/BZ, HDKI/BZ_Mass, HDKI/EFT_DE, PHENOM/binning, "
        "PHENOM/growth_index, PHENOM/growth_index_yukawa.")


def Kfuncs_to_tables_jax(
    k, pk, pk_now, *, z, Om, beyond_eds=True, fkpt_approximation=True,
    kmin=None, kmax=None, Nk_kernel=120, nquadSteps=300, NQ=10, NR=10,
    xnow=-3.912023, f0_kmax=None,
    mu1=1.0, mu2=1.0, mu3=1.0, mu4=1.0,
    z_div=1.0, z_TGR=10.0, z_tw=0.5, scale_bins=False,
    k_TGR=0.001, k_S=0.5, k_c=0.1, k_tw=0.01,
    rbao=104.0, pmax_bao=0.4, Np_bao=100,
    return_kernel_constants=True, return_raw_kfuncs=False, static_ctx=None,
    fk=None, f0=None, ingredients_fn=None,
    model='PHENOM', mg_variant='binning',
    w0=-1.0, wa=0.0,
    fR0_HS=0.0, beta2=1.0 / 6.0, n_HS=1,
    r_c=1.0e30,
    mu0=0.0,
    beta_1=1.0, lambda_1=1.0, exp_s=1.0,
    mu_kinf_BZmass=1.0, lambda_a_BZmass=0.0, lambda_dS_BZmass=0.0,
    eftde_eta_grid=None, eftde_h1_grid=None, eftde_h3_grid=None, eftde_h5_grid=None,
    eftde_scale_dependent=True, eftde_k2_pivot=0.01,
    gamma_0=0.54545, gamma_a=0.0, t_k=1000.0, d_s=0.0001,
):
    """Fully jax-traceable (jit/vmap-able) ``Kfuncs_to_tables``.

    Supports every model :mod:`fkptjax.mg_jax` implements: ``LCDM``/``GR``,
    ``HS`` (Hu-Sawicki f(R)), ``NDGP``, ``HDKI`` with ``mg_variant`` in
    ``{'mu_OmDE', 'BZ', 'BZ_Mass', 'EFT_DE'}``, and ``PHENOM`` with
    ``mg_variant`` in ``{'binning', 'growth_index', 'growth_index_yukawa'}``
    (the default, and the historical behaviour, is ``PHENOM``/``binning``).
    ``model`` and ``mg_variant`` are STATIC: they choose which mu expression
    is traced. See :mod:`fkptjax.mg_jax`'s module docstring for the exact
    mu(k,eta) formula used by each and for the ``eftde_*`` grid convention
    (EFT_DE takes h1/h3/h5(eta) pre-sampled on a grid, not callables).

    Same physics/outputs as :func:`Kfuncs_to_tables`, but the growth and
    beyond-EdS kernel ODEs are integrated with diffrax (``fkptjax.jax_ode``) on
    the jax RHS (``fkptjax.binning_jax``), and every scalar stays ``jnp`` (no
    ``float()``/``np`` concretisation), so the whole fkpt loop can be ``jax.jit``
    / ``jax.vmap``'d.  The legacy numpy/numba :func:`Kfuncs_to_tables` is
    unchanged.  (The jax ODE is fully converged, so results match the legacy path
    to the legacy RKQS truncation, ~1e-3 in the multipoles.)

    Parameters
    ----------
    fk : array or None, default=None
        Linear growth rate f(k) on the input grid ``k``.  ``None`` (default)
        integrates the binned-mu growth ODE internally, as before.  Supply it when
        ``pk``/``pk_now`` come from an emulator that also predicts the growth: the
        ODE and the emulated P_theta are otherwise independent statements about the
        same cosmology, so the loop kernels and the linear spectrum can describe
        subtly different models.  Extended to the internal extrapolated grid by
        edge-clamped interpolation.  Replaces the LINEAR growth only -- with
        ``beyond_eds=True`` the beyond-EdS kernel constants still come from
        ``kernel_constants_jax``, which takes the MG parameters directly.

        SIGN CAVEAT.  The internal ODE returns a SIGNED f = D'/D, which goes
        negative wherever the modification drives a decaying mode (e.g. mu < 0 deep
        enough in a bin).  A caller deriving f(k) from a spectrum ratio --
        sqrt(P_theta/P_delta), the usual route -- can only supply |f|, because
        P_theta ~ f^2 discards the sign.  In such regions the two sources genuinely
        disagree and neither is complete: the ODE has the sign, the spectrum ratio
        does not.  Measured for the binned model at mu1 = -2, the ODE reaches
        f = -0.47 (z=0.8) / -1.0 (z=0.3) between k_TGR and k_c where the spectrum
        ratio gives +0.47 / +1.0.  If your model can produce decaying modes and the
        RSD sign matters, supply a signed f(k) or leave ``fk=None``.
    f0 : float or None, default=None
        Scale-independent growth rate.  ``None`` estimates it by averaging ``fk``
        over ``k <= f0_kmax``.  Pass it explicitly alongside ``fk`` when the caller
        has its own definition (e.g. sqrt(P_theta/P_delta) at a fixed small k), so
        the two cannot drift apart.
    ingredients_fn : callable or None, default=None
        Same hook :func:`Kfuncs_to_tables` (the eager builder) accepts: a
        zero-extra-argument closure (e.g. ``fkptjax.ab_ingredients.
        IngredientsProvider.bind(**params)``) taking ``(k_ext_q, q_loop_Q,
        kminus_Q, x_r, k_r, p_r, f0)`` and returning the SAME 16-tuple
        :func:`fkptjax.MG_kernels.I1udd1_and_P13_grid` does -- substitutes a
        trained Chebyshev-predict + JAX-native interpolation for the live
        diffrax solve below when ``fkpt_approximation=False``. ``None``
        (default): unchanged behaviour (live ``I1udd1_and_P13_grid``). Unlike
        the eager builder, this path is ALREADY fully ``jnp``/traceable
        either way -- the point of wiring this in here is speed (avoiding a
        live per-quadrature-point ODE solve on every call), not jittability,
        which this function already has.
    return_raw_kfuncs : bool, default=False
        If ``True``, additionally return the raw ``fkptjax.types.KFunctionsOut``
        namedtuple from ``JaxCalculator.evaluate_jax`` (appended after
        ``kernel_constants`` when ``return_kernel_constants=True``, otherwise
        as the third element).  Useful for inspecting an individual, not yet
        combined, kernel piece -- e.g. ``kfuncs.P22uu`` (the pure G2-squared
        one-loop term, before it is added to ``P13uu`` in the assembled table)
        -- rather than only the fully combined/RSD-projected tables/multipoles.
    fkpt_approximation : bool, default=True
        ``True`` (default): beyond-EdS kernels use the fkPT large-scale-limit
        approximation (``kernel_constants_jax``), exactly as before --
        identical output to prior releases.  ``False``: the P22-loop
        (F2evQ/G2evQ -> P22dd/P22du/P22uu and the bias terms built from them)
        and the P13-loop (Gamma2evR/Gamma2fevR/C3Gamma3/C3Gamma3f ->
        P13dd/P13du/P13uu) are computed at the genuine, non-squeezed
        (external k, loop q, angle) triples of the Q/R quadrature grids via
        :mod:`fkptjax.MG_kernels`, instead of reusing large-scale-limit
        scalars for every mode. This was cross-checked against an independent
        from-scratch scipy ODE solve and a from-scratch Wolfram evaluation of
        the collaborator reference sMGPT's own equations (agreement to
        ~1e-6-1e-7 pointwise, and against a real sMGPT reference table
        end-to-end), and matches the ``True`` path exactly in the GR and
        scale-independent limits, as required. That process also found and
        fixed three real, pre-existing bugs in calculate_jax.py's
        Gamma2evR/Gamma2fevR/C3Gamma3/C3Gamma3f/G3K formulas along the way.
        The I1udd1-family bias/RSD kernels (I1udd1A/I2uud1A/I2uud2A/I3uuu2A/
        I3uuu3A) now (2026-09-01, THIRD attempt) use a genuine three-ordering
        (Q+R+N) sum here, mirroring sMGPT's own computeOneKBothExact/
        AKernelsT architecture exactly: all of Q ("tA"), R ("a") and N ("A")
        evaluated at the SAME (r,x) point on the SAME clipped Q-loop grid,
        combined before a single q-integral -- see calculate_jax.py's
        "have_N" branch. Two earlier attempts evaluated the R/N pieces on
        the separate, unclipped R-loop/P13-domain instead and failed (see
        ``.claude/plans/quizzical-mapping-catmull.md`` for that evidence
        trail); this one fixes the actual domain mismatch that caused those
        failures. Pending the same validation as the rest of this flag
        (GR-limit exact match, Wolfram cross-check) before being trusted for
        production use.  Requires a model
        supported by ``fkptjax.mg_jax.mu`` -- now every model this function
        accepts (see the class-level list above); raises
        ``NotImplementedError`` for anything ``_resolve_mg_kind`` doesn't
        recognize. Has no effect when ``beyond_eds=False``. More expensive
        than ``True`` -- see :mod:`fkptjax.MG_kernels`'s performance note.
    """
    import folps as folpsv2
    from folps.tools_jax import extrapolate_pklin, simpson, interp
    from fkptjax.calculate_jax import JaxCalculator
    from fkptjax.util import setup_kfunctions
    from fkptjax import mg_jax as _bj
    from fkptjax.jax_ode import DP_jax, kernel_constants_jax

    k = jnp.asarray(k); pk = jnp.asarray(pk); pk_now = jnp.asarray(pk_now)
    k_ext, pk_ext = extrapolate_pklin(k, pk)
    _, pk_now_ext = extrapolate_pklin(k, pk_now)

    # MG constants (Om and the MG parameters, INCLUDING z, may be traced --
    # xstop is kept as a jnp value, not concretized via float(), specifically
    # so z can be jax.jit/vmap/grad'd through this function; only kind is
    # STATIC (it selects which model's mu is traced at all -- see mg_jax's
    # module docstring for why that is a Python branch and not a jnp.where).
    kind = _resolve_mg_kind(model, mg_variant)
    P = _bj.pack_constants_jnp(
        om=Om, ol=1.0 - Om, kind=kind,
        mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4,
        z_div=z_div, z_TGR=z_TGR, z_tw=z_tw, scale_bins=scale_bins,
        k_TGR=k_TGR, k_c=k_c, k_S=k_S, k_tw=k_tw,
        mu_kinf=mu_kinf_BZmass, lambda_a=lambda_a_BZmass, lambda_dS=lambda_dS_BZmass,
        w0=w0, wa=wa,
        fR0_HS=fR0_HS, beta2=beta2, n_HS=n_HS,
        r_c=r_c, mu0=mu0,
        beta_1=beta_1, lambda_1=lambda_1, exp_s=exp_s,
        eftde_eta_grid=eftde_eta_grid, eftde_h1_grid=eftde_h1_grid,
        eftde_h3_grid=eftde_h3_grid, eftde_h5_grid=eftde_h5_grid,
        eftde_scale_dependent=eftde_scale_dependent, eftde_k2_pivot=eftde_k2_pivot,
        gamma_0=gamma_0, gamma_a=gamma_a, t_k=t_k, d_s=d_s)
    xstop = jnp.log(1.0 / (1.0 + jnp.asarray(z, dtype=jnp.float64)))

    # growth: either integrate the binned-mu ODE, or take f(k) from the caller.
    #
    # Only the RATIO D'/D is consumed on this route -- the amplitude D is discarded
    # immediately below -- so an external f(k) fully determines the linear-growth
    # sector here. (This is specific to the PHENOM/binning route: the rescaling
    # branch uses D itself, via (D_MG/D_GR)^2, and is untouched by this option.)
    #
    # Supplying f(k) matters when pk/pk_now come from an emulator: the ODE and the
    # emulated P_theta are then two independent statements about the same
    # cosmology's growth, and nothing forces them to agree, so the loop kernels and
    # the linear spectrum can describe subtly different models. It also skips a
    # diffrax solve over the whole extrapolated grid.
    #
    # Mirrors how the rescaling branch already accepts caller-supplied
    # ``kernel_constants`` in place of its own ODE.
    if fk is None:
        Y = DP_jax(k_ext, P, float(xnow), xstop)
        D_ext, Dp_ext = Y[0], Y[1]
        fk_ext = Dp_ext / D_ext
    else:
        # fk arrives on the INPUT grid `k`; lift it to the extrapolated grid.
        # jnp.interp (not extrapolate_pklin) on purpose: that helper fits a power
        # law, which is right for P(k) and wrong for f(k). f(k) tends to the
        # scale-independent f0 as k -> 0 and stays bounded and smooth at high k, so
        # clamping to the edge values -- exactly jnp.interp's out-of-range
        # behaviour -- is the physically correct extension.
        fk_in_arr = jnp.asarray(fk, dtype=jnp.float64)
        if fk_in_arr.shape != jnp.asarray(k).shape:
            raise ValueError(
                f"fk must have the same shape as k; got {fk_in_arr.shape} vs "
                f"{jnp.asarray(k).shape}")
        fk_ext = jnp.interp(k_ext, jnp.asarray(k), fk_in_arr)

    if kmin is None:
        kmin = float(jnp.minimum(1e-3, jnp.min(k)))
    if kmax is None:
        kmax = float(jnp.maximum(0.5, jnp.max(k)))
    if f0_kmax is None:
        f0_kmax = float(kmin)

    if f0 is not None:
        # Take the caller's f0 verbatim. The estimator below averages f(k) over
        # k <= f0_kmax, which is a DIFFERENT definition from, say, evaluating
        # sqrt(P_theta/P_delta) at a fixed small k -- close, but not equal, and the
        # difference would drift silently between caller and library.
        f0 = jnp.asarray(f0, dtype=jnp.float64)
    else:
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

    # Full (non-fkPT-approximated) beyond-EdS kernels: solved at the genuine
    # (external k, loop q, angle) triples of the Q/R grids instead of reusing
    # the large-scale-limit scalars A/ApOverf0/CFD3/CFD3p above. See
    # fkptjax.MG_kernels and the fkpt_approximation docstring entry.
    #
    # Three momentum "orderings" of the same physical triple {k_ext, q,
    # kminus=|k_ext-q|} are needed for the I1udd1-family (mirrors the
    # collaborator reference sMGPT's Q/R/N convention, 2_P22type.wl's
    # computeOneKBothExact/AKernelsT), ALL evaluated on the SAME clipped
    # Q-loop grid:
    #   Q ordering: output leg = k_ext, input legs = (q, kminus)
    #   R ordering: output leg = kminus, input legs = (k_ext, q)
    #   N ordering: output leg = q,      input legs = (k_ext, kminus)
    # plus the FUSED (caligraphic-A - caligraphic-B x^2) combination on the
    # SEPARATE R grid (output leg = kminus, angle sign flipped -- matches
    # sMGPT's AminusBx2h[-x,k_ext,q]) for the P13-loop's Gamma2/Gamma3-type
    # pieces only -- that one genuinely needs its own (unclipped) domain,
    # unlike the I1udd1-family above.
    A_Q = B_Q = ApOverf0_Q = BpOverf0_Q = None
    A_N = B_N = ApOverf0_N = BpOverf0_N = None
    A_RQ = B_RQ = ApOverf0_RQ = BpOverf0_RQ = None
    A_fused_R = ApOverf0_fused_R = None
    CFD3_R = CFD3p_R = None
    if bool(beyond_eds) and not bool(fkpt_approximation):
        from fkptjax import MG_kernels as _mgk

        k_ext_q = calculator.logk_grid_jax

        # Q ordering (P22-loop F2evQ/G2evQ, and the Q-ordering "tA" bias/RSD kernels)
        r_q = calculator.r_jax
        x_q = calculator.x_jax
        y_q = jnp.sqrt(1.0 + r_q * r_q - 2.0 * r_q * x_q)
        q_loop_Q = r_q * k_ext_q
        kminus_Q = k_ext_q * y_q

        # The I1udd1-family's three orderings (Q/R/N) -- matching sMGPT's own
        # computeOneKBothExact/AKernelsT architecture -- all evaluate the
        # SAME A_B_grid formula at the SAME (r,x) point on this SAME clipped
        # Q-loop grid, just with the {k_ext, q, kminus} triple permuted:
        #   Q: output leg k_ext,   input legs (q, kminus)
        #   N: output leg q,       input legs (k_ext, kminus)  -- sMGPT's Ah[q,k,kminus]
        #   R: output leg kminus,  input legs (k_ext, q)        -- sMGPT's Ah[kminus,k,q]
        # (An earlier version of this fix evaluated R/N on the separate,
        # unclipped R-loop grid instead, which only reproduces the correct
        # physics in the squeezed (scalar A=B) limit -- see
        # .claude/plans/quizzical-mapping-catmull.md for that evidence trail.)
        #
        # P13-loop's fused D2/D3 kernel (Gamma2/Gamma3-type terms, on the
        # SEPARATE R-loop grid) is independently cross-checked against a
        # from-scratch Wolfram evaluation of sMGPT's own equations (see
        # .claude/plans/quizzical-mapping-catmull.md; that process also
        # found and fixed three real, pre-existing bugs in
        # calculate_jax.py's Gamma2evR/Gamma2fevR/C3Gamma3/C3Gamma3f/G3K
        # formulas). P13dd/du/uu match the sMGPT reference cleanly.
        #
        # Rather than 4 separate top-level ODE dispatches (3x A_B_grid for
        # Q/N/R, plus D2_D3_fused_grid for the P13-loop -- each its own XLA
        # dispatch + host-device sync), MG_kernels.I1udd1_and_P13_grid
        # solves all of them in ONE combined batch: verified (2026-09-02) to
        # reproduce the separate-calls result to ~1e-10, for both a
        # scale-independent and a scale-dependent model, at ~1.8-2x the
        # speed on GPU.
        r_r = calculator.r_r_jax
        x_r = calculator.x_r_jax
        q_loop_R = r_r * k_ext_q

        if ingredients_fn is not None:
            # Trained ab_ingredients emulator (predict + interpolate) instead of a
            # live per-quadrature-point ODE solve -- see this session's validation
            # notes (mu_OmDE: <0.05% max relative error against 'adaptive'; BZ_Mass
            # needed a resolution boost first, see build_ingredients_provider_for_run).
            (A_Q, B_Q, ApOverf0_Q, BpOverf0_Q,
             A_N, B_N, ApOverf0_N, BpOverf0_N,
             A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ,
             A_fused_R, ApOverf0_fused_R, CFD3_R, CFD3p_R) = ingredients_fn(
                k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R,
                jnp.asarray(f0, dtype=jnp.float64))
        else:
            # solver='rk4': MG_kernels.py's own default is 'adaptive' because a
            # fixed-step solver gives wrong (sometimes wrong-sign) results near
            # the collinear (q~k_ext) singularity -- see its module docstring.
            # 'rk4' is used HERE specifically because this Q/R-loop domain is
            # already clipped well away from that singularity: verified
            # (2026-09-02) that y=kminus/k_ext's minimum stays ~0.52 (Q-grid) /
            # ~0.23 (R-grid), essentially UNCHANGED across Nk_kernel=16..120 (so
            # this holds regardless of the caller's k-bin count, not just at one
            # tested resolution) -- both far from the y->0 danger zone. At
            # n_steps=64, this matches the adaptive solver's own answer to
            # ~1e-6 for both a scale-independent (HDKI/mu_OmDE) and a
            # scale-dependent (Hu-Sawicki f(R)) model, while being ~3-20x
            # faster (adaptive's own step count/cost varies noticeably by
            # model). If this grid's construction ever changes (different
            # kmin/kmax/NQ/NR/nquadSteps) such that y could approach 0, this
            # choice needs re-validating -- see check_rk4_speed.py.
            (A_Q, B_Q, ApOverf0_Q, BpOverf0_Q,
             A_N, B_N, ApOverf0_N, BpOverf0_N,
             A_RQ, B_RQ, ApOverf0_RQ, BpOverf0_RQ,
             A_fused_R, ApOverf0_fused_R, CFD3_R, CFD3p_R) = _mgk.I1udd1_and_P13_grid(
                k_ext_q, q_loop_Q, kminus_Q, x_r, k_ext_q, q_loop_R, P, float(xnow), xstop, f0,
                solver='rk4', n_steps=64)

    kfuncs = calculator.evaluate_jax(
        Pk_in=pk_ext, Pk_nw_in=pk_now_ext, fk_in=fk_ext,
        A=A, ApOverf0=ApOverf0, CFD3=CFD3, CFD3p=CFD3p, sigma2v=0.0, f0=f0,
        A_Q=A_Q, B_Q=B_Q, ApOverf0_Q=ApOverf0_Q, BpOverf0_Q=BpOverf0_Q,
        A_fused_R=A_fused_R, ApOverf0_fused_R=ApOverf0_fused_R,
        CFD3_R=CFD3_R, CFD3p_R=CFD3p_R,
        A_N=A_N, B_N=B_N, ApOverf0_N=ApOverf0_N, BpOverf0_N=BpOverf0_N,
        A_RQ=A_RQ, B_RQ=B_RQ, ApOverf0_RQ=ApOverf0_RQ, BpOverf0_RQ=BpOverf0_RQ)

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

    if return_kernel_constants and return_raw_kfuncs:
        return table_w, table_nw, (A, ApOverf0 * f0, CFD3, CFD3p), kfuncs
    if return_kernel_constants:
        return table_w, table_nw, (A, ApOverf0 * f0, CFD3, CFD3p)
    if return_raw_kfuncs:
        return table_w, table_nw, kfuncs
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
