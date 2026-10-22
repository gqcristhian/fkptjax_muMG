"""
ide_bispectrum.py
-----------------
Tree-level redshift-space bispectrum multipoles with the IDE kernels.

The tree-level bispectrum needs only the second-order kernels, which for an IDE
model are exactly scale independent (see :mod:`fkptjax.ide`; Kamble & Ishak IDE notes, in prep., Sec. 4.2):

    F2 = Gamma1 alpha_s + [3/14 (L + WL) - Gamma1^2/2] (1 - mu^2)
    G2 = Gamma1 alpha_s + r_G (1 - mu^2),   r_G = 3/7 ((f/fQ) L + L'/(2 fQ)) - Gamma1

with alpha_s = 1 + mu/2 (k1/k2 + k2/k1) and theta = -div(v)/(a H fQ) (G1 = 1), so that the
velocity growth rate fQ = f + g plays the role of f everywhere in the redshift-space mapping
(notes Sec. 7, eqs. 57-58):

    Z1(k)      = b1 + fQ mu^2
    Z2(k1, k2) = b2/2 + bs/2 (mu12^2 - 1/3) + b1 F2 + fQ mu^2 G2
                 + fQ mu k/2 [mu1/k1 Z1(k2) + mu2/k2 Z1(k1)]

(k = k1 + k2, mu = its cosine with the line of sight). The BAO damping of each leg uses
the IDE displacement, Sigma^2_tot(mu) from :func:`fkptjax.ide.ir_sigma2_tot` (Gamma1, fPsi),
as in the power spectrum (:func:`fkptjax.rsd.tables_to_pkmu` with ``ide=``).

Everything else -- the AP transform, the EFT counterterms of Z1, the shot noise, the FoG
damping, the Sigma^2 / delta Sigma^2 / sigma_v^2 integrals and the Sugiyama / Scoccimarro
multipole quadratures -- is FOLPS's ``BispectrumCalculator_fk``, unchanged: only its
per-configuration integrand ``bispectrum()`` is replaced. With g = 0 (Gamma1 = 1, WL = 0,
fPsi = fQ = f) the result equals FOLPS's with calA = L, calAp = L' and f(k) = f.

The third-order Sigma_3 sector (S, WS) of the IDE growth system is not needed here;
it would enter a one-loop bispectrum only.
"""

import importlib
import os

import jax.numpy as jnp

from fkptjax.ide import F2_G2, ir_sigma2_tot

SUGIYAMA_MULTIPOLES = ('B000', 'B110', 'B220', 'B202', 'B022', 'B112', 'B222')
SCOCCIMARRO_MULTIPOLES = ('B0', 'B2', 'B4')

_CALCULATOR_CLS = None


def ide_Z2(ki, kj, xij, mui, muj, b1, b2, bs, ide):
    """Second-order redshift-space kernel Z2(ki, kj) with the IDE F2 / G2.

    ``xij`` is the cosine between ki and kj, ``mui``/``muj`` their cosines with the line
    of sight. ``ide`` is an :class:`fkptjax.ide.IDEKernelCoeffs` at the output redshift.
    """
    fQ = ide.fQ
    F2, G2 = F2_G2(xij, ki / kj + kj / ki, ide)
    km = ki * mui + kj * muj
    mu2 = km**2 / (ki**2 + kj**2 + 2. * ki * kj * xij)
    return (b2 / 2. + bs / 2. * (xij**2 - 1. / 3.)
            + b1 * F2 + fQ * mu2 * G2
            + km / 2. * fQ * (mui / ki * (b1 + fQ * muj**2) + muj / kj * (b1 + fQ * mui**2)))


def _folps():
    # FOLPS picks its backend once, at first import: make it JAX (fkptjax is jitted).
    os.environ.setdefault('FOLPS_BACKEND', 'jax')
    return importlib.import_module('folps.folps')


def get_calculator_cls():
    """:class:`IDEBispectrumCalculator`, built on first use (FOLPS is imported lazily)."""
    global _CALCULATOR_CLS
    if _CALCULATOR_CLS is not None:
        return _CALCULATOR_CLS
    ff = _folps()

    class IDEBispectrumCalculator(ff.BispectrumCalculator_fk):
        """FOLPS ``BispectrumCalculator_fk`` with the IDE second-order kernels and BAO damping.

        Parameters
        ----------
        ide : IDEKernelCoeffs
            Kernel coefficients at the output redshift (scalars). ``ide.fQ`` is the growth
            rate of every term; the ``f`` FOLPS passes along and the fourth row of
            ``k_pkl_pklnw_fk`` (f(k)) are not used.
        model : str
            As FOLPS (only labels the FoG printout).
        damping_ide : IDEKernelCoeffs, optional
            Coefficients for the BAO damping of each leg (Gamma1, fPsi); default ``ide``. With
            ``ide = eds_kernel_coefficients(c)`` and ``damping_ide = c``: EdS kernels, IDE damping.
        """

        _tables_cache = {}  # own Gauss-Legendre cache, not FOLPS's

        def __init__(self, ide, model='EFT', damping_ide=None):
            super().__init__(model=model)
            self.ide = ide
            self.damping_ide = ide if damping_ide is None else damping_ide

        def bispectrum(self, k1, k2, x12, mu1, phi, f, sigma2v, Sigma2, deltaSigma2,
                       bpars, qpar, qperp, k_pkl_pklnw_fk, damping='lor', interpolation_method='cubic',
                       mg_kernel_fn=None):
            """FOLPS ``BispectrumCalculator_fk.bispectrum`` line for line, except: Z2 is
            :func:`ide_Z2`, f is ``ide.fQ`` (Z1, Z2, shot noise, FoG) and the BAO damping of
            each leg is ``ir_sigma2_tot`` (Gamma1, fPsi). ``mg_kernel_fn`` (FOLPS's MG
            kernel hook, passed by newer FOLPS versions) must be None: the IDE kernels replace it."""
            if mg_kernel_fn is not None:
                raise ValueError("IDEBispectrumCalculator uses the IDE kernels; mg_kernel_fn must be None.")
            ide = self.ide
            fQ = ide.fQ
            # As FOLPS's split_bispectrum_pars: (..., Bshot, Pshot, Ashot, X_FoG), or the legacy 8-parameter vector
            # whose constant term is tied, Ashot = Pshot**2.
            if len(bpars) > 8:
                b1, b2, bs, c1, c2, Bshot, Pshot, Ashot, X_FoG_b = bpars[:9]
            else:
                b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG_b = bpars[:8]
                Ashot = Pshot**2

            cosphi = jnp.cos(phi)
            k1AP, k2AP, k3AP, x12AP, x23AP, x31AP, mu1AP, mu2AP, mu3AP, _ = self.APtransforms(
                k1, k2, x12, mu1, cosphi, qpar, qperp)
            k_, pkl_, pklnw_ = k_pkl_pklnw_fk[0], k_pkl_pklnw_fk[1], k_pkl_pklnw_fk[2]

            def leg(kAP, muAP):
                pk = self.interpolation_b(kAP, k_, pkl_, method=interpolation_method)
                pknw = self.interpolation_b(kAP, k_, pklnw_, method=interpolation_method)
                pkIR = pknw + (pk - pknw) * jnp.exp(-ir_sigma2_tot(muAP, Sigma2, deltaSigma2, self.damping_ide) * kAP**2)
                Z1eft = b1 + fQ * muAP**2 - (c1 * muAP**2 + c2 * muAP**4) * kAP**2
                return pkIR, Z1eft

            pkIR1, Z1eft1 = leg(k1AP, mu1AP)
            pkIR2, Z1eft2 = leg(k2AP, mu2AP)
            pkIR3, Z1eft3 = leg(k3AP, mu3AP)

            B12 = 2 * ide_Z2(k1AP, k2AP, x12AP, mu1AP, mu2AP, b1, b2, bs, ide) * Z1eft1 * pkIR1 * Z1eft2 * pkIR2
            B23 = 2 * ide_Z2(k2AP, k3AP, x23AP, mu2AP, mu3AP, b1, b2, bs, ide) * Z1eft2 * pkIR2 * Z1eft3 * pkIR3
            B31 = 2 * ide_Z2(k3AP, k1AP, x31AP, mu3AP, mu1AP, b1, b2, bs, ide) * Z1eft3 * pkIR3 * Z1eft1 * pkIR1

            W = ff.fog_damping((k1AP * mu1AP, X_FoG_b), (k2AP * mu2AP, X_FoG_b), (k3AP * mu3AP, X_FoG_b),
                               f=fQ, sigma2v=sigma2v, damping=damping)
            shot = ((b1 * Bshot + Pshot * fQ * mu1AP**2) * Z1eft1 * pkIR1
                    + (b1 * Bshot + Pshot * fQ * mu2AP**2) * Z1eft2 * pkIR2
                    + (b1 * Bshot + Pshot * fQ * mu3AP**2) * Z1eft3 * pkIR3
                    + Ashot)
            toret = (W * (B12 + B23 + B31) + shot) / (qpar * qperp**2)**2

            # desilike's redshift-smearing hook (``_patch_folps_bispectrum`` wraps FOLPS's own
            # classes only): D(k1 mu1) D(k2 mu2) D(k3 mu3) with the pre-AP angles.
            damping_fn = getattr(self, '_redshift_smearing', None)
            if damping_fn is not None:
                k3 = jnp.sqrt(k1**2 + k2**2 + 2. * k1 * k2 * x12)
                mu2 = jnp.sqrt(1. - mu1**2) * jnp.sqrt(1. - x12**2) * cosphi + mu1 * x12
                mu3 = -(k1 * mu1 + k2 * mu2) / k3
                toret = toret * damping_fn(k1 * mu1) * damping_fn(k2 * mu2) * damping_fn(k3 * mu3)
            return toret

    _CALCULATOR_CLS = IDEBispectrumCalculator
    return _CALCULATOR_CLS


def ide_bispectrum_poles(ide, bpars, k1k2, k_pkl_pklnw, qpar=1., qper=1., multipoles=('B000', 'B202'),
                         basis='sugiyama', precision=None, damping='lor', interpolation_method='linear',
                         bias_scheme='folps', renormalize=True, model='EFT', redshift_smearing=None):
    """Tree-level bispectrum multipoles with the IDE kernels.

    Parameters
    ----------
    ide : IDEKernelCoeffs
        Coefficients at the output redshift (scalars), e.g. ``ide_kernel_coeffs(z, bg)``;
        the same object the power-spectrum tables were built with.
    bpars : array, (9,) or (8,)
        FOLPS bispectrum nuisance vector (b1, b2, bs, c1, c2, Bshot, Pshot, Ashot, X_FoG), in
        ``bias_scheme``'s convention, or the legacy (b1, b2, bs, c1, c2, Bshot, Pshot, X_FoG), with Ashot = Pshot**2.
    k1k2 : array
        (N, 2) pairs (k1, k2) for ``basis='sugiyama'``; (N, 3) triplets (k1, k2, k3) for
        ``basis='scoccimarro'``.
    k_pkl_pklnw : sequence
        (k, P_lin, P_lin no-wiggle) of the cdm+baryon field at the output redshift (further
        rows, e.g. f(k) or FOLPS's calA/calAp, are ignored).
    qpar, qper : float
        AP parameters.
    multipoles : sequence of str
        Sugiyama (``'B000'``, ``'B202'``, ...) or Scoccimarro (``'B0'``, ``'B2'``, ``'B4'``).
    precision : sequence of int, optional
        Gauss-Legendre points, (Nphi, Nx, Nmu) for Sugiyama, (Nphi, Nmu) for Scoccimarro;
        FOLPS's defaults if None.
    damping, interpolation_method, bias_scheme, renormalize, model
        As FOLPS's ``Sugiyama_Bell`` / ``Scoccimarro_Bell`` (``renormalize``: Sugiyama only).
    redshift_smearing : callable, optional
        Single-field D(k mu) multiplying each leg (desilike's ``RedshiftSmearing`` kernel).

    Returns
    -------
    Array of shape (len(multipoles), N).
    """
    calc = get_calculator_cls()(ide, model=model)
    if redshift_smearing is not None:
        calc._redshift_smearing = redshift_smearing
    k = jnp.asarray(k_pkl_pklnw[0])
    table = [k, jnp.asarray(k_pkl_pklnw[1]), jnp.asarray(k_pkl_pklnw[2]), jnp.ones_like(k) * ide.fQ]
    multipoles = list(multipoles)
    kw = {} if precision is None else {'precision': list(precision)}
    if basis == 'sugiyama':
        bad = [ell for ell in multipoles if ell not in SUGIYAMA_MULTIPOLES]
        if bad:
            raise ValueError(f"Sugiyama multipoles must be in {SUGIYAMA_MULTIPOLES}, got {bad}")
        result = calc.Sugiyama_Bell(ide.fQ, bpars, table, k1k2pairs=k1k2, qpar=qpar, qper=qper,
                                    damping=damping, multipoles=multipoles, renormalize=renormalize,
                                    interpolation_method=interpolation_method, bias_scheme=bias_scheme, **kw)
    elif basis == 'scoccimarro':
        bad = [ell for ell in multipoles if ell not in SCOCCIMARRO_MULTIPOLES]
        if bad:
            raise ValueError(f"Scoccimarro multipoles must be in {SCOCCIMARRO_MULTIPOLES}, got {bad}")
        raw = calc.Scoccimarro_Bell(k1k2, ide.fQ, bpars, qpar, qper, table, damping=damping,
                                    interpolation_method=interpolation_method, multipoles=multipoles,
                                    bias_scheme=bias_scheme, **kw)
        result = [raw[SCOCCIMARRO_MULTIPOLES.index(ell)] for ell in multipoles]
    else:
        raise ValueError(f"basis must be 'sugiyama' or 'scoccimarro', got {basis!r}")
    return jnp.asarray(result)
