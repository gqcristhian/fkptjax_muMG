"""The HDKI/EFT_DE quasi-static source (2026-10-22): the finite coefficients (h1, D0, N3, N5) in the numpy and jax
paths, their equivalence with the legacy h1/h3/h5 inputs, the analytic no-slip / GR branch, the refusal of a missing
source (issue_eftde_full_kernels.md: it used to fall back to GR silently), and the eager full-kernel branch handing
the sampled source to mg_jax."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from fkptjax import mg_jax
from fkptjax.ode import ModelDerivatives, eftde_source_from_h

ETA = np.linspace(-4.2, 0.0, 64)
A = np.exp(ETA)
H1, H3, H5 = 1.0 + 0.1 * A**2, 50.0 * A, 80.0 * A       # the issue's test functions (h3 > 0: no pole)
OM = 0.3212


def _interp(values):
    return lambda eta: np.interp(eta, ETA, values)


LEGACY = eftde_source_from_h(_interp(H1), _interp(H3), _interp(H5))         # the h-function form, converted
FINITE = dict(eftde_h1_interp=_interp(H1), eftde_d0_interp=lambda eta: 1.0,
              eftde_n3_interp=_interp(H3), eftde_n5_interp=_interp(H5))
KS = np.geomspace(1e-4, 0.5, 25)
ETAS = np.linspace(-4.0, 0.0, 17)


def _derivs(**source):
    return ModelDerivatives(om=OM, ol=1 - OM, model='HDKI', mg_variant='EFT_DE', **source)


def test_finite_coefficients_reproduce_the_h_form():
    legacy, finite = _derivs(**LEGACY), _derivs(**FINITE)
    assert legacy.is_EFTDE_scale_dependent and finite.is_EFTDE_scale_dependent
    for eta in ETAS:
        assert np.allclose(finite.mu_mg(eta, KS), legacy.mu_mg(eta, KS), rtol=1e-14, atol=0)


def test_finite_form_is_regular_through_the_crossing():
    """D0 through zero (where h3 = N3 / D0, h5 = N5 / D0 have their pole): mu stays finite and equals
    h1 N5 / N3 at the crossing for every k; below the crossing the pole of the expression sits at
    k^2 = -D0 / N3 and is left non-finite rather than regularised."""
    d0 = lambda eta: eta + 2.0                      # zero at eta = -2
    d = _derivs(eftde_h1_interp=lambda eta: 1.1, eftde_d0_interp=d0,
                eftde_n3_interp=lambda eta: 0.5, eftde_n5_interp=lambda eta: 0.8)
    d.is_EFTDE_scale_dependent = True
    assert np.allclose(d.mu_mg(-2.0, KS), 1.1 * 0.8 / 0.5)
    assert np.all(np.isfinite(d.mu_mg(-1.0, KS)))
    k_pole = np.sqrt(-d0(-2.5) / 0.5)
    mu = d.mu_mg(-2.5, np.array([k_pole]))
    assert not np.isfinite(mu).all()


def test_noslip_and_gr_take_the_analytic_branch():
    """N5 == N3 (exact no-slip; exact GR with D0 = N3 = N5 = 0): mu = h1 bit for bit, no ratio formed."""
    n3 = _interp(H3)
    d = _derivs(eftde_h1_interp=_interp(H1), eftde_d0_interp=lambda eta: 0.0, eftde_n3_interp=n3, eftde_n5_interp=n3)
    assert not d.is_EFTDE_scale_dependent
    d.is_EFTDE_scale_dependent = True
    for eta in ETAS:
        assert np.all(d.mu_mg(eta, KS) == np.interp(eta, ETA, H1))
    gr = _derivs(eftde_h1_interp=lambda eta: 1.0, eftde_d0_interp=lambda eta: 0.0,
                 eftde_n3_interp=lambda eta: 0.0, eftde_n5_interp=lambda eta: 0.0)
    assert np.all(gr.mu_mg(-1.0, KS) == 1.0)


def test_missing_source_is_refused_not_gr():
    with pytest.raises(ValueError):
        _derivs()
    with pytest.raises(ValueError):
        _derivs(eftde_h1_interp=_interp(H1), eftde_n3_interp=_interp(H3))             # incomplete
    with pytest.raises(ValueError):
        mg_jax.pack_constants_jnp(om=OM, ol=1 - OM, kind=mg_jax.EFT_DE)
    with pytest.raises(ValueError):
        mg_jax.pack_constants_jnp(om=OM, ol=1 - OM, kind=mg_jax.EFT_DE, eftde_eta_grid=ETA, eftde_h1_grid=H1)


def test_jax_source_matches_numpy_on_the_sampled_grids():
    d = _derivs(**LEGACY)
    grids = d.eftde_grids(ETA)
    assert set(grids) >= {'eftde_eta_grid', 'eftde_h1_grid', 'eftde_d0_grid', 'eftde_n3_grid', 'eftde_n5_grid', 'eftde_scale_dependent'}
    P = mg_jax.pack_constants_jnp(om=OM, ol=1 - OM, kind=mg_jax.EFT_DE, **grids)
    mu_jax = jax.jit(lambda eta, k: mg_jax.mu(eta, k, P))
    for eta in ETAS:
        # the callables are linear interpolants on ETA itself, so jnp.interp on the same nodes reproduces them
        assert np.allclose(np.asarray(mu_jax(eta, jnp.asarray(KS))), d.mu_mg(eta, KS), rtol=1e-12, atol=0)
    # the h-form sampled as D0 = 1 grids is the same source
    P_legacy = mg_jax.pack_constants_jnp(om=OM, ol=1 - OM, kind=mg_jax.EFT_DE, eftde_eta_grid=ETA, eftde_h1_grid=H1,
                                         eftde_d0_grid=np.ones_like(ETA), eftde_n3_grid=H3, eftde_n5_grid=H5, eftde_scale_dependent=True)
    for eta in ETAS:   # both eager: bit for bit (the jitted version differs from eager by XLA's fusion, 2e-16)
        assert np.array_equal(np.asarray(mg_jax.mu(eta, jnp.asarray(KS), P_legacy)), np.asarray(mg_jax.mu(eta, jnp.asarray(KS), P)))
    # the no-slip identity survives interpolation: identical grids give mu == h1
    P_ns = mg_jax.pack_constants_jnp(om=OM, ol=1 - OM, kind=mg_jax.EFT_DE, eftde_eta_grid=ETA, eftde_h1_grid=H1,
                                     eftde_d0_grid=np.zeros_like(ETA), eftde_n3_grid=H3, eftde_n5_grid=H3, eftde_scale_dependent=True)
    for eta in ETAS:
        h1 = jnp.interp(eta, jnp.asarray(ETA), jnp.asarray(H1))
        assert np.all(np.asarray(mg_jax.mu(eta, jnp.asarray(KS), P_ns)) == float(h1))


def test_eager_full_kernel_branch_hands_the_sampled_source_to_mg_jax(monkeypatch):
    """issue_eftde_full_kernels.md: the eager builder's full-kernel branch called pack_constants_jnp without any
    eftde_* input, so mg_jax used GR. Now it samples the source (with the numpy path's scale-dependence verdict);
    intercept the call to check what it receives, without running the kernels."""
    pytest.importorskip('diffrax')
    from fkptjax.kfuncs_to_tables import Kfuncs_to_tables

    class Captured(Exception):
        pass

    received = {}

    def fake_pack(*args, **kwargs):
        received.update(kwargs)
        raise Captured()

    monkeypatch.setattr(mg_jax, 'pack_constants_jnp', fake_pack)
    k = np.geomspace(1e-4, 10.0, 200)
    pk = 1e4 * (k / 0.02) / (1.0 + (k / 0.02)**2)**1.6
    with pytest.raises(Captured):
        Kfuncs_to_tables(k=k, pk=pk, pk_now=pk, z=0.5, Om=OM, kmin=1e-3, kmax=0.5, Nk_kernel=8, nquadSteps=20,
                         NQ=3, NR=3, beyond_eds=True, fkpt_approximation=False, model='HDKI', mg_variant='EFT_DE',
                         fk=np.full_like(k, 0.5), f0=0.5, **LEGACY)
    assert received['kind'] == mg_jax.EFT_DE
    for name in ('eftde_eta_grid', 'eftde_h1_grid', 'eftde_d0_grid', 'eftde_n3_grid', 'eftde_n5_grid'):
        assert received[name] is not None and np.all(np.isfinite(received[name]))
    assert received['eftde_scale_dependent'] is True
    eta = received['eftde_eta_grid']
    assert np.allclose(received['eftde_h1_grid'], np.interp(eta, ETA, H1))
    assert np.allclose(received['eftde_n3_grid'], np.interp(eta, ETA, H3)) and np.all(received['eftde_d0_grid'] == 1.0)
