"""Independent regression tests for smooth-CPL background growth."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fkptjax import binning_numba, mg_jax
from fkptjax.ode import ModelDerivatives


def _background(eta, om, ol, w0, wa):
    """Independent CPL H(a), Omega_m(a), and 2 + dlnH/dlna."""
    a = np.exp(eta)
    dark_energy = ol * a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(
        3.0 * wa * (a - 1.0)
    )
    matter = om * a**-3.0
    curvature = (1.0 - om - ol) * a**-2.0
    ea2 = matter + curvature + dark_energy
    omega_m = matter / ea2
    w_de = w0 + wa * (1.0 - a)
    friction = 2.0 - (
        1.5 * matter + curvature + 1.5 * (1.0 + w_de) * dark_energy
    ) / ea2
    return ea2, omega_m, friction


@pytest.mark.parametrize("w0, wa", [(-1.0, 0.0), (-0.8, 0.0), (-1.0, 0.3)])
def test_cpl_friction_matches_hubble_derivative(w0, wa):
    om, ol, eta = 0.3, 0.7, -0.4
    derivs = ModelDerivatives(om=om, ol=ol, w0=w0, wa=wa, model="LCDM")

    step = 1.0e-5
    ep = np.sqrt(_background(eta + step, om, ol, w0, wa)[0])
    em = np.sqrt(_background(eta - step, om, ol, w0, wa)[0])
    finite_difference = 2.0 + (np.log(ep) - np.log(em)) / (2.0 * step)

    assert derivs.friction(eta) == pytest.approx(finite_difference, rel=2.0e-10)

    params = mg_jax.pack_constants_jnp(
        om, ol, w0=w0, wa=wa, kind=mg_jax.LCDM
    )
    assert float(mg_jax.friction(eta, params)) == pytest.approx(
        finite_difference, rel=2.0e-10
    )


@pytest.mark.parametrize(
    "w0, wa, expected_f",
    [(-0.8, 0.0, 0.6370309908), (-1.0, 0.3, 0.6639771705)],
)
def test_cpl_growth_rate_against_independent_equation(w0, wa, expected_f):
    om, ol = 0.3, 0.7
    xnow = np.log(0.02)
    xstop = np.log(1.0 / 1.3)
    initial = np.array([np.exp(xnow), np.exp(xnow)])
    derivs = ModelDerivatives(om=om, ol=ol, w0=w0, wa=wa, model="LCDM")

    actual = solve_ivp(
        lambda eta, y: derivs.firstOrder(eta, y, 0.1),
        (xnow, xstop),
        initial,
        rtol=1.0e-11,
        atol=1.0e-13,
    ).y[:, -1]

    def standard_rhs(eta, y):
        _, omega_m, friction = _background(eta, om, ol, w0, wa)
        return np.array([y[1], 1.5 * omega_m * y[0] - friction * y[1]])

    expected = solve_ivp(
        standard_rhs,
        (xnow, xstop),
        initial,
        rtol=1.0e-11,
        atol=1.0e-13,
    ).y[:, -1]

    np.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-12)
    assert actual[1] / actual[0] == pytest.approx(expected_f, rel=2.0e-9)


@pytest.mark.parametrize("w0, wa", [(-0.8, 0.0), (-1.0, 0.3)])
def test_growth_index_mu_uses_the_same_cpl_background(w0, wa):
    """The ISiTGR mu formula should make f=Omega_m**gamma an ODE solution."""
    om, ol = 0.3, 0.7
    gamma = 6.0 / 11.0
    xnow = np.log(0.02)
    xstop = np.log(1.0 / 1.3)
    omega_m_initial = _background(xnow, om, ol, w0, wa)[1]
    initial = np.array([1.0, omega_m_initial**gamma])
    derivs = ModelDerivatives(
        om=om,
        ol=ol,
        w0=w0,
        wa=wa,
        model="PHENOM",
        mg_variant="growth_index",
        gamma_0=gamma,
        gamma_a=0.0,
    )

    solution = solve_ivp(
        lambda eta, y: derivs.firstOrder(eta, y, 0.1),
        (xnow, xstop),
        initial,
        rtol=1.0e-11,
        atol=1.0e-13,
    ).y[:, -1]
    expected_f = _background(xstop, om, ol, w0, wa)[1] ** gamma

    assert solution[1] / solution[0] == pytest.approx(expected_f, rel=5.0e-9)


@pytest.mark.skipif(not binning_numba.HAVE_NUMBA, reason="numba is unavailable")
@pytest.mark.parametrize(
    "background",
    [
        {"om": 0.3, "ol": 0.7, "w0": -0.8, "wa": 0.0},
        {"om": 0.3, "ol": 0.65, "w0": -1.0, "wa": 0.0},
    ],
)
def test_numba_binning_rejects_non_flat_lcdm_background(background):
    with pytest.raises(NotImplementedError, match="flat LambdaCDM"):
        ModelDerivatives(
            **background,
            model="PHENOM",
            mg_variant="binning",
            use_numba=True,
        )


def test_jax_first_order_uses_cpl_friction(monkeypatch):
    eta, om, ol, w0, wa = -0.4, 0.3, 0.7, -0.8, 0.2
    state = jnp.asarray([[0.7, 0.8], [0.4, 0.5]])
    k = jnp.asarray([0.05, 0.2])
    params = mg_jax.pack_constants_jnp(
        om, ol, w0=w0, wa=wa, kind=mg_jax.LCDM
    )
    # Isolate this RHS test from the independent neutrino-table interpolation.
    monkeypatch.setattr(
        mg_jax, "mu", lambda eta, wave_number, p: jnp.ones_like(wave_number)
    )

    _, omega_m, friction = _background(eta, om, ol, w0, wa)
    expected = np.stack(
        [np.asarray(state[1]), 1.5 * omega_m * np.asarray(state[0])
         - friction * np.asarray(state[1])]
    )
    np.testing.assert_allclose(
        np.asarray(mg_jax.firstOrder(eta, state, k, params)),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
