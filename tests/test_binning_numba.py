"""
Bit-for-bit validation: numba binning RHS vs the reference numpy ModelDerivatives.

Builds two ModelDerivatives (use_numba False/True) with identical PHENOM/binning
constants and asserts firstOrder/secondOrder/thirdOrder (and mu) agree to ~machine
precision on random inputs.

Run:
    python -m pytest tests/test_binning_numba.py -q
or  python tests/test_binning_numba.py
"""
import numpy as np
from fkptjax.ode import ModelDerivatives

CONST = dict(
    om=0.31, ol=0.69, model="PHENOM", mg_variant="binning",
    mu1=1.2, mu2=0.8, mu3=1.5, mu4=0.9,
    z_div=1.0, z_TGR=2.0, z_tw=0.05, scale_bins=True,
    k_TGR=0.01, k_c=0.1, k_S=0.2, k_tw=0.001,
)


def _pair():
    ref = ModelDerivatives(use_numba=False, **CONST)
    nb = ModelDerivatives(use_numba=True, **CONST)
    assert nb._use_numba_binning, "numba fast path did not activate"
    return ref, nb


def test_mu_scalar_and_array():
    ref, nb = _pair()
    rng = np.random.default_rng(0)
    for _ in range(200):
        eta = float(rng.uniform(-4.0, 0.0))
        k = float(rng.uniform(1e-4, 5.0))
        a = float(ref.mu(eta, k))
        b = float(nb._bnb.nb_mu(eta, k, nb._nb_P))
        assert abs(a - b) <= 1e-12 + 1e-12 * abs(a), (eta, k, a, b)
    # array path (used by firstOrder via DP)
    k_arr = np.geomspace(1e-4, 10.0, 256)
    for eta in [-3.5, -2.0, -0.7, 0.0]:
        a = np.asarray(ref.mu(eta, k_arr))
        b = nb._bnb._mu_arr(eta, k_arr, nb._nb_P)
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_firstOrder():
    ref, nb = _pair()
    rng = np.random.default_rng(1)
    k_arr = np.geomspace(1e-4, 10.0, 64)
    for _ in range(50):
        x = float(rng.uniform(-4.0, 0.0))
        Y = rng.uniform(0.1, 2.0, size=(2, k_arr.size))
        a = np.asarray(ref.firstOrder(x, Y, k_arr))
        b = np.asarray(nb.firstOrder(x, Y, k_arr))
        assert a.shape == b.shape == (2, k_arr.size)
        assert np.allclose(a, b, rtol=0, atol=1e-11)


def test_secondOrder():
    ref, nb = _pair()
    rng = np.random.default_rng(2)
    for _ in range(300):
        eta = float(rng.uniform(-4.0, 0.0))
        x = float(rng.uniform(-0.99, 0.99))
        k = float(rng.uniform(1e-3, 2.0))
        p = float(rng.uniform(1e-3, 2.0))
        y = rng.uniform(0.1, 2.0, size=6)
        a = np.asarray(ref.secondOrder(eta, y, x, k, p))
        b = np.asarray(nb.secondOrder(eta, y, x, k, p))
        assert np.allclose(a, b, rtol=0, atol=1e-11), (eta, x, k, p)


def test_thirdOrder():
    ref, nb = _pair()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(500):
        eta = float(rng.uniform(-4.0, 0.0))
        x = float(rng.uniform(-0.99, 0.99))
        k = float(rng.uniform(1e-3, 2.0))
        p = float(rng.uniform(1e-3, 2.0))
        y = rng.uniform(0.1, 2.0, size=10)
        a = np.asarray(ref.thirdOrder(eta, y, x, k, p))
        b = np.asarray(nb.thirdOrder(eta, y, x, k, p))
        worst = max(worst, float(np.max(np.abs(a - b))))
        assert np.allclose(a, b, rtol=0, atol=1e-10), (eta, x, k, p, a - b)
    print("thirdOrder worst abs diff:", worst)


if __name__ == "__main__":
    test_mu_scalar_and_array(); print("mu OK")
    test_firstOrder(); print("firstOrder OK")
    test_secondOrder(); print("secondOrder OK")
    test_thirdOrder(); print("thirdOrder OK")
    print("ALL BIT-FOR-BIT CHECKS PASSED")
