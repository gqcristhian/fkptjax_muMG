"""
jax_ode.py
----------
JAX (jit/vmap-able) growth and kernel-constant ODE solves for the PHENOM/binning
modified-gravity model, using ``diffrax`` adaptive integration with the JAX RHS
in :mod:`binning_jax`.  These replace the pure-Python RKQS solves
(``ode.DP`` / ``ode.kernel_constants``) in the JAX backend of
``Kfuncs_to_tables`` -- they reproduce the same ODE system and initial conditions,
so the physics matches the numpy path to the ODE tolerance.

Requires ``jax_enable_x64=True`` (folps/fkptjax are float64).
"""

import jax
import jax.numpy as jnp
import diffrax

from . import binning_jax as bj


# Default adaptive tolerances.  Validated: the JAX solve is fully converged here
# (rtol 1e-9->1e-11 changes D(k) by ~1e-10), while the legacy Python RKQS
# (eps=1e-4) carries ~8e-4 truncation error in D(k) -- i.e. the JAX ODE is the
# *more accurate* one, and the JAX-vs-legacy difference is the legacy's error.
# 1e-8/1e-11 is converged to ~1e-6 (far below the emulator's ~0.16% floor).
_RTOL = 1e-8
_ATOL = 1e-11
_MAXSTEPS = 100000

# Fixed-step RK4 solver: a uniform step count makes the solve identical work for
# every batch element, so jax.vmap maps cleanly (unlike the adaptive solver,
# which pads to the batch's max step count).  _N_STEPS is validated to converge
# the growth + kernel ODEs to ~1e-6 over xnow..xstop (smooth ODEs, short range).
_N_STEPS = 128


def _solve(rhs, y0, xnow, xstop, rtol=_RTOL, atol=_ATOL):
    """Adaptive diffrax solve (vmaps with step-count padding overhead)."""
    term = diffrax.ODETerm(rhs)
    solver = diffrax.Tsit5()
    ctrl = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term, solver, t0=xnow, t1=xstop, dt0=0.01, y0=y0,
        stepsize_controller=ctrl, saveat=diffrax.SaveAt(t1=True),
        max_steps=_MAXSTEPS,
    )
    return sol.ys[-1]  # state at t1 = xstop


def _solve_rk4(rhs, y0, xnow, xstop, n_steps=_N_STEPS):
    """Fixed-step classical RK4 via lax.scan -- vmap-friendly (uniform work)."""
    h = (xstop - xnow) / n_steps

    def step(carry, _):
        t, y = carry
        k1 = rhs(t, y, None)
        k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1, None)
        k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2, None)
        k4 = rhs(t + h, y + h * k3, None)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return (t + h, y), None

    (_, yf), _ = jax.lax.scan(step, (xnow, y0), None, length=n_steps)
    return yf


def _run_solver(rhs, y0, xnow, xstop, solver, rtol, atol, n_steps):
    if solver == 'rk4':
        return _solve_rk4(rhs, y0, xnow, xstop, n_steps)
    return _solve(rhs, y0, xnow, xstop, rtol, atol)


def DP_jax(k_arr, P, xnow, xstop, rtol=_RTOL, atol=_ATOL,
           solver='rk4', n_steps=_N_STEPS):
    """Linear growth D(k), D'(k) at xstop for a 1D array of k.

    Returns array of shape (2, nk): [D, D'].  Mirrors ode.DP (batched first-order
    ODE; IC D = D' = exp(xnow)).  ``solver='rk4'`` (default) is a fixed-step
    lax.scan RK4 (vmap-friendly); ``solver='adaptive'`` uses diffrax.
    """
    k_arr = jnp.asarray(k_arr)
    nk = k_arr.shape[0]
    y0 = jnp.exp(xnow) * jnp.ones((2, nk), dtype=jnp.float64)

    def rhs(t, y, args):
        return bj.firstOrder(t, y, k_arr, P)

    return _run_solver(rhs, y0, xnow, xstop, solver, rtol, atol, n_steps)


def kernel_constants_jax(f0, P, xnow, xstop, KMIN=1e-8, x=0.0, rtol=_RTOL, atol=_ATOL,
                         solver='rk4', n_steps=_N_STEPS):
    """Beyond-EdS kernel constants (ALS, AprimeLS, KR1LS, KR1pLS).

    Mirrors ode.kernel_constants (non-EFT path): solve the 10-dim third-order ODE
    at k = p = KMIN, x=0, then form the constants.  ``solver='rk4'`` (default)
    fixed-step (vmap-friendly); ``solver='adaptive'`` uses diffrax.
    """
    k = float(KMIN); p = float(KMIN); x = float(x)
    e1 = jnp.exp(xnow)
    e2 = jnp.exp(2.0 * xnow)
    e3 = jnp.exp(3.0 * xnow)
    one_m_x2 = 1.0 - x * x
    pk = p / k
    ang = 1.0 / (1.0 + pk * pk + 2.0 * pk * x) + 1.0 / (1.0 + pk * pk - 2.0 * pk * x)

    y0 = jnp.array([
        e1, e1, e1, e1,                                  # Dk, Dk', Dp, Dp'
        3.0 * e2 / 7.0 * one_m_x2, 6.0 * e2 / 7.0 * one_m_x2,   # D2+, D2+'
        3.0 * e2 / 7.0 * one_m_x2, 6.0 * e2 / 7.0 * one_m_x2,   # D2-, D2-'
        (5.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang,          # D3
        (15.0 / 63.0) * e3 * one_m_x2 * one_m_x2 * ang,         # D3'
    ], dtype=jnp.float64)

    def rhs(t, y, args):
        return bj.thirdOrder(t, y, x, k, p, P)

    Y = _run_solver(rhs, y0, xnow, xstop, solver, rtol, atol, n_steps)
    Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3 = [Y[i] for i in range(10)]

    KR1LS = (21.0 / 5.0) * D3 / (Dk * Dp * Dp)
    KR1pLS = (21.0 / 5.0) * dD3 / (Dk * Dp * Dp) / (3.0 * f0)
    C = (3.0 / 7.0) * Dk * Dp
    Cp = (3.0 / 7.0) * (dDk * Dp + Dk * dDp)
    ALS = D2p / C
    AprimeLS = dD2p / C - D2p * Cp / (C * C)
    return ALS, AprimeLS, KR1LS, KR1pLS
