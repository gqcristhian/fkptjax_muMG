# The following implementation is a translation of the NR C code used in FKPT,
# provided to allow direct comparison for testing and validation purposes.

from math import copysign
from typing import Callable, Tuple, Optional, List

import numpy as np

TINY = 1.0e-30

def odeint(
    ystart: np.ndarray,
    x1: float,
    x2: float,
    derivs: Callable[[float, np.ndarray], np.ndarray],
    eps: float = 1e-4,
    h1: float = 2./5.,
    hmin: float = 0.0,
    maxnsteps: int = 10000,
    *,
    dxsav: Optional[float] = None,
    kmax: int = 0,
) -> Tuple[np.ndarray, int, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Integrate y'(x) = f(x, y) from x1 to x2 with adaptive steps and quality control.

    Parameters
    ----------
    ystart : array_like (nvar,)
        Initial condition at x1. Will not be modified in-place.
    x1, x2 : float
        Integration limits.
    derivs : function
        derivs(x, y) -> dydx
    eps : float
        Desired relative accuracy (passed to stepper).
    h1 : float
        Initial step size guess (sign is adjusted to match x2 - x1).
    hmin : float
        Minimum allowed step size magnitude.
    maxnsteps : int
        Safety cap on number of steps.
    dxsav : float, optional
        If provided and kmax>0, save (x,y) every ~dxsav in |x|.
    kmax : int
        Max number of saved output points (0 disables saving).

    Returns
    -------
    yfinal : np.ndarray
        y at x = x2 (final state).
    nok : int
        Count of successful steps with hdid == htry.
    nbad : int
        Count of steps where htry was reduced (hdid != htry).
    xp : np.ndarray or None
        Saved x samples (None if kmax==0 or dxsav is None).
    yp : np.ndarray or None
        Saved y samples with shape (kount, nvar) aligned with xp.
    """
    y = np.array(ystart, dtype=float, copy=True)
    nvar = y.size
    x = float(x1)
    # Non-finite inputs (a NaN growth template from a failed Boltzmann call upstream) must fail
    # at once: with x = NaN the termination test (x - x2) * (x2 - x1) >= 0 is never true and
    # abs(hnext) <= hmin never fires, so the loop would grind through all maxnsteps NaN steps
    # (observed 2026-09-13: ~0.6 s per NaN step, 1.7 h per point, stalling a whole MPI chain).
    # RuntimeError is what callers already catch for 'Too many steps' / 'stepsize underflow'.
    if not (np.isfinite(x) and np.isfinite(x2) and np.all(np.isfinite(y))):
        raise RuntimeError("non-finite initial condition or limits in odeint")
    h = copysign(abs(h1), x2 - x1)

    nok = 0
    nbad = 0

    # Output sampling (mimics NR globals: xp, yp, dxsav, kmax, kount)
    save_output = (kmax > 0 and dxsav is not None and dxsav > 0.0)
    dxsav_val = dxsav if dxsav is not None else 1.0
    xsav = x - 2.0 * dxsav_val  # force save on first eligible step
    xp: List[float] = []
    yp: List[np.ndarray] = []
    kount = 0

    for _ in range(1, maxnsteps + 1):
        dydx = derivs(x, y)
        yscal = np.abs(y) + np.abs(dydx * h) + TINY

        # Save if enough distance since last save (and capacity remains)
        if save_output and (kount < kmax - 1) and (abs(x - xsav) > abs(dxsav_val)):
            xp.append(x)
            yp.append(y.copy())
            kount += 1
            xsav = x

        # Shorten last step to land exactly on x2
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        # Take a step with quality control
        ynew, xnew, hdid, hnext = rkqs(y, dydx, x, h, eps, yscal, derivs)

        # Bookkeeping: “good” vs “bad” steps
        # NR compares hdid == h literally; we allow machine epsilon slack.
        if abs(hdid - h) <= max(1.0, abs(h)) * 1e-15:
            nok += 1
        else:
            nbad += 1

        y, x = ynew, xnew
        # Same reason as the entry guard: a NaN abscissa or state can never reach x2.
        if not (np.isfinite(x) and np.all(np.isfinite(y))):
            raise RuntimeError("non-finite state in odeint")

        # Reached (or passed) the end?
        if (x - x2) * (x2 - x1) >= 0.0:
            # Final save and return
            if save_output and kount < kmax:
                xp.append(x)
                yp.append(y.copy())
                kount += 1
            return y, nok, nbad, (np.array(xp) if save_output else None), (np.vstack(yp) if save_output else None)

        if abs(hnext) <= hmin:
            raise RuntimeError("Step size too small in odeint")

        h = hnext

    raise RuntimeError("Too many steps in routine odeint")

SAFETY = 0.9
PGROW  = -0.2
PSHRNK = -0.25
ERRCON = 1.89e-4

def rkck(
    y: np.ndarray,
    dydx: np.ndarray,
    x: float,
    h: float,
    derivs: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cash–Karp Runge–Kutta step.
    Returns:
      yout : 5th-order solution estimate
      yerr : error estimate (y5 - y4), used for step-size control
    """
    # Cash–Karp coefficients (match the C code literals)
    a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875

    b21 = 0.2

    b31, b32 = 3.0/40.0, 9.0/40.0

    b41, b42, b43 = 0.3, -0.9, 1.2

    b51, b52, b53, b54 = -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0

    b61 = 1631.0/55296.0
    b62 = 175.0/512.0
    b63 = 575.0/13824.0
    b64 = 44275.0/110592.0
    b65 = 253.0/4096.0

    # 5th-order solution (c*)
    c1, c3, c4, c6 = 37.0/378.0, 250.0/621.0, 125.0/594.0, 512.0/1771.0

    # Differences y5 - y4 (dc*)
    dc1 = c1 - 2825.0/27648.0
    dc3 = c3 - 18575.0/48384.0
    dc4 = c4 - 13525.0/55296.0
    dc5 = -277.0/14336.0
    dc6 = c6 - 0.25

    k1 = dydx
    k2 = derivs(x + a2*h, y + h*(b21*k1))
    k3 = derivs(x + a3*h, y + h*(b31*k1 + b32*k2))
    k4 = derivs(x + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    k5 = derivs(x + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    k6 = derivs(x + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

    yout = y + h*(c1*k1 + c3*k3 + c4*k4 + c6*k6)
    yerr = h*(dc1*k1 + dc3*k3 + dc4*k4 + dc5*k5 + dc6*k6)

    return yout, yerr


def rkqs(
    y: np.ndarray,
    dydx: np.ndarray,
    x: float,
    htry: float,
    eps: float,
    yscal: np.ndarray,
    derivs: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, float, float, float]:
    """
    Quality-controlled single step (Numerical Recipes 'rkqs') using 'rkck'.

    Parameters
    ----------
    y      : current state at x (not modified in-place)
    dydx   : derivative at (x, y)
    x      : current independent variable
    htry   : trial step size
    eps    : desired accuracy
    yscal  : scaling vector (typically |y| + |dydx*h| + TINY)
    derivs : function f(x, y) -> dy/dx

    Returns
    -------
    ynew  : state after accepted step
    xnew  : x + hdid
    hdid  : actual step taken
    hnext : proposed next step
    """
    h = float(htry)

    while True:
        ytemp, yerr = rkck(y, dydx, x, h, derivs)

        # errmax = max_i |yerr_i / yscal_i| / eps
        errmax = float(np.max(np.abs(yerr / yscal)))
        errmax /= eps

        if errmax <= 1.0:
            # accepted
            break

        # rejected: shrink h
        htemp = SAFETY * h * (errmax ** PSHRNK)
        if h >= 0.0:
            h = max(htemp, 0.1 * h)
        else:
            h = min(htemp, 0.1 * h)

        xnew = x + h
        if xnew == x:
            raise RuntimeError("stepsize underflow in rkqs")

    # propose next step
    if errmax > ERRCON:
        hnext = SAFETY * h * (errmax ** PGROW)
    else:
        hnext = 5.0 * h

    hdid = h
    xnew = x + hdid
    ynew = ytemp

    return ynew, xnew, hdid, hnext
