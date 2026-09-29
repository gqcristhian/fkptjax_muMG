import numpy as np

from fkptjax.types import KFunctionsInitData, KFunctionsOut, Float64NDArray, AbsCalculator


class NumpyCalculator(AbsCalculator):
    """NumPy-based k-functions calculator implementing the AbsCalculator interface."""

    def __init__(self) -> None:
        """Initialize an empty calculator. Call initialize() before evaluate()."""
        self.k_in = None
        self.logk_grid = None
        self.kk_grid = None
        self.xxQ = None
        self.wwQ = None
        self.xxR = None
        self.wwR = None
        # Precomputed spline interpolation coefficients
        self.spline_logk = None
        self.spline_kk = None
        self.spline_y = None
        self.spline_y_r = None
        # Precomputed Q-function quantities
        self.logk_grid2 = None
        self.dkk = None
        self.dkk_reshaped = None
        self.scale_Q = None
        self.r = None
        self.r2 = None
        self.x = None
        self.w = None
        self.x2 = None
        self.y2 = None
        self.y = None
        # Precomputed R-function quantities
        self.r_r = None
        self.r2_r = None
        self.x_r = None
        self.w_r = None
        self.x2_r = None
        self.y2_r = None
        self.AngleEvR = None
        self.AngleEvR2 = None
        self.dkk_r = None

    def initialize(self, data: KFunctionsInitData) -> None:
        """Initialize the calculator with grid data and quadrature points.

        Args:
            data: Initialization data containing k-grid, quadrature points, etc.
        """
        self.k_in = data.k_in
        self.logk_grid = data.logk_grid
        self.kk_grid = data.kk_grid
        self.xxQ = data.xxQ
        self.wwQ = data.wwQ
        self.xxR = data.xxR
        self.wwR = data.wwR

        # Pre-compute commonly used grid quantities
        self.logk_grid2 = self.logk_grid * self.logk_grid
        self.dkk = np.diff(self.kk_grid)
        self.dkk_reshaped = self.dkk.reshape(-1, 1, 1)
        self.scale_Q = 0.25 * self.logk_grid2 / np.pi ** 2

        # Pre-compute spline coefficients for fixed interpolation grids
        # These grids don't change between evaluate() calls, so we can
        # pre-compute the interpolation coefficients once here

        # 1. Coefficients for interpolating onto logk_grid
        self.spline_logk = self._init_cubic_spline(self.k_in, self.logk_grid)

        # 2. Coefficients for interpolating onto kk_grid
        self.spline_kk = self._init_cubic_spline(self.k_in, self.kk_grid)

        # 3. Pre-compute Q-function quantities and spline coefficients
        # Compute variable integration limits for mu (local variables only needed here)
        rmax = self.k_in[-1] / self.logk_grid
        rmin = self.k_in[0] / self.logk_grid
        rmax2 = rmax * rmax
        rmin2 = rmin * rmin

        self.r = self.kk_grid[1:].reshape(-1, 1, 1) / self.logk_grid
        self.r2 = np.square(self.r)

        mumin = np.maximum(-1.0, (1.0 + self.r2 - rmax2) / (2.0 * self.r))
        mumax = np.minimum(1.0, (1.0 + self.r2 - rmin2) / (2.0 * self.r))
        mumax = np.divide(0.5, self.r, out=mumax, where=self.r >= 0.5)

        # Scale Gauss-Legendre nodes and weights to [mumin, mumax]
        dmu = mumax - mumin
        xGL = 0.5 * (dmu * self.xxQ.reshape(-1, 1, 1, 1) + (mumax + mumin))
        wGL = 0.5 * dmu * self.wwQ.reshape(-1, 1, 1, 1)

        # Compute x, w, x2, y2, y values for Q-function integration
        self.x = xGL
        self.w = wGL
        self.x2 = self.x * self.x
        self.y2 = 1.0 + self.r2 - 2.0 * self.r * self.x
        self.y = np.sqrt(self.y2)

        # Pre-compute coefficients for interpolating at logk_grid * y
        self.spline_y = self._init_cubic_spline(self.k_in, self.logk_grid * self.y)

        # Pre-compute R-function quantities
        # R-function uses r from kk[1:-1] (indices 1 to nquadSteps-2)
        self.r_r = self.kk_grid[1:-1].reshape(-1, 1, 1) / self.logk_grid
        self.r2_r = np.square(self.r_r)

        # Gauss-Legendre points in [-1, 1] (fixed limits for R-functions)
        self.x_r = self.xxR.reshape(-1, 1, 1, 1)
        self.w_r = self.wwR.reshape(-1, 1, 1, 1)
        self.x2_r = self.x_r * self.x_r
        self.y2_r = 1.0 + self.r2_r - 2.0 * self.r_r * self.x_r

        # R-function angles (independent of input parameters)
        self.AngleEvR = -self.x_r
        self.AngleEvR2 = np.square(self.AngleEvR)

        # R-function trapezoidal integration spacing
        self.dkk_r = self.dkk_reshaped[:-1].reshape(-1, 1, 1)

        # Pre-compute coefficients for interpolating P(k), f(k) at the R-grid's
        # "kminus" = k*sqrt(1+r_r^2-2 r_r x_r) points -- needed by Gamma2fevR
        # below (sMGPT's own formula uses the growth rate at kminus, not at
        # k_ext -- see the Gamma2fevR fix in evaluate()).
        self.y_r = np.sqrt(self.y2_r)
        self.spline_y_r = self._init_cubic_spline(self.k_in, self.logk_grid * self.y_r)

    def _calc_2nd_derivs(self, x: Float64NDArray, y: Float64NDArray) -> Float64NDArray:
        """Initialize a cubic spline interpolator by precomputing 2nd derivatives."""
        n = len(x)
        y = np.moveaxis(y, -1, 0)
        y2 = np.zeros_like(y)
        u = np.zeros_like(y)
        # Forward sweep
        for i in range(1, n-1):
            sig = (x[i] - x[i-1]) / (x[i+1] - x[i-1])
            p = sig * y2[i-1] + 2.0
            y2[i] = (sig - 1.0) / p
            udiff = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1])
            u[i] = (6.0 * udiff / (x[i+1] - x[i-1]) - sig * u[i-1]) / p
        # Back substitution
        for k in range(n-2, -1, -1):
            y2[k] = y2[k] * y2[k+1] + u[k if k < n-1 else n-2]
        return np.moveaxis(y2, 0, -1)

    def _init_cubic_spline(self, xa: Float64NDArray, x: Float64NDArray) -> dict:
        """Pre-compute cubic spline interpolation coefficients for given xa and x.

        Args:
            xa: The x-coordinates of the data points (must be increasing)
            x: The x-coordinates where we want to evaluate the spline

        Returns:
            Dictionary containing precomputed coefficients needed for evaluation
        """
        idx_hi = np.searchsorted(xa, x, side='right')
        idx_hi = np.clip(idx_hi, 1, xa.size - 1)
        idx_lo = idx_hi - 1
        h = xa[idx_hi] - xa[idx_lo]
        a = (xa[idx_hi] - x) / h
        b = (x - xa[idx_lo]) / h

        return {
            'idx_lo': idx_lo,
            'idx_hi': idx_hi,
            'h': h,
            'a': a,
            'b': b,
            'h2_div_6': (h**2) / 6.0,
            'a3_minus_a': a**3 - a,
            'b3_minus_b': b**3 - b
        }

    def _eval_cubic_spline(self, ya: Float64NDArray, y2a: Float64NDArray, coeffs: dict) -> Float64NDArray:
        """Evaluate the cubic spline using precomputed coefficients.

        Args:
            ya: The y-coordinates of the data points
            y2a: The second derivatives at the data points
            coeffs: Dictionary of precomputed coefficients from _init_cubic_spline

        Returns:
            Interpolated values at the x-coordinates specified during initialization
        """
        return np.moveaxis(
            (coeffs['a'] * ya[..., coeffs['idx_lo']] +
             coeffs['b'] * ya[..., coeffs['idx_hi']] +
             (coeffs['a3_minus_a'] * y2a[..., coeffs['idx_lo']] +
              coeffs['b3_minus_b'] * y2a[..., coeffs['idx_hi']]) * coeffs['h2_div_6']),
            0, -1)

    def evaluate(self, Pk_in: Float64NDArray, Pk_nw_in: Float64NDArray,
                 fk_in: Float64NDArray, A: float, ApOverf0: float, CFD3: float,
                 CFD3p: float, sigma2v: float, f0: float) -> KFunctionsOut:
        """Evaluate k-functions given input power spectra.

        Args:
            Pk_in: Linear power spectrum values at k_in grid points
            Pk_nw_in: No-wiggle linear power spectrum values at k_in grid points
            fk_in: Growth rate f(k) values at k_in grid points (normalized by f0)
            A: Cosmological parameter A
            ApOverf0: Cosmological parameter Ap/f0
            CFD3: Cosmological parameter CFD3
            CFD3p: Cosmological parameter CFD3p
            sigma2v: Velocity dispersion parameter
            f0: Reference growth rate

        Returns:
            KFunctionsOut containing all computed k-functions
        """
        # Stack input power spectra for interpolation
        Y = np.stack([Pk_in, Pk_nw_in, fk_in / f0], axis=0)  # shape (3, n_k_in)

        # Use instance variables for grid data
        k_in = self.k_in
        logk_grid = self.logk_grid
        kk_grid = self.kk_grid
        xxQ = self.xxQ
        wwQ = self.wwQ
        xxR = self.xxR
        wwR = self.wwR

        # Compute second derivatives for cubic spline interpolation
        Y2 = self._calc_2nd_derivs(k_in, Y)

        # Interpolate onto output grid using precomputed coefficients
        Pout, Pout_nw, fout = self._eval_cubic_spline(Y, Y2, self.spline_logk).T

        # Interpolate onto quadrature grid using precomputed coefficients
        Pkk, Pkk_nw, fkk = self._eval_cubic_spline(Y, Y2, self.spline_kk).T

        # Use precomputed grid quantities
        logk_grid2 = self.logk_grid2
        dkk = self.dkk

        # ============================================================================
        # Q-FUNCTIONS: Vectorized over ALL dimensions
        # ============================================================================

        # Use precomputed Q-function quantities
        r = self.r
        r2 = self.r2
        x = self.x
        w = self.w
        x2 = self.x2
        y2 = self.y2
        y = self.y

        # Loop over quadrature k values
        # fp shape: (nquadSteps-1, 1, 1) - will broadcast to (nquadSteps-1, 1, Nk)
        fp = fkk[1:].reshape(-1, 1, 1)

        # Interpolate power spectra at (ki * y) points using precomputed coefficients
        psl_w, psl_nw, fkmp = self._eval_cubic_spline(Y, Y2, self.spline_y).T
        psl = np.concatenate((psl_w.T, psl_nw.T), axis=2) # shape (NQ, nquadSteps-1, 2, Nk)
        fkmp = fkmp.T # shape (NQ, nquadSteps-1, 1, Nk)

        # Compute SPT kernels F2evQ and G2evQ
        AngleEvQ = (x - r) / y
        AngleEvQ2 = np.square(AngleEvQ)
        fsum = fp + fkmp

        S2evQ = AngleEvQ2 - 1./3.
        F2evQ = (1.0/2.0 + 3.0/14.0 * A + (1.0/2.0 - 3.0/14.0 * A) * AngleEvQ2 +
                 AngleEvQ / 2.0 * (y/r + r/y))
        G2evQ = (3.0/14.0 * A * fsum + 3.0/14.0 * ApOverf0 +
                 (1.0/2.0 * fsum - 3.0/14.0 * A * fsum - 3.0/14.0 * ApOverf0) * AngleEvQ2 +
                 AngleEvQ / 2.0 * (fkmp * y/r + fp * r/y))

        # Accumulate over mu dimension (sum along axis 0)
        # Shapes after summation: (nquadSteps-1, Nk)

        # Precompute some temporary expressions that are used multiple times
        wpsl = w * psl
        fkmpr2 = fkmp * r2
        rx = r * x
        y4 = y2 ** 2

        # P22 kernels
        P22dd_B = np.sum(wpsl * (2.0 * r2 * F2evQ**2), axis=0)
        P22du_B = np.sum(wpsl * (2.0 * r2 * F2evQ * G2evQ), axis=0)
        P22uu_B = np.sum(wpsl * (2.0 * r2 * G2evQ**2), axis=0)

        # ========== 5 THREE-POINT CORRELATION FUNCTION KERNELS (Q-part) ==========

        # I1udd1tA
        I1udd1tA_B = np.sum(wpsl * (
            2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * F2evQ
            ), axis=0)

        # I2uud1tA
        I2uud1tA_B = np.sum(wpsl * (-fp * fkmpr2 * (1.0 - x2) / y2 * F2evQ), axis=0)

        # I2uud2tA
        I2uud2tA_B = np.sum(wpsl * (
            2.0 * (fp * rx + fkmpr2 * (1.0 - rx) / y2) * G2evQ
            + fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * F2evQ
            ), axis=0)

        # I3uuu2tA
        I3uuu2tA_B = np.sum(wpsl * (fp * fkmpr2 * (x2 - 1.0) / y2 * G2evQ), axis=0)

        # I3uuu3tA
        I3uuu3tA_B = np.sum(wpsl * (
            fp * fkmp * (r2 * (1.0 - 3.0 * x2) + 2.0 * rx) / y2 * G2evQ
            ), axis=0)

        # ========== 7 BpC TERM KERNELS (Q-part, will become D-terms) ==========

        # I2uudd1BpC
        I2uudd1BpC_B = np.sum(wpsl * (
            1.0 / 4.0 * (1.0 - x2) * (fp * fp + np.square(fkmpr2) / y4)
            + fp * fkmpr2 * (-1.0 + x2) / y2 / 2.0
            ), axis=0)

        # I2uudd2BpC
        I2uudd2BpC_B = np.sum(wpsl * (
            (
                fp * fp * (-1.0 + 3.0 * x2)
                + 2.0 * fkmp * fp * r * (r + 2.0 * x - 3.0 * r * x2) / y2
                + fkmp * fkmpr2 * (2.0 - 4.0 * rx + r2 * (-1.0 + 3.0 * x2)) / y4
            )
            / 4.0
            ), axis=0)

        # I3uuud2BpC
        I3uuud2BpC_B = np.sum(wpsl * (
            -(
                fkmp * fp * (
                    fkmp * (-2.0 + 3.0 * rx) * r2
                    - fp * (-1.0 + 3.0 * rx) * (1.0 - 2.0 * rx + r2)
                )
                * (-1.0 + x2)
            )
            / (2.0 * y2 * y2)
            ), axis=0)

        # I3uuud3BpC
        I3uuud3BpC_B = np.sum(wpsl * (
            (
                fkmp * fp * (
                    -(
                        fp
                        * (1.0 - 2.0 * rx + r2)
                        * (1.0 - 3.0 * x2 + rx * (-3.0 + 5.0 * x2))
                    )
                    + fkmp * r * (2.0 * x + r * (2.0 - 6.0 * x2 + rx * (-3.0 + 5.0 * x2)))
                )
            )
            / (2.0 * y4)
            ), axis=0)

        # I4uuuu2BpC
        I4uuuu2BpC_B = np.sum(wpsl * (
            3.0 * np.square(fkmp) * np.square(fp) * r2 * np.square(-1.0 + x2) / (16.0 * y4)
            ), axis=0)

        # I4uuuu3BpC
        I4uuuu3BpC_B = np.sum(wpsl * (
            -(
                np.square(fkmp) * np.square(fp) * (-1.0 + x2) * (2.0 + 3.0 * r * (-4.0 * x + r * (-1.0 + 5.0 * x2)))
            )
            / (8.0 * y2 * y2)
            ), axis=0)

        # I4uuuu4BpC
        I4uuuu4BpC_B = np.sum(wpsl * (
            (
                np.square(fkmp) * np.square(fp) * (
                    -4.0
                    + 8.0 * rx * (3.0 - 5.0 * x2)
                    + 12.0 * x2
                    + r2 * (3.0 - 30.0 * x2 + 35.0 * np.square(x2))
                )
            )
            / (16.0 * y4)
            ), axis=0)

        # Left and right endpoints for power spectra
        PSLB = np.stack((Pkk[1:], Pkk_nw[1:]), axis=1)[:,:,None] # shape (nQuadSteps-1, 2, 1)

        # Use precomputed dkk_reshaped and scale_Q
        dkk_reshaped = self.dkk_reshaped
        scale_Q = self.scale_Q

        # Bias
        Pb1b2_B = np.sum(wpsl * (r2 * F2evQ), axis=0)
        Pb1bs2_B = np.sum(wpsl * (r2 * F2evQ * S2evQ), axis=0)
        Pratio = PSLB / psl
        PratioInv = psl / PSLB
        Pb22_B = np.sum(wpsl * (
            1.0 / 2.0 * r2 * (1.0 / 2.0 * (1.0 - Pratio) + 1.0 / 2.0 * (1.0 - PratioInv))
            ), axis=0)
        Pb2s2_B = np.sum(wpsl * (
            1.0 / 2.0 * r2 * (
                1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * Pratio)
                + 1.0 / 2.0 * (S2evQ - 2.0 / 3.0 * PratioInv)
            )
            ), axis=0)
        Ps22_B = np.sum(wpsl * (
            1.0 / 2.0 * r2
            * (
                1.0 / 2.0 * (np.square(S2evQ) - 4.0 / 9.0 * Pratio)
                + 1.0 / 2.0 * (np.square(S2evQ) - 4.0 / 9.0 * PratioInv)
            )
            ), axis=0)
        Pb2theta_B = np.sum(wpsl * (r2 * G2evQ), axis=0)
        Pbs2theta_B = np.sum(wpsl * (r2 * S2evQ * G2evQ), axis=0)

        # Apply trapezoidal rule with optimized operations
        def trapsumQ(B: Float64NDArray) -> Float64NDArray:
            # Scale input array (in-place is slightly slower)
            B = B * (scale_Q * PSLB)
            # Use fused trapezoidal sum (more efficient than cumsum approach)
            return np.sum((B[:-1] + B[1:]) * dkk_reshaped[1:], axis=0) + B[0] * dkk_reshaped[0]

        P22dd = trapsumQ(P22dd_B)
        P22du = trapsumQ(P22du_B)
        P22uu = trapsumQ(P22uu_B)

        I1udd1tA = trapsumQ(I1udd1tA_B)
        I2uud1tA = trapsumQ(I2uud1tA_B)
        I2uud2tA = trapsumQ(I2uud2tA_B)
        I3uuu2tA = trapsumQ(I3uuu2tA_B)
        I3uuu3tA = trapsumQ(I3uuu3tA_B)

        I2uudd1BpC = trapsumQ(I2uudd1BpC_B)
        I2uudd2BpC = trapsumQ(I2uudd2BpC_B)
        I3uuud2BpC = trapsumQ(I3uuud2BpC_B)
        I3uuud3BpC = trapsumQ(I3uuud3BpC_B)
        I4uuuu2BpC = trapsumQ(I4uuuu2BpC_B)
        I4uuuu3BpC = trapsumQ(I4uuuu3BpC_B)
        I4uuuu4BpC = trapsumQ(I4uuuu4BpC_B)

        # Bias terms
        Pb1b2 = trapsumQ(Pb1b2_B)
        Pb1bs2 = trapsumQ(Pb1bs2_B)
        Pb22 = trapsumQ(Pb22_B)
        Pb2s2 = trapsumQ(Pb2s2_B)
        Ps22 = trapsumQ(Ps22_B)
        Pb2theta = trapsumQ(Pb2theta_B)
        Pbs2theta = trapsumQ(Pbs2theta_B)

        # ============================================================================
        # R-FUNCTIONS: Also fully vectorized (NR, nquadSteps-2, Nk) - note nquadSteps-2!
        # ============================================================================

        # Get f(k) at output k values
        fk = fout  # already computed above

        # Use precomputed R-function quantities
        r_r = self.r_r
        r2_r = self.r2_r
        x_r = self.x_r
        w_r = self.w_r
        x2_r = self.x2_r
        y2_r = self.y2_r
        AngleEvR = self.AngleEvR
        AngleEvR2 = self.AngleEvR2

        # R-function uses fp from kk[1:-1] and psl from Pkk[1:-1]
        fp_r = fkk[1:-1].reshape(-1 , 1, 1)
        psl_r = np.stack((Pkk[1:-1], Pkk_nw[1:-1]), axis=1)[:,:,None] # shape (nquadSteps-2, 2, 1)

        # P(kminus)/f(kminus) -- needed by Gamma2fevR below.
        _, _, fkminusp_r = self._eval_cubic_spline(Y, Y2, self.spline_y_r).T
        fkminusp_r = fkminusp_r.T  # shape (NR, nquadSteps-2, 1, Nk)

        F2evR = (1.0/2.0 + 3.0/14.0 * A + (1.0/2.0 - 3.0/14.0 * A) * AngleEvR2 +
                 AngleEvR / 2.0 * (1.0/r_r + r_r))
        G2evR = (3.0/14.0 * A * (fp_r + fk) + 3.0/14.0 * ApOverf0 +
                 (1.0/2.0 * (fp_r + fk) - 3.0/14.0 * A * (fp_r + fk) -
                  3.0/14.0 * ApOverf0) * AngleEvR2 +
                 AngleEvR / 2.0 * (fk/r_r + fp_r * r_r))

        # ========== 5 THREE-POINT CORRELATION FUNCTION KERNELS (R-part) ==========
        # Accumulate over mu (sum along axis 0)
        # Shapes after summation: (nquadSteps-2, Nk)
        wpsl_r = w_r * psl_r

        Gamma2evR  = A *(1. - x2_r)
        # Gamma2fevR: sMGPT's own formula (3_P13type.wl) uses the growth rate
        # at kminus (the output leg of the underlying D2 solve) here, not at
        # k_ext -- a from-scratch re-derivation against that source (see
        # calculate_jax.py's matching fix / .claude/plans/quizzical-mapping-catmull.md)
        # found this used to read fk (external-k growth) instead of
        # fkminusp_r, a pre-existing bug.
        Gamma2fevR = A *(1. - x2_r)*(fkminusp_r + fp_r)/2. + 1./2. * ApOverf0 *(1 - x2_r)
        C3Gamma3  = 2.*5./21. * CFD3  *(1 - x2_r)*(1 - x2_r)/y2_r
        # C3Gamma3f previously carried an extra "(fk+2*fp_r)/3" growth-rate
        # weighting with no counterpart in sMGPT's C3gamma3fLCDM (3_P13type.wl:
        # 2*(5/21)*lcdmCFprime*(1-x^2)^2/y^2, no growth-rate factor). Removing
        # it was confirmed against sMGPT's own reference output (see the
        # matching fix in calculate_jax.py).
        C3Gamma3f = 2.*5./21. * CFD3p *(1 - x2_r)*(1 - x2_r)/y2_r
        # NOTE: the middle term below uses fk (external-k growth), matching
        # sMGPT's G3K term "1/3*C2*(fk/f0)*(k^2-kpx)/(k^2+p^2-2pkx)*gamma2evR"
        # (3_P13type.wl) exactly -- this used to read fp_r (loop-q growth), a
        # pre-existing bug (see calculate_jax.py's matching fix). The OTHER
        # fp_r below (last term) is correct as-is -- sMGPT's own formula
        # calls for fp (loop q) there.
        G3K = (
            C3Gamma3f/ 2. + (2 * Gamma2fevR * x_r)/(7. * r_r) - (fk  * x2_r)/(6 * r2_r)
            + fk * Gamma2evR*(1 - r_r * x_r)/(7 * y2_r)
            - 1./7.*(fp_r * Gamma2evR + 2 *Gamma2fevR) * (1. - x2_r)/y2_r)
        F3K = C3Gamma3/6. - x2_r/(6 * r2_r) + (Gamma2evR * x_r *(1 - r_r * x_r))/(7. *r_r *y2_r)

        P13dd_B = np.sum(wpsl_r * (6.* r2_r * F3K), axis=0)
        P13du_B = np.sum(wpsl_r * (3.* r2_r * G3K + 3.* r2_r * F3K * fk), axis=0)
        P13uu_B = np.sum(wpsl_r * (6.* r2_r * G3K * fk), axis=0)

        sigma32PSL_B = np.sum(wpsl_r * (
            ( 5.0* r2_r * (7. - 2*r2_r + 4*r_r*x_r + 6*(-2 + r2_r)*x2_r - 12*r_r*x2_r*x_r + 9*np.square(x2_r)))
            / (24.0 * y2_r)
        ), axis=0)

        # I1udd1a
        I1udd1a_B = np.sum(wpsl_r * (
            2.0 * r2_r * (1.0 - r_r * x_r) / y2_r * G2evR + 2.0 * fp_r * r_r * x_r * F2evR
        ), axis=0)

        # I2uud1a
        I2uud1a_B = np.sum(wpsl_r * (
            -fp_r * r2_r * (1.0 - x2_r) / y2_r * G2evR
        ), axis=0)

        # I2uud2a
        I2uud2a_B = np.sum(wpsl_r * (
            ((r2_r * (1.0 - 3.0 * x2_r) + 2.0 * r_r * x_r) / y2_r * fp_r +
                    fk * 2.0 * r2_r * (1.0 - r_r * x_r) / y2_r) * G2evR + 2.0 * x_r * r_r * fp_r * fk * F2evR
        ), axis=0)

        # I3uuu2a
        # Note this is similar to I2uud1a_B but with additional factor of fk inside the sum
        I3uuu2a_B = np.sum(wpsl_r * (
            -fp_r * r2_r * (1.0 - x2_r) / y2_r * G2evR * fk
        ), axis=0)

        # I3uuu3a
        I3uuu3a_B = np.sum(wpsl_r * (
            (r2_r * (1.0 - 3.0 * x2_r) + 2.0 * r_r * x_r) / y2_r * fp_r * fk * G2evR
        ), axis=0)

        # Calculate scaling for R-functions
        pkl_k = np.vstack([Pout, Pout_nw])  # shape (2, Nk)
        scale_R = logk_grid2 / (8.0 * np.pi ** 2) * pkl_k

        # Use precomputed dkk_r for trapezoidal integration
        dkk_r = self.dkk_r

        def trapsumR(B: Float64NDArray) -> Float64NDArray:
            # Scale input array (in-place is slightly slower)
            B = B * scale_R
            # Use fused trapezoidal sum (more efficient than cumsum approach)
            return np.sum((B[:-1] + B[1:]) * dkk_r[1:], axis=0) + B[0] * dkk_r[0]

        I1udd1a = trapsumR(I1udd1a_B)
        I2uud1a = trapsumR(I2uud1a_B)
        I2uud2a = trapsumR(I2uud2a_B)
        I3uuu2a = trapsumR(I3uuu2a_B)
        I3uuu3a = trapsumR(I3uuu3a_B)

        P13uu = trapsumR(P13uu_B)
        P13du = trapsumR(P13du_B)
        P13dd = trapsumR(P13dd_B)

        sigma32PSL = trapsumR(sigma32PSL_B)

        # ============================================================================
        # Combine Q and R functions
        # ============================================================================
        I1udd1A = I1udd1tA + 2.0 * I1udd1a
        I2uud1A = I2uud1tA + 2.0 * I2uud1a
        I2uud2A = I2uud2tA + 2.0 * I2uud2a
        I3uuu2A = I3uuu2tA + 2.0 * I3uuu2a
        I3uuu3A = I3uuu3tA + 2.0 * I3uuu3a

        # ============================================================================
        # D-TERMS (B + C - G corrections)
        # ============================================================================
        # Note: BpC terms already calculated from Q-functions above
        # Now apply G-corrections (sigma2v damping) to specific terms in-place
        fk_grid = fk  # Already normalized by f0

        # I2uudd1D (subtract k^2 * sigma2v * P_L(k))
        I2uudd1BpC -= logk_grid2 * sigma2v * pkl_k

        # I3uuud2D (subtract 2 * k^2 * sigma2v * f(k) * P_L(k))
        I3uuud2BpC -= 2.0 * logk_grid2 * sigma2v * fk_grid * pkl_k

        # I4uuuu3 (subtract k^2 * sigma2v * f(k)^2 * P_L(k))
        I4uuuu3BpC -= logk_grid2 * sigma2v * np.square(fk_grid) * pkl_k

        return KFunctionsOut(
            P22dd, P22du, P22uu,
            I1udd1A, I2uud1A, I2uud2A,
            I3uuu2A, I3uuu3A,
            I2uudd1BpC, I2uudd2BpC,
            I3uuud2BpC, I3uuud3BpC,
            I4uuuu2BpC, I4uuuu3BpC, I4uuuu4BpC,
            Pb1b2, Pb1bs2, Pb22, Pb2s2, Ps22,
            Pb2theta, Pbs2theta,
            P13dd, P13du, P13uu,
            sigma32PSL,
            pkl_k
        )
