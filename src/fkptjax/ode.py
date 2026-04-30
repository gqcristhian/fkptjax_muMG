from typing import Callable, Tuple, Union

import numpy as np

import scipy.integrate

from fkptjax.types import Float64NDArray
from fkptjax.odeint import odeint


class ModelDerivatives:
    """Hu-Sawicki f(R) modified gravity model for fkPT calculations.

    This class implements the Hu-Sawicki f(R) modified gravity model with chameleon
    screening for perturbation theory calculations. It provides methods to compute
    scale-dependent growth functions, source terms for second and third-order
    perturbations, and ODE derivatives for Lagrangian Perturbation Theory (LPT).

    The f(R) modification to gravity introduces:
    - Scale-dependent growth factor μ(k, η) that modifies the Poisson equation
    - Effective scalar field mass m(η) that determines screening scale
    - Chameleon screening effects via M2 and M3 coefficients
    - Frame-lagging corrections (KFL terms) from the Newtonian potential
    - Differential interaction terms (dI) from the scalar field dynamics

    We also have the following models:
    - LCDM: here we fix mu(a,k) = 1
    - HDKI: This models consist of 3 variants at this point
    i) mg_variant: "mu_OmDE" --- mu(a) = 1 + mu_0 * Omega_DE/Omega_Lambda
    i) mg_variant: "BZ" --- Bertschinger-Zukin parameterization.
    iii) mg_variant: "binning" --- binnings in redshift and scale for mu.

    References
    ----------
    See equations in csrc/paper.pdf for the fkPT formalism and implementation details
    in csrc/models.c (lines 306-840) for the C reference implementation.

    Notes
    -----
    This implementation mirrors the Hu-Sawicky Model functions in csrc/models.c.
    All calculations are performed in conformal time η = ln(a) where a is the scale factor.
    """
     
    def __init__(
        self,
        om: float,
        ol: float,
        w0: float = -1.0,
        wa: float = 0.0,
        # --- model switches
        model: str = "HS",
        mg_variant: str = "mu_OmDE",
        # --- HS / f(R) parameters
        fR0_HS: float = 0.0,
        beta2: float = 1.0/6.0,
        n_HS: int = 1,
        screening: int = 1,
        omegaBD: float = 0.0,
        # --- nDGP
        r_c: float = 1.0e30,
        # --- HDKI-like: mu_OmDE
        mu0: float = 0.0,
        # --- HDKI-like: BZ
        beta_1: float = 1.0,
        lambda_1: float = 1.0,
        exp_s: float = 1.0,
        # --- PHENOM: binning
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
        # --- PHENOM: growth-index
        gamma_0: float = 1.0,
        gamma_a: float = 0.0,
        t_k: float = 0.0,
        d_s: float = 0.0,
        # --- EFTCAMB_HORNDESKI: 1D interpolators over eta=ln(a) for h1, h3, h5
        # These are built from EFTCAMB's get_eft_functions() output.
        # mu(k, eta) = h1(eta) * (1 + k^2 * h5(eta)) / (1 + k^2 * h3(eta))
        # GR limit: h1=1, h3=0, h5=0  ->  mu=1
        eftcamb_h1_interp=None,   # callable: eta -> h1(eta)
        eftcamb_h3_interp=None,   # callable: eta -> h3(eta)
        eftcamb_h5_interp=None,   # callable: eta -> h5(eta)
    ) -> None:
        """Initialize the Hu-Sawicki f(R) model parameters.

        Parameters
        ----------
        om : float
            Matter density parameter Ωₘ at present epoch (z=0).
        ol : float
            Dark energy density parameter Ωₗ at present epoch (z=0).
        fR0_HS : float
            Present-day value of the f(R) modification parameter |f_R0|.
            Typically negative, controls the strength of the fifth force.
        beta2 : float, optional
            Coupling strength parameter β² = 1/(3(1 + 4*λ²)), default is 1/6.
            For Hu-Sawicki model, β² = 1/6 corresponds to conformal coupling.
        n_HS : int, optional
            Power-law index n in the Hu-Sawicki model, default is 1.
            Controls the redshift evolution of the screening.
        screening : int, optional
            Screening toggle: 1 to include screening (default), 0 to disable.
            When disabled, sets M2=M3=0 removing chameleon screening effects.
        omegaBD : float, optional
            Brans-Dicke parameter ω_BD, default is 0.0.
            Used in Jordan frame calculations for specific scalar-tensor theories.

        Notes
        -----
        The class stores invH0 = c/H₀ = 2997.92458 Mpc/h for converting between
        physical and comoving scales.
        """
        self.invH0 = 2997.92458

        # background
        self.om = float(om)
        self.ol = float(ol)
        self.w0 = float(w0)
        self.wa = float(wa)

        # switches
        self.model = str(model)
        self.mg_variant = str(mg_variant)

        # HS / f(R)
        self.fR0_HS = float(fR0_HS)
        self.beta2 = float(beta2)
        self.n_HS = int(n_HS)
        self.screening = int(screening)
        self.omegaBD = float(omegaBD)

        # nDGP
        self.r_c = float(r_c)

        # HDKI: mu_OmDE
        self.mu0 = float(mu0)

        # HDKI: BZ-like
        self.beta_1 = float(beta_1)
        self.lambda_1 = float(lambda_1)
        self.exp_s = float(exp_s)

        # PHENOM: binning
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.mu3 = float(mu3)
        self.mu4 = float(mu4)
        self.z_div = float(z_div)
        self.z_TGR = float(z_TGR)
        self.z_tw = float(z_tw)
        self.scale_bins = bool(scale_bins)
        self.k_TGR = float(k_TGR)
        self.k_S = float(k_S)
        self.k_c = float(k_c)
        self.k_tw = float(k_tw)

        # PHENOM: growth index
        self.gamma_0 = float(gamma_0)
        self.gamma_a = float(gamma_a)
        self.t_k = float(t_k)
        self.d_s = float(d_s)

        # EFTCAMB_HORNDESKI: 1D interpolators eta -> h1/h3/h5
        self.eftcamb_h1_interp = eftcamb_h1_interp
        self.eftcamb_h3_interp = eftcamb_h3_interp
        self.eftcamb_h5_interp = eftcamb_h5_interp

    def _isitgr_k_windows(self, k):
        # Mirrors Fortran ISiTGR_k_windows
        t1 = np.tanh((k - self.k_TGR) / self.k_tw)
        t2 = np.tanh((k - self.k_c)   / self.k_tw)
        t3 = np.tanh((k - self.k_S)   / self.k_tw)

        W1 = 0.5 * (1.0 - t1)
        W2 = 0.5 * (t1 - t2)
        W3 = 0.5 * (t2 - t3)
        W4 = 0.5 * (1.0 + t3)
        return W1, W2, W3, W4

    def _mu_Z1(self, k):
        W1, W2, W3, W4 = self._isitgr_k_windows(k)
        # low-z: GR, mu1, mu2, GR
        return 1.0 * W1 + self.mu1 * W2 + self.mu2 * W3 + 1.0 * W4

    def _mu_Z2(self, k):
        W1, W2, W3, W4 = self._isitgr_k_windows(k)
        # high-z: GR, mu3, mu4, GR
        return 1.0 * W1 + self.mu3 * W2 + self.mu4 * W3 + 1.0 * W4

    def mu(self, eta: Union[float, Float64NDArray], k: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute scale-dependent modification to the Poisson equation μ(k, η).

        The μ function quantifies how the gravitational force is modified in modified gravity.

        Supports:
          - model='LCDM' (or 'GR'): μ = 1
          - model='HS'            : Hu-Sawicki f(R)
          - model='NDGP'          : nDGP braneworld
          - model='HDKI'          : Horndeski-like, with mg_variant in {'mu_OmDE', 'BZ'}
          - model='PHENOM'        : phenomenological parameterizations, with
                                    mg_variant in {'binning', 'growth_index', 'growth_index_yukawa'}

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.
        k : float or array_like
            Comoving wavenumber in units of h/Mpc.

        Returns
        -------
        float or ndarray
            Scale-dependent growth modification μ(k, η).

        Notes
        -----
        Implements the model-dependent μ(k, η) used by the fkPT ODE system.
        """

        k2 = np.square(k)
        model = getattr(self, "model", "HS").upper()

        # ------------------------------------------------------------
        # LCDM / GR
        # ------------------------------------------------------------
        if model in ("LCDM", "GR"):
            return 1.0

        # ------------------------------------------------------------
        # HS (Hu-Sawicki): mu = 1 + 2*beta2*k^2/(k^2 + a^2 m(eta)^2)
        # ------------------------------------------------------------
        if model == "HS":
            if self.fR0_HS == 0.0:
                return 1.0

            a = np.exp(eta)
            invH0 = self.invH0

            # Present-day DE density parameter
            omega_de0 = self.ol

            # CPL background DE contribution Y(a)
            # Defaults to LCDM if w0=-1 and wa=0
            w0 = getattr(self, "w0", -1.0)
            wa = getattr(self, "wa", 0.0)

            Y_a = omega_de0 * a**(-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
            Y_0 = omega_de0  # because at a=1, Y(a)=Omega_DE,0

            m = (
                (1.0 / invH0)
                * np.sqrt(1.0 / (2.0 * np.abs(self.fR0_HS)))
                * np.power(self.om * a**(-3.0) + 4.0 * Y_a, (2.0 + self.n_HS) / 2.0)
                / np.power(self.om + 4.0 * Y_0, (1.0 + self.n_HS) / 2.0)
            )

            return 1.0 + 2.0 * self.beta2 * k2 / (k2 + (a * m) ** 2)

        # ------------------------------------------------------------
        # nDGP: mu(a) = 1 + 1/(3 beta(a))
        # beta(a) = 1 + 2 E(a) r_c [1 - 0.5 Omega_m(a)]
        # using a flat LCDM background with (om, ol)
        # ------------------------------------------------------------
        if model == "NDGP":
            a = np.exp(eta)
            Ea2 = self.om * np.power(a, -3.0) + self.ol
            E = np.sqrt(Ea2)
            Om = (self.om * np.power(a, -3.0)) / Ea2

            beta = 1.0 + 2.0 * E * self.r_c * (1.0 - 0.5 * Om)
            return 1.0 + 1.0 / (3.0 * beta)

        # ------------------------------------------------------------
        # HDKI-like parameterizations
        #   - mu_OmDE: 1 + mu0 * Omega_DE(a)/Omega_Lambda
        #   - BZ:      (1 + beta1 * x) / (1 + x), x = lambda1^2 a^s k^2
        #   - EFT_DE:  EFTCAMB Horndeski μ(k,eta) from h1/h3/h5 interpolators
        # ------------------------------------------------------------
        if model == "HDKI":
            v = str(getattr(self, "mg_variant", "mu_OmDE")).strip().lower()
            a = np.exp(eta)

            if v in ("mu_omde", "muomde"):
                OmDE_over_OmL = 1.0 / (self.ol + self.om * np.power(a, -3.0))
                return 1.0 + self.mu0 * OmDE_over_OmL

            if v == "bz":
                x = np.power(self.lambda_1, 2.0) * k2 * np.power(a, self.exp_s)
                return (1.0 + self.beta_1 * x) / (1.0 + x)

            if v in ("eft_de", "eftde", "eftcamb_horndeski", "horndeski"):
                if (
                    self.eftcamb_h1_interp is None
                    or self.eftcamb_h3_interp is None
                    or self.eftcamb_h5_interp is None
                ):
                    return 1.0   # GR fallback

                h1 = self.eftcamb_h1_interp(eta)
                h3 = self.eftcamb_h3_interp(eta)
                h5 = self.eftcamb_h5_interp(eta)

                return h1 * (1.0 + k2 * h5) / (1.0 + k2 * h3)

            raise ValueError(
                f"Unknown HDKI mg_variant={v!r} "
                "(expected 'mu_OmDE', 'BZ', or 'EFT_DE')"
            )

        # ------------------------------------------------------------
        # Phenomenological parameterizations
        #   - binning
        #   - growth_index
        #   - growth_index_yukawa
        # ------------------------------------------------------------
        if model == "PHENOM":
            v = str(getattr(self, "mg_variant", "binning")).strip().lower()
            a = np.exp(eta)

            if v == "binning":
                z = 1.0 / a - 1.0

                Tz_div = np.tanh((z - self.z_div) / self.z_tw)
                Tz_TGR = np.tanh((z - self.z_TGR) / self.z_tw)

                if self.scale_bins:
                    mu_z1 = self._mu_Z1(k)
                    mu_z2 = self._mu_Z2(k)

                    return 0.5 * (
                        1.0 + mu_z1
                        + (mu_z2 - mu_z1) * Tz_div
                        + (1.0 - mu_z2) * Tz_TGR
                    )

                z1 = 1.0 * self.z_TGR / 4.0
                z2 = 2.0 * self.z_TGR / 4.0
                z3 = 3.0 * self.z_TGR / 4.0
                z4 = 4.0 * self.z_TGR / 4.0

                T1 = np.tanh((z - z1) / self.z_tw)
                T2 = np.tanh((z - z2) / self.z_tw)
                T3 = np.tanh((z - z3) / self.z_tw)
                T4 = np.tanh((z - z4) / self.z_tw)

                return (
                    0.5 * (1.0 + self.mu1)
                    + 0.5 * (self.mu2 - self.mu1) * T1
                    + 0.5 * (self.mu3 - self.mu2) * T2
                    + 0.5 * (self.mu4 - self.mu3) * T3
                    + 0.5 * (1.0 - self.mu4) * T4
                )

            if v == "growth_index":
                Ea2 = self.om * a**(-3.0) + self.ol
                Om = (self.om * a**(-3.0)) / Ea2

                gamma = self.gamma_0 + self.gamma_a * (1.0 - a)
                gammap = -self.gamma_a * a
                logOm = np.log(Om)

                mu_gi = (2.0 / 3.0) * Om**(gamma - 1.0) * (
                    Om**gamma
                    + (2.0 - 3.0 * gamma)
                    + 3.0 * (gamma - 0.5) * Om
                    + gammap * logOm
                )

                mu_gi_pivot = (2.0 / 3.0) * Om**(0.545454 - 1.0) * (
                    Om**0.545454
                    + (2.0 - 3.0 * 0.545454)
                    + 3.0 * (0.545454 - 0.5) * Om
                )

                aH_over_c = a * np.sqrt(Ea2) / self.invH0

                ds = self.d_s
                tk = self.t_k

                if (ds is None) or (tk is None) or (ds <= 0.0) or (tk <= 0.0):
                    return mu_gi

                arg = (k - tk * aH_over_c) / ds
                Fk = 0.5 * (1.0 + np.tanh(arg))

                return mu_gi_pivot + (mu_gi - mu_gi_pivot) * Fk

            if v == "growth_index_yukawa":
                Ea2 = self.om * a**(-3.0) + self.ol
                Om = (self.om * a**(-3.0)) / Ea2

                gamma = self.gamma_0 + self.gamma_a * (1.0 - a)
                gammap = -self.gamma_a * a
                logOm = np.log(Om)

                mu_gi = (2.0 / 3.0) * Om**(gamma - 1.0) * (
                    Om**gamma
                    + (2.0 - 3.0 * gamma)
                    + 3.0 * (gamma - 0.5) * Om
                    + gammap * logOm
                )

                mu_gi_pivot = (2.0 / 3.0) * Om**(0.545454 - 1.0) * (
                    Om**0.545454
                    + (2.0 - 3.0 * 0.545454)
                    + 3.0 * (0.545454 - 0.5) * Om
                )

                aH_over_c = a * np.sqrt(Ea2) / self.invH0

                n = self.d_s
                alpha = self.t_k

                if (n is None) or (alpha is None) or (n <= 0.0) or (alpha <= 0.0):
                    return mu_gi

                kc2 = (alpha * aH_over_c) ** 2
                kk2 = k * k
                Fk = (kk2 / (kk2 + kc2)) ** n

                return mu_gi_pivot + (mu_gi - mu_gi_pivot) * Fk

            raise ValueError(f"Unknown PHENOM mg_variant={v!r}")
            
        raise ValueError(
            f"Unknown model={model!r} "
            "(expected 'LCDM'/'GR', 'HS', 'NDGP', 'HDKI', or 'PHENOM')"
        )

    def f1(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute logarithmic growth rate f₁(η) = d ln D/d ln a.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.

        Returns
        -------
        float or ndarray
            Linear growth rate f₁(η) ≈ Ωₘ(η)^0.55 for ΛCDM.

        Notes
        -----
        Implements f1_HS from csrc/models.c. For ΛCDM, f₁ = 3Ωₘ/(2Ωₘ + 2Ωₗa³).
        """
        return 3 / (2 * (1 + self.ol / self.om * np.exp(3 * eta)))

    def kpp(self, x: Union[float, Float64NDArray], k: Union[float, Float64NDArray], p: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute magnitude of vector sum |k + p| given k, p, and cosine x = k·p/(kp).

        Parameters
        ----------
        x : float or array_like
            Cosine of angle between k and p: x = k·p/(kp).
        k : float or array_like
            Magnitude of wavenumber k in h/Mpc.
        p : float or array_like
            Magnitude of wavenumber p in h/Mpc.

        Returns
        -------
        float or ndarray
            Magnitude |k + p| in h/Mpc.
        """
        return np.sqrt(np.square(k) + np.square(p) + 2 * k * p * x)

    def S2a(self, eta: float, x: float, k: float, p: float) -> float:
        """Compute symmetric second-order source term S2a.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.

        Returns
        -------
        float
            S2a(η, x, k, p) = f₁μ(|k+p|).

        Notes
        -----
        Implements S2a_HS from csrc/models.c. Similar to source_a but for k+p mode.
        """
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * self.mu(eta, kplusp)  # type: ignore[return-value]

    def S2b(self, eta: float, x: float, k: float, p: float) -> float:
        """Compute symmetric second-order source term S2b.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.

        Returns
        -------
        float
            S2b(η, x, k, p) differential μ response.

        Notes
        -----
        Implements S2b_HS from csrc/models.c. Similar to source_b but for k+p mode.
        """
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * (self.mu(eta, k) + self.mu(eta, p) - self.mu(eta, kplusp))  # type: ignore[return-value]

    def S2FL(self, eta: float, x: float, k: float, p: float) -> float:
        """Compute symmetric frame-lagging source S2FL for third-order.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.

        Returns
        -------
        float
            S2FL frame-lagging contribution.

        Notes
        -----
        Implements S2FL_HS from csrc/models.c.
        """
        kp = self.kpp(x, k, p)
        f1v = self.f1(eta)

        mu_k  = self.mu(eta, k)
        mu_p  = self.mu(eta, p)
        mu_kp = self.mu(eta, kp)

        r  = p / k
        ri = k / p

        return f1v * (mu_kp * (r + ri) * x - ri * x * mu_k - r * x * mu_p)

    def SD2(self, eta: float, x: float, k: float, p: float) -> float:
        """Compute total symmetric second-order source SD2 for third-order kernels.

        Combines all second-order source contributions for the symmetric third-order
        kernel computation.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.

        Returns
        -------
        float
            SD2 = S2a - S2b·x² + S2FL - S2dI.

        Notes
        -----
        Implements SD2_HS from csrc/models.c. Used in third-order ODE source terms.
        """
        return self.S2a(eta, x, k, p) - self.S2b(eta, x, k, p) * np.square(x) + self.S2FL(eta, x, k, p)

    def S3IIplus(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float) -> float:
        """Compute S3II+ contribution for k+p mode in third-order source.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.
        Dpk : float
            First-order density perturbation D(k, η).
        Dpp : float
            First-order density perturbation D(p, η).
        D2f : float
            Second-order perturbation for k+p mode.

        Returns
        -------
        float
            S3IIplus contribution.

        Notes
        -----
        Implements S3IIplus_HS from csrc/models.c.
        """
        kplusp = self.kpp(x, k, p)

        f1 = self.f1(eta)
        mu_k  = self.mu(eta, k)
        mu_p  = self.mu(eta, p)
        mu_kp = self.mu(eta, kplusp)
        return (
            -f1 * (mu_p + mu_kp - 2.0 * mu_k) * Dpp * (D2f + Dpk * Dpp * x * x)
            -f1 * (mu_kp - mu_k + mu_kp*(p/k+k/p)*x - k*x/p*mu_k - p*x/k*mu_p) * Dpk * Dpp * Dpp
            #-f1 * (mu_kp - mu_k) * Dpk * Dpp * Dpp
        )


    def S3FLplus(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float) -> float:
        """Compute S3FL+ frame-lagging contribution for k+p mode.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.
        Dpk : float
            First-order density perturbation D(k, η).
        Dpp : float
            First-order density perturbation D(p, η).
        D2f : float
            Second-order perturbation for k+p mode.

        Returns
        -------
        float
            S3FLplus frame-lagging contribution.

        Notes
        -----
        Implements S3FLplus_HS from csrc/models.c.
        """
        k2 = k * k
        p2 = p * p
        pk = p * k

        # Guard against pathological zero division (shouldn't happen in practice if k,p>0)
        if p2 == 0.0 or pk == 0.0:
            return 0.0

        denom = k2 + p2 + 2.0 * pk * x  # = |k+p|^2
        if denom == 0.0:
            return 0.0

        kplusp = float(self.kpp(x, k, p))

        mu_k  = float(self.mu(eta, k))
        mu_p  = float(self.mu(eta, p))
        mu_kp = float(self.mu(eta, kplusp))

        # --- coefficients appearing in the screenshot ---
        c1 = (p2 + pk * x) / denom
        c2 = (p2 + pk * x) / p2
        c3 = (p2 + k2) * (x * x / p2 + x / pk)

        # --- pieces (exactly as in screenshot) ---
        term1 = c1 * (mu_p - mu_k) * (D2f * Dpp)

        term2 = c2 * (mu_kp - mu_k) * (
            (D2f * Dpp) + (1.0 + x * x) * (Dpk * Dpp * Dpp)
        )

        term3 = c3 * (mu_kp - mu_k) * (Dpk * Dpp * Dpp)

        return self.f1(eta) * (term1 + term2 + term3)
# This function is only causing the small differences:
#        kplusp = self.kpp(x, k, p)
#        return self.f1(eta) * (self.M1(eta) / (3.0 * self.PiF(eta, k))) * (
#            (2.0 * np.square(p + k * x) / np.square(kplusp) - 1.0 - (k * x) / p)
#            * (self.mu(eta, p) - 1.0) * D2f * Dpp
#            + ((np.square(p) + 3.0 * k * p * x + 2.0 * k * k * x * x) / np.square(kplusp))
#            * (self.mu(eta, kplusp) - 1.0) * self.D2phiplus(eta, x, k, p, Dpk, Dpp, D2f) * Dpp
#            + 3.0 * np.square(x) * (self.mu(eta, k) + self.mu(eta, p) - 2.0) * Dpk * Dpp * Dpp
#        )

    # Main third order source functions
    def S3I(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float, D2mf: float) -> float:
        """Compute third-order source S3I (Type I kernel contribution).

        This source combines both k+p and k-p modes with angular factors,
        representing the leading contribution to third-order density kernels.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.
        Dpk : float
            First-order density perturbation D(k, η).
        Dpp : float
            First-order density perturbation D(p, η).
        D2f : float
            Second-order perturbation for k+p mode.
        D2mf : float
            Second-order perturbation for k-p mode.

        Returns
        -------
        float
            S3I third-order source term.

        Notes
        -----
        Implements S3I_HS from csrc/models.c. Contains (1-x²) angular factors.
        """
        kplusp = self.kpp(x, k, p)
        kpluspm = self.kpp(-x, k, p)
        return (
            (
                self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kplusp) - self.mu(eta, k)) * D2f * Dpp
                + self.SD2(eta, x, k, p) * Dpk * Dpp * Dpp
            ) * (1.0 - np.square(x)) / (1.0 + np.square(p / k) + 2.0 * (p / k) * x)
            + (
                self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kpluspm) - self.mu(eta, k)) * D2mf * Dpp
                + self.SD2(eta, -x, k, p) * Dpk * Dpp * Dpp
            ) * (1.0 - np.square(x)) / (1.0 + np.square(p / k) - 2.0 * (p / k) * x)
        )

    def S3II(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float, D2mf: float) -> float:
        """Compute third-order source S3II (Type II kernel contribution).

        Combines S3IIplus and S3IIminus contributions from both k±p modes.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.
        Dpk : float
            First-order density perturbation D(k, η).
        Dpp : float
            First-order density perturbation D(p, η).
        D2f : float
            Second-order perturbation for k+p mode.
        D2mf : float
            Second-order perturbation for k-p mode.

        Returns
        -------
        float
            S3II = S3IIplus + S3IIminus.

        Notes
        -----
        Implements S3II_HS from csrc/models.c.
        """
        return self.S3IIplus(eta, x, k, p, Dpk, Dpp, D2f) + self.S3IIplus(eta, -x, k, p, Dpk, Dpp, D2mf)

    def S3FL(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float, D2mf: float) -> float:
        """Compute third-order frame-lagging source S3FL.

        Combines frame-lagging contributions from both k±p modes.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p.
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.
        Dpk : float
            First-order density perturbation D(k, η).
        Dpp : float
            First-order density perturbation D(p, η).
        D2f : float
            Second-order perturbation for k+p mode.
        D2mf : float
            Second-order perturbation for k-p mode.

        Returns
        -------
        float
            S3FL = S3FLplus + S3FLminus.

        Notes
        -----
        Implements S3FL_HS from csrc/models.c.
        """
        #print("debugging:")
        #print(self.S3FLplus(eta, x, k, p, Dpk, Dpp, D2f))
        return self.S3FLplus(eta, x, k, p, Dpk, Dpp, D2f) + self.S3FLplus(eta, -x, k, p, Dpk, Dpp, D2mf)

    def firstOrder(self, x: float, y: Float64NDArray, k: Union[float, Float64NDArray]) -> Float64NDArray:
        """Compute ODE derivatives for first-order perturbation growth.

        Solves the coupled system for D(k, η) and D'(k, η) = dD/dη where
        D is the linear density perturbation growth factor.

        Parameters
        ----------
        x : float
            Conformal time η = ln(a).
        y : ndarray
            State vector [D, D'] of length 2.
        k : float or 1D numpy array of floats
            Comoving wavenumber in h/Mpc.

        Returns
        -------
        ndarray
            Derivatives [D', D''] of length 2.

        Notes
        -----
        Implements derivsFirstOrder from csrc/gsm_diffeqs.c.
        The second-order ODE is: D'' + (2-f₁)D' - f₁μ(k)D = 0.
        """
        f1x = self.f1(x)
        return np.array([y[1], f1x * self.mu(x, k) * y[0] - (2 - f1x) * y[1]])

    def secondOrder(self, eta: float, y: Float64NDArray, x: float, k: float, p: float) -> Float64NDArray:
        """
        Notebook-style 2nd order system:
          y = [Dk, Dk', Dp, Dp', D2, D2']
        where D2 corresponds to the k_f = |k+p| mode and depends on angle x = cos(theta).
        """
        f2 = self.f1(eta)       # in your code: f1(eta) = 3/2 * Omega_m = notebook's f2
        f1 = 2.0 - f2           # friction term = 2 + H'/H = 2 - 3/2 Omega_m = notebook's f1

        kf = float(self.kpp(x, k, p))
        src = self.SD2(eta, x, k, p)   # = S2a - x^2 S2b + S2FL (matches notebook)

        return np.array([
            y[1],
            f2 * self.mu(eta, k)  * y[0] - f1 * y[1],

            y[3],
            f2 * self.mu(eta, p)  * y[2] - f1 * y[3],

            y[5],
            f2 * self.mu(eta, kf) * y[4] - f1 * y[5] + src * y[0] * y[2],
        ])

    def thirdOrder(self, eta: float, y: Float64NDArray, x: float, k: float, p: float) -> Float64NDArray:
        """Compute ODE derivatives for third-order kernel calculation.

        Solves coupled ODEs for two input modes (k, p) at angle x = k·p/(kp),
        two second-order modes (k+p and k-p), and one third-order mode.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        y : ndarray
            State vector [Dₖ, Dₖ', Dₚ, Dₚ', D₂₊, D₂₊', D₂₋, D₂₋', D₃, D₃'] of length 10 where:
            - (Dₖ, Dₖ'): First-order growth for k
            - (Dₚ, Dₚ'): First-order growth for p
            - (D₂₊, D₂₊'): Second-order for k+p mode
            - (D₂₋, D₂₋'): Second-order for k-p mode
            - (D₃, D₃'): Third-order symmetric kernel
        x : float
            Cosine of angle between k and p: x = k·p/(kp).
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.

        Returns
        -------
        ndarray
            Derivatives [Dₖ', Dₖ'', Dₚ', Dₚ'', D₂₊', D₂₊'', D₂₋', D₂₋'', D₃', D₃''] of length 10.

        Notes
        -----
        Implements derivsThirdOrder from csrc/gsm_diffeqs.c.
        Third-order source combines S3I, S3II, S3FL, and S3dI terms.
        """
        f1eta = self.f1(eta)
        f2eta = 2 - f1eta
        kplusp = self.kpp(x, k, p)
        kpluspm = self.kpp(-x, k, p)
        Dpk = y[0]
        Dpp = y[2]
        D2f = y[4]
        D2mf = y[6]
        return np.array([
            y[1], f1eta * self.mu(eta, k) * y[0] - f2eta * y[1],
            y[3], f1eta * self.mu(eta, p) * y[2] - f2eta * y[3],
            y[5], f1eta * self.mu(eta, kplusp) * y[4] - f2eta * y[5] + self.SD2(eta, x, k, p) * y[0] * y[2],
            y[7], f1eta * self.mu(eta, kpluspm) * y[6] - f2eta * y[7] + self.SD2(eta, -x, k, p) * y[0] * y[2],
            y[9], f1eta * self.mu(eta, k) * y[8] - f2eta * y[9]
                + self.S3I(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
                + self.S3II(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
                + self.S3FL(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
        ])

class ODESolver:
    """ODE solver class to integrate ODEs from xnow to xstop.

    Use RKQS for compatibility the original FKPT C code, or scipy_ivp to use
    scipy's built-in ODE solver."""
    def __init__(self, zout: float, xnow: float = -4, method: str = 'RKQS') -> None:
        self.xstop: float = np.log(1.0/(1.0+zout))
        self.xnow: float = xnow
        if method not in ['RKQS', 'scipy_ivp']:
            raise ValueError(f"Unknown ODE solver method: {method}")
        self.method: str = method

    def __call__(self, dydx: Callable[[float, Float64NDArray], Float64NDArray], y0: Float64NDArray) -> Float64NDArray:
        if self.method == 'scipy_ivp':
            soln = scipy.integrate.solve_ivp(dydx, (self.xnow, self.xstop), y0)
            return soln.y[:, -1]
        else:
            soln = odeint(y0, self.xnow, self.xstop, dydx)
            return soln[0]

def DP(k: Union[float, Float64NDArray], derivs, solver):
    """Integrate first-order growth ODE to get D(k, η) and D'(k, η).

    Parameters
    ----------
    k : float or 1D numpy array of floats
        Comoving wavenumber in h/Mpc.
    derivs : ModelDerivatives
        Model instance providing ODE derivative functions.
    solver : ODESolver
        ODE solver configured with initial time xnow and final time xstop.

    Returns
    -------
    ndarray
        Solution [D(k, xstop), D'(k, xstop)] at final time.

    Notes
    -----
    Initial conditions at xnow: D(k, xnow) = D'(k, xnow) = exp(xnow).
    This normalization ensures D ∝ a in matter domination.
    """
    # Normalize k to a 1D array for uniform handling
    k_array = np.atleast_1d(k).astype(float)
    nk = k_array.size

    # Initial conditions: D = D' = exp(xnow)
    y0_flat = np.exp(solver.xnow) * np.ones(2 * nk, dtype=float)

    # RHS for the batched system
    def rhs(x: float, y_flat: Float64NDArray) -> Float64NDArray:
        Y = y_flat.reshape(2, nk)
        dYdx = derivs.firstOrder(x, Y, k_array)
        return np.ravel(dYdx)

    # Integrate
    y_flat = solver(rhs, y0_flat)

    # Reshape final result to (2, nk)
    Y = np.reshape(y_flat, (2, nk))

    # Return shape (2,) for scalar k
    return Y[:, 0] if np.isscalar(k) else Y

def growth_factor(k: Union[float, Float64NDArray], derivs, solver):
    """Compute logarithmic growth rate f(k, η) = d ln D / d ln a.

    Parameters
    ----------
    k : float or 1D numpy array of floats
        Comoving wavenumber in h/Mpc.
    derivs : ModelDerivatives
        Model instance providing ODE derivative functions.
    solver : ODESolver
        ODE solver configured with initial and final times.

    Returns
    -------
    float or 1D numpy array of floats
        Growth rate f(k) = D'(k)/D(k) at final time.

    Notes
    -----
    For ΛCDM, f ≈ Ωₘ^0.55. In modified gravity, f becomes scale-dependent
    through μ(k, η).
    """
    y = DP(k, derivs, solver)
    return y[1] / y[0]

def D2v2(x: float, k: float, p: float, derivs: ModelDerivatives, solver: ODESolver) -> Float64NDArray:
    """
    Returns [Dk, Dk', Dp, Dp', D2, D2'] at xstop, where D2 is for k_f = |k+p|.
    """
    y0 = np.empty(6)
    y0[:4] = np.exp(solver.xnow)  # Dk, Dk', Dp, Dp' = a in EdS

    # EdS IC used in the notebook:
    # D2 = (3/7) a^2 (1 - x^2), and D2' = 2 D2 because d/deta(a^2)=2a^2
    y0[4] = (3.0/7.0) * np.exp(2.0 * solver.xnow) * (1.0 - x*x)
    y0[5] = 2.0 * y0[4]

    return solver(lambda eta, y: derivs.secondOrder(eta, y, x, k, p), y0)

def D3v2(x: float, k: float, p: float, derivs: ModelDerivatives, solver: ODESolver) -> Float64NDArray:
    """Integrate third-order kernel ODE for modes k and p at angle x.

    Computes third-order symmetric kernel from two input modes k and p
    separated by angle cos⁻¹(x).

    Parameters
    ----------
    x : float
        Cosine of angle between k and p: x = k·p/(kp).
    k : float
        Wavenumber k in h/Mpc.
    p : float
        Wavenumber p in h/Mpc.
    derivs : ModelDerivatives
        Model instance providing ODE derivative functions.
    solver : ODESolver
        ODE solver configured with initial and final times.

    Returns
    -------
    ndarray
        Solution [Dₖ, Dₖ', Dₚ, Dₚ', D₂₊, D₂₊', D₂₋, D₂₋', D₃, D₃'] at final time.

    Notes
    -----
    Initial conditions at xnow for ΛCDM-like EdS universe:
    - First-order: Dₖ = Dₚ = exp(xnow), Dₖ' = Dₚ' = exp(xnow)
    - Second-order: D₂± = (3/7)exp(2*xnow)(1-x²), D₂±' = (6/7)exp(2*xnow)(1-x²)
    - Third-order: D₃ = (5/63)exp(3*xnow)(1-x²)² [sum over ±],
                   D₃' = (15/63)exp(3*xnow)(1-x²)² [sum over ±]
    """
    y0 = np.empty(10)
    y0[:4] = np.exp(solver.xnow)
    y0[4:8] = 3.0 * np.exp(2.0 * solver.xnow) / 7.0 * (1.0 - np.square(x))
    y0[5:8:2] *= 2.0
    y0[8] = (5.0 / (7.0 * 9.0)) * np.exp(3.0 * solver.xnow) * np.square(1.0 - np.square(x)) * (
        1.0 / (1.0 + np.square(p / k) + 2.0 * (p / k) * x)
        + 1.0 / (1.0 + np.square(p / k) - 2.0 * (p / k) * x)
    )
    y0[9] = (15.0 / (7.0 * 9.0)) * np.exp(3.0 * solver.xnow) * np.square(1.0 - np.square(x)) * (
        1.0 / (1.0 + np.square(p / k) + 2.0 * (p / k) * x)
        + 1.0 / (1.0 + np.square(p / k) - 2.0 * (p / k) * x)
    )
    return solver(lambda eta, y: derivs.thirdOrder(eta, y, x, k, p), y0)

def kernel_constants(
    f0: float,
    derivs: ModelDerivatives,
    solver: ODESolver,
    KMIN: float = 1e-8,
    x: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Compute one-loop kernel constants (ALS, AprimeLS, KR1LS, KR1pLS).

    For ordinary models, use the standard ODE-based notebook definitions.

    For EFTCAMB Horndeski models encoded as:
        model = "HDKI"
        mg_variant = "EFT_DE"

    use h1(eta) for the large-scale A kernel and dh1/deta for A',
    while still using the ODE system for KR1LS and KR1pLS.
    """
    model_u = str(getattr(derivs, "model", "")).upper()
    variant = str(getattr(derivs, "mg_variant", "")).strip().lower()

    is_eft_de = (
        model_u == "HDKI"
        and variant in ("eft_de", "eftde", "eftcamb_horndeski", "horndeski")
    )

    Dk, dDk, Dp, dDp, D2p, dD2p, D2m, dD2m, D3, dD3 = D3v2(
        x, KMIN, KMIN, derivs, solver
    )

    KR1LS = (21.0 / 5.0) * D3 / (Dk * Dp * Dp)
    KR1pLS = (21.0 / 5.0) * dD3 / (Dk * Dp * Dp) / (3.0 * f0)

    if is_eft_de:
        eta_out = solver.xstop
        h1_interp = getattr(derivs, "eftcamb_h1_interp", None)

        if h1_interp is None:
            ALS = 1.0
            AprimeLS = 0.0
        else:
            ALS = float(h1_interp(eta_out))
            deps = 1e-4
            AprimeLS = float(
                (h1_interp(eta_out + deps) - h1_interp(eta_out - deps))
                / (2.0 * deps)
            )

        return ALS, AprimeLS, KR1LS, KR1pLS

    C = (3.0 / 7.0) * Dk * Dp
    Cp = (3.0 / 7.0) * (dDk * Dp + Dk * dDp)

    ALS = D2p / C
    AprimeLS = dD2p / C - D2p * Cp / (C * C)

    return ALS, AprimeLS, KR1LS, KR1pLS