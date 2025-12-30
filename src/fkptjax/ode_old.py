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
        # --- HS / f(R) parameters
        fR0: float = 0.0,
        beta2: float = 1.0/6.0,
        nHS: int = 1,
        screening: int = 1,
        omegaBD: float = 0.0,
        # --- model switches
        model: str = "HS",
        mg_variant: str = "mu_OmDE",
        # --- HDKI: mu_OmDE
        mu0: float = 0.0,
        # --- HDKI: BZ-like
        beta_1: float = 1.0,
        lambda_1: float = 0.0,
        exp_s: float = 0.0,
        # --- HDKI: binning (defaults correspond to GR-ish)
        mu1: float = 0.0,
        mu2: float = 0.0,
        mu3: float = 0.0,
        mu4: float = 0.0,
        k_c: float = 0.1,
        k_tw: float = 0.01,
        z_div: float = 1.0,
        z_TGR: float = 10.0,
        z_tw: float = 0.5,
    ) -> None:
        """Initialize the Hu-Sawicki f(R) model parameters.

        Parameters
        ----------
        om : float
            Matter density parameter Ωₘ at present epoch (z=0).
        ol : float
            Dark energy density parameter Ωₗ at present epoch (z=0).
        fR0 : float
            Present-day value of the f(R) modification parameter |f_R0|.
            Typically negative, controls the strength of the fifth force.
        beta2 : float, optional
            Coupling strength parameter β² = 1/(3(1 + 4*λ²)), default is 1/6.
            For Hu-Sawicki model, β² = 1/6 corresponds to conformal coupling.
        nHS : int, optional
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

        # HS / f(R)
        self.fR0 = float(fR0)
        self.beta2 = float(beta2)
        self.nHS = int(nHS)
        self.screening = int(screening)
        self.omegaBD = float(omegaBD)

        # switches
        self.model = str(model)
        self.mg_variant = str(mg_variant)

        # HDKI: mu_OmDE
        self.mu0 = float(mu0)

        # HDKI: BZ-like
        self.beta_1 = float(beta_1)
        self.lambda_1 = float(lambda_1)
        self.exp_s = float(exp_s)

        # HDKI: binning
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.mu3 = float(mu3)
        self.mu4 = float(mu4)
        self.k_c = float(k_c)
        self.k_tw = float(k_tw)
        self.z_div = float(z_div)
        self.z_TGR = float(z_TGR)
        self.z_tw = float(z_tw)


    def mass(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute effective mass of the scalar field (scalaron) in f(R) gravity.

        The mass determines the Compton wavelength λ = 1/m of the fifth force,
        controlling the transition between screened and unscreened regimes.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.

        Returns
        -------
        float or ndarray
            Effective mass m(η) in units of h/Mpc.

        Notes
        -----
        Implements mass_HS from csrc/models.c. The mass evolves with redshift
        according to the Hu-Sawicki model power law with index nHS.
        """
        model = getattr(self, "model", "HS").upper() # in case people select e.g. 'hdki' instead of 'HDKI' (so we force capital letters)

        # ------------------------------------------------------------
        # LCDM (GR): mu = 1
        # ------------------------------------------------------------
        if model in ("LCDM", "GR"):
            return 1e30

        # ------------------------------------------------------------
        # HS (Hu-Sawicki): mu = 1 + 2*beta2*k^2/(k^2 + a^2 m(eta)^2)
        # ------------------------------------------------------------
        if model == "HS":
            return (1 / self.invH0 * np.sqrt(1 / (2 * np.abs(self.fR0)))
            * np.power(self.om * np.exp(-3 * eta) + 4 * self.ol, (2 + self.nHS) / 2)
            / np.power(self.om + 4 * self.ol, (1 + self.nHS) / 2)
            )

        # ------------------------------------------------------------
        # HDKI: mu depends on mg_variant (matches models.c)
        #   - mu_OmDE:  1 + mu0 * Omega_DE(a)
        #   - BZ:       (1 + beta1 * x) / (1 + x), x = lambda1^2 a^s k^2
        #   - binning:  tanh transitions in k and z
        # ------------------------------------------------------------
        if model == "HDKI":
            small = 1.0e-60
            a = np.exp(eta)
            v = getattr(self, "mg_variant", "mu_OmDE")

            if v == "mu_OmDE":
                # if you truly want h3=0 for this variant
                return 1e30

            elif v == "BZ":
                # to match mu_BZ exactly in the ratio form: h3 = lambda_1^2 * a^exp_s
                h3 = np.power(self.lambda_1, 2.0) * np.power(a, self.exp_s)
                return np.sqrt(1.0 / (h3 + small)) / a

            elif v == "binning":
                # mass is determined by the large-scale and small-scale limits of the binning methods.
                # from the large-scale (low-k) limit we get:
                #h1 = 
                ## from the small-scale (high-k) limit we get:
                #alphat = 
                ## now we substitute in the general formula (binning is constant through these limits):
                #mu = 
                return self.k_c/a

            else:
                raise ValueError(f"Unknown HDKI mg_variant={v!r} for mass()")

        raise ValueError(f"Unknown model={model!r} (expected 'LCDM'/'GR', 'HS', or 'HDKI')")


    def mu(self, eta: Union[float, Float64NDArray], k: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute scale-dependent modification to the Poisson equation μ(k, η).

        The μ function quantifies how the gravitational force is modified in modified gravity.

        Supports:
          - model='LCDM' (or 'GR'): μ = 1
          - model='HS'           :  Hu-Sawicki f(R) (mu -> 1 on small scales (screened) and 1+2β² on large scales (unscreened).)
          - model='HDKI'         :  Horndeski-like, with mg_variant in {'mu_OmDE','BZ','binning'}

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.
        k : float or array_like
            Comoving wavenumber in units of h/Mpc.

        Returns
        -------
        float or ndarray
            Scale-dependent growth modification μ(k, η) ≥ 1.

        Notes
        -----
        Implements mu_HS from csrc/models.c. The transition scale is k_screening ~ a*m(η).
        For k >> k_screening: μ → 1 (screened, GR recovered)
        For k << k_screening: μ → 1 + 2β² (unscreened, enhanced gravity)
        """

        k2 = np.square(k)
        model = getattr(self, "model", "HS").upper() # in case people select e.g. 'hdki' instead of 'HDKI' (so we force capital letters)

        # ------------------------------------------------------------
        # LCDM (GR): mu = 1
        # ------------------------------------------------------------
        if model in ("LCDM", "GR"):
            return 1.0

        # ------------------------------------------------------------
        # HS (Hu-Sawicki): mu = 1 + 2*beta2*k^2/(k^2 + a^2 m(eta)^2)
        # ------------------------------------------------------------
        if model == "HS":
            return 1.0 + 2.0 * self.beta2 * k2 / (k2 + np.exp(2.0 * eta) * np.square(self.mass(eta)))

        # ------------------------------------------------------------
        # HDKI: mu depends on mg_variant (matches models.c)
        #   - mu_OmDE:  1 + mu0 * Omega_DE(a)
        #   - BZ:       (1 + beta1 * x) / (1 + x), x = lambda1^2 a^s k^2
        #   - binning:  tanh transitions in k and z
        # ------------------------------------------------------------
        if model == "HDKI":
            v = getattr(self, "mg_variant", "mu_OmDE")
            a = np.exp(eta)

            if v == "mu_OmDE":
                OmDE_over_OmL = 1.0 / (self.ol + self.om * np.power(a, -3.0))
                return 1.0 + self.mu0 * OmDE_over_OmL

            if v == "BZ":
                x = np.power(self.lambda_1, 2.0) * k2 * np.power(a, self.exp_s)
                return (1.0 + self.beta_1 * x) / (1.0 + x)

            if v == "binning":
                z = 1.0 / a - 1.0
                Tk = np.tanh((k - self.k_c) / self.k_tw)
                mu_z1 = 0.5*(self.mu2 + self.mu1) + 0.5*(self.mu2 - self.mu1)*Tk
                mu_z2 = 0.5*(self.mu4 + self.mu3) + 0.5*(self.mu4 - self.mu3)*Tk
                Tz_div = np.tanh((z - self.z_div) / self.z_tw)
                Tz_TGR = np.tanh((z - self.z_TGR) / self.z_tw)
                return (
                    0.5*(1.0 + mu_z1)
                    + 0.5*(mu_z2 - mu_z1)*Tz_div
                    + 0.5*(1.0 - mu_z2)*Tz_TGR
                )

            raise ValueError(f"Unknown HDKI mg_variant={v!r}")

        raise ValueError(f"Unknown model={model!r} (expected 'LCDM'/'GR', 'HS', or 'HDKI')")

    def PiF(self, eta: Union[float, Float64NDArray], k: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute the scalar field propagator function Π_F(k, η).

        The propagator appears in the denominator of screening terms and determines
        how the scalar field responds to matter perturbations.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.
        k : float or array_like
            Comoving wavenumber in units of h/Mpc.

        Returns
        -------
        float or ndarray
            Propagator Π_F(k, η) = k²/a² + m²(η) in units of (h/Mpc)².

        Notes
        -----
        Implements PiF_HS from csrc/models.c. This is the inverse Green's function
        for the scalar field equation of motion.
        """
        return np.square(k) / np.exp(2 * eta) + np.square(self.mass(eta))

    def M2(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute second-order chameleon screening coefficient M₂(η).

        M₂ controls the strength of second-order screening effects in the
        differential interaction terms.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.

        Returns
        -------
        float or ndarray
            Screening coefficient M₂(η) in units of (h/Mpc)⁴.

        Notes
        -----
        Implements M2_HS from csrc/models.c. When screening=0, returns 0.
        M₂ scales as |f_R0|⁻² and increases toward the past.
        """

        model = getattr(self, "model", "HS").upper() # in case people select e.g. 'hdki' instead of 'HDKI' (so we force capital letters)

        if model in ("HS"):
            return self.screening * (
                9 / (4 * np.square(self.invH0)) * np.square(1 / np.abs(self.fR0))
                * np.power(self.om * np.exp(-3 * eta) + 4 * self.ol, 5)
                / np.power(self.om + 4 * self.ol, 4)
            )
        else:
            return 0.0

    def OmM(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute matter density parameter Ωₘ(η) as a function of conformal time.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.

        Returns
        -------
        float or ndarray
            Matter density parameter Ωₘ(η) = Ωₘ,₀/(Ωₘ,₀ + Ωₗ,₀a³).

        Notes
        -----
        Implements OmM_HS from csrc/models.c.
        """
        return 1 / (1 + self.ol / self.om * np.exp(3 * eta))

    def H(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute normalized Hubble parameter H(η)/H₀.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.

        Returns
        -------
        float or ndarray
            Normalized Hubble parameter H(η)/H₀ = √(Ωₘa⁻³ + Ωₗ).

        Notes
        -----
        Implements H_HS from csrc/models.c.
        """
        return np.sqrt(self.om * np.exp(-3 * eta) + self.ol)

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

    def A0(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute normalization coefficient A₀(η) = 3Ωₘ(η)H²(η)/(2(c/H₀)²).

        This appears in the linearized equation of motion for density perturbations.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a) where a is the scale factor.

        Returns
        -------
        float or ndarray
            Normalization A₀(η) in units of (h/Mpc)².

        Notes
        -----
        Implements A0_HS from csrc/models.c. This is the coefficient of δ in the
        continuity equation d²δ/dη² + (2-f₁)dδ/dη - (3/2)Ωₘμδ = 0.
        """
        return 1.5 * self.OmM(eta) * np.square(self.H(eta)) / np.square(self.invH0)

    def source_a(self, eta: float, kf: float) -> float:
        """Compute second-order source term 'a' for symmetric kernel.

        This is the leading-order contribution to the second-order growth equation,
        proportional to f₁μ(kf).

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        kf : float
            Final wavenumber |k₁ + k₂| in h/Mpc.

        Returns
        -------
        float
            Source term a(η, kf).

        Notes
        -----
        Implements sourceA_HS from csrc/models.c (the 'a' component).
        """
        return self.f1(eta) * self.mu(eta, kf)  # type: ignore[return-value]

    def source_b(self, eta: float, kf: float, k1: float, k2: float) -> float:
        """Compute second-order source term 'b' for velocity divergence.

        This term accounts for the differential response of the scalar field
        to the two incoming modes k₁ and k₂.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        kf : float
            Final wavenumber |k₁ + k₂| in h/Mpc.
        k1 : float
            First input wavenumber in h/Mpc.
        k2 : float
            Second input wavenumber in h/Mpc.

        Returns
        -------
        float
            Source term b(η, kf, k₁, k₂).

        Notes
        -----
        Implements sourceb_HS from csrc/models.c.
        """
        return self.f1(eta) * (self.mu(eta, k1) + self.mu(eta, k2) - self.mu(eta, kf))  # type: ignore[return-value]

    def KFL(self, eta: float, k: float, k1: float, k2: float) -> float:
        """Compute frame-lagging kernel KFL for second-order perturbations.

        The frame-lagging term arises from the time derivative of the Newtonian
        potential in the scalar field equation.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        k : float
            Output wavenumber in h/Mpc.
        k1 : float
            First input wavenumber in h/Mpc.
        k2 : float
            Second input wavenumber in h/Mpc.

        Returns
        -------
        float
            Frame-lagging kernel KFL(η, k, k₁, k₂).

        Notes
        -----
        Implements KFL_HS from csrc/models.c. Vanishes in GR (μ=1).
        """
        k2_  = np.square(k)
        k12  = np.square(k1)
        k22  = np.square(k2)
        num  = np.square(k2_ - k12 - k22)
        term0 = 0.5 * num / (k12 * k22) * (self.mu(eta, k1) + self.mu(eta, k2) - 2.0)
        term1 = 0.5 * (k2_ - k12 - k22) / k12 * (self.mu(eta, k1) - 1.0)
        term2 = 0.5 * (k2_ - k12 - k22) / k22 * (self.mu(eta, k2) - 1.0)
        return term0 + term1 + term2

    def source_FL(self, eta: float, kf: float, k1: float, k2: float) -> float:
        """Compute frame-lagging source term for second-order perturbations.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        kf : float
            Final wavenumber |k₁ + k₂| in h/Mpc.
        k1 : float
            First input wavenumber in h/Mpc.
        k2 : float
            Second input wavenumber in h/Mpc.

        Returns
        -------
        float
            Frame-lagging source term.

        Notes
        -----
        Implements sourceFL_HS from csrc/models.c.
        """
        return self.f1(eta) * np.square(self.mass(eta)) / self.PiF(eta, kf) * self.KFL(eta, kf, k1, k2)  # type: ignore[return-value]

    def source_dI(self, eta: float, kf: float, k1: float, k2: float) -> float:
        """Compute differential interaction source term for second-order.

        This term arises from the chameleon screening mechanism and involves
        the M2 coefficient and three propagators.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        kf : float
            Final wavenumber |k₁ + k₂| in h/Mpc.
        k1 : float
            First input wavenumber in h/Mpc.
        k2 : float
            Second input wavenumber in h/Mpc.

        Returns
        -------
        float
            Differential interaction source term.

        Notes
        -----
        Implements sourcedI_HS from csrc/models.c. Proportional to M2 screening coefficient.
        """
        return (
            1/6 * np.sqrt(self.OmM(eta) * self.H(eta) / (np.exp(eta) * self.invH0))
            * np.sqrt(kf) * self.M2(eta) / (self.PiF(eta, kf) * self.PiF(eta, k1) * self.PiF(eta, k2))
        )

    def source_A(self, eta: float, kf: float, k1: float, k2: float) -> float:
        """Compute total source term A for second-order density perturbations.

        This is the complete source term appearing in the equation for the
        symmetric second-order kernel (A-term in RSD).

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        kf : float
            Final wavenumber |k₁ + k₂| in h/Mpc.
        k1 : float
            First input wavenumber in h/Mpc.
        k2 : float
            Second input wavenumber in h/Mpc.

        Returns
        -------
        float
            Total source A = source_a + source_FL - source_dI.

        Notes
        -----
        Combines contributions from basic growth (a), frame-lagging (FL), and
        differential interaction (dI) terms.
        """
        return self.source_a(eta, kf) + self.source_FL(eta, kf, k1, k2) - self.source_dI(eta, kf, k1, k2)

    # Third order helper functions
    def M1(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute M₁(η) = 3m²(η) used in third-order frame-lagging.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a).

        Returns
        -------
        float or ndarray
            M₁(η) = 3m²(η) in units of (h/Mpc)².

        Notes
        -----
        Implements M1_HS from csrc/models.c.
        """
        return 3.0 * np.square(self.mass(eta))

    def M3(self, eta: Union[float, Float64NDArray]) -> Union[float, Float64NDArray]:
        """Compute third-order chameleon screening coefficient M₃(η).

        M₃ controls the strength of third-order screening effects in the
        differential interaction terms.

        Parameters
        ----------
        eta : float or array_like
            Conformal time η = ln(a).

        Returns
        -------
        float or ndarray
            Screening coefficient M₃(η) in units of (h/Mpc)⁶.

        Notes
        -----
        Implements M3_HS from csrc/models.c. When screening=0, returns 0.
        M₃ scales as |f_R0|⁻³.
        """

        model = getattr(self, "model", "HS").upper() # in case people select e.g. 'hdki' instead of 'HDKI' (so we force capital letters)

        if model in ("HS"):
            return self.screening * (
                45.0 / (8.0 * np.square(self.invH0)) * np.power(1 / np.abs(self.fR0), 3.0)
                * np.power(self.om * np.exp(-3.0 * eta) + 4.0 * self.ol, 7.0)
                / np.power(self.om + 4.0 * self.ol, 6.0)
            )
        else:
            return 0.0

    def KFL2(self, eta: float, x: float, k: float, p: float) -> float:
        """Compute second-order frame-lagging kernel KFL2 for third-order.

        Parameters
        ----------
        eta : float
            Conformal time η = ln(a).
        x : float
            Cosine of angle between k and p: x = k·p/(kp).
        k : float
            Wavenumber k in h/Mpc.
        p : float
            Wavenumber p in h/Mpc.

        Returns
        -------
        float
            Frame-lagging kernel KFL2(η, x, k, p).

        Notes
        -----
        Implements KFL2_HS from csrc/models.c. Used in JFL and third-order sources.
        """
        return (
            2.0 * np.square(x) * (self.mu(eta, k) + self.mu(eta, p) - 2.0)
            + (p * x / k) * (self.mu(eta, k) - 1.0)
            + (k * x / p) * (self.mu(eta, p) - 1.0)
        )

    def JFL(self, eta: float, x: float, k: float, p: float) -> float:
        """Compute normalized frame-lagging combination JFL.

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
            JFL(η, x, k, p) in units of (h/Mpc)².

        Notes
        -----
        Implements JFL_HS from csrc/models.c. Combines KFL2 with propagators.
        """
        result = (
            9.0 / (2.0 * self.A0(eta))
            * self.KFL2(eta, x, k, p) * self.PiF(eta, k) * self.PiF(eta, p)
        )
        return result  # type: ignore[return-value]

    def D2phiplus(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float) -> float:
        """Compute second-order potential derivative D²φ₊ for k+p combination.

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
            D²φ₊(η, x, k, p) including screening corrections.

        Notes
        -----
        Implements D2phiplus_HS from csrc/models.c. Used in third-order FL terms.
        """
        return (
            (1.0 + np.square(x))
            - (2.0 * self.A0(eta) / 3.0)
            * (
                (self.M2(eta) + self.JFL(eta, x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / (3.0 * self.PiF(eta, k) * self.PiF(eta, p))
            )
        ) * Dpk * Dpp + D2f

    def D2phiminus(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2mf: float) -> float:
        """Compute second-order potential derivative D²φ₋ for k-p combination.

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
        D2mf : float
            Second-order perturbation for k-p mode.

        Returns
        -------
        float
            D²φ₋(η, x, k, p) including screening corrections.

        Notes
        -----
        Implements D2phiminus_HS from csrc/models.c. Used in third-order FL terms.
        """
        return (
            (1.0 + np.square(x))
            - (2.0 * self.A0(eta) / 3.0)
            * (
                (self.M2(eta) + self.JFL(eta, -x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / (3.0 * self.PiF(eta, k) * self.PiF(eta, p))
            )
        ) * Dpk * Dpp + D2mf

    def K3dI(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float, D2mf: float) -> float:
        """Compute third-order differential interaction kernel K3dI.

        This is the most complex term in third-order perturbation theory,
        arising from chameleon screening at cubic order. It involves M₂ and M₃
        screening coefficients, multiple propagators, and both k±p modes.

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
            K3dI kernel summing six distinct contributions (t1-t6).

        Notes
        -----
        Implements K3dI_HS from csrc/models.c. The six terms correspond to:
        - t1, t2: k→0 limit contributions (infrared)
        - t3, t4: k+p mode contributions with screening
        - t5, t6: k-p mode contributions with screening
        """
        kplusp = self.kpp(x, k, p)
        kpluspm = self.kpp(-x, k, p)

        t1 = (
            2.0 * np.square(self.OmM(eta) * self.H(eta) / self.invH0)
            * (self.M2(eta) / (self.PiF(eta, k) * self.PiF(eta, 0)))
        )

        t2 = (
            (1.0 / 3.0) * (np.power(self.OmM(eta), 3.0) * np.power(self.H(eta), 4.0) / np.power(self.invH0, 4))
            * (
                self.M3(eta) - self.M2(eta) * (self.M2(eta) + self.JFL(eta, -1.0, p, p) * (3.0 + 2.0 * self.omegaBD))
                / self.PiF(eta, 0)
            ) / (np.square(self.PiF(eta, p)) * self.PiF(eta, k))
        )

        t3 = (
            np.square(self.OmM(eta) * self.H(eta) / self.invH0)
            * (self.M2(eta) / (self.PiF(eta, p) * self.PiF(eta, kplusp)))
            * (1.0 + np.square(x) + D2f / (Dpk * Dpp))
        )

        t4 = (
            (1.0 / 3.0) * (np.power(self.OmM(eta), 3.0) * np.power(self.H(eta), 4.0) / np.power(self.invH0, 4))
            * (
                self.M3(eta) - self.M2(eta) * (self.M2(eta) + self.JFL(eta, x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / self.PiF(eta, kplusp)
            ) / (np.square(self.PiF(eta, p)) * self.PiF(eta, k))
        )

        t5 = (
            np.square(self.OmM(eta) * self.H(eta) / self.invH0)
            * (self.M2(eta) / (self.PiF(eta, p) * self.PiF(eta, kpluspm)))
            * (1.0 + np.square(x) + D2mf / (Dpk * Dpp))
        )

        t6 = (
            (1.0 / 3.0) * (np.power(self.OmM(eta), 3.0) * np.power(self.H(eta), 4.0) / np.power(self.invH0, 4))
            * (
                self.M3(eta) - self.M2(eta) * (self.M2(eta) + self.JFL(eta, -x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / self.PiF(eta, kpluspm)
            ) / (np.square(self.PiF(eta, p)) * self.PiF(eta, k))
        )

        return t1 + t2 + t3 + t4 + t5 + t6

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
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * (  # type: ignore[return-value]
            self.M1(eta) / (3.0 * self.PiF(eta, kplusp))
            * self.KFL2(eta, x, k, p)
        )

    def S2dI(self, eta: float, x: float, k: float, p: float) -> float:
        """Compute symmetric differential interaction source S2dI for third-order.

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
            S2dI screening contribution proportional to M₂.

        Notes
        -----
        Implements S2dI_HS from csrc/models.c.
        """
        kplusp = self.kpp(x, k, p)
        return (
            (1.0 / 6.0) * np.square(self.OmM(eta) * self.H(eta) / (np.exp(eta) * self.invH0))
            * (np.square(kplusp) * self.M2(eta) / (self.PiF(eta, kplusp) * self.PiF(eta, k) * self.PiF(eta, p)))
        )

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
        return (
            self.S2a(eta, x, k, p) - self.S2b(eta, x, k, p) * np.square(x)
            + self.S2FL(eta, x, k, p) - self.S2dI(eta, x, k, p)
        )

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
        return (
            -self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kplusp) - 2.0 * self.mu(eta, k))
            * Dpp * (D2f + Dpk * Dpp * np.square(x))
            - self.f1(eta) * (self.mu(eta, kplusp) - self.mu(eta, k)) * Dpk * Dpp * Dpp
            - (
                (self.M1(eta) / (3.0 * self.PiF(eta, kplusp))) * self.f1(eta) * self.KFL2(eta, x, k, p)
                - np.square(self.OmM(eta) * self.H(eta) / self.invH0)
                * (self.M2(eta) * kplusp * kplusp * np.exp(-2.0 * eta))
                / (6.0 * self.PiF(eta, kplusp) * self.PiF(eta, k) * self.PiF(eta, p))
            ) * Dpk * Dpp * Dpp
        )

    def S3IIminus(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2mf: float) -> float:
        """Compute S3II- contribution for k-p mode in third-order source.

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
        D2mf : float
            Second-order perturbation for k-p mode.

        Returns
        -------
        float
            S3IIminus contribution.

        Notes
        -----
        Implements S3IIminus_HS from csrc/models.c.
        """
        kpluspm = self.kpp(-x, k, p)
        return (
            -self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kpluspm) - 2.0 * self.mu(eta, k))
            * Dpp * (D2mf + Dpk * Dpp * np.square(x))
            - self.f1(eta) * (self.mu(eta, kpluspm) - self.mu(eta, k)) * Dpk * Dpp * Dpp
            - (
                (self.M1(eta) / (3.0 * self.PiF(eta, kpluspm))) * self.f1(eta) * self.KFL2(eta, -x, k, p)
                - np.square(self.OmM(eta) * self.H(eta) / self.invH0)
                * (self.M2(eta) * kpluspm * kpluspm * np.exp(-2.0 * eta))
                / (6.0 * self.PiF(eta, kpluspm) * self.PiF(eta, k) * self.PiF(eta, p))
            ) * Dpk * Dpp * Dpp
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
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * (self.M1(eta) / (3.0 * self.PiF(eta, k))) * (
            (2.0 * np.square(p + k * x) / np.square(kplusp) - 1.0 - (k * x) / p)
            * (self.mu(eta, p) - 1.0) * D2f * Dpp
            + ((np.square(p) + 3.0 * k * p * x + 2.0 * k * k * x * x) / np.square(kplusp))
            * (self.mu(eta, kplusp) - 1.0) * self.D2phiplus(eta, x, k, p, Dpk, Dpp, D2f) * Dpp
            + 3.0 * np.square(x) * (self.mu(eta, k) + self.mu(eta, p) - 2.0) * Dpk * Dpp * Dpp
        )

    def S3FLminus(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2mf: float) -> float:
        """Compute S3FL- frame-lagging contribution for k-p mode.

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
        D2mf : float
            Second-order perturbation for k-p mode.

        Returns
        -------
        float
            S3FLminus frame-lagging contribution.

        Notes
        -----
        Implements S3FLminus_HS from csrc/models.c.
        """
        kpluspm = self.kpp(-x, k, p)
        return self.f1(eta) * (self.M1(eta) / (3.0 * self.PiF(eta, k))) * (
            (2.0 * np.square(p - k * x) / np.square(kpluspm) - 1.0 + (k * x) / p)
            * (self.mu(eta, p) - 1.0) * D2mf * Dpp
            + ((np.square(p) - 3.0 * k * p * x + 2.0 * k * k * x * x) / np.square(kpluspm))
            * (self.mu(eta, kpluspm) - 1.0) * self.D2phiminus(eta, x, k, p, Dpk, Dpp, D2mf) * Dpp
            + 3.0 * np.square(x) * (self.mu(eta, k) + self.mu(eta, p) - 2.0) * Dpk * Dpp * Dpp
        )

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
        return self.S3IIplus(eta, x, k, p, Dpk, Dpp, D2f) + self.S3IIminus(eta, x, k, p, Dpk, Dpp, D2mf)

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
        return self.S3FLplus(eta, x, k, p, Dpk, Dpp, D2f) + self.S3FLminus(eta, x, k, p, Dpk, Dpp, D2mf)

    def S3dI(self, eta: float, x: float, k: float, p: float, Dpk: float, Dpp: float, D2f: float, D2mf: float) -> float:
        """Compute third-order differential interaction source S3dI.

        This is the chameleon screening contribution at third order,
        proportional to M₂ and M₃ screening coefficients via K3dI.

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
            S3dI third-order screening source.

        Notes
        -----
        Implements S3dI_HS from csrc/models.c. Vanishes when screening=0.
        """
        return (
            -(np.square(k) / np.exp(2.0 * eta))
            * (1.0 / (6.0 * self.PiF(eta, k)))
            * self.K3dI(eta, x, k, p, Dpk, Dpp, D2f, D2mf) * Dpk * Dpp * Dpp
        )

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

    def secondOrder(self, x: float, y: Float64NDArray, kf: float, k1: float, k2: float) -> Float64NDArray:
        """Compute ODE derivatives for second-order kernel calculation.

        Solves coupled ODEs for two input modes (k₁, k₂) and two output modes
        (A-term and B-term) at final wavenumber kf = |k₁ + k₂|.

        Parameters
        ----------
        x : float
            Conformal time η = ln(a).
        y : ndarray
            State vector [D₁, D₁', D₂, D₂', A, A', B, B'] of length 8 where:
            - (D₁, D₁'): First-order growth for k₁
            - (D₂, D₂'): First-order growth for k₂
            - (A, A'): Second-order symmetric kernel (A-term)
            - (B, B'): Second-order antisymmetric kernel (B-term)
        kf : float
            Final wavenumber |k₁ + k₂| in h/Mpc.
        k1 : float
            First input wavenumber in h/Mpc.
        k2 : float
            Second input wavenumber in h/Mpc.

        Returns
        -------
        ndarray
            Derivatives [D₁', D₁'', D₂', D₂'', A', A'', B', B''] of length 8.

        Notes
        -----
        Implements derivsSecondOrder from csrc/gsm_diffeqs.c.
        Source terms source_A and source_b couple first-order solutions to
        second-order kernels.
        """
        f1x = self.f1(x)
        f2x = 2 - f1x
        srcA = self.source_A(x, kf, k1, k2)
        srcB = self.source_b(x, kf, k1, k2)
        return np.array([
            y[1], f1x * self.mu(x, k1) * y[0] - f2x * y[1],
            y[3], f1x * self.mu(x, k2) * y[2] - f2x * y[3],
            y[5], f1x * self.mu(x, kf) * y[4] - f2x * y[5] + srcA * y[0] * y[2],
            y[7], f1x * self.mu(x, kf) * y[6] - f2x * y[7] + srcB * y[0] * y[2]
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
                + self.S3dI(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
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

def D2v2(kf: float, k1: float, k2: float, derivs: ModelDerivatives, solver: ODESolver) -> Float64NDArray:
    """Integrate second-order kernel ODEs for modes k₁ and k₂.

    Computes second-order density kernels (A and B terms) from two input
    linear modes combining at wavenumber kf = |k₁ + k₂|.

    Parameters
    ----------
    kf : float
        Final wavenumber |k₁ + k₂| in h/Mpc.
    k1 : float
        First input wavenumber in h/Mpc.
    k2 : float
        Second input wavenumber in h/Mpc.
    derivs : ModelDerivatives
        Model instance providing ODE derivative functions.
    solver : ODESolver
        ODE solver configured with initial and final times.

    Returns
    -------
    ndarray
        Solution [D₁, D₁', D₂, D₂', A, A', B, B'] at final time.

    Notes
    -----
    Initial conditions at xnow for ΛCDM-like EdS universe:
    - First-order: D₁ = D₂ = exp(xnow), D₁' = D₂' = exp(xnow)
    - Second-order: A = B = (3/7)exp(2*xnow), A' = B' = (6/7)exp(2*xnow)
    """
    y0 = np.empty(8)
    y0[:4] = np.exp(solver.xnow)
    y0[4:] = 3 * np.exp(2 * solver.xnow) / 7
    y0[5::2] *= 2
    return solver(lambda x, y: derivs.secondOrder(x, y, kf, k1, k2), y0)

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

def kernel_constants(f0: float, derivs: ModelDerivatives, solver: ODESolver) -> Tuple[float, float, float, float]:
    """Compute normalized kernel constants in the squeezed (k→0) limit.

    These constants characterize the IR behavior of perturbation kernels
    and are used for consistency checks and analytical approximations.

    Parameters
    ----------
    f0 : float
        Linear growth rate f₀ = f(k→0) at final redshift.
    derivs : ModelDerivatives
        Model instance providing ODE derivative functions.
    solver : ODESolver
        ODE solver configured with initial and final times.

    Returns
    -------
    tuple of float
        (KA_LCDM, KAp_LCDM, KR1_LCDM, KR1p_LCDM) where:
        - KA_LCDM: Second-order symmetric kernel constant (A-term),
                   expected value 1.0 for ΛCDM
        - KAp_LCDM: Derivative of A-term kernel constant,
                    expected value 0.0 for ΛCDM
        - KR1_LCDM: Third-order symmetric kernel constant (R1-term),
                    expected value 1.0 for ΛCDM
        - KR1p_LCDM: Derivative of R1-term kernel constant,
                     expected value 0.0 for ΛCDM

    Notes
    -----
    Uses k = 1e-20 h/Mpc to approach IR limit. Initial conditions assume
    EdS-like scaling. Deviations from (1, 0, 1, 0) indicate modified gravity
    effects or numerical issues.
    """
    KMIN = 1e-20
    # 2nd order
    Dpk1D2, _, Dpk2D2, _, DA2D2, DA2primeD2, DB2D2, _ = D2v2(KMIN, KMIN, KMIN, derivs, solver)
    KA_LCDM = DA2D2 / ((3/7) * Dpk1D2 * Dpk2D2)
    KAp_LCDM = DA2primeD2 / ((3/7) * Dpk1D2 * Dpk2D2) - 2 * DA2D2 / ((3/7) * Dpk1D2 * Dpk2D2) * f0
    # 3rd order
    DpkD3, _, DppD3, _, D2fD3, _, D2mfD3, _, D3symmD3, D3symmprimeD3 = D3v2(1e-7, KMIN, KMIN, derivs, solver)
    KR1_LCDM = (21/5) * D3symmD3 / (DpkD3 * DppD3 * DppD3)
    KR1p_LCDM = (21/5) * D3symmprimeD3 / (DpkD3 * DppD3 * DppD3) / (3 * f0)
    return KA_LCDM, KAp_LCDM, KR1_LCDM, KR1p_LCDM
