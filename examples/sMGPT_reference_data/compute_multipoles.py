"""
ir_multipoles.py

Fast post-processing module for IR-resummed EFT multipoles.

This module uses the already-computed AllFunctions tables as input.  The
expensive Mathematica part generates these two files:

    AllFunctions_<suffix>.dat
    AllFunctions_<suffix>_nw.dat

Then this Python module can be used many times with different nuisance
parameters, for example inside an MCMC.

Example
-------

from src.ir_multipoles import IRMultipolesModel, ParamsPost

model = IRMultipolesModel(
    "outputs/BGS_F5/FullMG/AllFunctions_BGS_F5_FullMG.dat",
    "outputs/BGS_F5/FullMG/AllFunctions_BGS_F5_FullMG_nw.dat",
)

pars = ParamsPost(b1=1.44, b2=-1.0, bs2="auto", b3nl="auto",
                alpha0=-10, alpha2=-44, alpha4=-2.9,
                ctilde=-0.38, alphashot0=1, alphashot2=0,
                PshotP=1/0.000208)

result = model.compute(pars, ells=(0, 2, 4))
# result["table"] has columns k, P0, P2, P4

Notes
-----
- The formulas are a direct translation of src/IREFTMultipoles.m.
- By default, column mapping is detected automatically so both the historical
  43-column table and the table with a duplicated Pbs2b1 column are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Union, Any

import numpy as np

try:
    from scipy.interpolate import InterpolatedUnivariateSpline
    from scipy.integrate import quad
    from scipy.special import spherical_jn, eval_legendre
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ir_multipoles.py needs scipy. Install it with: pip install scipy"
    ) from exc

AutoValue = Union[str, float, int]


@dataclass(frozen=True)
class ParamsPost:
    """Nuisance parameters for the IR-resummed multipoles."""

    b1: float = 1.44
    b2: float = -1.0
    bs2: AutoValue = "auto"
    b3nl: AutoValue = "auto"
    alpha0: float = -10.0
    alpha2: float = -44.0
    alpha4: float = -2.9
    ctilde: float = -0.38
    alphashot0: float = 1.0
    alphashot2: float = 0.0
    PshotP: float = 1.0 / 0.000208

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ParamsPost":
        allowed = cls.__dataclass_fields__.keys()
        clean = {k: v for k, v in values.items() if k in allowed}
        return cls(**clean)

    def resolved(self) -> Dict[str, float]:
        b1 = float(self.b1)
        bs2 = -4.0 / 7.0 * (b1 - 1.0) if _is_auto(self.bs2) else float(self.bs2)
        b3nl = 32.0 / 315.0 * (b1 - 1.0) if _is_auto(self.b3nl) else float(self.b3nl)
        return {
            "b1": b1,
            "b2": float(self.b2),
            "bs2": bs2,
            "b3nl": b3nl,
            "alpha0": float(self.alpha0),
            "alpha2": float(self.alpha2),
            "alpha4": float(self.alpha4),
            "ctilde": float(self.ctilde),
            "alphashot0": float(self.alphashot0),
            "alphashot2": float(self.alphashot2),
            "PshotP": float(self.PshotP),
        }


def _is_auto(value: AutoValue) -> bool:
    return isinstance(value, str) and value.lower() in {"auto", "automatic", "none"}


def _read_numeric_table(path: Union[str, Path]) -> np.ndarray:
    """Read a Mathematica-exported table, skipping text/header lines robustly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
    with path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Mathematica sometimes quotes string header rows. Skip any line
            # whose tokens cannot all be parsed as floats.
            parts = stripped.replace(",", " ").split()
            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    ncols = len(rows[0])
    bad = [i for i, r in enumerate(rows) if len(r) != ncols]
    if bad:
        raise ValueError(
            f"Inconsistent number of columns in {path}; first bad row index {bad[0]}"
        )

    return np.asarray(rows, dtype=float)


class _Spline:
    def __init__(self, k: np.ndarray, y: np.ndarray, order: int = 3):
        order = int(min(order, len(k) - 1))
        self._spline = InterpolatedUnivariateSpline(k, y, k=order, ext=0)

    def __call__(self, x):
        return self._spline(x)


class TheoryTable:
    """Interpolated AllFunctions table."""

    def __init__(
        self,
        file_name: Union[str, Path],
        interpolation_order: int = 3,
        legacy_column_order: Union[str, bool] = "auto",
    ):
        self.file_name = str(file_name)
        data = _read_numeric_table(file_name)
        if data.ndim != 2 or data.shape[1] < 43:
            raise ValueError(
                f"{file_name} must have at least 43 numeric columns; got {data.shape}"
            )

        self.data = data
        self.ncols = data.shape[1]
        self.k = data[:, 0]
        self._order = interpolation_order

        bias_cols = self._bias_column_indices(legacy_column_order)

        self.PSL = self._interp(1)
        self.fk = self._interp(2)
        self.P22dd = self._interp(3)
        self.P22dt = self._interp(4)
        self.P22tt = self._interp(5)
        self.P13dd = self._interp(6)
        self.P13dt = self._interp(7)
        self.P13tt = self._interp(8)

        self.I1udd1A = self._interp(9)
        self.I2uud1A = self._interp(10)
        self.I2uud2A = self._interp(11)
        self.I3uuu2A = self._interp(12)
        self.I3uuu3A = self._interp(13)

        self.I2uudd1B = self._interp(14)
        self.I2uudd2B = self._interp(15)
        self.I3uuud1B = self._interp(16)
        self.I3uuud2B = self._interp(17)
        self.I3uuud3B = self._interp(18)
        self.I4uuuu1B = self._interp(19)
        self.I4uuuu2B = self._interp(20)
        self.I4uuuu3B = self._interp(21)
        self.I4uuuu4B = self._interp(22)

        self.I2uudd1C = self._interp(23)
        self.I2uudd2C = self._interp(24)
        self.I3uuud1C = self._interp(25)
        self.I3uuud2C = self._interp(26)
        self.I3uuud3C = self._interp(27)
        self.I4uuuu1C = self._interp(28)
        self.I4uuuu2C = self._interp(29)
        self.I4uuuu3C = self._interp(30)
        self.I4uuuu4C = self._interp(31)

        self.sigma2 = float(data[0, 32])
        self.sigma2v = float(data[0, 33])
        self.f0 = float(data[0, 34])

        self.Pb2b1 = self._interp(35)
        self.Pbs2b1 = self._interp(36)
        self.Pb22 = self._interp(bias_cols["Pb22"])
        self.Pb2s2 = self._interp(bias_cols["Pb2s2"])
        self.Pbs22 = self._interp(bias_cols["Pbs22"])
        self.Pb2t = self._interp(bias_cols["Pb2t"])
        self.Pbs2t = self._interp(bias_cols["Pbs2t"])
        self.sigma32pk = self._interp(bias_cols["sigma32pk"])

        self.bias_column_indices = bias_cols

    def _interp(self, zero_based_col: int) -> _Spline:
        return _Spline(self.k, self.data[:, zero_based_col], self._order)

    def _bias_column_indices(self, legacy_column_order: Union[str, bool]) -> Dict[str, int]:
        """Return 0-based column indices for bias terms.

        The clean 43-column layout has:
          col 36 Pb2b1, 37 Pbs2b1, 38 Pb22, ..., 43 sigma32pk
        in 1-based Mathematica notation.

        Some historical files have an extra duplicated Pbs2b1 column after
        column 37, shifting Pb22...sigma32pk by one.  In auto mode we detect
        this by checking whether numeric columns 37 and 38 are identical.
        """
        ncols = self.ncols

        if legacy_column_order == "auto":
            has_duplicate = False
            if ncols >= 44:
                a = self.data[:, 36]
                b = self.data[:, 37]
                scale = max(1.0, float(np.nanmax(np.abs(a))))
                has_duplicate = bool(np.nanmax(np.abs(a - b)) < 1e-10 * scale)
            start = 38 if has_duplicate else 37  # zero-based Pb22 index
        elif bool(legacy_column_order):
            # Reproduce IREFTMultipoles.m with LegacyColumnOrder -> True.
            start = 37
        else:
            # Use the shifted layout when a duplicated column is present.
            start = 38 if ncols >= 44 else 37

        needed = start + 5
        if needed >= ncols:
            raise ValueError(
                f"Not enough columns for bias terms: ncols={ncols}, Pb22 index={start}"
            )

        return {
            "Pb22": start,
            "Pb2s2": start + 1,
            "Pbs22": start + 2,
            "Pb2t": start + 3,
            "Pbs2t": start + 4,
            "sigma32pk": start + 5,
        }


class IRMultipolesModel:
    """Cached model for fast nuisance-parameter scans."""

    def __init__(
        self,
        file_ini: Union[str, Path],
        file_ini_nw: Union[str, Path],
        *,
        interpolation_order: int = 3,
        legacy_column_order: Union[str, bool] = "auto",
        kosc: float = 1.0 / 104.0,
        ks: float = 0.4,
        kmin_ir: float = 0.0001,
        nx: int = 16,
    ):
        self.w = TheoryTable(file_ini, interpolation_order, legacy_column_order)
        self.nw = TheoryTable(file_ini_nw, interpolation_order, legacy_column_order)
        self.kosc = float(kosc)
        self.ks = float(ks)
        self.kmin_ir = float(kmin_ir)
        self.nx = int(nx)
        self.x_gl, self.w_gl = np.polynomial.legendre.leggauss(self.nx)
        self.sigma2, self.delta_sigma2 = self._compute_ir_sigmas()

    def _compute_ir_sigmas(self) -> tuple[float, float]:
        def integrand_sigma(q: float) -> float:
            return float(
                self.nw.PSL(q)
                * (1.0 - spherical_jn(0, q / self.kosc) + 2.0 * spherical_jn(2, q / self.kosc))
            )

        def integrand_delta(q: float) -> float:
            return float(self.nw.PSL(q) * spherical_jn(2, q / self.kosc))

        sigma2 = quad(integrand_sigma, self.kmin_ir, self.ks, limit=200)[0] / (6.0 * np.pi**2)
        delta = quad(integrand_delta, self.kmin_ir, self.ks, limit=200)[0] / (2.0 * np.pi**2)
        return float(sigma2), float(delta)

    def _sigma2_total(self, mu):
        mu = np.asarray(mu)
        return (
            (1.0 + self.w.f0 * mu**2 * (2.0 + self.nw.f0)) * self.sigma2
            + self.w.f0**2 * mu**2 * (mu**2 - 1.0) * self.delta_sigma2
        )

    @staticmethod
    def _pddL(th: TheoryTable, k):
        return th.PSL(k)

    @staticmethod
    def _pdtL(th: TheoryTable, k):
        return (th.fk(k) / th.f0) * th.PSL(k)

    @staticmethod
    def _pttL(th: TheoryTable, k):
        return (th.fk(k) / th.f0) ** 2 * th.PSL(k)

    @staticmethod
    def _pddloop(th: TheoryTable, k):
        return th.P22dd(k) + th.P13dd(k)

    @staticmethod
    def _pdtloop(th: TheoryTable, k):
        return th.P22dt(k) + th.P13dt(k)

    @staticmethod
    def _pttloop(th: TheoryTable, k):
        return th.P22tt(k) + th.P13tt(k)

    def _pddXloop(self, th: TheoryTable, k, p: Mapping[str, float]):
        b1, b2, bs2, b3nl = p["b1"], p["b2"], p["bs2"], p["b3nl"]
        return (
            b1**2 * self._pddloop(th, k)
            + 2.0 * b1 * b2 * th.Pb2b1(k)
            + 2.0 * b1 * bs2 * th.Pbs2b1(k)
            + b2**2 * th.Pb22(k)
            + bs2**2 * th.Pbs22(k)
            + 2.0 * b2 * bs2 * th.Pb2s2(k)
            + 2.0 * b1 * b3nl * th.sigma32pk(k)
        )

    def _pdtXloop(self, th: TheoryTable, k, p: Mapping[str, float]):
        b1, b2, bs2, b3nl = p["b1"], p["b2"], p["bs2"], p["b3nl"]
        return (
            b1 * self._pdtloop(th, k)
            + b2 * th.Pb2t(k)
            + bs2 * th.Pbs2t(k)
            + b3nl * th.fk(k) / th.f0 * th.sigma32pk(k)
        )

    def _pttXloop(self, th: TheoryTable, k):
        return self._pttloop(th, k)

    @staticmethod
    def _af(th: TheoryTable, k, mu, f):
        return (
            f * mu**2 * th.I1udd1A(k)
            + f**2 * (mu**2 * th.I2uud1A(k) + mu**4 * th.I2uud2A(k))
            + f**3 * (mu**4 * th.I3uuu2A(k) + mu**6 * th.I3uuu3A(k))
        )

    @staticmethod
    def _bf(th: TheoryTable, k, mu, f):
        return (
            f**2 * (mu**2 * th.I2uudd1B(k) + mu**4 * th.I2uudd2B(k))
            + f**3
            * (
                mu**2 * th.I3uuud1B(k)
                + mu**4 * th.I3uuud2B(k)
                + mu**6 * th.I3uuud3B(k)
            )
            + f**4
            * (
                mu**2 * th.I4uuuu1B(k)
                + mu**4 * th.I4uuuu2B(k)
                + mu**6 * th.I4uuuu3B(k)
                + mu**8 * th.I4uuuu4B(k)
            )
        )

    @staticmethod
    def _cf(th: TheoryTable, k, mu, f):
        return (
            f**2 * (mu**2 * th.I2uudd1C(k) + mu**4 * th.I2uudd2C(k))
            + f**3
            * (
                mu**2 * th.I3uuud1C(k)
                + mu**4 * th.I3uuud2C(k)
                + mu**6 * th.I3uuud3C(k)
            )
            + f**4
            * (
                mu**2 * th.I4uuuu1C(k)
                + mu**4 * th.I4uuuu2C(k)
                + mu**6 * th.I4uuuu3C(k)
                + mu**8 * th.I4uuuu4C(k)
            )
        )

    def _atns(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        b1 = p["b1"]
        return b1**3 * self._af(th, k, mu, th.f0 / b1)

    def _btns(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        b1 = p["b1"]
        return b1**4 * self._bf(th, k, mu, th.f0 / b1)

    def _ctns(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        b1 = p["b1"]
        return b1**4 * self._cf(th, k, mu, th.f0 / b1)

    def _gtns(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        b1 = p["b1"]
        return (
            -th.f0**2 * mu**2 * k**2 * th.sigma2v * b1**2 * self._pddL(th, k)
            -2.0 * th.f0**3 * mu**4 * k**2 * th.sigma2v * b1 * self._pdtL(th, k)
            -th.f0**4 * mu**6 * k**2 * th.sigma2v * self._pttL(th, k)
        )

    def _pPMEloop(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        return (
            self._pddXloop(th, k, p)
            + 2.0 * mu**2 * th.f0 * self._pdtXloop(th, k, p)
            + mu**4 * th.f0**2 * self._pttXloop(th, k)
            + self._atns(th, k, mu, p)
            + self._btns(th, k, mu, p)
            + self._ctns(th, k, mu, p)
            + self._gtns(th, k, mu, p)
        )

    @staticmethod
    def _pKaiser(th: TheoryTable, k, mu, p: Mapping[str, float]):
        return (p["b1"] + mu**2 * th.fk(k)) ** 2 * th.PSL(k)

    def _pctilde(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        return p["ctilde"] * (mu * k * th.f0) ** 4 * th.sigma2v**2 * self._pKaiser(th, k, mu, p)

    def _peft(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        return (
            (p["alpha0"] + p["alpha2"] * mu**2 + p["alpha4"] * mu**4) * k**2 * th.PSL(k)
            + self._pctilde(th, k, mu, p)
        )

    @staticmethod
    def _pshot(k, mu, p: Mapping[str, float]):
        return p["PshotP"] * (p["alphashot0"] + p["alphashot2"] * k**2 * mu**2)

    def _ploop(self, th: TheoryTable, k, mu, p: Mapping[str, float]):
        return self._pPMEloop(th, k, mu, p) + self._peft(th, k, mu, p) + self._pshot(k, mu, p)

    def pk_ir(self, k, mu, params: Union[ParamsPost, Mapping[str, Any]]):
        """IR-resummed P(k,mu). k and mu can be arrays with broadcastable shapes."""
        p = params.resolved() if isinstance(params, ParamsPost) else ParamsPost.from_mapping(params).resolved()
        k = np.asarray(k, dtype=float)
        mu = np.asarray(mu, dtype=float)
        s2 = self._sigma2_total(mu)
        damp = np.exp(-k**2 * s2)
        wiggle_linear = self.w.PSL(k) - self.nw.PSL(k)
        return (
            (p["b1"] + self.w.fk(k) * mu**2) ** 2
            * (self.nw.PSL(k) + damp * wiggle_linear * (1.0 + k**2 * s2))
            + damp * self._ploop(self.w, k, mu, p)
            + (1.0 - damp) * self._ploop(self.nw, k, mu, p)
        )

    def compute(
        self,
        params: Union[ParamsPost, Mapping[str, Any]],
        *,
        kgrid: Union[None, Sequence[float], np.ndarray] = None,
        ells: Iterable[int] = (0, 2, 4),
    ) -> Dict[str, Any]:
        """Return multipoles for a set of nuisance parameters.

        Returns a dictionary with keys:
          - "k"
          - "ells"
          - "multipoles": dict ell -> array
          - "table": array with columns k, P_ell...
          - "parameters": resolved nuisance parameters
        """
        p_obj = params if isinstance(params, ParamsPost) else ParamsPost.from_mapping(params)
        p = p_obj.resolved()
        k = self.w.k if kgrid is None else np.asarray(kgrid, dtype=float)
        ell_list = tuple(int(x) for x in ells)

        kk = k[:, None]
        mu = self.x_gl[None, :]
        pk = self.pk_ir(kk, mu, p_obj)

        multipoles: Dict[int, np.ndarray] = {}
        for ell in ell_list:
            leg = eval_legendre(ell, self.x_gl)[None, :]
            multipoles[ell] = (2.0 * ell + 1.0) / 2.0 * np.sum(self.w_gl[None, :] * pk * leg, axis=1)

        table = np.column_stack([k] + [multipoles[ell] for ell in ell_list])
        return {
            "k": k,
            "ells": ell_list,
            "multipoles": multipoles,
            "table": table,
            "parameters": p,
            "sigma2": self.sigma2,
            "delta_sigma2": self.delta_sigma2,
            "bias_column_indices_wiggle": self.w.bias_column_indices,
            "bias_column_indices_nowiggle": self.nw.bias_column_indices,
        }


def save_multipoles(path: Union[str, Path], result: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ells = result["ells"]
    header = " ".join(["k[h/Mpc]"] + [f"Pell{ell}" for ell in ells])
    np.savetxt(path, result["table"], header=header)
