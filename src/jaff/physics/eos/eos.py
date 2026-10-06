# ABOUTME: Symbolic equation-of-state infrastructure: EosProps config, EosFactory
# ABOUTME: builder dispatch and the Eos wrapper exposing volumetric/specific forms

from __future__ import annotations

from functools import cached_property
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, Tuple

from sympy import Expr, Float, symbols

from ..constants import N_A, k_B

if TYPE_CHECKING:
    from ...core import Network


class EosFactory:
    """Build the symbolic :class:`Eos` selected by an :class:`EosProps`.

    Each EOS type maps to a builder method through :attr:`_BUILDERS`; the
    valid types must match :attr:`EosProps._REQUIRED`.
    """

    _BUILDERS: Dict[str, str] = {
        "ideal": "ideal",
        "multi_gamma": "multi_gamma",
        "fermi_degenerate": "fermi_degenerate",
        "relativistic_fermi_degenerate": "relativistic_fermi_degenerate",
    }

    def __init__(self, net: Network, props: EosProps):

        self.props: EosProps = props
        self._net: Network = net
        self._tgas = symbols("tgas")

    def generate(self):
        if self.props.type not in self._BUILDERS:
            raise ValueError(
                f"Invalid eos: '{self.props.type}'. "
                f"Valid eos types are: {', '.join(self._BUILDERS)}"
            )

        return getattr(self, self._BUILDERS[self.props.type])

    def ideal(self):
        return self._net.ntot * k_B.cgs.value * self._tgas / (self.props.gamma - 1.0)  # type: ignore

    def multi_gamma(self) -> Eos:
        e = Float(0.0)
        for sp in self._net.species:
            e += (
                self._net.ndens[sp.index]
                * k_B.cgs.value
                * self._tgas
                / (self.props.gamma_map.get(sp.name, self.props.default_gamma))  # type: ignore
            )

        return Eos(e, self._net)

    def fermi_degenerate(self):
        raise NotImplementedError()

    def relativistic_fermi_degenerate(self):
        raise NotImplementedError()


class Eos:
    def __init__(self, expr: Expr, net: Network):
        self._net: Network = net
        self._vol_expr: Expr = expr

    @cached_property
    def volumetric(self) -> Expr:
        return self._vol_expr

    @cached_property
    def specific(self) -> Expr:
        return self._vol_expr / self._net.rho

    @cached_property
    def per_particle(self) -> Expr:
        return self._vol_expr / self._net.ntot

    @cached_property
    def molar(self) -> Expr:
        return self.per_particle * N_A.cgs.value


class EosProps:
    """EOS configuration passed to :class:`~jaff.core.network.Network`.

    ``type`` selects the EOS; the keyword arguments are the parameters that
    type requires (see :attr:`_REQUIRED`).  Every argument is validated on
    construction, so a bad configuration fails here rather than in codegen.

    Attributes
    ----------
    type : str
        EOS type, one of the keys of :attr:`_REQUIRED`.
    gamma : float
        Adiabatic index (``ideal``); must be > 1.
    default_gamma : float
        Adiabatic index for species absent from ``gamma_map``
        (``multi_gamma``); must be > 1.
    gamma_map : dict[str, float]
        Per-species adiabatic index keyed by species name (``multi_gamma``);
        every value must be > 1.
    """

    _REQUIRED: Dict[str, Tuple[str, ...]] = {
        "ideal": ("gamma",),
        "multi_gamma": ("default_gamma", "gamma_map"),
        "fermi_degenerate": (),
        "relativistic_fermi_degenerate": (),
    }

    def __init__(self, type: str, **kwargs: Any):
        """Validate and store the EOS configuration.

        Raises
        ------
        ValueError
            If ``type`` is unknown, a required key is missing, an unexpected
            key is given, or an adiabatic index is not > 1.
        TypeError
            If an adiabatic index is not a real number or ``gamma_map`` is not
            a ``dict``.
        """
        if type not in self._REQUIRED:
            raise ValueError(
                f"Invalid eos: '{type}'. Valid eos types are: {', '.join(self._REQUIRED)}"
            )

        required = self._REQUIRED[type]
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"eos '{type}' requires: {', '.join(missing)}")

        unknown = [k for k in kwargs if k not in required]
        if unknown:
            raise ValueError(
                f"Unexpected parameters for eos '{type}': {', '.join(unknown)}"
            )

        self.type: str = type
        for key in ("gamma", "default_gamma"):
            if key in kwargs:
                setattr(self, key, self._validate_gamma(key, kwargs[key]))

        if "gamma_map" in kwargs:
            self.gamma_map: Dict[str, float] = self._validate_gamma_map(
                kwargs["gamma_map"]
            )

    @staticmethod
    def _validate_gamma(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"'{name}' must be a real number, got {value!r}")

        if value <= 1.0:
            raise ValueError(f"'{name}' must be > 1, got {value}")

        return float(value)

    @classmethod
    def _validate_gamma_map(cls, value: Any) -> Dict[str, float]:
        if not isinstance(value, dict):
            raise TypeError(f"'gamma_map' must be a dict, got {type(value).__name__}")

        return {
            name: cls._validate_gamma(f"gamma_map['{name}']", gamma)
            for name, gamma in value.items()
        }
