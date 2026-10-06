# ABOUTME: Symbolic equation-of-state infrastructure: EosProps config, EosFactory
# ABOUTME: builder dispatch and the Eos wrapper exposing volumetric/specific forms

from __future__ import annotations

from functools import cached_property
from numbers import Real
from typing import TYPE_CHECKING, Any, Dict, Tuple

from sympy import Expr, Float, Integer, Symbol, symbols

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

    def __init__(self, net: Network, props: EosProps) -> None:
        """Bind the factory to a network and an EOS configuration.

        Parameters
        ----------
        net : Network
            Network supplying the symbolic densities (``ntot``, ``ndens``,
            ``rho``) and species list.
        props : EosProps
            Validated EOS configuration selecting the builder.
        """
        self.props: EosProps = props
        self._net: Network = net
        self._tgas: Symbol = symbols("tgas")

    def generate(self) -> Eos:
        """Build the EOS selected by ``props.type``.

        Returns
        -------
        Eos
            Symbolic internal energy of the bound network.

        Raises
        ------
        ValueError
            If ``props.type`` has no entry in :attr:`_BUILDERS`.
        """
        if self.props.type not in self._BUILDERS:
            raise ValueError(
                f"Invalid eos: '{self.props.type}'. "
                f"Valid eos types are: {', '.join(self._BUILDERS)}"
            )

        return getattr(self, self._BUILDERS[self.props.type])()

    def ideal(self) -> Eos:
        """Volumetric ideal-gas internal energy with a single adiabatic index.

        ``E = n_tot · k_B · T_gas / (γ − 1)`` [erg cm⁻³].

        Returns
        -------
        Eos
            EOS wrapping the volumetric internal energy [erg cm⁻³].
        """
        e = self._net.ntot * k_B.cgs.value * self._tgas / (self.props.gamma - 1.0)  # type: ignore
        return Eos(e, self._net)

    def multi_gamma(self) -> Eos:
        """Internal energy summed over species with per-species adiabatic indices.

        Each species uses ``props.gamma_map[name]``, falling back to
        ``props.default_gamma``.

        Returns
        -------
        Eos
            EOS wrapping the volumetric internal energy [erg cm⁻³].
        """
        e = Float(0.0)
        for sp in self._net.species:
            e += (
                self._net.ndens[sp.index]
                * k_B.cgs.value
                * self._tgas
                / (self.props.gamma_map.get(sp.name, self.props.default_gamma))  # type: ignore
            )

        return Eos(e, self._net)

    def fermi_degenerate(self) -> Eos:
        """Non-relativistic degenerate Fermi gas EOS (not implemented).

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError()

    def relativistic_fermi_degenerate(self) -> Eos:
        """Relativistic degenerate Fermi gas EOS (not implemented).

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError()


class Eos:
    """Symbolic internal energy of a network in several normalisations.

    Wraps a volumetric internal energy ``E`` [erg cm⁻³] and derives the
    specific, per-particle and molar forms from the bound network's
    :attr:`~jaff.core.network.Network.rho` and
    :attr:`~jaff.core.network.Network.ntot`.
    """

    def __init__(self, expr: Expr, net: Network) -> None:
        """Wrap a volumetric internal energy.

        Parameters
        ----------
        expr : sympy.Expr
            Volumetric internal energy [erg cm⁻³].
        net : Network
            Network whose ``rho`` and ``ntot`` normalise *expr*.
        """
        self._net: Network = net
        self._vol_expr: Expr = expr

    def normaliser(self, form: str) -> Expr:
        """Density dividing the volumetric energy to give *form*.

        Parameters
        ----------
        form : str
            ``"volumetric"`` (``1``), ``"specific"`` (``ρ``),
            ``"per_particle"`` (``n_tot``) or ``"molar"`` (``n_tot / N_A``).

        Returns
        -------
        sympy.Expr
            Symbolic normaliser.

        Raises
        ------
        ValueError
            If *form* is not one of the forms above.
        """
        if form == "volumetric":
            return Integer(1)
        if form == "specific":
            return self._net.rho
        if form == "per_particle":
            return self._net.ntot
        if form == "molar":
            return self._net.ntot / N_A.cgs.value

        raise ValueError(
            f"Invalid eos form: '{form}'. "
            "Valid forms are: volumetric, specific, per_particle, molar"
        )

    @cached_property
    def volumetric(self) -> Expr:
        """Internal energy per unit volume [erg cm⁻³]."""
        return self._vol_expr

    @cached_property
    def specific(self) -> Expr:
        """Internal energy per unit mass, ``E / ρ`` [erg g⁻¹]."""
        return self._vol_expr / self.normaliser("specific")

    @cached_property
    def per_particle(self) -> Expr:
        """Internal energy per particle, ``E / n_tot`` [erg]."""
        return self._vol_expr / self.normaliser("per_particle")

    @cached_property
    def molar(self) -> Expr:
        """Internal energy per mole, ``N_A · E / n_tot`` [erg mol⁻¹]."""
        return self._vol_expr / self.normaliser("molar")


class EosProps:
    """EOS configuration passed to :class:`~jaff.core.network.Network`.

    ``type`` selects the EOS; the keyword arguments are the parameters that
    type requires (see :attr:`_REQUIRED`), with any omitted parameter taken
    from :attr:`_DEFAULTS`.  Every argument is validated on
    construction, so a bad configuration fails here rather than in codegen.

    Attributes
    ----------
    type : str
        EOS type, one of the keys of :attr:`_REQUIRED`.
    gamma : float
        Adiabatic index (``ideal``); must be > 1.  Default ``1.6666666666667``
        (monoatomic ideal gas).
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

    _DEFAULTS: Dict[str, Dict[str, Any]] = {
        "ideal": {"gamma": 1.6666666666667},
    }

    def __init__(self, type: str, **kwargs: Any) -> None:
        """Validate and store the EOS configuration.

        Parameters
        ----------
        type : str
            EOS type, one of the keys of :attr:`_REQUIRED`.
        **kwargs : Any
            Parameters listed for *type* in :attr:`_REQUIRED`; any omitted
            one is taken from :attr:`_DEFAULTS`.

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

        kwargs = {**self._DEFAULTS.get(type, {}), **kwargs}
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
        """Return *value* as a float, requiring a real number > 1."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"'{name}' must be a real number, got {value!r}")

        if value <= 1.0:
            raise ValueError(f"'{name}' must be > 1, got {value}")

        return float(value)

    @classmethod
    def _validate_gamma_map(cls, value: Any) -> Dict[str, float]:
        """Return a copy of *value* with every adiabatic index validated."""
        if not isinstance(value, dict):
            raise TypeError(f"'gamma_map' must be a dict, got {type(value).__name__}")

        return {
            name: cls._validate_gamma(f"gamma_map['{name}']", gamma)
            for name, gamma in value.items()
        }
