from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from sympy import Float, Symbol, symbols

from ..constants import k_B
from .eos import Eos
from .eos_props import EosProps

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
