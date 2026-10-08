# ABOUTME: Symbolic equation-of-state infrastructure: EosProps config, EosFactory
# ABOUTME: builder dispatch and the Eos wrapper exposing volumetric/specific forms

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Callable, ClassVar

from sympy import Expr

from ..constants import N_A

if TYPE_CHECKING:
    from ...core import Network


class InternalEnergy:
    """Symbolic internal energy of a network in several normalisations.

    Wraps a volumetric internal energy ``E`` [erg cm⁻³] and derives the
    specific, per-particle and molar forms from the bound network's
    :attr:`~jaff.core.network.Network.rho` and
    :attr:`~jaff.core.network.Network.ntot`.
    """

    #: Form name -> accessor of the matching normalised property.
    _NORMALISED: ClassVar[dict[str, Callable[[InternalEnergy], Expr]]] = {
        "volumetric": lambda e: e.volumetric,
        "specific": lambda e: e.specific,
        "per_particle": lambda e: e.per_particle,
        "molar": lambda e: e.molar,
    }

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
        """Internal energy in the requested normalisation.

        Parameters
        ----------
        form : str
            ``"volumetric"``, ``"specific"``, ``"per_particle"`` or ``"molar"``.

        Returns
        -------
        sympy.Expr
            The internal energy normalised to *form* (see the property of the
            same name).

        Raises
        ------
        ValueError
            If *form* is not one of the forms above.
        """
        if form not in self._NORMALISED:
            raise ValueError(
                f"Invalid eos form: '{form}'. "
                f"Valid forms are: {', '.join(self._NORMALISED)}"
            )

        return self._NORMALISED[form](self)

    @cached_property
    def volumetric(self) -> Expr:
        """Internal energy per unit volume [erg cm⁻³]."""
        return self._vol_expr

    @cached_property
    def specific(self) -> Expr:
        """Internal energy per unit mass, ``E / ρ`` [erg g⁻¹]."""
        return self._vol_expr / self._net.rho

    @cached_property
    def per_particle(self) -> Expr:
        """Internal energy per particle, ``E / n_tot`` [erg]."""
        return self._vol_expr / self._net.ntot

    @cached_property
    def molar(self) -> Expr:
        """Internal energy per mole, ``N_A · E / n_tot`` [erg mol⁻¹]."""
        return self._vol_expr * N_A.cgs.value / self._net.ntot


class DEDt(InternalEnergy):
    pass
