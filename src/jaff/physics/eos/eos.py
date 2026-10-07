# ABOUTME: Symbolic equation-of-state infrastructure: EosProps config, EosFactory
# ABOUTME: builder dispatch and the Eos wrapper exposing volumetric/specific forms

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from sympy import Expr, Integer

from ..constants import N_A

if TYPE_CHECKING:
    from ...core import Network


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
