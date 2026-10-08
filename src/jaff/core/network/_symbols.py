# ABOUTME: NetworkSymbols — canonical symbols and symbolic quantities of a Network
# ABOUTME: (fixed physics symbols, densities, introspection, standardization)
"""Canonical symbols and symbolic quantities of a :class:`~jaff.Network`.

Reached as ``net.symbols``.  Fixed physics symbols are class attributes, so code
without a network can use ``NetworkSymbols.tgas``.  Runtime imports are limited to
SymPy, the standard library and :mod:`jaff.errors` so that any module may import this
one without creating an import cycle.
"""

from __future__ import annotations

from functools import cached_property, reduce
from typing import TYPE_CHECKING, ClassVar

from sympy import Basic, Expr, Float, Function, IndexedBase, Symbol
from sympy.core.function import UndefinedFunction

if TYPE_CHECKING:
    from .network import Network


class NetworkSymbols:
    """Canonical symbols and symbolic quantities of a network.

    Attributes
    ----------
    tgas, tdust : Symbol
        Gas and dust temperature [K].
    av : Symbol
        Visual extinction [mag].
    crate : Symbol
        Cosmic-ray ionisation rate [s⁻¹].
    chi : Symbol
        UV field scaling (Draine units).
    chi_pe : Symbol
        Photoelectric-band field placeholder, replaced by :meth:`standardize`.
    zd : Symbol
        Grain charge (``Zd``).
    vdisp : Symbol
        Velocity dispersion [cm s⁻¹].
    photorates : UndefinedFunction
        Placeholder function for photo-reaction rates.
    """

    tgas: ClassVar[Symbol] = Symbol("tgas")
    tdust: ClassVar[Symbol] = Symbol("tdust")
    av: ClassVar[Symbol] = Symbol("av")
    crate: ClassVar[Symbol] = Symbol("crate")
    chi: ClassVar[Symbol] = Symbol("chi")
    chi_pe: ClassVar[Symbol] = Symbol("chi_pe")
    zd: ClassVar[Symbol] = Symbol("Zd")
    vdisp: ClassVar[Symbol] = Symbol("vdisp")
    photorates: ClassVar[UndefinedFunction] = Function("photorates")  # type: ignore

    def __init__(self, net: Network) -> None:
        """Bind to *net*; derived quantities are computed lazily from it.

        Parameters
        ----------
        net : Network
            Network whose species, reactions and settings the symbols describe.
        """
        self._net: Network = net
        self._element_sums: dict[str, Expr | None] = {}

    @staticmethod
    def ncol(name: str) -> Symbol:
        """Column-density symbol ``ncol_<name>`` [cm⁻²].

        Parameters
        ----------
        name : str
            Species name as used in shielding configuration (e.g. ``"H2"``).

        Returns
        -------
        Symbol
            ``Symbol(f"ncol_{name}")``.
        """
        return Symbol(f"ncol_{name}")

    @staticmethod
    def free_symbols(expr: Basic) -> set[Basic]:
        """Free symbols of *expr*, excluding ``nden`` entries.

        ``nden[i]`` references are internal index variables, not user-visible
        physical symbols.

        Parameters
        ----------
        expr : Basic
            A SymPy expression.

        Returns
        -------
        set[Basic]
            Free symbols that do not involve ``"nden"``.
        """
        return {fs for fs in expr.free_symbols if "nden" not in str(fs)}

    @cached_property
    def ndens(self) -> IndexedBase:
        """Symbolic ``nden`` indexed base for species number densities.

        A SymPy :class:`~sympy.tensor.indexed.IndexedBase` that provides
        scalar-indexed access. Entry ``nden[i]`` is the number density of the
        species with index ``i``.  Cached so every consumer shares one symbol.

        Returns
        -------
        sympy.IndexedBase
            The ``nden`` indexed base symbol.
        """
        return IndexedBase("nden", shape=(self._net.species.count,))

    @cached_property
    def ntot(self) -> Expr:
        """Total number density ``Σ_i nden[i]`` over all species.

        Returns
        -------
        sympy.Expr
            Symbolic sum of every entry of :attr:`ndens`.
        """
        return sum(self.ndens[i] for i in range(self._net.species.count))

    @cached_property
    def rho(self) -> Expr:
        """Mass density ``Σ_i m_i · nden[i]`` over all species.

        Each species contributes its mass ``m_i`` times its number density.
        Species with an unset mass (``mass is None``) contribute ``0``.

        Returns
        -------
        sympy.Expr
            Symbolic mass density.
        """
        return reduce(
            lambda x, y: x + y,
            [(s.mass or 0.0) * self.ndens[s.index] for s in self._net.species],
        )

    @cached_property
    def n_hnuc(self) -> Expr:
        """Total hydrogen-nuclei number density ``Σ_i n_H(i) · nden[i]``.

        Each species contributes its hydrogen-atom count (``H2`` counts twice,
        ``H+`` once, ...) times its number density, so the sum is the total H
        nuclei density rather than a molecular count.  Equivalent to the
        ``n_H_nuc`` grammar token; used directly by the dust radiation-moment
        source terms (see :mod:`jaff.physics._equations`), cached so every
        consumer shares one expression.

        Returns
        -------
        sympy.Expr
            Symbolic total hydrogen-nuclei number density.  ``Float(0.0)`` when
            the network contains no H-bearing species.
        """
        total = self.element_sum("H")

        return total if total is not None else Float(0.0)

    def element_sum(self, element: str) -> Expr | None:
        """Nucleus density of *element*: ``Σ_i count_i · nden[i]`` (memoised).

        Parameters
        ----------
        element : str
            Canonical element symbol (e.g. ``"H"``, ``"He"``).

        Returns
        -------
        Expr | None
            The sum, or ``None`` when no species bears *element*.
        """
        if element not in self._element_sums:
            terms = [
                count * self.ndens[i]
                for i, spec in enumerate(self._net.species)
                if (count := spec.exploded.count(element)) > 0
            ]
            self._element_sums[element] = sum(terms) if terms else None

        return self._element_sums[element]
