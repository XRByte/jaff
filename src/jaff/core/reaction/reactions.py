"""Ordered, doubly-indexed :class:`Reactions` catalogue for JAFF networks.

Reactions can be looked up by verbatim string
(``reactions["H + H2O+ -> H2 + OH+"]``) or by serialized form
(``reactions["H.H2O+__H2.OH+"]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sympy import Basic

from ...types import Catalogue, Vector
from ..species import Species
from .reaction import Reaction

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class Reactions(Catalogue[Reaction]):
    """Ordered, doubly-indexed catalogue of ``Reaction`` objects.

    Reactions can be looked up by verbatim string (``reactions["H + H2O+ -> H2 + OH+"]``)
    or by serialized form (``reactions["H_H2O+__H2_OH+"]``).
    """

    def __init__(self, reactions: list[Reaction] | None = None):
        """Initialise the reactions catalogue.

        Parameters
        ----------
        reactions : list[Reaction] | None, optional
            Initial reactions.  If ``None``, an empty catalogue is created.
        """
        _by_name: dict[str, Reaction] | None = None
        _by_serialized: dict[str, Reaction] = {}

        if reactions is not None:
            _by_name = {r.verbatim: r for r in reactions}
            _by_serialized = {r.serialized: r for r in reactions}

        super().__init__(reactions, _by_name)
        self._by_serialized = _by_serialized

    def __repr__(self):
        return f"Catalogue({self.verbatim()!r})"

    def add(self, reaction: Reaction) -> None:
        """Append a reaction to the catalogue (duplicates are not checked here).

        Parameters
        ----------
        reaction : Reaction

        Raises
        ------
        ValueError
            If *reaction* is not a ``Reaction`` instance.
        """
        if not isinstance(reaction, Reaction):
            raise ValueError(f"'{reaction}' must be an instance of 'Reaction'")

        self._by_name[reaction.verbatim] = reaction
        self._by_serialized[reaction.serialized] = reaction
        self._list.append(reaction)
        self.count = len(self._list)

    def from_serialized(self, serialized: str) -> Reaction:
        """Look up a reaction by its serialized form.

        Parameters
        ----------
        serialized : str
            Canonical form ``"<sorted_reactants>__<sorted_products>"``.

        Returns
        -------
        Reaction
        """
        return self._by_serialized[serialized]

    def from_verbatim(self, verbatim: str, type: str | None = None) -> Reaction | None:
        """Look up a reaction by its verbatim string.

        Parameters
        ----------
        verbatim : str
            Human-readable string (e.g. ``"H + H2O+ -> H2 + OH+"``).
        type : str or None, optional
            If supplied, return ``None`` when the reaction type does not match.

        Returns
        -------
        Reaction or None
        """
        rea = self._by_name[verbatim]
        if type is None or rea.type == type:
            return rea

    def get_list(self) -> list[Reaction]:
        """Return the underlying ordered list of ``Reaction`` objects.

        Returns
        -------
        list[Reaction]
        """
        return self._list

    def get(self, reaction: str, type: str | None = None) -> Reaction | None:
        """Look up a reaction by name or serialized form, with optional type filter.

        Parameters
        ----------
        reaction : str
            Verbatim string or serialized form.
        type : str or None, optional
            If given, return ``None`` when the reaction type does not match.

        Returns
        -------
        Reaction or None
        """
        rea = self[reaction]
        if type is None or rea.type == type:
            return rea

    def with_type(self, type: str):
        """Return all reactions matching the given reaction type.

        Parameters
        ----------
        type : str
            One of ``"photo"``, ``"cosmic_ray"``, ``"3_body"``, ``"unknown"``.

        Returns
        -------
        Vector[Reaction]
        """
        return Vector([r for r in self if r.type == type])

    def verbatim(self) -> Vector[str]:
        """Return a ``Vector`` of verbatim reaction strings.

        Returns
        -------
        Vector[str]
        """
        return Vector([r.verbatim for r in self])

    def types(self) -> Vector[str]:
        """Return a ``Vector`` of reaction type strings.

        Returns
        -------
        Vector[str]
        """
        return Vector([r.type for r in self])

    def reactants(self) -> Vector[Species]:
        """Return a ``Vector`` of reactant ``Species`` catalogues.

        Returns
        -------
        Vector[Species]
        """
        return Vector([r.reactants for r in self])

    def products(self) -> Vector[Species]:
        """Return a ``Vector`` of product ``Species`` catalogues.

        Returns
        -------
        Vector[Species]
        """
        return Vector([r.products for r in self])

    def rates(self) -> Vector[Basic]:
        """Return a ``Vector`` of rate SymPy expressions.

        Returns
        -------
        Vector[Basic]
        """
        return Vector([r.rate for r in self])

    def tmins(self) -> Vector[float | None]:
        """Return a ``Vector`` of lower temperature bounds (K or ``None``).

        Returns
        -------
        Vector[float | None]
        """
        return Vector([r.tmin for r in self])

    def tmaxes(self) -> Vector[float | None]:
        """Return a ``Vector`` of upper temperature bounds (K or ``None``).

        Returns
        -------
        Vector[float | None]
        """
        return Vector([r.tmax for r in self])

    def dE(self) -> Vector[Basic]:
        """Return a ``Vector`` of energy-change SymPy expressions (erg).

        Returns
        -------
        Vector[Basic]
        """
        return Vector([r.dE for r in self])

    def dRad(self) -> Vector[Basic]:
        """Return a ``Vector`` of radiation-rate SymPy expressions.

        Returns
        -------
        Vector[Basic]
        """
        return Vector([r.dRad for r in self])

    def serialized(self) -> Vector[str]:
        """Return a ``Vector`` of name-level serialized reaction strings.

        Returns
        -------
        Vector[str]
        """
        return Vector([r.serialized for r in self])

    def serialized_exploded(self) -> Vector[str]:
        """Return a ``Vector`` of atom-level (isomer-insensitive) serialized strings.

        Returns
        -------
        Vector[str]
        """
        return Vector([r.serialized_exploded for r in self])

    def photo_reactions(self) -> Vector[Reaction]:
        """Return all photo-reactions (``type == "photo"``).

        Returns
        -------
        Vector[Reaction]
        """
        return Vector([r for r in self if r.type == "photo"])

    def photo_reaction_truths(self) -> Vector[int]:
        """Return a binary ``Vector`` marking photo-reactions with ``1``.

        Returns
        -------
        Vector[int]
        """
        return Vector([int(reaction.type == "photo") for reaction in self])

    def photo_reaction_indices(self) -> Vector[int]:
        """Return the integer indices of photo-reactions within this catalogue.

        Returns
        -------
        Vector[int]
        """
        return Vector([i for i, reaction in enumerate(self) if reaction.type == "photo"])

    def plot_rates(self, **kwargs: Any) -> tuple[plt.Figure, Any] | None:
        """Plot the rate coefficients of every reaction in the catalogue.

        Thin wrapper over :func:`jaff.plotting.plot_rates` that passes all
        reactions in this catalogue as one overlay.  Accepts the same keyword
        arguments (``tmin``, ``tmax``, ``shade``, ``save``, ``filename`` ...).

        Reactions whose rate cannot be evaluated numerically (e.g. photo
        reactions) are skipped with a warning.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] or None
        """
        from ...plotting import plot_rates

        return plot_rates(list(self._list), **kwargs)

    def plot_xsecs(self, **kwargs: Any) -> tuple[plt.Figure, Any] | None:
        """Plot the photo cross sections of the catalogue's reactions.

        Thin wrapper over :func:`jaff.plotting.plot_xsecs`.  Reactions without
        cross-section data are skipped.  Accepts the same keyword arguments
        (``processes``, ``energy_unit``, ``shade``, ``show_bands`` ...).

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] or None
        """
        from ...plotting import plot_xsecs

        return plot_xsecs(list(self._list), **kwargs)
