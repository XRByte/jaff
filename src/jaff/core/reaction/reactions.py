"""Ordered, doubly-indexed :class:`Reactions` catalogue for JAFF networks.

Reactions can be looked up by verbatim string
(``reactions["H + H2O+ -> H2 + OH+"]``) or by serialized form
(``reactions["H.H2O+__H2.OH+"]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsIndex, cast, overload

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

    Both indexes map a key to a *list* of reactions: several reactions may
    share a verbatim string or serialized form when they differ only by
    mechanism/``type`` (e.g. thermal vs cosmic-ray desorption).
    """

    _by_prop: dict[str, list[Reaction]]  # type: ignore[assignment]
    _by_serialized: dict[str, list[Reaction]]  # type: ignore[assignment]

    def __init__(self, reactions: list[Reaction] | None = None):
        """Initialise the reactions catalogue.

        Parameters
        ----------
        reactions : list[Reaction] | None, optional
            Initial reactions.  If ``None``, an empty catalogue is created.
        """
        _by_name: dict[str, list[Reaction]] | None = None
        _by_serialized: dict[str, list[Reaction]] = {}

        if reactions is not None:
            _by_name = {}
            for r in reactions:
                _by_name.setdefault(r.verbatim, []).append(r)
                _by_serialized.setdefault(r.serialized, []).append(r)

        super().__init__(
            reactions,
            cast("dict[Any, Reaction | list[Reaction]] | None", _by_name),
            check_length=False,
        )
        self._by_serialized = _by_serialized

    @overload
    def __getitem__(self, key: str | SupportsIndex) -> Reaction: ...

    @overload
    def __getitem__(self, key: tuple[str, str]) -> Reaction: ...

    @overload
    def __getitem__(self, key: slice) -> list[Reaction]: ...

    def __getitem__(self, key):
        """Look up a reaction by serialized/verbatim string, ``(id, type)``, or index.

        Several reactions may share a serialized form or verbatim string when
        they differ only by mechanism/``type``, so string lookup is
        *scalar-or-raise*:

        * ``str`` — return the sole reaction with that serialized form (checked
          first) or verbatim string.  Raise :exc:`KeyError` if the key is
          ambiguous (use the ``(id, type)`` form or :meth:`all`) or absent.
        * ``tuple`` — ``(id, type)``: return the reaction whose serialized form
          or verbatim string is *id* and whose ``type`` is *type* (exactly one
          must match).
        * ``int`` / ``slice`` — positional access, delegated to the base list.

        Parameters
        ----------
        key : str, tuple[str, str], int, or slice

        Returns
        -------
        Reaction or list[Reaction]
            A single ``Reaction`` for string/tuple/int keys; a list for slices.

        Raises
        ------
        KeyError
            If a string key is ambiguous or missing, or a tuple key does not
            match exactly one reaction.
        """
        if isinstance(key, tuple):
            ident, rtype = key
            group = self._by_serialized.get(ident) or self._by_prop.get(ident, [])
            matches = [r for r in group if r.type == rtype]

            if len(matches) != 1:
                raise KeyError(f"{key!r}: {len(matches)} matches")

            return matches[0]

        if isinstance(key, str):
            # Serialized form first, then verbatim string; both map to a list.
            for index in (self._by_serialized, self._by_prop):
                group = index.get(key)
                if group is not None:
                    if len(group) > 1:
                        raise KeyError(
                            f"{key!r} ambiguous ({len(group)} mechanisms); "
                            f"use reactions[{key!r}, type] or .all({key!r})"
                        )

                    return group[0]

            raise KeyError(f"{key!r} not found in catalogue")

        # int / slice — delegate to base (positional list access)
        return super().__getitem__(key)

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

        self._by_prop.setdefault(reaction.verbatim, []).append(reaction)
        self._by_serialized.setdefault(reaction.serialized, []).append(reaction)
        self._list.append(reaction)
        self.count = len(self._list)

    def __contains__(self, item) -> bool:
        """Test membership by ``Reaction``, name/serialized string, or
        ``(serialized, type)`` tuple.

        Parameters
        ----------
        item : Reaction, str, or tuple[str, str]
            * ``str`` — checks verbatim names and serialized forms.
            * ``tuple`` — ``(serialized, type)``: true if a reaction with that
              serialized form *and* type exists.
            * otherwise — positional membership in the list.

        Returns
        -------
        bool
        """
        if isinstance(item, tuple):
            ident, rtype = item
            group = self._by_serialized.get(ident) or self._by_prop.get(ident, [])

            return any(r.type == rtype for r in group)

        if isinstance(item, str):
            return item in self._by_prop or item in self._by_serialized

        return item in self._list

    def from_serialized(self, serialized: str) -> list[Reaction]:
        """Return all reactions sharing a serialized form.

        A serialized form (species-only) can be shared by several reactions
        that differ by mechanism/``type`` (e.g. thermal vs cosmic-ray
        desorption).  All matching reactions are returned.

        Parameters
        ----------
        serialized : str
            Canonical form ``"<sorted_reactants>__<sorted_products>"``.

        Returns
        -------
        list[Reaction]
        """
        return self._by_serialized[serialized]

    def all(self, serialized: str) -> Vector[Reaction]:
        """Return all reactions sharing *serialized* (empty if none).

        Unlike :meth:`from_serialized`, a missing key yields an empty
        ``Vector`` rather than raising.

        Parameters
        ----------
        serialized : str
            Canonical ``"<sorted_reactants>__<sorted_products>"`` form.

        Returns
        -------
        Vector[Reaction]
        """
        return Vector(self._by_serialized.get(serialized, []))

    def from_verbatim(self, verbatim: str, type: str | None = None) -> Reaction | None:
        """Look up a reaction by its verbatim string.

        Parameters
        ----------
        verbatim : str
            Human-readable string (e.g. ``"H + H2O+ -> H2 + OH+"``).
        type : str or None, optional
            If supplied, return ``None`` when the reaction type does not match.
            Also disambiguates when several reactions share a verbatim string.

        Returns
        -------
        Reaction or None
        """
        for rea in self._by_prop.get(verbatim, []):
            if type is None or rea.type == type:
                return rea

        return None

    def get_list(self) -> list[Reaction]:
        """Return the underlying ordered list of ``Reaction`` objects.

        Returns
        -------
        list[Reaction]
        """
        return self._list

    def get(self, reaction: str, type: str | None = None) -> Reaction | None:
        """Look up a reaction by name or serialized form, with optional type filter.

        When *type* is supplied and *reaction* is a serialized form shared by
        several mechanisms, the exact ``(serialized, type)`` match is returned.
        Never raises: a missing or ambiguous lookup yields ``None``.

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
        if type is not None and (
            reaction in self._by_serialized or reaction in self._by_prop
        ):
            try:
                return self[reaction, type]
            except KeyError:
                return None

        try:
            rea = self[reaction]
        except KeyError:
            return None

        if type is None or rea.type == type:
            return rea

    def with_type(self, type: str):
        """Return all reactions matching the given reaction type.

        Parameters
        ----------
        type : str
            A reaction type string, e.g. ``"photo"``, ``"cosmic_ray"``,
            ``"3_body"``, ``"unknown"``, or a grain surface-mechanism type
            (see :class:`~jaff.core.reaction.Reaction`).

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
