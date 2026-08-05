"""Ordered, name-indexed :class:`Species` catalogue for JAFF networks.

``Species`` supports look-up by name (``species["H2O"]``), by serialized form
(``species["+/H/H/O"]``), and by integer index (``species[0]``).
"""

from __future__ import annotations

from functools import cached_property

from ...types import Catalogue, Vector
from .._typing import ElementProps
from ..elements import Elements
from .specie import Specie


class Species(Catalogue[Specie]):
    """Ordered, name-indexed collection of ``Specie`` objects.

    Species supports look-up by name (``species["H2O"]``), by serialized form
    (``species["+/H/H/O"]``), and by integer index (``species[0]``).  The
    ``ne`` parameter on many accessor methods excludes the electron species
    (``"e-"``), which is often treated separately in network solvers.
    """

    _mass_dict: dict | None = None

    @classmethod
    def configure(cls, mass_dict: dict[str, ElementProps]) -> None:
        """Override the mass dictionary and propagate it to ``Specie``.

        Parameters
        ----------
        mass_dict : dict[str, ElementProps]
            Custom mass dictionary.
        """
        cls._mass_dict = mass_dict
        Specie.configure(mass_dict)

    def __init__(
        self,
        species: list[Specie] | list[str] | None = None,
        check_length: bool = True,
    ):
        """Initialise the species catalogue.

        Parameters
        ----------
        species : list[Specie] | list[str] | None, optional
            Initial species.  Plain strings are converted to ``Specie``
            objects with indices assigned in list order.  If ``None``, an
            empty catalogue is created (items can be added with ``add``).
        check_length : bool, optional
            If ``True`` (default), the base ``Catalogue`` verifies that the
            list and name-dict have the same length.  Set to ``False`` when
            constructing a catalogue from reactants/products that may contain
            duplicate species.
        """
        _by_name: dict[str, Specie] | None = None
        _by_serialized: dict[str, Specie] = {}

        if species is not None:
            if species and isinstance(species[0], str):
                species = [Specie(name, idx) for idx, name in enumerate(species)]  # type: ignore[arg-type]
            _by_name = {sp.name: sp for sp in species}  # type: ignore
            _by_serialized = {sp.serialized: sp for sp in species}  # type: ignore

        _species: list[Specie] = species  # type: ignore

        super().__init__(_species, _by_name, check_length)
        self._by_serialized = _by_serialized

    def __repr__(self):
        return f"Catalogue({self.names()!r})"

    def add(self, specie: Specie) -> None:
        """Append a new species to the catalogue if not already present.

        Duplicates (by name) are silently ignored.

        Parameters
        ----------
        specie : Specie
            The species to add.

        Raises
        ------
        ValueError
            If *specie* is not a ``Specie`` instance.
        """
        if not isinstance(specie, Specie):
            raise ValueError(f"'{specie}' must be an instance of 'Specie'")

        if specie.name not in self._by_prop:
            self._by_prop[specie.name] = specie
            self._by_serialized[specie.serialized] = specie
            self._list.append(specie)
            self.count = len(self._list)
            # Invalidate the cached core/special sub-catalogues on mutation.
            self.__dict__.pop("core", None)
            self.__dict__.pop("special", None)

    @cached_property
    def special(self) -> "Species":
        """Sub-catalogue of the special pseudo-species (``is_special``).

        Returns
        -------
        Species
        """
        return Species([s for s in self._list if s.is_special], check_length=False)

    @cached_property
    def core(self) -> "Species":
        """Sub-catalogue of the core (real) species (``is_core``).

        Used wherever only physically integrated species should participate —
        e.g. the mass-action density product and the ODE assembly — so the
        special pseudo-species are excluded from the kinetics.

        Returns
        -------
        Species
        """
        return Species([s for s in self._list if s.is_core], check_length=False)

    def from_serialized(self, serialized: str) -> Specie:
        """Return the species matching the given serialized form.

        Parameters
        ----------
        serialized : str
            Canonical serialized string (e.g. ``"+/H/H/O"``).

        Returns
        -------
        Specie
        """
        return self._by_serialized[serialized]

    def from_name(self, name: str) -> Specie:
        """Return the species with the given chemical name.

        Parameters
        ----------
        name : str
            Species name (e.g. ``"H2O+"``).

        Returns
        -------
        Specie
        """
        return self._by_prop[name]

    def get_list(self) -> list[Specie]:
        """Return the underlying ordered list of ``Specie`` objects.

        Returns
        -------
        list[Specie]
        """
        return self._list

    def names(self, ne: bool = False) -> Vector[str]:
        """Return a ``Vector`` of species names.

        Parameters
        ----------
        ne : bool, optional
            If ``True``, exclude the electron species (``"e-"``),
            by default ``False``.

        Returns
        -------
        Vector[str]
        """
        return Vector([s.name for s in self if not (ne and s.name == "e-")])

    def masses(self, ne: bool = False) -> Vector[float | None]:
        """Return a ``Vector`` of species masses in grams (CGS).

        Parameters
        ----------
        ne : bool, optional
            If ``True``, exclude the electron species, by default ``False``.

        Returns
        -------
        Vector[float | None]
        """
        return Vector([s.mass for s in self if not (ne and s.name == "e-")])

    def exploded(self, ne: bool = False) -> Vector[list[str]]:
        """Return a ``Vector`` of atom lists (one per species).

        Parameters
        ----------
        ne : bool, optional
            If ``True``, exclude the electron species, by default ``False``.

        Returns
        -------
        Vector[list[str]]
        """
        return Vector([s.exploded for s in self if not (ne and s.name == "e-")])

    def latex(self, dollars: bool = True, ne: bool = False) -> Vector[str]:
        """Return a ``Vector`` of LaTeX species strings.

        Parameters
        ----------
        dollars : bool, optional
            Wrap each string in ``$...$`` math delimiters, by default ``True``.
        ne : bool, optional
            If ``True``, exclude the electron species, by default ``False``.

        Returns
        -------
        Vector[str]
        """
        return Vector([s.latex(dollars) for s in self if not (ne and s.name == "e-")])

    def charges(self, ne: bool = False) -> Vector[int]:
        """Return a ``Vector`` of net charges in units of elementary charge.

        Parameters
        ----------
        ne : bool, optional
            If ``True``, exclude the electron species, by default ``False``.

        Returns
        -------
        Vector[int]
        """
        return Vector([s.charge for s in self if not (ne and s.name == "e-")])

    def serialized(self, ne: bool = False) -> Vector[str]:
        """Return a ``Vector`` of canonical serialized species strings.

        Parameters
        ----------
        ne : bool, optional
            If ``True``, exclude the electron species, by default ``False``.

        Returns
        -------
        Vector[str]
        """
        return Vector([s.serialized for s in self if not (ne and s.name == "e-")])

    def elements(self, ne: bool = False) -> Vector[Elements]:
        """Return a ``Vector`` of ``Elements`` collections (one per species).

        Parameters
        ----------
        ne : bool, optional
            If ``True``, exclude the electron species, by default ``False``.

        Returns
        -------
        Vector[Elements]
        """
        return Vector([s.elements for s in self if not (ne and s.name == "e-")])

    def e_idx(self) -> int | None:
        """Return the integer index of the electron species, or ``None``.

        Returns
        -------
        int or None
            ``None`` when ``"e-"`` is not in this catalogue.
        """
        if "e-" in self:
            return self["e-"].index

    def normalized_names(self, pos: str = "p", neg: str = "n") -> Vector[str]:
        """Return species names normalized for use as code identifiers.

        All characters are lower-cased; ``"+"`` is replaced with *pos* and
        ``"-"`` with *neg*.

        Parameters
        ----------
        pos : str, optional
            Replacement for ``"+"``, by default ``"p"``.
        neg : str, optional
            Replacement for ``"-"``, by default ``"n"``.

        Returns
        -------
        Vector[str]
        """
        return Vector([s.name.lower().replace("+", pos).replace("-", neg) for s in self])

    def neutral(self, attr: str = "") -> Vector[Specie | int]:
        """Return neutral (charge == 0) species or one of their attributes.

        Parameters
        ----------
        attr : str, optional
            If given, return the named attribute of each neutral species
            instead of the ``Specie`` object itself.  Must be one of the
            values in ``Specie._ATTRS``.

        Returns
        -------
        Vector[Specie | int]

        Raises
        ------
        ValueError
            If *attr* is not a valid ``Specie`` attribute name.
        """
        if attr:
            if attr not in Specie._ATTRS:
                raise ValueError(f"Invalid attribute passed: {attr}")

            return Vector([getattr(s, attr) for s in self if s.charge == 0])

        return Vector([s for s in self if s.charge == 0])

    def charged(self, attr: str = "", ne: bool = False) -> Vector[Specie]:
        """Return charged (charge != 0) species or one of their attributes.

        Parameters
        ----------
        attr : str, optional
            If given, return the named attribute of each charged species.
            Must be one of the values in ``Specie._ATTRS``.
        ne : bool, optional
            If ``True``, exclude the electron species (``"e-"``),
            by default ``False``.

        Returns
        -------
        Vector[Specie]

        Raises
        ------
        ValueError
            If *attr* is not a valid ``Specie`` attribute name.
        """
        if attr:
            if attr not in Specie._ATTRS:
                raise ValueError(f"Invalid attribute passed: {attr}")

            return Vector(
                [
                    getattr(s, attr)
                    for s in self
                    if s.charge != 0 and not (ne and s.name == "e-")
                ]
            )

        return Vector([s for s in self if s.charge != 0 and not (ne and s.name == "e-")])

    def charge_truths(self, ne: bool = False) -> Vector[int]:
        """Return a binary ``Vector`` indicating whether each species is charged.

        Parameters
        ----------
        ne : bool, optional
            If ``True``, exclude the electron species, by default ``False``.

        Returns
        -------
        Vector[int]
            ``1`` for charged species, ``0`` for neutral.
        """
        return Vector([int(bool(s.charge)) for s in self if not (ne and s.name == "e-")])
