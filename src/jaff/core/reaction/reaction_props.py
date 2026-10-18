# ABOUTME: Per-reaction property groups split out of Reaction: photo-reaction state
# ABOUTME: (cross sections, bands) and radiation coupling (energy added to the field)

from __future__ import annotations

from typing import TYPE_CHECKING, List

from sympy import Expr, Float

if TYPE_CHECKING:
    from ...physics import RadiationGroup, XsecProps


class PhotoReactionProps:
    """Photo-reaction state, held as ``Reaction.photo`` (``None`` otherwise).

    Attributes
    ----------
    xsec : XsecProps or None
        Cross sections of the reaction's single decay channel, or ``None``
        when no database has an entry.
    pi_database : str or None
        Per-reaction photoionization database override (``"norad"``,
        ``"verner"`` or ``"leiden"``); ``None`` uses the global
        :attr:`RadiationProps.pi_database`.
    groups : list[RadiationGroup]
        Radiation bands this reaction contributes to, populated when a
        radiation field is configured.  See :attr:`Reaction.band_xsecs`.
    custom_rad_rate : bool
        ``True`` when the rate comes from a ``.jfunc`` aux function rather
        than from the cross sections.
    xsec_tot : float or None
        Cross section averaged over the whole spectrum (cm²), set when the
        rate is computed from the cross sections.
    """

    def __init__(
        self,
        xsec: XsecProps | None = None,
        pi_database: str | None = None,
        groups: List[RadiationGroup] | None = None,
        custom_rad_rate: bool = False,
        xsec_tot: float | None = None,
    ):
        """Store the photo-reaction state; see the class attributes.

        Parameters
        ----------
        groups : list[RadiationGroup] or None, optional
            Initial bands; ``None`` (default) starts with an empty list.
        """
        self.xsec: XsecProps | None = xsec
        self.pi_database: str | None = pi_database
        self.groups: List[RadiationGroup] = groups if groups is not None else []
        self.custom_rad_rate: bool = custom_rad_rate
        self.xsec_tot: float | None = xsec_tot


class ReactionRadiationProps:
    """Radiation coupling of any reaction, held as ``Reaction.rad``.

    Attributes
    ----------
    dRad : Expr
        Radiation energy the reaction adds to the field per unit photon energy
        (erg/eV), as a SymPy expression in ``E`` (eV).  Zero by default.
    """

    def __init__(self, explicit: bool, dRad: Expr | None = None):
        """Store the radiation coupling; ``dRad=None`` (default) means zero."""
        self.dRad: Expr = dRad if dRad is not None else Float(0.0)
        self.explicit: bool = explicit
