# ABOUTME: Per-reaction property groups split out of Reaction: photo-reaction state
# ABOUTME: (cross sections, bands) and radiation coupling (energy added to the field)

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

from sympy import Expr, Float

from ...core.parsers import AuxiliaryFunctionProps

if TYPE_CHECKING:
    from ...physics import XsecProps


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
    shielding : ShieldingProps or None
        Shielding applied to the photo rate, or ``None`` when the reaction is
        not shielded.
    """

    def __init__(
        self,
        xsec: XsecProps | None = None,
        pi_database: str | None = None,
        custom_rad_rate: bool = False,
        rgrouped: List[GroupedPhotoReactionProps] | None = None,
        shielding: ShieldingProps | None = None,
    ):
        """Store the photo-reaction state; see the class attributes.

        Parameters
        ----------
        groups : list[RadiationGroup] or None, optional
            Initial bands; ``None`` (default) starts with an empty list.
        """
        self.xsec: XsecProps | None = xsec
        self.pi_database: str | None = pi_database
        self.custom_rad_rate: bool = custom_rad_rate
        self.rgrouped: List[GroupedPhotoReactionProps] | None = rgrouped
        self.shielding: ShieldingProps | None = shielding


class GroupedPhotoReactionProps:
    def __init__(self, k: Expr, xsec: float | None, xsec_frac: float, delta_rad: float):
        self.k: Expr = k
        self.xsec: float | None = xsec
        self.xsec_frac: float = xsec_frac
        self.detla_rad: float = delta_rad


class ReactionRadiationProps:
    """Radiation coupling of any reaction, held as ``Reaction.rad``.

    Attributes
    ----------
    dRad : Expr
        Radiation energy the reaction adds to the field per unit photon energy
        (erg/eV), as a SymPy expression in ``E`` (eV).  Zero by default.
    """

    def __init__(self, dRad: Expr | None = None):
        """Store the radiation coupling; ``dRad=None`` (default) means zero."""
        self.dRad: Expr = dRad if dRad is not None else Float(0.0)


class ShieldingProps:
    """Shielding of a photo rate, held as ``Reaction.photo.shielding``.

    Attributes
    ----------
    type : str
        Shielding model (e.g. ``"leiden"``, ``"db1996"``).
    props : dict[str, Any]
        Model-specific settings (e.g. ``shielded_by``, ``min_ncol``).  A
        ``value`` entry fixes the factor; the computed factor is cached there.
    """

    def __init__(self, type: str, props: Dict[str, Any] | None = None):
        """Store the model; ``props=None`` (default) starts empty."""
        self.type: str = type
        self.props: Dict[str, Any] = props if props is not None else {}


@dataclass
class AuxFunctions:
    chemrate: AuxiliaryFunctionProps | None = None
    deltae: AuxiliaryFunctionProps | None = None
    deltarad: AuxiliaryFunctionProps | None = None
