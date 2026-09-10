"""Dust physics for a JAFF network.

:class:`Dust` is the container for the network's dust-related physics.  It is
attached to a :class:`~jaff.core.network.Network` when the network is built
with ``Network(..., dust_props=DustProps(...))`` and groups the individual
dust processes as sub-objects (currently photoelectric emission and tabular
dust cross sections; more may be added later).

The symbols exposed by these sub-objects (e.g. the scaled photoelectric-band
radiation field ``chi_pe`` supplied by :attr:`Dust.pe`) are substituted into
reaction-rate expressions during network standardisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import DustProps
from .photoelectric_emission import PhotoelectricEmission
from .tabular import Tabular

if TYPE_CHECKING:
    from ...core.network import Network


class Dust:
    """Dust-physics container for a network.

    Parameters
    ----------
    network : Network
        The parent network this dust model belongs to.
    props : DustProps
        Configuration object holding the dust model selection and radiation
        reduction settings.

    Attributes
    ----------
    net : Network
        Back-reference to the parent network.
    pe : PhotoelectricEmission
        Photoelectric-emission model, source of the ``chi_pe`` symbol
        substitution (the Draine field scaled to the photoelectric band).
    rv : float
        Total-to-selective extinction ratio R_V (from ``props``) selecting
        the WD01/Draine dust model.
    u_reduction : str | None
        Dust cross-section kind used to attenuate the radiation *density*
        (0th) moment (from ``props``).
    f_reduction : str | None
        Dust cross-section kind used to attenuate the radiation *flux*
        (1st) moment (from ``props``).
    tabular : Tabular
        Tabulated per-H-nucleus dust cross sections for the selected R_V.
    """

    def __init__(
        self,
        network: Network,
        props: DustProps,
    ):
        """Build the dust model and its process sub-objects.

        Parameters
        ----------
        network : Network
            The parent network.
        props : DustProps
            Configuration object supplying the R_V dust-model selection
            (``rv``, default ``3.1``), the radiation reduction kinds
            (``u_reduction``, ``f_reduction``), and the photoelectric band
            edges (``pe_threshold_low``, ``pe_threshold_high``).
        """

        self.net: Network = network
        self.pe: PhotoelectricEmission = PhotoelectricEmission(
            network, props.pe_threshold_low, props.pe_threshold_high
        )
        self.rv: float = props.rv
        self.u_reduction: str | None = props.u_reduction
        self.f_reduction: str | None = props.f_reduction
        self.tabular: Tabular = Tabular(self)
