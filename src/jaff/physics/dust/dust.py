"""Dust physics for a JAFF network.

:class:`Dust` is the container for the network's dust-related physics.  It is
attached to a :class:`~jaff.core.network.Network` when the network is built
with ``dust=True`` and groups the individual dust processes as sub-objects
(currently photoelectric emission; more may be added later).

The symbols exposed by these sub-objects (e.g. the scaled photoelectric-band
radiation field ``chi_pe`` supplied by :attr:`Dust.pe`) are substituted into
reaction-rate expressions during network standardisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    Attributes
    ----------
    network : Network
        Back-reference to the parent network.
    pe : PhotoelectricEmission
        Photoelectric-emission model, source of the ``chi_pe`` symbol
        substitution (the Draine field scaled to the photoelectric band).
    """

    _valid_rv: tuple[float, ...] = (3.1, 4.0, 5.5)

    _valid_reductions: tuple[str | None, ...] = (
        "extinction",
        "absorption",
        "scattering",
        "transport",
        "none",
        None,
    )

    def __init__(
        self,
        network: Network,
        rv: float = 5.5,
        u_reduction: str | None = "absorption",
        f_reduction: str | None = "transport",
    ):
        """Build the dust model and its process sub-objects.

        Parameters
        ----------
        network : Network
            The parent network.
        rv : float, optional
            Total-to-selective extinction ratio R_V selecting the WD01/Draine
            dust model.  Must be one of :attr:`_valid_rv` (``3.1``, ``4.0``,
            ``5.5``); default ``5.5``.
        u_reduction : str | None, optional
            Dust cross-section kind used to attenuate the radiation
            *density* (0th) moment: ``"extinction"``, ``"absorption"``,
            ``"scattering"``, ``"transport"``, or ``"none"`` to disable the
            term.  Default ``"absorption"``.
        f_reduction : str | None, optional
            Dust cross-section kind used to attenuate the radiation *flux*
            (1st) moment; same choices as *u_reduction*.  Default
            ``"transport"``.
        """
        if rv not in self._valid_rv:
            raise ValueError(
                f"Invalid R_V {rv!r} for dust model. "
                f"Available models: {', '.join(str(v) for v in self._valid_rv)}"
            )

        for label, value in (("u_reduction", u_reduction), ("f_reduction", f_reduction)):
            if value not in self._valid_reductions:
                raise ValueError(
                    f"Invalid {label} {value!r} for dust model. Available "
                    f"reductions: {', '.join(str(v) for v in self._valid_reductions)}"
                )

        self.net: Network = network
        self.pe: PhotoelectricEmission = PhotoelectricEmission(network)
        self.rv: float = rv
        self.u_reduction: str | None = u_reduction
        self.f_reduction: str | None = f_reduction
        self.tabular: Tabular = Tabular(self)
