from astropy import units as u

from ...errors import ParserError


class DustProps:
    """Dust configuration passed to :class:`~jaff.core.network.Network`.

    Holds the user-facing dust settings that select the dust cross-section
    model and control how the dust attenuates the radiation field.  An
    instance is passed as ``Network(..., dust_props=DustProps(...))`` and its
    values are consumed when the network builds its :class:`Dust` model.

    Attributes
    ----------
    rv : float
        Total-to-selective extinction ratio R_V selecting the WD01/Draine
        dust model.  Intended values are those in :attr:`_valid_rv`
        (``3.1``, ``4.0``, ``5.5``).
    u_reduction : str | None
        Dust cross-section kind used to attenuate the radiation *density*
        (0th) moment; one of :attr:`_valid_reductions`.
    f_reduction : str | None
        Dust cross-section kind used to attenuate the radiation *flux*
        (1st) moment; one of :attr:`_valid_reductions`.
    pe_threshold_low : float
        Lower edge of the photoelectric band, in eV.
    pe_threshold_high : float
        Upper edge of the photoelectric band, in eV.
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
        rv: float = 3.1,
        u_reduction: str | None = "absorption",
        f_reduction: str | None = "transport",
        pe_threshold_low: float = 6.0,
        pe_threshold_high: float = 13.6,
    ):
        """Validate and store the dust configuration.

        Parameters
        ----------
        rv : float, optional
            Total-to-selective extinction ratio R_V selecting the WD01/Draine
            dust model.  Must be one of :attr:`_valid_rv` (``3.1``,
            ``4.0``, ``5.5``); default ``3.1``.
        u_reduction : str | None, optional
            Dust cross-section kind used to attenuate the radiation *density*
            (0th) moment.  One of :attr:`_valid_reductions`: ``"extinction"``,
            ``"absorption"``, ``"scattering"``, ``"transport"``, ``"none"`` or
            ``None`` to disable the term.  Default ``"absorption"``.
        f_reduction : str | None, optional
            Dust cross-section kind used to attenuate the radiation *flux*
            (1st) moment; same accepted values as *u_reduction*.  Default
            ``"transport"``.
        pe_threshold_low : float, optional
            Lower edge of the photoelectric band, in eV.  Default ``6.0``.
        pe_threshold_high : float, optional
            Upper edge of the photoelectric band, in eV.  Default ``13.6``.

        Raises
        ------
        ParserError
            If any argument fails its validator (see :meth:`_validate_rv`,
            :meth:`_validate_reductions`, :meth:`_validate_pe_thresholds`).
        """
        self.rv: float = self._validate_rv(rv)
        self.u_reduction: str | None
        self.f_reduction: str | None
        self.u_reduction, self.f_reduction = self._validate_reductions(
            u_reduction, f_reduction
        )
        self.pe_threshold_low: float
        self.pe_threshold_high: float
        self.pe_threshold_low, self.pe_threshold_high = self._validate_pe_thresholds(
            pe_threshold_low, pe_threshold_high
        )

    def _validate_rv(self, rv: float) -> float:
        """Validate ``rv``: must be numeric and one of :attr:`_valid_rv`.

        Raises :class:`ParserError` on a non-numeric value or an R_V that has no
        corresponding grain-model table.
        """
        if not isinstance(rv, (int, float)):
            raise ParserError(f"rv must be of type <float>. Found {type(rv)}")

        if rv not in self._valid_rv:
            raise ParserError(
                f"Invalid rv {rv}. Supported values are "
                f"{', '.join(str(v) for v in self._valid_rv)}"
            )

        return rv

    def _validate_reductions(
        self, u_reduction: str | None, f_reduction: str | None
    ) -> tuple[str | None, str | None]:
        """Validate both reduction kinds against :attr:`_valid_reductions` and
        return them lower-cased; raises :class:`ParserError` on an invalid value."""
        if any(
            (not isinstance(r, str) and r is not None) for r in [u_reduction, f_reduction]
        ):
            raise ParserError(
                f"Invalid radiation reduction type due to dust. Valid types are {', '.join([str(r) for r in self._valid_reductions])}"
            )

        if any((r not in self._valid_reductions) for r in [u_reduction, f_reduction]):
            raise ParserError(
                f"Invalid radiation reduction type due to dust. Valid types are {', '.join([str(r) for r in self._valid_reductions])}"
            )

        return (
            u_reduction.lower() if isinstance(u_reduction, str) else u_reduction,
            f_reduction.lower() if isinstance(f_reduction, str) else f_reduction,
        )

    def _validate_pe_thresholds(self, low, high) -> tuple[float, float]:
        """Type-check both photoelectric thresholds (int/float) and return them;
        raises :class:`ParserError` if either is non-numeric."""
        if all(isinstance(v, (int, float)) for v in [low, high]):
            return low, high

        raise ParserError(
            "Photoelectric emission thresholds in dust must be of type float."
        )
