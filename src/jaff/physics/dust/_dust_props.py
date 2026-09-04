from astropy import units as u

from ...errors import ParserError


class DustProps:
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
        if isinstance(rv, (int, float)):
            return rv

        raise ParserError(f"rv must be of type <float>. Found {type(rv)}")

    def _validate_reductions(
        self, u_reduction: str | None, f_reduction: str | None
    ) -> tuple[str | None, str | None]:
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
        if all(isinstance(v, (int, float)) for v in [low, high]):
            return low, high

        raise ParserError(
            "Photoelectric emission thresholds in dust must be of type float."
        )
