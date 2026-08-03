from typing import TYPE_CHECKING

from sympy import Basic, Piecewise, symbols

from ....types import Catalogue
from . import RateSegment

if TYPE_CHECKING:
    from sympy import Expr


class RateSegments(Catalogue[RateSegment]):
    def __init__(self, segments: list[RateSegment], mode: str):
        _by_temp: dict[tuple[float | None, float | None], RateSegment] = {}

        if segments is not None:
            _by_temp = {(rs.tmin, rs.tmax): rs for rs in segments}

        super().__init__(segments, _by_temp)
        self.mode: str = mode

    def add(self, segment: RateSegment):
        if not isinstance(segment, RateSegment):
            raise ValueError(f"'{segment}' must be an instance of 'RateSegment'")

        tup = (segment.tmin, segment.tmax)
        if tup not in self._by_prop:
            self._by_prop[tup] = segment
        else:
            raise KeyError(
                f"A rate coefficient already exists for ({segment.tmin}, {segment.tmax})"
            )

        self._list.append(segment)
        self.count = len(self._list)

    def __repr__(self):
        return "<RateSegment Object>"

    def evaluate_equivalent_rate(self) -> Expr:
        tgas = symbols("tgas")
        ls = self._list

        # No piecewise needed when the (single) segment has no temperature
        # dependence or no proper temperature range.
        if not ls[0].rate.has(tgas) or ls[0].tmin is None or ls[0].tmax is None:
            return ls[0].rate

        first = ls[0]
        last = ls[-1]
        segs: list[tuple[Expr, Basic | bool]] = []

        if self.mode == "clip":
            segs.append((first.rate.xreplace({tgas: first.tmin}), tgas < first.tmin))

        segs.append((first.rate, tgas < first.tmax))

        # Interpolation gaps + subsequent ranges.
        for i, seg in enumerate(ls[1:]):
            prev = ls[i]
            if prev.tmax is None or seg.tmin is None:
                raise ValueError(
                    "Multiple temperature range reactions should have well defined temperature"
                )
            if prev.tmax > seg.tmin:
                raise ValueError(
                    "Temperature ranges shouldn't overlap for multi-temperature range reactions"
                )

            a = prev.tmax  # left boundary
            b = seg.tmin  # right boundary
            # Harmonic mean
            interp = (prev.rate * (b - tgas) + seg.rate * (tgas - a)) / (b - a)

            segs.append((interp, tgas < seg.tmin))
            segs.append((seg.rate, tgas < seg.tmax))

        if self.mode == "clip":
            segs.append((last.rate.xreplace({tgas: last.tmax}), True))
        else:
            segs[-1] = (segs[-1][0], True)

        return Piecewise(*segs)

    def sort(self) -> "RateSegments":
        self._list = sorted(self._list, key=lambda s: s.tmin)

        return self
