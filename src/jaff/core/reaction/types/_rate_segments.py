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

        if not (tup := (segment.tmin, segment.tmax) in self._by_prop):
            self._by_prop[tup] = segment
        else:
            raise KeyError(
                f"A rate coefficent already exists for ({segment.tmin}, {segment.tmax}"
            )

        self._list.append(segment)
        self.count = len(self._list)

    def __repr__(self):
        return "<RateSegment Object>"

    def evaluate_equivalent_rate(self) -> Expr:
        tgas = symbols("tgas")
        ls = self._list

        # There sholdn't be multiple reactions if rate doesn't depend on temperature
        # Same goes if a proper temperature range is not defined
        if not ls[0].rate.has(tgas) or ls[0].tmin is None or ls[0].tmax is None:
            return ls[0].rate

        segs: list[tuple[Expr, Basic | bool]] = []
        if self.count > 1:
            segs = self._get_piecewise_rate()

        first = ls[0]
        last = ls[-1]

        if self.mode == "clip":
            segs.append((last.rate.xreplace({tgas, last.tmax}), tgas > last.tmax))
            segs.insert(0, (first.rate.xreplace({tgas, first.tmin}), tgas < first.tmin))
        elif self.mode == "extrapolate":
            if self.count == 1:
                return first.rate

            segs[0] = (segs[0][0], tgas < first.tmax)
            segs[-1] = (segs[-1][0], tgas > last.tmin)

        return Piecewise(segs)

    def _get_piecewise_rate(self) -> list[tuple[Expr, Basic | bool]]:
        tgas = symbols("tgas")
        ls = self._list
        first = ls[0]

        segs: list[tuple[Expr, Basic | bool]] = [
            (ls[0].rate, first.tmin < tgas < first.tmax)
        ]
        for i, seg in enumerate(ls[1:]):
            prev = ls[i]
            if prev.tmax is None or seg.tmin is None:
                raise ValueError(
                    "Multiple temperature range reactions should have well defined temperature"
                )

            # Harmonic interpolation
            if prev.tmax > seg.tmin:
                raise ValueError(
                    "Temperature ranges shouldn't overlap for multi-temperature range reactions"
                )

            lw = 1 / (tgas - prev.tmax)  # left weight
            rw = 1 / (seg.tmin - tgas)  # right weight
            interp = (prev.rate * lw + seg.rate * rw) / (lw + rw)

            segs.extend(
                [
                    (interp, prev.tmax < tgas < seg.tmin),
                    (seg.rate, seg.tmin < tgas < seg.tmax),
                ]
            )

        return segs
