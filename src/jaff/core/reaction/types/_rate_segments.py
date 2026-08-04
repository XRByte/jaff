from typing import TYPE_CHECKING

from sympy import Basic, Piecewise, symbols

from ....types import Catalogue
from . import RateSegment

if TYPE_CHECKING:
    from sympy import Expr


class RateSegments(Catalogue[RateSegment]):
    """Ordered catalogue of a reaction's rate pieces over temperature ranges.

    A reaction whose rate is defined piecewise across several disjoint
    temperature ranges holds one :class:`RateSegment` per range, keyed by its
    ``(tmin, tmax)`` bounds.  The pieces are collapsed into a single SymPy
    expression by :meth:`evaluate_equivalent_rate`.

    Parameters
    ----------
    segments : list[RateSegment]
        Initial rate segments.
    mode : str
        Out-of-range behaviour.  ``"clip"`` holds the boundary rate below the
        first / above the last range; any other value lets the outermost
        segments extend unbounded.
    """

    def __init__(self, segments: list[RateSegment], mode: str):
        _by_temp: dict[tuple[float | None, float | None], RateSegment] = {}

        if segments is not None:
            _by_temp = {(rs.tmin, rs.tmax): rs for rs in segments}

        super().__init__(segments, _by_temp)
        self.mode: str = mode

    def add(self, segment: RateSegment):
        """Append a rate segment, keyed by its ``(tmin, tmax)`` range.

        If a segment already exists for the same ``(tmin, tmax)`` range it is
        replaced by *segment*.

        Parameters
        ----------
        segment : RateSegment

        Raises
        ------
        ValueError
            If *segment* is not a :class:`RateSegment`.
        """
        if not isinstance(segment, RateSegment):
            raise ValueError(f"'{segment}' must be an instance of 'RateSegment'")

        tup = (segment.tmin, segment.tmax)
        if tup in self._by_prop:
            self._list.remove(self._by_prop[tup])

        self._by_prop[tup] = segment
        self._list.append(segment)
        self.count = len(self._list)

    def __repr__(self):
        return "<RateSegment Object>"

    def evaluate_equivalent_rate(self) -> Expr:
        """Collapse the segments into a single SymPy ``Piecewise`` rate.

        Segments must be sorted by ascending temperature (call :meth:`sort`
        first).  Each range contributes its own rate; the gap between two
        adjacent ranges is bridged by linear interpolation of the two bounding
        rates.  Out-of-range behaviour follows :attr:`mode` (``"clip"`` holds
        the boundary value, otherwise the end segments extend unbounded).

        A lone segment with a temperature-independent rate, or with no proper
        range, is returned as-is without wrapping in a ``Piecewise``.

        Returns
        -------
        Expr
            The equivalent rate expression in ``tgas``.

        Raises
        ------
        ValueError
            If adjacent segments have undefined bounds or overlapping ranges.
        """
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
            # Linear interpolation of the two rates across the gap [a, b].
            interp = (prev.rate * (b - tgas) + seg.rate * (tgas - a)) / (b - a)

            segs.append((interp, tgas < seg.tmin))
            segs.append((seg.rate, tgas < seg.tmax))

        if self.mode == "clip":
            segs.append((last.rate.xreplace({tgas: last.tmax}), True))
        else:
            segs[-1] = (segs[-1][0], True)

        return Piecewise(*segs)

    def sort(self) -> "RateSegments":
        """Sort the segments in place by ascending lower temperature bound.

        Returns
        -------
        RateSegments
            ``self``, to allow chaining (e.g. ``segs.sort().evaluate_...()``).
        """
        self._list = sorted(self._list, key=lambda s: s.tmin)

        return self
