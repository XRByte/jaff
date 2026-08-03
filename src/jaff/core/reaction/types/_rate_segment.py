from dataclasses import dataclass

from sympy import Expr


@dataclass
class RateSegment:
    rate: Expr
    tmin: float | None
    tmax: float | None
