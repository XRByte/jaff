from dataclasses import dataclass
from typing import List

from sympy import Expr

from ...physics import RadiationGroup, XsecProps


@dataclass
class PhotoReactionProps:
    dRad: Expr
    custom_rad_rate: bool
    rad_groups: List[RadiationGroup]
    rad_xsec: float | None = None
    xsec_props: XsecProps | None = None
    pi_database: str | None = None
