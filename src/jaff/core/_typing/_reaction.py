from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from sympy import Expr

if TYPE_CHECKING:
    from ...physics import XsecProps
    from .. import Network, Reaction, Specie, Species
else:
    Specie = "Specie"
    Species = "Species"
    Reaction = "Reaction"
    Network = "Network"


RateSegmentProps = TypedDict(
    "RateSegmentProps",
    {
        "rate": Expr,
        "tmin": float | None,
        "tmax": float | None,
    },
)


ReactionProps = TypedDict(
    "ReactionProps",
    {
        "reactants": list["Specie"],
        "products": list["Specie"],
        "rate": Expr,
        "dE": Expr,
        "custom_rad_rate": bool,
        "reaction_type": str,
        "format": NotRequired[str | None],
        "tmin": float | None,
        "tmax": float | None,
        "t_cutoff": NotRequired[str],
        "rate_segments": NotRequired[list["RateSegmentProps"]],
        "original_string": str,
        "index": NotRequired[int],
        "pi_database": NotRequired[str | None],
        "shielding": NotRequired[dict[str, Any] | None],
        "xsec_props": NotRequired["XsecProps | None"],
        "spectral_yield": NotRequired[Expr | None],
    },
)
