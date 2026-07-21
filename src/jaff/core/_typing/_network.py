from pathlib import Path
from typing import NotRequired, TypedDict

NetworkProps = TypedDict(
    "NetworkProps",
    {
        "fname": str | Path,
        "config": NotRequired[str | Path | None],
        "errors": NotRequired[bool],
        "label": NotRequired[str],
        "funcfile": NotRequired[bool | str | Path],
        "replace_nH": NotRequired[bool],
        "rad_bands": NotRequired[list],
        "rad_powerlaw_index": NotRequired[int | float],
        "rad_energy_density": NotRequired[bool],
        "c": NotRequired[float],
        "_from_cli": NotRequired[bool],
        "_metadata": NotRequired[dict],
    },
)
