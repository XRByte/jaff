"""The :class:`NetworkProps` container of :class:`~jaff.Network` constructor arguments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..physics import constants


@dataclass
class NetworkProps:
    """Constructor arguments for :class:`~jaff.Network`.

    A single mutable container for every keyword accepted by
    :meth:`jaff.core.network.Network.__init__`.  ``Network`` bundles its own
    arguments into one of these (exposed as :attr:`Network.params`), and the
    ``jaffgen`` CLI accumulates resolved configuration into one before building
    the network.  Field defaults mirror the ``Network`` constructor defaults
    exactly, so an empty ``NetworkProps()`` reproduces the library defaults.

    Attributes
    ----------
    fname : str | Path
        Network file path or predefined network name.  The placeholder default
        (an empty ``Path``) is always overwritten before use.
    """

    fname: Path = Path()
    config: str | Path | None = None
    errors: bool = False
    label: str | None = None
    funcfile: bool | str | Path = True
    replace_nH: bool = True
    rad_bands: list = field(default_factory=list)
    rad_powerlaw_index: int | float = 0
    rad_energy_density: bool = False
    c: float = constants.c.cgs.value
    _from_cli: bool = False
    _metadata: dict[str, Any] = field(default_factory=dict)
