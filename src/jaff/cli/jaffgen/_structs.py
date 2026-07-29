from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...config import JAFF_DIR
from ...drivers import Toml
from ...physics import constants

#: Fallback output directory (``<repo_root>/generated``) when none is supplied.
DEFAULT_OUTPUT: Path = JAFF_DIR.parent.parent / "generated"


@dataclass
class NetworkProps:
    """Constructor arguments forwarded to :class:`~jaff.Network`.

    Defaults mirror :meth:`jaff.core.network.Network.__init__` exactly, except
    for ``fname`` (a placeholder ``Path`` here, always overwritten before use)
    and ``_from_cli`` (``True``, since jaffgen prints its own MOTD banner).
    """

    fname: str | Path = Path()
    config: str | Path | None = None
    errors: bool = False
    label: str | None = None
    funcfile: bool | str | Path = True
    replace_nH: bool = True
    rad_bands: list = field(default_factory=list)
    rad_powerlaw_index: int | float = 0
    rad_energy_density: bool = False
    c: float = constants.c.cgs.value
    _from_cli: bool = True
    _metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedPath:
    """A path paired with the relative path used to mirror it under ``outdir``."""

    abspath: Path
    relpath: Path


@dataclass
class State:
    """Accumulating resolved configuration for a single ``jaffgen`` run."""

    template: str | None = None
    config_file: ResolvedPath | None = None
    config_dir: ResolvedPath | None = None
    config_raw: Toml | None = None
    network_file: ResolvedPath | None = None
    network_dir: ResolvedPath | None = None
    network_props: NetworkProps = field(default_factory=NetworkProps)
    input_dir: ResolvedPath | None = None
    input_files: list[ResolvedPath] = field(default_factory=list)
    output_dir: ResolvedPath = field(
        default_factory=lambda: ResolvedPath(DEFAULT_OUTPUT, Path())
    )
    output_files: list[ResolvedPath] = field(default_factory=list)
    lang: str | None = None
