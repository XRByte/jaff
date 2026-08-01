"""The :class:`NetworkParams` normalized parameters of a :class:`~jaff.Network`."""

from pathlib import Path
from typing import Any

from ..config import NETWORKS_DIR, predefined_networks
from ..errors import ParserError


class NetworkParams:
    """Normalized construction parameters of a :class:`~jaff.Network`.

    ``Network`` bundles its constructor arguments into one of these (exposed as
    :attr:`Network.params`) and reads them back from it.  Construction parses
    the raw arguments into their canonical forms:

    * ``fname`` is resolved to an absolute path (a filesystem path or a
      predefined network name; see :meth:`resolve_network_path`);
    * ``funcfile`` is validated and coerced to ``bool`` or :class:`~pathlib.Path`;
    * ``config`` is coerced from ``str`` to :class:`~pathlib.Path`.

    Every field is required — ``Network`` always supplies all of them, so the
    library defaults live in one place (the ``Network`` constructor signature).

    Attributes
    ----------
    fname : Path
        Resolved, absolute path to the network file.
    config : Path | None
        Resolved path to the ``jaff.toml`` config file, if any.
    funcfile : bool | Path
        ``True`` to scan the network directory, ``False`` to skip, or a path.
    """

    def __init__(
        self,
        fname: str | Path,
        config: str | Path | None,
        errors: bool,
        label: str | None,
        funcfile: bool | str | Path,
        replace_nH: bool,
        rad_bands: list,
        rad_powerlaw_index: int | float,
        rad_energy_density: bool,
        c: float,
        _from_cli: bool,
        _metadata: dict[str, Any],
    ):
        self.fname: Path = self.resolve_network_path(fname)
        self.config: Path | None = Path(config) if isinstance(config, str) else config
        self.errors: bool = errors
        self.label: str | None = label

        if not isinstance(funcfile, (bool, str, Path)):
            raise ParserError(f"funcfile accepts True/False/str/Path, got {funcfile!r}")
        # True: scan the network directory.  False: skip.  Path: use as given.
        self.funcfile: bool | Path = (
            funcfile if isinstance(funcfile, bool) else Path(funcfile)
        )

        self.replace_nH: bool = replace_nH
        self.rad_bands: list = rad_bands
        self.rad_powerlaw_index: int | float = rad_powerlaw_index
        self.rad_energy_density: bool = rad_energy_density
        self.c: float = c
        self._from_cli: bool = _from_cli
        self._metadata: dict[str, Any] = _metadata

    @staticmethod
    def resolve_network_path(fname: str | Path) -> Path:
        """Resolve *fname* to a network file path or a predefined network name.

        A predefined network name wins over a same-named path on disk.  A
        predefined name is a sub-directory of :data:`~jaff.config.NETWORKS_DIR`
        that contains exactly one ``.jet`` file.  Non-predefined names are
        treated as filesystem paths, with relative paths resolved against the
        current working directory.

        Parameters
        ----------
        fname : str | Path
            A filesystem path or a predefined network name.

        Returns
        -------
        Path
            The resolved, absolute network file path.

        Raises
        ------
        FileNotFoundError
            If *fname* is neither an existing file nor a predefined name, or a
            predefined network directory has no ``.jet`` file.
        ParserError
            If a predefined network directory has more than one ``.jet`` file.
        """
        p = Path(fname)
        abspath = p.resolve()

        names = predefined_networks()
        if p.name not in names:
            if not abspath.exists():
                raise FileNotFoundError(f"Network file '{fname}' not found")

            return abspath

        ndir = NETWORKS_DIR / p.name
        jets = sorted(f for f in ndir.iterdir() if f.suffix.lower() == ".jet")
        if not jets:
            raise FileNotFoundError(f"No .jet file in predefined network '{p.name}'")

        if len(jets) > 1:
            raise ParserError(
                f"Predefined network '{p.name}' has multiple .jet files: "
                f"{[j.name for j in jets]}"
            )

        return jets[0].resolve()
