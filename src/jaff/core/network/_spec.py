"""The :class:`NetworkSpec` normalized parameters of a :class:`~jaff.Network`."""

from pathlib import Path
from typing import Any

from ...config import NETWORKS_DIR, predefined_networks
from ...drivers import Toml
from ...errors import ParserError
from ..parsers import AuxiliaryFunctionParser


class NetworkSpec:
    """Normalized construction parameters of a :class:`~jaff.Network`.

    ``Network`` bundles its constructor arguments into one of these (exposed as
    :attr:`Network.spec`) and reads them back from it.  Construction parses
    the raw arguments into their canonical forms:

    * ``fname`` is resolved to an absolute path (a filesystem path or a
      predefined network name; see :meth:`resolve_network_path`);
    * ``config`` (a ``jaff.toml`` path, or auto-detected next to the network
      file) is loaded into its parsed ``[network]`` section — a dict, since
      that is what the network actually consumes.
    * ``funcfile`` is coerced to ``bool`` / :class:`~pathlib.Path`, then
      resolved to the actual ``.jfunc`` file (scanning next to the network
      file when ``True``) and its contents are parsed into :attr:`aux_funcs`.

    Every field is required — ``Network`` always supplies all of them, so the
    library defaults live in one place (the ``Network`` constructor signature).

    Attributes
    ----------
    fname : Path
        Resolved, absolute path to the network file.
    config : dict
        Parsed ``[network]`` section of the ``jaff.toml`` config (``{}`` when
        none is supplied or auto-detected).
    errors : bool
        If ``True``, conservation violations and duplicate reactions are fatal.
    label : str or None
        Optional human-readable label for the network.
    funcfile : bool | Path
        ``True`` to scan the network directory, ``False`` to skip, or the
        resolved path of the ``.jfunc`` file actually used.
    aux_funcs : dict
        Parsed auxiliary functions from the ``.jfunc`` file (``{}`` when none).
        Exposed for inspection/debugging.
    duplicate_policy : str
        How to resolve two rate coefficients sharing a reaction, mechanism, and
        temperature range: ``"preserve-first"`` (keep the first, drop later
        ones), ``"preserve-last"`` (keep the last), or ``"error"`` (raise).
        Resolved from the constructor argument, else the network ``jaff.toml``
        ``[network].duplicate_policy`` key, else ``"preserve-first"``.
    replace_nH : bool
        Whether density symbols are rewritten in terms of ``nH``.
    rad_bands : list
        Radiation-field band definitions.
    rad_powerlaw_index : int | float
        Spectral power-law index of the radiation field.
    rad_energy_density : bool
        Whether the radiation field is given as an energy density.
    c : float or str
        Speed of light in the unit system used by the network.
    _from_cli : bool
        Internal flag marking construction from the command-line interface.
    _metadata : dict
        Internal per-reaction/network metadata carried through construction.
    """

    DUPLICATE_POLICIES = ["preserve-first", "preserve-last", "error"]

    def __init__(
        self,
        fname: str | Path,
        config: str | Path | None,
        errors: bool,
        label: str | None,
        funcfile: bool | str | Path,
        duplicate_policy: str | None,
        replace_nH: bool,
        rad_bands: list,
        rad_powerlaw_index: int | float,
        rad_energy_density: bool,
        c: float | str,
        _from_cli: bool,
        _metadata: dict[str, Any],
    ):
        self.fname: Path = self._resolve_network_path(fname)
        self.config: dict[str, Any] = self._load_config(config, self.fname)
        self.errors: bool = errors
        self.label: str | None = label

        if not isinstance(funcfile, (bool, str, Path)):
            raise ParserError(f"funcfile accepts True/False/str/Path, got {funcfile!r}")
        # True: scan the network directory.  False: skip.  Path: use as given.
        self.funcfile: bool | Path = (
            funcfile if isinstance(funcfile, bool) else Path(funcfile)
        )
        resolved_policy = (
            duplicate_policy or self.config.get("duplicate_policy") or "preserve-first"
        )
        if resolved_policy not in self.DUPLICATE_POLICIES:
            raise ValueError(
                f"Invalid duplicate policy: {resolved_policy}\n"
                f"Valid duplicate policies are {', '.join(self.DUPLICATE_POLICIES)}"
            )
        self.duplicate_policy: str = resolved_policy
        # Resolves funcfile to the actual .jfunc path (when True) and parses it.
        self.aux_funcs: dict = self._load_aux_funcs()
        self.replace_nH: bool = replace_nH
        self.rad_bands: list = rad_bands
        self.rad_powerlaw_index: int | float = rad_powerlaw_index
        self.rad_energy_density: bool = rad_energy_density
        self.c: float = c
        self._from_cli: bool = _from_cli
        self._metadata: dict[str, Any] = _metadata

    def _load_aux_funcs(self) -> dict:
        """Detect and parse the auxiliary ``.jfunc`` file into a dict.

        When :attr:`funcfile` is ``True``, scans next to the network file for
        ``<network>.jfunc`` (both ``<name>.jfunc`` and ``<stem>.jfunc``) and
        rewrites :attr:`funcfile` to the resolved path.  ``False`` skips loading.

        Returns
        -------
        dict
            Parsed auxiliary functions, or ``{}`` when there is no ``.jfunc``.

        Raises
        ------
        FileNotFoundError
            If :attr:`funcfile` is an explicit path that does not exist.
        """
        if self.funcfile is False:
            return {}

        if self.funcfile is True:
            candidates = [
                Path(f"{self.fname}.jfunc"),
                self.fname.with_suffix(".jfunc"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.funcfile = candidate
                    break
            else:
                return {}

        assert isinstance(self.funcfile, Path)
        if not self.funcfile.exists():
            raise FileNotFoundError(self.funcfile)

        with AuxiliaryFunctionParser(self.funcfile) as afp:
            return afp.get_dict()

    @staticmethod
    def _load_config(config: str | Path | None, fname: Path) -> dict[str, Any]:
        """Locate and parse a ``jaff.toml`` config, returning its ``[network]`` section.

        When *config* is ``None``, auto-detects ``<network_dir>/jaff.toml`` next
        to the (resolved) network file *fname*.

        Parameters
        ----------
        config : str | Path | None
            Explicit path to a ``jaff.toml`` config file, or ``None`` to
            auto-detect.
        fname : Path
            Resolved network file path, used to auto-detect a sibling config.

        Returns
        -------
        dict
            The parsed ``[network]`` section, or ``{}`` when no config exists.
        """
        if config is None:
            candidate = fname.parent / "jaff.toml"
            if not candidate.exists():
                return {}
            config = candidate

        return Toml(Path(config).resolve()).get_key("network") or {}

    @staticmethod
    def _resolve_network_path(fname: str | Path) -> Path:
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
