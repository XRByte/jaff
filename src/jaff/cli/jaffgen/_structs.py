from dataclasses import dataclass, field
from pathlib import Path

from ...config import JAFF_DIR
from ...core import NetworkProps
from ...drivers import Toml

#: Fallback output directory (``<repo_root>/generated``) when none is supplied.
DEFAULT_OUTPUT: Path = JAFF_DIR.parent.parent / "generated"


@dataclass(frozen=True)
class ResolvedPath:
    """A path paired with the relative path used to mirror it under ``outdir``.

    Attributes
    ----------
    abspath : Path
        Absolute, resolved path to the source file or directory.
    relpath : Path
        Path used when writing the file under the output directory
        (``outdir / relpath``), so the source tree structure is preserved.
    """

    abspath: Path
    relpath: Path


@dataclass
class State:
    """Accumulating resolved configuration for a single ``jaffgen`` run.

    Attributes
    ----------
    template : str or None
        Name of the resolved built-in template collection, if any.
    config_file, config_dir : ResolvedPath or None
        The loaded ``jaffgen.toml`` and its containing directory (used to
        resolve config-relative paths).
    config_raw : Toml or None
        Parsed ``jaffgen.toml`` contents.
    network_file : ResolvedPath or None
        Config-provided network file, resolved against the config file's
        directory.  A CLI ``--network`` bypasses this and writes straight to
        :attr:`network_props`.
    network_props : NetworkProps
        Keyword arguments forwarded to the :class:`~jaff.Network` constructor.
    input_dir : ResolvedPath or None
        Resolved ``--indir`` / config ``input_dir`` directory.
    input_files : list of ResolvedPath
        Config-provided individual template files, resolved but not yet added
        to :attr:`output_files`.
    output_dir : ResolvedPath
        Directory the generated files are written to (defaults to
        :data:`DEFAULT_OUTPUT`).
    output_files : list of ResolvedPath
        Every file to render, gathered from the template, ``--indir``,
        ``--files``, and the config file.
    lang : str or None
        Default code-generation language for files whose extension is not
        recognised.
    """

    template: str | None = None
    config_file: ResolvedPath | None = None
    config_dir: ResolvedPath | None = None
    config_raw: Toml | None = None
    network_file: ResolvedPath | None = None
    # _from_cli=True: jaffgen prints its own MOTD banner, so Network must not.
    network_props: NetworkProps = field(
        default_factory=lambda: NetworkProps(_from_cli=True)
    )
    input_dir: ResolvedPath | None = None
    input_files: list[ResolvedPath] = field(default_factory=list)
    output_dir: ResolvedPath = field(
        default_factory=lambda: ResolvedPath(DEFAULT_OUTPUT, Path())
    )
    output_files: list[ResolvedPath] = field(default_factory=list)
    lang: str | None = None
