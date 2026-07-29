import logging
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import typer

from ... import Network
from ...codegen import TemplateParser
from ...common import motd
from ...config import JAFF_DIR, TEMPLATES_DIR
from ...drivers import Toml
from ...errors import ParserError
from ...io import JaffLogger, jaff_progress
from .._helper import funcfile_arg
from ._structs import ResolvedPath, State


class JaffGen:
    def __init__(self, args: SimpleNamespace):
        self.logger: logging.Logger = JaffLogger().get_logger()
        self.args: SimpleNamespace = args
        self.state: State = State()
        self.parse_args()
        self.process_files()

    def parse_args(self) -> None:
        print(motd("jaffgen"))
        self.set_config(self.args.config)
        self.set_template(self.state.template or self.args.template)
        if self.state.config_file is None and self.state.template is not None:
            self.detect_config_file(self.state.output_files)

        self.set_input_dir(self.args.indir)
        self.set_input_files(self.args.files)
        self.set_output_dir(self.args.outdir)

        if self.state.config_file:
            self.state.output_files.append(self.state.config_file)

        self.state.output_dir.abspath.mkdir(parents=True, exist_ok=True)

    def set_config(self, config_file: Path | str | None):
        if config_file is None:
            return

        if isinstance(config_file, str):
            config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(config_file)

        if not config_file.is_file():
            raise FileNotFoundError(f"{config_file} is not a file")

        self.state.config_file = ResolvedPath(config_file.resolve(), config_file)
        self.state.config_dir = ResolvedPath(
            self.state.config_file.abspath.parent, self.state.config_file.relpath.parent
        )

        self.state.config_raw = Toml(self.state.config_file.abspath)
        self.set_state_from_config()

    def set_template(self, template: str | None) -> None:
        if template is None:
            return

        gen_dir = TEMPLATES_DIR / "generator"
        pproc_dir = TEMPLATES_DIR / "preprocessor"
        valid_templates: list[str] = [f.name for f in gen_dir.iterdir() if f.is_dir()]

        # Validate that the requested template exists
        if template not in valid_templates:
            raise ValueError(
                f"Invalid template name. Supported templates are {valid_templates}"
            )

        gtemplate_dir = gen_dir / template
        ptemplate_dir = pproc_dir / template
        gfiles = [
            ResolvedPath(f, f.relative_to(gtemplate_dir))
            for f in gtemplate_dir.rglob("*")
            if not f.is_dir()
        ]
        pfiles = [
            ResolvedPath(f, f.relative_to(ptemplate_dir))
            for f in ptemplate_dir.rglob("*")
            if not f.is_dir()
        ]
        extras = {f.relpath for f in pfiles} - {f.relpath for f in gfiles}

        ss = self.state
        ss.output_files.extend(gfiles)
        ss.output_files.extend([f for f in pfiles if f.relpath in extras])
        self.state.template = template

    def set_input_dir(self, input_dir: str | None) -> None:
        if input_dir is None:
            return

        idir = self.get_resolved_path(input_dir, Path.cwd())
        self.state.input_dir = idir
        self.state.output_files.extend(
            [
                self.get_resolved_path(f, Path.cwd())
                for f in idir.abspath.rglob("*")
                if not f.is_dir()
            ]
        )

    def set_input_files(self, files: list[str] | None) -> None:
        if files is None:
            return

        self.state.output_files.extend(
            [self.get_resolved_path(f, Path.cwd()) for f in files]
        )

    def set_output_dir(self, output_dir: str | None) -> None:
        if output_dir is None:
            self.logger.warning("No output directory has been supplied.")
            self.logger.warning(
                f"Files will be generated at {JAFF_DIR.parent.parent / 'generated'}"
            )
            return

        odir = self.get_resolved_path(output_dir, Path.cwd())
        self.state.output_dir = odir

    def set_state_from_config(self):
        ss = self.state
        assert ss.config_dir is not None
        assert ss.config_file is not None

        jgp = self.get_prop("jaffgen")
        if jgp:
            ss.template = jgp.get("template") or ss.template

            if d := jgp.get("input_dir") is not None:
                ss.input_dir = self.get_resolved_path(d, ss.config_dir.abspath)

            if fs := jgp.get("input_files") is not None:
                ss.input_files = [
                    self.get_resolved_path(f, ss.config_dir.abspath) for f in fs
                ]

            if d := jgp.get("output_dir") is not None:
                ss.output_dir = self.get_resolved_path(d, ss.config_dir.abspath)

            if f := jgp.get("network_file") is not None:
                ss.network_file = self.get_resolved_path(f, ss.config_dir.abspath)

            ss.network_dir = ResolvedPath(
                ss.network_file.abspath.parent, ss.network_file.relpath.parent
            )
            ss.network_props.fname = ss.network_file.abspath
            ss.lang = jgp.get("lang") or ss.lang

        np = self.get_prop("network")
        if np:
            sn = ss.network_props
            sn.label = np.get("label") or sn.label
            sn.errors = sn.errors if np.get("errors") is None else np.get("errors")
            sn.config = np.get("config") or sn.config
            sn.replace_nH = (
                sn.replace_nH if np.get("replace_nH") is None else np.get("replace_nH")
            )
            sn.funcfile = (
                sn.funcfile if np.get("funcfile") is None else np.get("funcfile")
            )

            nr = np.get("radiation")
            if nr:
                sn.rad_bands = nr.get("rad_bands") or sn.rad_bands
                sn.rad_powerlaw_index = (
                    sn.rad_powerlaw_index
                    if nr.get("power_law_index") is None
                    else nr.get("power_law_index")
                )
                sn.rad_energy_density = (
                    sn.rad_energy_density
                    if nr.get("energy_density") is None
                    else np.get("energy_density")
                )
                sn.c = nr.get("rsl") or sn.c

    def detect_config_file(self, files: list[ResolvedPath]) -> None:
        count: int = 0
        index: int = -1
        for i, f in enumerate(files):
            if f.abspath.name == "jaffgen.toml":
                index = i
                count += 1

        if count == 0:
            return

        if count > 1:
            raise ParserError(
                "More than one jaffgen.toml file found in template directory"
            )

        self.state.config_file = files[index]
        self.state.config_dir = ResolvedPath(
            files[index].abspath.parent, files[index].relpath.parent
        )
        self.set_config(self.state.config_file.abspath)

    def get_prop(self, key: str, prop: str | None = None) -> Any | None:
        if self.state.config_raw is None:
            return None

        return (
            self.state.config_raw.get_key(key)
            if prop is None
            else (self.state.config_raw.get_key(key) or {}).get(prop, {})
        )

    def get_resolved_path(self, f: Any, relative_to: Path) -> ResolvedPath:
        if not isinstance(f, str):
            raise ParserError("input_dir must be a string")

        f: Path = Path(f)
        return ResolvedPath(f.absolute(), f.relative_to(relative_to))

    def process_files(self) -> None:
        for file in jaff_progress.track(
            self.state.output_files, description="Processing files"
        ):
            fparser: TemplateParser = TemplateParser(
                Network(**asdict(self.state.network_props)), file.abspath, self.state.lang
            )

            lines: str = fparser.parse_file()

            outfile: Path = self.state.output_dir.abspath / file.relpath
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_text(lines)

            self.logger.info(
                f"[cyan]{file.relpath.name}[/] created at {self.state.output_dir.abspath}"
            )

        self.logger.info("[green]Successfully generated files[/]")
        self.logger.info(
            f"Generated files can be found at {self.state.output_dir.abspath}"
        )

    def __handle_data_tables(self, props: list) -> None:
        """
        Process ``[[table]]`` entries from the TOML config and write outputs.

        Each entry is passed to :class:`~jaff.cli.ConfigTable` for parsing.
        Depending on the target format declared in the table block, the result
        is written as either an HDF5 file or a CSV file.

        Parameters
        ----------
        props : list of dict
            List of table configuration dictionaries from the ``[[table]]``
            TOML array.

        Returns
        -------
        None
        """
        assert self.jaffgen_config["config_file"] is not None

        for table_props in props:
            ct = ConfigTable(
                table_props,
                self.jaffgen_config["config_file"],
                self.jaffgen_config["network_file"],
            )
            parsed_out = ct.parse()

            # Write HDF5 output when the target format is HDF5.
            if isinstance(parsed_out, HDF5Dict):
                HDF5().from_dict(
                    self.jaffgen_config["output_dir"] / ct.target_props["path"],
                    parsed_out,
                )

            # Write CSV output when the target format is CSV.
            if isinstance(parsed_out, pd.DataFrame):
                parsed_out.to_csv(
                    self.jaffgen_config["output_dir"] / ct.target_props["path"],
                    sep=ct.target_props["delimiter"],
                )


class Lang(str, Enum):
    """Supported ``--lang`` values for default code generation."""

    c = "c"
    cxx = "cxx"
    fortran = "fortran"
    python = "python"
    rust = "rust"
    julia = "julia"


app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Generate code for chemical reaction networks in multiple programming languages.",
)


@app.command()
def generate(
    network: Optional[str] = typer.Option(
        None,
        "--network",
        metavar="FILE",
        help="Path to chemical reaction network file (required)",
    ),
    config: Optional[str] = typer.Option(
        None, "--config", metavar="FILE", help="Path to jaff config file"
    ),
    label: Optional[str] = typer.Option(
        None,
        "--label",
        metavar="TEXT",
        help="Network will be generated by the supplied label. Defaults to network file name",
    ),
    funcfile: Optional[str] = typer.Option(
        None,
        "--funcfile",
        metavar="FILE",
        help="Path to auxiliary function file. Checks network dir for <network_name>.jfunc by default ('true'). Pass 'false' to skip",
    ),
    replace_nh: Optional[bool] = typer.Option(
        None,
        "--replace-nH/--no-replace-nH",
        help="Standardizes symbols when true",
    ),
    errors: Optional[bool] = typer.Option(
        None,
        "--errors/--no-errors",
        help="Stops parsing if physical errors are encountered",
    ),
    network_config: Optional[str] = typer.Option(
        None,
        "--network-config",
        metavar="FILE",
        help="Path to a jaff.toml network config (temperature cutoffs). "
        "Defaults to <network_dir>/jaff.toml auto-detected by Network",
    ),
    outdir: Optional[str] = typer.Option(
        None,
        "--outdir",
        metavar="DIR",
        help="Output directory for generated files (default: jaff/generated)",
    ),
    indir: Optional[str] = typer.Option(
        None,
        "--indir",
        metavar="DIR",
        help="Directory containing template files to process",
    ),
    files: Optional[List[str]] = typer.Option(
        None,
        "--files",
        metavar="FILE",
        help="Individual template file(s) to process",
    ),
    template: Optional[str] = typer.Option(
        None,
        "--template",
        metavar="NAME",
        help="Name of predefined template collection in jaff/templates/generator/",
    ),
    lang: Optional[Lang] = typer.Option(
        None,
        "--lang",
        metavar="LANGUAGE",
        help="Default programming language for unsupported files",
    ),
):
    """
    Generate code for chemical reaction networks in multiple programming
    languages.

    Examples:

      # Generate from a template directory
      jaffgen --network networks/react_COthin --indir templates/ --outdir output/

      # Use a predefined template collection
      jaffgen --network networks/react_COthin --template chemistry_solver --outdir output/

      # Process specific files with Rust
      jaffgen --network networks/test.dat --files rates.txt odes.txt --lang rust --outdir output/

      # Combine template and custom files
      jaffgen --network networks/test.dat --template base --files custom.cpp --outdir output/

    Supported Languages: c, cxx (c++, cpp), fortran (f90), python (py),
    rust (rs), julia (jl), r
    """
    args = SimpleNamespace(
        network=network,
        config=config,
        label=label,
        # Map "true"/"false" onto booleans, matching the old argparse type.
        funcfile=funcfile_arg(funcfile) if funcfile is not None else None,
        replace_nH=replace_nh,
        errors=errors,
        network_config=network_config,
        outdir=outdir,
        indir=indir,
        files=files,
        template=template,
        lang=lang.value if lang is not None else None,
    )
    JaffGen(args)


def main():
    """Entry point registered as the ``jaffgen`` console script."""
    app()


if __name__ == "__main__":
    main()
