# ABOUTME: TemplateParser facade of the new $$JAFF engine: same API as the old engine
# ABOUTME: Picks the language from the file extension, renders, wraps Fortran output

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .._languages import Language, wrap_fortran_source
from ..codegen import Codegen
from .context import RenderContext
from .engine import split_blocks
from .render import render_blocks

if TYPE_CHECKING:
    from ...core.network.network import Network


class TemplateParser:
    """Render a ``$$JAFF`` template for a network.

    Examples:
        >>> net = Network("networks/react_COthin")
        >>> output = TemplateParser(net, Path("template.cpp")).parse_file()
    """

    def __init__(
        self,
        network: Network,
        file: Path,
        default_lang: Optional[str] = None,
        echo_directives: bool = True,
    ) -> None:
        """Pick the target language from the file extension.

        Args:
            network: Network to generate code for.
            file: Template path.
            default_lang: Language used when the extension is not recognised.
            echo_directives: Whether directives appear as comments in the output.

        Raises:
            RuntimeError: If the extension is unknown and *default_lang* is ``None``.
        """
        self.net = network
        self.file = Path(file)
        self.echo_directives = echo_directives
        ext = self.file.suffix[1:].lower()
        found = Language.from_extension(ext)
        lang = found.name if found is not None else default_lang
        if lang is None:
            kind = f"{ext} files" if ext else "files without an extension"
            raise RuntimeError(
                f"{kind} are not yet supported\n"
                "Please use the --lang option to specify a language for\n"
                "unsupported files if you are using jaffgen"
            )
        self.cg = Codegen(network=self.net, lang=lang)

    def parse_file(self) -> str:
        """Render the template and return the generated code.

        Raises:
            ParserError: If the template is invalid.
        """
        ctx = RenderContext(self.net, self.cg, self.file, self.echo_directives)
        text = self.file.read_text(encoding="utf-8")
        output = render_blocks(split_blocks(text, self.file), ctx)
        if self.cg.lang.name == "fortran":
            output = wrap_fortran_source(output)
        return output
