# ABOUTME: Base class for $$JAFF commands, built from a parsed directive header
# ABOUTME: Subclasses add the header's fields plus validate and render for their body

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Self, Sequence, Set

from ..placeholders import split_line
from ..resolve import apply_replacements, compile_replacements

if TYPE_CHECKING:
    from ..context import RenderContext
    from ..options import Options


class CommandError(ValueError):
    """A problem with a command's header, reported at the directive's line."""


@dataclass(frozen=True)
class Command:
    """A parsed ``$$JAFF`` command; subclasses add the header's fields.

    Attributes:
        has_body: Whether the command owns the lines up to ``$$JAFF END``.
        groups: Option groups the command accepts, with their keys.
    """

    has_body: ClassVar[bool] = True
    groups: ClassVar[Dict[str, Set[str]]] = {}

    @classmethod
    def from_items(cls, items: Sequence[Any]) -> Self:
        """Build the command from its header parts, in grammar order.

        Args:
            items: Header parts as plain strings or tuples of names.

        Returns:
            The command instance.
        """
        return cls(*items)

    @property
    def keyword(self) -> str:
        """The command's keyword as written in templates, e.g. ``SUB``."""
        return type(self).__name__.upper()

    def validate(self, options: Options, ctx: RenderContext) -> None:
        """Check the header against the network before anything is rendered.

        Raises:
            CommandError: If the header names something the command cannot use.
        """

    def render(self, body: List[str], options: Options, ctx: RenderContext) -> str:
        """Render the body lines into output text.

        Args:
            body: Raw body lines, with their line endings.
            options: Layered options of the block.
            ctx: Render context.

        Returns:
            The generated text, one ``\\n``-terminated line per output line.
        """
        raise NotImplementedError(f"{self.keyword} rendering is not implemented")


def render_each_line(
    body: List[str], transform: Callable[[str], str], options: Options
) -> str:
    """Render line by line as the old engine did for SUB, GET and HAS.

    Each line keeps its leading spaces; its stripped content goes through
    *transform* and then the block's REPLACE pairs.
    """
    replacements = compile_replacements(options)
    out = []
    for line in body:
        indent, content = split_line(line)
        out.append(indent + apply_replacements(transform(content), replacements) + "\n")

    return "".join(out)


def check_names(
    keyword: str, kind: str, names: Sequence[str], known: Sequence[str]
) -> None:
    """Every name in *names* must be one of *known*.

    Raises:
        CommandError: Listing the unknown names and the known ones.
    """
    unknown = [name for name in names if name not in known]
    if unknown:
        raise CommandError(
            f"{keyword} has no {kind} {', '.join(unknown)} "
            f"(known: {', '.join(sorted(known))})"
        )
