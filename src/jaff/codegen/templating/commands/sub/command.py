# ABOUTME: SUB command: substitute network-wide tokens such as $nspec$ in the body
# ABOUTME: Each body line keeps its indentation; REPLACE runs on every rendered line

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

from ...placeholders import substitute
from ..base import Command, check_names, render_each_line
from .props import TOKENS

if TYPE_CHECKING:
    from ...context import RenderContext
    from ...resolve import Resolved


@dataclass(frozen=True)
class Sub(Command):
    """``SUB tokens``."""

    tokens: Tuple[str, ...]

    def validate(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> None:
        """Every listed token must be a known SUB token."""
        check_names(self.keyword, "token", self.tokens, tuple(TOKENS))

    def render(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> str:
        """Substitute the listed tokens on every body line."""
        values = {name: (lambda get=TOKENS[name]: get(ctx)) for name in self.tokens}

        return render_each_line(
            body, lambda text: substitute(text, values), resolved.replacements
        )
