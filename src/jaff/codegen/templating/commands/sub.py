# ABOUTME: SUB command: substitute network-wide tokens such as $nspec$ in the body
# ABOUTME: Each body line keeps its indentation; REPLACE runs on every rendered line

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from ..placeholders import substitute
from .base import Command, check_names, render_each_line

if TYPE_CHECKING:
    from ..context import RenderContext
    from ..resolve import Resolved

TOKENS: Tuple[str, ...] = (
    "nspec",
    "nelem",
    "nreact",
    "nbands",
    "label",
    "filename",
    "filepath",
    "dedt",
    "dtdt",
    "e_idx",
)


def _values(ctx: RenderContext) -> Dict[str, Callable[[], Any]]:
    """Token name to a callable computing its value for *ctx*."""
    net, cg = ctx.net, ctx.cg
    return {
        "nspec": lambda: net.species.count,
        "nelem": lambda: net.elements.count,
        "nreact": lambda: net.reactions.count,
        "nbands": lambda: net.radiation.nbands if net.radiation else 0,
        "label": lambda: net.label,
        "filename": lambda: ctx.file.name,
        "filepath": lambda: ctx.file,
        "dedt": cg.get_dedt,
        "dtdt": cg.get_dtdt,
        "e_idx": net.species.e_idx,
    }


@dataclass(frozen=True)
class Sub(Command):
    """``SUB tokens``."""

    tokens: Tuple[str, ...]

    def validate(self, resolved: Resolved, ctx: RenderContext) -> None:
        """Every listed token must be a known SUB token."""
        check_names(self.keyword, "token", self.tokens, TOKENS)

    def render(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> str:
        """Substitute the listed tokens on every body line."""
        values = {
            name: value for name, value in _values(ctx).items() if name in self.tokens
        }

        return render_each_line(
            body, lambda text: substitute(text, values), resolved.replacements
        )
