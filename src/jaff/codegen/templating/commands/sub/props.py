# ABOUTME: SUB's tokens: the names templates may substitute and how each value is computed
# ABOUTME: Add a token to TOKENS and to token_values(); the SUB command reads both

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

if TYPE_CHECKING:
    from ...context import RenderContext

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


def token_values(ctx: RenderContext) -> Dict[str, Callable[[], Any]]:
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
