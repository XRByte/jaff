# ABOUTME: HAS command: substitute 1 or 0 for whether an entity is in the network
# ABOUTME: The placeholder is the entity kind, e.g. $specie$; REPLACE runs per line

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from ..placeholders import substitute
from .base import Command, check_names, render_each_line

if TYPE_CHECKING:
    from ..context import RenderContext
    from ..options import Options

KINDS = ("specie", "reaction", "element")


def _collections(ctx: RenderContext) -> Dict[str, Any]:
    """Entity kind to the network collection that holds it."""
    net = ctx.net
    return {"specie": net.species, "reaction": net.reactions, "element": net.elements}


@dataclass(frozen=True)
class Has(Command):
    """``HAS kind entity``."""

    kind: str
    entity: str

    def validate(self, options: Options, ctx: RenderContext) -> None:
        """The kind must be ``specie``, ``reaction`` or ``element``."""
        check_names(self.keyword, "kind", (self.kind,), KINDS)

    def render(self, body: List[str], options: Options, ctx: RenderContext) -> str:
        """Substitute ``$<kind>$`` with 1 or 0 on every body line."""
        collection = _collections(ctx)[self.kind]
        values: Dict[str, Callable[[], int]] = {
            self.kind: lambda: int(self.entity in collection)
        }

        return render_each_line(body, lambda text: substitute(text, values), options)
