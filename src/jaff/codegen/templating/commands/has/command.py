# ABOUTME: HAS command: substitute 1 or 0 for whether an entity is in the network
# ABOUTME: The placeholder is the entity kind, e.g. $specie$; REPLACE runs per line

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

from ...placeholders import substitute
from ..base import Command, check_names, entity_collections, render_each_line
from .props import KINDS

if TYPE_CHECKING:
    from ...context import RenderContext
    from ...resolve import Resolved


@dataclass(frozen=True)
class Has(Command):
    """``HAS kind entity``."""

    kind: str
    entity: str

    def validate(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> None:
        """The kind must be ``specie``, ``reaction`` or ``element``."""
        check_names(self.keyword, "kind", (self.kind,), KINDS)

    def render(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> str:
        """Substitute ``$<kind>$`` with 1 or 0 on every body line."""
        collection = entity_collections(ctx)[self.kind]
        values: Dict[str, Callable[[], int]] = {
            self.kind: lambda: int(self.entity in collection)
        }

        return render_each_line(
            body, lambda text: substitute(text, values), resolved.replacements
        )
