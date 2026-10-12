# ABOUTME: GET command: substitute properties of a species, reaction, element or aux func
# ABOUTME: Each body line keeps its indentation; REPLACE runs on every rendered line

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from ..placeholders import substitute
from .base import Command, CommandError, check_names, entity_collections, render_each_line

if TYPE_CHECKING:
    from ..context import RenderContext
    from ..options import Options

# Property name -> (entity kind, getter taking the context and the entity)
_Getter = Callable[["RenderContext", str], Any]
_PROPERTIES: Dict[str, Tuple[str, _Getter]] = {
    "element_idx": ("element", lambda c, e: c.net.elements.symbols().index(e)),
    "specie_idx": ("specie", lambda c, s: c.net.species[s].index),
    "reaction_idx": ("reaction", lambda c, r: c.net.reactions[r].catalogue_index),
    "specie_mass": ("specie", lambda c, s: c.net.species[s].mass),
    "specie_charge": ("specie", lambda c, s: c.net.species[s].charge),
    "specie_latex": ("specie", lambda c, s: c.net.species[s].latex()),
    "reaction_tmin": ("reaction", lambda c, r: c.net.reactions[r].tmin),
    "reaction_tmax": ("reaction", lambda c, r: c.net.reactions[r].tmax),
    "reaction_verbatim": ("reaction", lambda c, r: c.net.reactions[r].verbatim),
    "aux_func": (
        "aux",
        lambda c, f: c.cg.lang.code_gen(
            c.net.spec.aux_funcs[f]["def"], strict=False, allow_unknown_functions=True
        ),
    ),
}
PROPS: Tuple[str, ...] = tuple(_PROPERTIES)

_KIND_NAMES = {
    "element": "an element",
    "specie": "a species",
    "reaction": "a reaction",
    "aux": "an auxiliary function",
}


def _has_entity(ctx: RenderContext, kind: str, entity: str) -> bool:
    """Whether *entity* exists as *kind* and resolves to a single item for GET."""
    if kind == "aux":
        return entity in ctx.net.spec.aux_funcs

    if entity not in entity_collections(ctx)[kind]:
        return False

    # A reaction name shared by several mechanisms gives a list, which has no tmin etc.
    return kind != "reaction" or not isinstance(ctx.net.reactions[entity], list)


@dataclass(frozen=True)
class Get(Command):
    """``GET props IN entity``."""

    props: Tuple[str, ...]
    entity: str

    def validate(self, options: Options, ctx: RenderContext) -> None:
        """Every property must be known and the entity must exist as its kind."""
        check_names(self.keyword, "property", self.props, PROPS)
        for name in self.props:
            kind = _PROPERTIES[name][0]
            if not _has_entity(ctx, kind, self.entity):
                raise CommandError(
                    f"{self.keyword} {name} needs {_KIND_NAMES[kind]}; "
                    f"{self.entity} is not one in this network"
                )

    def render(self, body: List[str], options: Options, ctx: RenderContext) -> str:
        """Substitute the listed properties of the entity on every body line."""
        values = {
            name: (lambda get=_PROPERTIES[name][1]: get(ctx, self.entity))
            for name in self.props
        }

        return render_each_line(body, lambda text: substitute(text, values), options)
