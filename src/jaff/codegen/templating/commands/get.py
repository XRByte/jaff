# ABOUTME: GET command: substitute properties of one species, reaction or element
# ABOUTME: Each body line keeps its indentation; REPLACE runs on every rendered line

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from ..placeholders import substitute
from .base import Command, check_names, render_each_line

if TYPE_CHECKING:
    from ..context import RenderContext
    from ..options import Options

PROPS: Tuple[str, ...] = (
    "element_idx",
    "specie_idx",
    "reaction_idx",
    "specie_mass",
    "specie_charge",
    "specie_latex",
    "reaction_tmin",
    "reaction_tmax",
    "reaction_verbatim",
    "aux_func",
)


def _lookups(ctx: RenderContext) -> Dict[str, Callable[[str], Any]]:
    """Property name to a callable taking the entity and returning the value."""
    net, cg = ctx.net, ctx.cg
    return {
        "element_idx": lambda e: net.elements.symbols().index(e),
        "specie_idx": lambda s: net.species[s].index,
        "reaction_idx": lambda r: net.reactions[r].catalogue_index,
        "specie_mass": lambda s: net.species[s].mass,
        "specie_charge": lambda s: net.species[s].charge,
        "specie_latex": lambda s: net.species[s].latex(),
        "reaction_tmin": lambda r: net.reactions[r].tmin,
        "reaction_tmax": lambda r: net.reactions[r].tmax,
        "reaction_verbatim": lambda r: net.reactions[r].verbatim,
        "aux_func": lambda f: cg.lang.code_gen(
            net.spec.aux_funcs[f]["def"], strict=False, allow_unknown_functions=True
        ),
    }


@dataclass(frozen=True)
class Get(Command):
    """``GET props IN entity``."""

    props: Tuple[str, ...]
    entity: str

    def validate(self, options: Options, ctx: RenderContext) -> None:
        """Every listed property must be a known GET property."""
        check_names(self.keyword, "property", self.props, PROPS)

    def render(self, body: List[str], options: Options, ctx: RenderContext) -> str:
        """Substitute the listed properties of the entity on every body line."""
        lookups = _lookups(ctx)
        values = {
            name: (lambda lookup=lookups[name]: lookup(self.entity))
            for name in self.props
        }

        return render_each_line(body, lambda text: substitute(text, values), options)
