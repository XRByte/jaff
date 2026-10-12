# ABOUTME: REDUCE command: expand $( ... )$ into a sum over network properties
# ABOUTME: The i-th summand takes the i-th value of every listed property

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..errors import TemplateError
from ..placeholders import split_line
from .base import Command, check_names, finish_line

if TYPE_CHECKING:
    from ..context import RenderContext
    from ..resolve import Resolved

# Property name -> (placeholder variable, getter of the list of values)
_Getter = Callable[["RenderContext"], List[Any]]
_PROPERTIES: Dict[str, Tuple[str, _Getter]] = {
    "specie_charges": ("specie_charge", lambda c: c.net.species.charges()),
    "specie_masses": ("specie_mass", lambda c: c.net.species.masses()),
    "charge_truths": ("charge_truth", lambda c: c.net.species.charge_truths()),
    "photo_reaction_truths": (
        "photo_reaction_truth",
        lambda c: c.net.reactions.photo_reaction_truths(),
    ),
    "photo_reaction_indices": (
        "photo_reaction_index",
        lambda c: c.net.reactions.photo_reaction_indices(),
    ),
    "tmins": ("tmin", lambda c: c.net.reactions.tmins()),
    "tmaxes": ("tmax", lambda c: c.net.reactions.tmaxes()),
    "specie_masses_ne": ("specie_mass_ne", lambda c: c.net.species.masses(ne=True)),
    "specie_charges_ne": (
        "specie_charge_ne",
        lambda c: c.net.species.charges(ne=True),
    ),
    "charged_specie_charges_ne": (
        "charged_specie_charge_ne",
        lambda c: c.net.species.charged("charge", ne=True),
    ),
    "charged_specie_charges": (
        "charged_specie_charge",
        lambda c: c.net.species.charged("charge"),
    ),
    "charge_truths_ne": (
        "charge_truth_ne",
        lambda c: c.net.species.charge_truths(ne=True),
    ),
    "neutral_specie_indices": (
        "neutral_specie_index",
        lambda c: c.net.species.neutral("index"),
    ),
    "charged_specie_indices": (
        "charged_specie_index",
        lambda c: c.net.species.charged("index"),
    ),
    "charged_specie_indices_ne": (
        "charged_specie_index_ne",
        lambda c: c.net.species.charged("index", ne=True),
    ),
    "charged_specie_masses_ne": (
        "charged_specie_mass_ne",
        lambda c: c.net.species.charged("mass", ne=True),
    ),
    "neutral_specie_masses": (
        "neutral_specie_mass",
        lambda c: c.net.species.neutral("mass"),
    ),
    "charged_specie_masses": (
        "charged_specie_mass",
        lambda c: c.net.species.charged("mass"),
    ),
}
PROPS: Tuple[str, ...] = tuple(_PROPERTIES)

# $( ... )$: group 1 is the whole reduction, group 2 the summand template
_REDUCTION = re.compile(r"(\$\((.*?)\)\$)")


@dataclass(frozen=True)
class Reduce(Command):
    """``REDUCE vars IN props``."""

    vars: Tuple[str, ...]
    props: Tuple[str, ...]

    def validate(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> None:
        """Properties must be known and every variable must belong to one of them."""
        check_names(self.keyword, "property", self.props, PROPS)
        known = [_PROPERTIES[prop][0] for prop in self.props]
        check_names(self.keyword, "variable", self.vars, known)

    def render(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> str:
        """Expand every ``$( ... )$`` that uses a listed variable, on every line.

        Other text is copied. REPLACE runs on the content of every line.
        """
        columns: Optional[Dict[str, List[Any]]] = None

        def expand(match: re.Match[str], index: int) -> str:
            nonlocal columns
            summand = match.group(2)
            if not any(f"${var}$" in summand for var in self.vars):
                return match.group(1)
            if columns is None:
                columns = self._columns(ctx)
            return self._sum(summand, columns, index)

        out = []
        for index, raw in enumerate(body):
            indent, line = split_line(raw)
            line = _REDUCTION.sub(lambda m: expand(m, index), line)
            out.append(finish_line(indent, line, resolved.replacements))

        return "".join(out)

    def _columns(self, ctx: RenderContext) -> Dict[str, List[Any]]:
        """Placeholder variable to its list of values, in the order of the props."""
        columns = {}
        for prop in self.props:
            var, getter = _PROPERTIES[prop]
            columns[var] = getter(ctx)

        return columns

    def _sum(self, summand: str, columns: Dict[str, List[Any]], index: int) -> str:
        """``((s0) + (s1) + ...)`` over the values, or ``0`` without any.

        The first property sets the number of terms; extra values in later ones
        are ignored, as in the old engine.
        """
        count = len(next(iter(columns.values())))
        terms = []
        for i in range(count):
            term = summand
            for var, values in columns.items():
                if i >= len(values):
                    raise TemplateError(
                        f"{self.keyword} properties differ in length: "
                        f"{', '.join(self.props)}",
                        body_index=index,
                    )

                term = term.replace(f"${var}$", str(values[i]))
            terms.append(f"({term})")

        return f"({' + '.join(terms)})" if terms else "0"
