# ABOUTME: REPEAT command: expand body lines once per item of a network property
# ABOUTME: Vertical with idx (one line per item), else horizontal into a bracketed list

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Set, Tuple

from ...errors import TemplateError
from ...placeholders import split_line
from ..base import Command, check_names, finish_line
from .props import GROUPS, PROPERTIES, RepeatProp, option_kwargs
from .expand import as_flattened, expand_horizontal, expand_vertical, find_idx

if TYPE_CHECKING:
    from ...context import RenderContext
    from ...resolve import Resolved


def _active(line: str, names: Tuple[str, ...]) -> bool:
    """Whether a body line holds a placeholder of this REPEAT.

    Placeholders are ``$name$`` for *names* (the item and the declared special
    variables such as ``cse``), ``$idx$`` with or without arithmetic, and
    ``$IDX$``. Other lines are copied, with REPLACE applied.
    """
    pattern = "|".join(re.escape(name) for name in names)
    placeholder = re.search(rf"\$(?:{pattern})\$", line)
    return bool(placeholder or find_idx(line) or "$IDX$" in line)


@dataclass(frozen=True)
class Repeat(Command):
    """``REPEAT vars IN prop``."""

    groups: ClassVar[Dict[str, Set[str]]] = GROUPS
    vars: Tuple[str, ...]
    prop: str

    def validate(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> None:
        """Check the property, its variables, its options and the CSE line."""
        check_names(self.keyword, "property", (self.prop,), tuple(PROPERTIES))
        prop = PROPERTIES[self.prop]
        check_names(f"{self.keyword} IN {self.prop}", "variable", self.vars, prop.vars)
        self._kwargs(body, resolved)

    def render(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> str:
        """Expand every line holding a placeholder; copy the others once.

        REPLACE runs on the content of every output line, expanded or copied.
        """
        prop = PROPERTIES[self.prop]
        # Recomputed rather than kept from validate: commands are immutable, and
        # the options and CSE scan are cheap next to codegen.
        kwargs = self._kwargs(body, resolved)
        items: Any = None
        out: List[str] = []
        for index, raw in enumerate(body):
            indent, line = split_line(raw)
            contents = [line]
            if _active(line, self._names(prop)):
                if items is None:
                    items = prop.source(ctx, kwargs)
                contents = self._expand(prop, items, line, index)

            out.extend(finish_line(indent, c, resolved.replacements) for c in contents)

        return "".join(out)

    def _expand(self, prop: RepeatProp, items: Any, line: str, index: int) -> List[str]:
        """The contents of the output lines for one line holding a placeholder."""
        extras: Dict[str, Any] = {}
        if prop.special:
            extras, items = items["extras"], items["expressions"]

        if "idx" not in self.vars:
            expanded = expand_horizontal(items, prop.item, line)
            return [line if expanded is None else expanded]

        for var in prop.special:
            if var in self.vars and f"${var}$" in line:
                return expand_vertical(extras[var], line, f"${var}$", index)

        return expand_vertical(as_flattened(items), line, f"${prop.item}$", index)

    def _names(self, prop: RepeatProp) -> Tuple[str, ...]:
        """The item variable and the special variables this REPEAT declares."""
        return (prop.item, *(var for var in prop.special if var in self.vars))

    def _kwargs(self, body: List[str], resolved: Resolved) -> Dict[str, Any]:
        """Keyword arguments for the property's source: options, then CSE naming."""
        prop = PROPERTIES[self.prop]
        kwargs = option_kwargs(self.prop, resolved.options)
        if "cse" in prop.special:
            kwargs.update(self._cse_kwargs(body, prop))

        return kwargs

    def _cse_kwargs(self, body: List[str], prop: RepeatProp) -> Dict[str, Any]:
        """CSE switch and temporary names, taken from the ``$cse$`` line.

        The ``$cse$`` line must be the first line that expands, and names its
        temporaries with an identifier around a plain ``$idx$``, e.g. ``cse$idx$``.

        Raises:
            TemplateError: At the offending body line.
        """
        if "cse" not in self.vars:
            return {"use_cse": False}

        index = next(
            (i for i, raw in enumerate(body) if _active(raw.strip(), self._names(prop))),
            None,
        )
        if index is None:
            return {"use_cse": True}

        line = body[index].strip()

        def error(message: str) -> TemplateError:
            return TemplateError(message, body_index=index)

        if "$cse$" not in line:
            raise error(
                "The $cse$ line must be the first line of the REPEAT block "
                "so CSE temporaries are declared before they are used"
            )
        spans = find_idx(line)
        if not spans:
            raise error("No valid idx variable detected")
        (begin, end), op, _ = spans[0]
        if op:
            raise error(
                "CSE temporaries cannot use arithmetic on $idx$; use $idx$ in the $cse$ name"
            )

        before = re.search(r"\w*$", line[:begin])
        after = re.match(r"\w*", line[end:])
        prefix = before.group() if before else ""
        suffix = after.group() if after else ""
        if not re.fullmatch(r"[A-Za-z_]\w*", prefix):
            raise error(
                "CSE temporary must be an identifier like cse$idx$ or tmp$idx$_value; "
                f"got '{prefix}$idx${suffix}'"
            )

        return {"use_cse": True, "cse_var": prefix, "cse_suffix": suffix}
