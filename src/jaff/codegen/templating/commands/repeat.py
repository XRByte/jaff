# ABOUTME: REPEAT command: expand body lines once per item of a network property
# ABOUTME: Vertical with idx (one line per item), else horizontal into a bracketed list

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set, Tuple

from ..errors import TemplateError
from ..placeholders import split_line
from ..resolve import apply_replacements
from .base import Command, check_names
from .repeat_props import GROUPS, PROPERTIES, RepeatProp, option_kwargs
from .repeat_render import as_flattened, expand_horizontal, expand_vertical, find_idx

if TYPE_CHECKING:
    from ..context import RenderContext
    from ..resolve import Resolved


def _active(line: str, prop: RepeatProp) -> bool:
    """Whether a body line holds a placeholder of this REPEAT.

    Placeholders are ``$item$``, special variables such as ``$cse$``, ``$idx$``
    with or without an offset, and ``$IDX$``. Other lines are copied unchanged.
    """
    names = "|".join(re.escape(var) for var in prop.vars[1:])
    placeholder = re.search(rf"\$(?:{names})\$", line)
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
        """Expand every body line that uses the property's variables."""
        prop = PROPERTIES[self.prop]
        # Recomputed rather than kept from validate: commands are immutable, and
        # the options and CSE scan are cheap next to codegen.
        kwargs = self._kwargs(body, resolved)
        items: Any = None
        out: List[str] = []
        for index, raw in enumerate(body):
            indent, line = split_line(raw)
            if not _active(line, prop):
                out.append(raw)
                continue

            if items is None:
                items = prop.source(ctx, kwargs)

            expanded = self._expand(prop, items, indent, line, index, resolved)
            out.append(raw if expanded is None else expanded)

        return "".join(out)

    def _expand(
        self,
        prop: RepeatProp,
        items: Any,
        indent: str,
        line: str,
        index: int,
        resolved: Resolved,
    ) -> Optional[str]:
        """One active line expanded, or ``None`` to copy it unchanged.

        REPLACE runs where the old engine ran it: on the expanded lines with
        their indentation in vertical mode, on the content in horizontal mode.
        """
        replacements = resolved.replacements
        extras: Dict[str, Any] = {}
        if prop.special:
            extras, items = items["extras"], items["expressions"]

        if "idx" not in self.vars:
            expanded = expand_horizontal(items, prop.item, line)
            if expanded is None:
                return None

            return indent + apply_replacements(expanded, replacements) + "\n"

        for var in prop.special:
            if f"${var}$" in line:
                text = expand_vertical(extras[var], indent, line, f"${var}$", index)
                # The old engine ended this block with an extra newline.
                return apply_replacements(text, replacements) + "\n"

        text = expand_vertical(as_flattened(items), indent, line, f"${prop.item}$", index)
        return apply_replacements(text, replacements)

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
            (i for i, raw in enumerate(body) if _active(raw.strip(), prop)), None
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
        (begin, end), offset = spans[0]
        if offset:
            raise error(
                "CSE temporaries cannot use an offset index; use $idx$ in the $cse$ name"
            )

        prefix = re.search(r"\w*$", line[:begin]).group()
        suffix = re.match(r"\w*", line[end:]).group()
        if not re.fullmatch(r"[A-Za-z_]\w*", prefix):
            raise error(
                "CSE temporary must be an identifier like cse$idx$ or tmp$idx$_value; "
                f"got '{prefix}$idx${suffix}'"
            )

        return {"use_cse": True, "cse_var": prefix, "cse_suffix": suffix}
