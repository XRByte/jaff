# ABOUTME: REPEAT's line expansion: vertical (a line per item) or horizontal (inline list)
# ABOUTME: Pure functions over IndexedList items; callers add indentation and REPLACE

import re
from typing import Any, List, Optional, Pattern, Tuple

from .....types import IndexedList
from ...errors import TemplateError
from ...placeholders import arithmetic

# $idx$ or $idx op N$, written like any other placeholder
_IDX = re.compile(r"\$idx(?:\s*([-+*/])\s*(\d+))?\s*\$")

IdxToken = Tuple[Tuple[int, int], Optional[str], int]


def find_idx(line: str) -> List[IdxToken]:
    """Span, operator and operand of every ``$idx$`` / ``$idx op N$`` in *line*."""
    return [(m.span(), m.group(1), int(m.group(2) or 0)) for m in _IDX.finditer(line)]


def as_flattened(items: Any) -> IndexedList:
    """*items* as a flattened IndexedList: one entry per leaf, with all indices."""
    if not isinstance(items, IndexedList):
        return IndexedList(items, flatten=True)

    return items if items.type() == "flattened" else items.flatten()


def as_nested(items: Any) -> IndexedList:
    """*items* as a nested IndexedList, sub-lists kept as IndexedList values."""
    if not isinstance(items, IndexedList):
        return IndexedList(items, nested=True)

    return items if items.type() == "nested" else items.nested()


def expand_vertical(items: Any, line: str, token: str, index: int) -> List[str]:
    """One output line per item: ``$idx±N$`` take its indices, *token* its value.

    ``$IDX$`` takes the first index, through the first ``$idx$``'s arithmetic if
    the line has one. A line without ``$idx$`` is repeated without indices.

    Args:
        items: Iterable of ``(indices, value)`` pairs, e.g. a flattened IndexedList.
        line: The body line's stripped content.
        token: Placeholder for the value, e.g. ``"$specie$"``.
        index: Position of the line in the body, for errors.

    Returns:
        The content of each expanded line, without indentation or line end.

    Raises:
        TemplateError: If the line has ``$idx$`` placeholders, but not one per
            index of the items.
    """
    spans = find_idx(line)
    out = []
    for indices, value in items:
        if spans and len(indices) != len(spans):
            raise TemplateError(
                f"expected {len(indices)} $idx$ placeholders on this line",
                body_index=index,
            )

        text = line
        for ((begin, end), op, operand), position in reversed(list(zip(spans, indices))):
            text = text[:begin] + str(arithmetic(position, op, operand)) + text[end:]

        text = text.replace(token, str(value))
        first = arithmetic(indices[0], *spans[0][1:]) if spans else indices[0]
        text = text.replace("$IDX$", str(first))

        out.append(text)

    return out


def _horizontal_pattern(item: str) -> Pattern[str]:
    """``<bracket> <quote>$item$<quote> <separator> <bracket>``, e.g. ``["$x$", ]``.

    Brackets are ``( { < [`` and Fortran's ``(/``, closed by ``) } > ]`` or ``/)``.
    """
    return re.compile(
        rf"""
        (\(/|[\(\{{<\[])
        \s*
        (["']?)
        \${re.escape(item)}\$
        \2
        ([,\;\:\s]*)
        \s*
        (/\)|[\)\}}>\]])
        """,
        re.VERBOSE,
    )


def expand_horizontal(items: Any, item: str, line: str) -> Optional[str]:
    """Replace the line's bracketed ``$item$`` with every item, nested lists nested.

    Brackets, quotes and the separator come from the template; the separator
    defaults to ``", "``.

    Args:
        items: The property's items, a list or an IndexedList.
        item: The item variable, e.g. ``"specie"``.
        line: The body line's stripped content.

    Returns:
        The expanded line, or ``None`` if the line has no bracketed ``$item$``.
    """
    match = _horizontal_pattern(item).search(line)
    if not match:
        return None

    left, quote, separator, right = match.group(1, 2, 3, 4)
    separator = separator or ", "

    def build(level: IndexedList) -> str:
        parts = [
            build(entry.value)
            if isinstance(entry.value, IndexedList)
            else f"{quote}{entry.value}{quote}"
            for entry in level
        ]

        return left + separator.join(parts) + right

    begin, end = match.span()

    return line[:begin] + build(as_nested(items)) + line[end:]
