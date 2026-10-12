# ABOUTME: REPEAT's line expansion: vertical (a line per item) or horizontal (inline list)
# ABOUTME: Pure functions over IndexedList items, kept byte-compatible with the old engine

import re
from typing import Any, List, Optional, Pattern, Tuple

from ....types import IndexedList
from ..errors import TemplateError

# $idx$, $idx+N$ or $idx-N$
_IDX = re.compile(r"\$idx([+-]\d+)?\$")


def find_idx(line: str) -> List[Tuple[Tuple[int, int], int]]:
    """Span and offset of every ``$idx$``, ``$idx+N$`` or ``$idx-N$`` in *line*."""
    return [(m.span(), int(m.group(1) or 0)) for m in _IDX.finditer(line)]


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


def expand_vertical(items: Any, indent: str, line: str, token: str, index: int) -> str:
    """One output line per item: ``$idx±N$`` take its indices, *token* its value.

    ``$IDX$`` takes the first index plus the first ``$idx$``'s offset (none
    without one). A line without ``$idx$`` is repeated without indices.

    Args:
        items: Iterable of ``(indices, value)`` pairs, e.g. a flattened IndexedList.
        indent: Leading spaces of the body line.
        line: The body line's stripped content.
        token: Placeholder for the value, e.g. ``"$specie$"``.
        index: Position of the line in the body, for errors.

    Returns:
        The expanded lines, each with *indent* and ending in ``\\n``.

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
        for ((begin, end), offset), position in reversed(list(zip(spans, indices))):
            text = text[:begin] + str(position + offset) + text[end:]

        text = text.replace(token, str(value))
        first = indices[0] + (spans[0][1] if spans else 0)
        text = text.replace("$IDX$", str(first))

        out.append(f"{indent}{text}\n")

    return "".join(out)


def _horizontal_pattern(item: str) -> Pattern[str]:
    """``<bracket> <quote>$item$<quote> <separator> <bracket>``, e.g. ``["$x$", ]``."""
    return re.compile(
        rf"""
        ([\(\{{<\[])
        \s*
        (["']?)
        \${re.escape(item)}\$
        \2
        ([,\;\:\s]*)
        \s*
        ([\)\}}>\]])
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
