# ABOUTME: Placeholder substitution shared by the commands: $name$ and $name op N$
# ABOUTME: Also splits body lines into indentation (spaces and tabs) and content

import operator
import re
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

_OPS: Dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.floordiv,
}


def split_line(line: str) -> Tuple[str, str]:
    """Split a body line into its indentation and the rest, without the line end.

    Indentation is all leading spaces and tabs, kept as written (tabs matter in
    Python and Makefiles); the rest keeps its trailing whitespace.
    """
    text = line.rstrip("\r\n")
    content = text.lstrip(" \t")

    return text[: len(text) - len(content)], content


def arithmetic(value: Any, op: Optional[str], operand: int) -> Any:
    """``value op operand`` when *op* is given and *value* is an integer.

    ``op`` is one of ``+ - * /`` (``/`` is floor division); anything else comes
    back unchanged.
    """
    if op and isinstance(value, int):
        return _OPS[op](value, operand)

    return value


def substitute(text: str, values: Mapping[str, Callable[[], Any]]) -> str:
    """Replace ``$name$`` and ``$name op N$`` placeholders in *text*.

    Each value is computed only when its placeholder occurs. ``op`` is one of
    ``+ - * /`` (``/`` is floor division, so results stay integers) and applies
    only when the value is an integer; otherwise the value is inserted unchanged.

    Args:
        text: Text to substitute into.
        values: Placeholder name to a callable returning its value.

    Returns:
        *text* with every matching placeholder replaced by ``str(value)``.
    """
    if not values:
        return text

    names = "|".join(re.escape(name) for name in values)
    pattern = re.compile(rf"\$({names})(?:\s*([-+*/])\s*(\d+))?\s*\$")

    def replace(match: re.Match[str]) -> str:
        value = values[match.group(1)]()
        return str(arithmetic(value, match.group(2), int(match.group(3) or 0)))

    return pattern.sub(replace, text)
