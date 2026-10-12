# ABOUTME: Placeholder substitution shared by the commands: $name$ and $name op N$
# ABOUTME: Also splits body lines into indentation and content the way the old engine did

import operator
import re
from typing import Any, Callable, Dict, Mapping, Tuple

_OPS: Dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def split_line(line: str) -> Tuple[str, str]:
    """Split a body line into its indentation and the rest, without the line end.

    Indentation is all leading spaces and tabs, kept as written (tabs matter in
    Python and Makefiles); the rest keeps its trailing whitespace.
    """
    text = line.rstrip("\r\n")
    content = text.lstrip(" \t")
    return text[: len(text) - len(content)], content


def substitute(text: str, values: Mapping[str, Callable[[], Any]]) -> str:
    """Replace ``$name$`` and ``$name op N$`` placeholders in *text*.

    Each value is computed only when its placeholder occurs. ``op`` is one of
    ``+ - * /`` and applies only when the value is an integer; otherwise the
    value is inserted unchanged.

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
        op = match.group(2)
        if op and isinstance(value, int):
            return str(_OPS[op](value, int(match.group(3))))

        return str(value)

    return pattern.sub(replace, text)
