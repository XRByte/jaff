# ABOUTME: Parsers for option-group values, which reach commands as raw strings
# ABOUTME: Each raises ValueError with a short reason; callers add the group and its line

from typing import Callable


def parse_bool(value: str) -> bool:
    """``True`` or ``False``, case-insensitively.

    Raises:
        ValueError: For any other text.
    """
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"

    raise ValueError(f"expects True or False, got {value!r}")


def parse_int(value: str) -> int:
    """A decimal integer.

    Raises:
        ValueError: If *value* is not an integer.
    """
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"expects an integer, got {value!r}") from None


def parse_text(value: str) -> str:
    """Any text, unchanged."""
    return value


def parse_choice(*choices: str) -> Callable[[str], str]:
    """A parser accepting one of *choices*, case-insensitively, returned lowercase.

    Args:
        choices: The accepted values.

    Returns:
        A parser raising ``ValueError`` for anything else.
    """
    options = tuple(choice.lower() for choice in choices)

    def parse(value: str) -> str:
        if value.lower() in options:
            return value.lower()
        raise ValueError(f"expects one of {', '.join(options)}, got {value!r}")

    return parse
