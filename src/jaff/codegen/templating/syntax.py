# ABOUTME: Parses one $$JAFF directive header with Lark into typed header and option objects
# ABOUTME: Syntax only: no network, codegen or registry knowledge lives here

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from lark import Lark, Token, Transformer, UnexpectedInput, UnexpectedToken

from ...errors import ParserError

GRAMMAR = Path(__file__).with_name("grammar.lark")


@dataclass(frozen=True)
class RepeatHeader:
    vars: Tuple[str, ...]
    prop: str


@dataclass(frozen=True)
class SubHeader:
    tokens: Tuple[str, ...]


@dataclass(frozen=True)
class GetHeader:
    props: Tuple[str, ...]
    entity: str


@dataclass(frozen=True)
class HasHeader:
    kind: str
    entity: str


@dataclass(frozen=True)
class ReduceHeader:
    vars: Tuple[str, ...]
    props: Tuple[str, ...]


@dataclass(frozen=True)
class DefineHeader:
    name: str


@dataclass(frozen=True)
class EndHeader:
    pass


Header = Union[
    RepeatHeader, SubHeader, GetHeader, HasHeader, ReduceHeader, DefineHeader, EndHeader
]


@dataclass
class Options:
    """Options of one directive block, in the order they were written."""

    replace: List[Tuple[str, str]] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)
    groups: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Directive:
    header: Header
    options: Optional[Options]


def _text(token: Token) -> str:
    """Token text; double-quoted strings lose their quotes, escapes stay raw."""
    if token.type == "STRING":
        return token[1:-1]

    return str(token)


def _unescape(token: Token) -> str:
    """Token text with quotes removed and ``\\"`` turned into ``"``."""
    return _text(token).replace('\\"', '"')


# One raw option: (kind, value); groups carry (name, line, [(key, value, line)])
_Part = Tuple[str, Any]


class _ToDirective(Transformer):
    """Lark transformer turning the parse tree into a header and raw option parts."""

    def start(self, items: List[Any]) -> Tuple[Header, Optional[List[_Part]]]:
        return items[0], (items[1] if len(items) > 1 else None)

    def names(self, items: List[Any]) -> Tuple[str, ...]:
        return tuple(str(t) for t in items)

    def repeat(self, items: List[Any]) -> RepeatHeader:
        return RepeatHeader(items[0], str(items[1]))

    def sub(self, items: List[Any]) -> SubHeader:
        return SubHeader(items[0])

    def get(self, items: List[Any]) -> GetHeader:
        return GetHeader(items[0], _text(items[1]))

    def has(self, items: List[Any]) -> HasHeader:
        return HasHeader(str(items[0]), _text(items[1]))

    def reduce(self, items: List[Any]) -> ReduceHeader:
        return ReduceHeader(items[0], items[1])

    def define(self, items: List[Any]) -> DefineHeader:
        return DefineHeader(str(items[0]))

    def end(self, items: List[Any]) -> EndHeader:
        return EndHeader()

    def replacement(self, items: List[Any]) -> Tuple[str, str]:
        return (_text(items[0]), _unescape(items[1]))

    def setting(self, items: List[Any]) -> Tuple[str, str, int]:
        return (str(items[0]), _unescape(items[1]), items[0].line)

    def replace(self, items: List[Any]) -> _Part:
        return ("replace", items)

    def use(self, items: List[Any]) -> _Part:
        return ("use", str(items[0]))

    def group(self, items: List[Any]) -> _Part:
        return ("group", (str(items[0]), items[0].line, items[1:]))

    def options(self, items: List[Any]) -> List[_Part]:
        return items


@cache
def _parser() -> Lark:
    return Lark(GRAMMAR.read_text(), parser="lalr", propagate_positions=True)


class _OptionError(ValueError):
    """Option error carrying the 1-based line, relative to the header text."""

    def __init__(self, message: str, rel_line: int) -> None:
        super().__init__(message)
        self.rel_line = rel_line


def _add_group(options: Options, name: str, rel_line: int, settings: List[Any]) -> None:
    if name in options.groups:
        raise _OptionError(f"option group {name} given twice", rel_line)
    seen: Dict[str, int] = {}
    for key, _, key_line in settings:
        if key in seen:
            raise _OptionError(f"{name} sets {key} more than once", key_line)

        seen[key] = key_line
    options.groups[name] = {key: value for key, value, _ in settings}


def _build_options(parts: List[_Part]) -> Options:
    """Collect option parts; a repeated group or key in one block is an error."""
    options = Options()
    for kind, value in parts:
        if kind == "replace":
            options.replace.extend(value)
        elif kind == "use":
            options.uses.append(value)
        else:
            _add_group(options, *value)

    return options


def _terminal_label(name: str) -> str:
    """Terminal name; anonymous terminals (e.g. ``->``) show their literal text."""
    if not name.startswith("__ANON"):
        return name
    for terminal in _parser().terminals:
        if terminal.name == name:
            return f"'{terminal.pattern.value}'"

    return name


def _describe(e: UnexpectedInput) -> str:
    """Readable message for a Lark failure: what was found and what was expected."""
    if isinstance(e, UnexpectedToken):
        found = "end of header" if e.token.type == "$END" else f"'{e.token}'"
        names = e.expected
    else:
        found = f"'{getattr(e, 'char', '')}'"
        names = getattr(e, "allowed", None)

    message = f"Invalid directive: unexpected {found} at column {max(e.column, 1)}"
    if names:
        shown = sorted(_terminal_label(name) for name in names)[:8]
        message += f"; expected one of: {', '.join(shown)}"

    return message


class IncompleteHeaderError(ParserError):
    """The header ended inside an open option block; more lines may complete it."""


def parse_header(text: str, line: int, file: Path) -> Directive:
    """Parse the header text that follows ``$$JAFF``.

    Args:
        text: Header text with the ``$$JAFF`` marker removed; may span lines.
        line: 1-based template line on which the header starts.
        file: Template path, for error messages.

    Returns:
        The parsed directive.

    Raises:
        IncompleteHeaderError: If the text ends inside an open option block.
        ParserError: On a syntax error or a repeated option group or key.
    """
    lines = text.splitlines() or [""]

    def error(
        message: str, rel_line: int, cls: Type[ParserError] = ParserError
    ) -> ParserError:
        rel = rel_line if 0 < rel_line <= len(lines) else 1
        return cls(message, lines[rel - 1], line + rel - 1, file)

    try:
        tree = _parser().parse(text)
    except UnexpectedToken as e:
        incomplete = e.token.type == "$END" and "RBRACE" in e.expected
        cls = IncompleteHeaderError if incomplete else ParserError

        raise error(_describe(e), e.line, cls) from None
    except UnexpectedInput as e:
        raise error(_describe(e), e.line) from None

    header, parts = _ToDirective().transform(tree)
    if parts is None:
        return Directive(header, None)
    try:
        return Directive(header, _build_options(parts))
    except _OptionError as e:
        raise error(str(e), e.rel_line) from None
