# ABOUTME: Single pass over a template that splits it into text and directive blocks
# ABOUTME: Lark decides where a header ends; the engine also collects the raw body up to $$JAFF END

from pathlib import Path
from typing import List, Tuple

from ...errors import ParserError
from .blocks import Block, CommandBlock, TextBlock
from .syntax import (
    DefineHeader,
    Directive,
    EndHeader,
    IncompleteHeaderError,
    parse_header,
)

MARKER = "$$JAFF"


def is_directive(line: str) -> bool:
    """Whether *line* starts a ``$$JAFF`` directive."""
    return line.lstrip().startswith(MARKER)


def _split_lines(text: str) -> List[str]:
    """Split on ``\\n`` only, keeping line ends (so ``\\r\\n`` stays attached)."""
    lines = [part + "\n" for part in text.split("\n")]
    lines[-1] = lines[-1][:-1]

    return lines if lines[-1] else lines[:-1]


def _src(line: str) -> str:
    """Source line without its trailing line break, for error messages."""
    return line.rstrip("\r\n")


def _read_header(lines: List[str], start: int, file: Path) -> Tuple[Directive, int]:
    """Parse the directive starting at *start*; return it and the next line index.

    The header continues onto the next line only while the parser reports that it
    ended inside an open option block.
    """
    text = lines[start].lstrip()[len(MARKER) :]
    end = start + 1
    while True:
        try:
            return parse_header(text, start + 1, file), end
        except IncompleteHeaderError:
            if end == len(lines) or is_directive(lines[end]):
                raise ParserError(
                    "Unclosed { in directive header", _src(lines[start]), start + 1, file
                ) from None

            text += lines[end]
            end += 1

        except ParserError as e:
            if end == start + 1 or e.nline == start + 1:
                raise

            hint = f" (in directive opened at line {start + 1}"
            hint += (
                "; missing '}'?)" if e.message.startswith("Invalid directive") else ")"
            )

            raise ParserError(e.message + hint, e.line, e.nline, e.fname) from None


def _read_body(
    lines: List[str], opened: int, start: int, file: Path
) -> Tuple[List[str], int]:
    """Collect body lines from *start* up to ``$$JAFF END``.

    *opened* is the index of the directive's first header line, for errors.
    Returns the body lines and the index after the END line.
    """
    i = start
    while i < len(lines):
        if is_directive(lines[i]):
            directive, after = _read_header(lines, i, file)
            if not isinstance(directive.header, EndHeader):
                raise ParserError(
                    "Nested $$JAFF directive (missing $$JAFF END?)",
                    _src(lines[i]),
                    i + 1,
                    file,
                )

            if directive.options is not None:
                raise ParserError(
                    "$$JAFF END takes no options", _src(lines[i]), i + 1, file
                )

            return lines[start:i], after

        i += 1

    raise ParserError("Missing $$JAFF END", _src(lines[opened]), opened + 1, file)


def split_blocks(text: str, file: Path) -> List[Block]:
    """Split a template into text and directive blocks in one pass.

    Args:
        text: Full template text.
        file: Template path, for error messages.

    Returns:
        Blocks in template order.

    Raises:
        ParserError: On a malformed directive or block structure.
    """
    lines = _split_lines(text)
    blocks: List[Block] = []
    i = 0

    while i < len(lines):
        if not is_directive(lines[i]):
            if blocks and isinstance(blocks[-1], TextBlock):
                blocks[-1].lines.append(lines[i])
            else:
                blocks.append(TextBlock([lines[i]], i + 1))

            i += 1
            continue

        directive, after = _read_header(lines, i, file)
        if isinstance(directive.header, EndHeader):
            raise ParserError(
                "$$JAFF END without a directive", _src(lines[i]), i + 1, file
            )

        body: List[str] = []
        end = after
        if not isinstance(directive.header, DefineHeader):
            body, end = _read_body(lines, i, after, file)

        blocks.append(CommandBlock(directive, lines[i:after], body, i + 1))
        i = end

    return blocks
